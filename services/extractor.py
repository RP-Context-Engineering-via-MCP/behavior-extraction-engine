"""
Behavior extraction orchestrator.
Handles the workflow: raw prompt -> GPT extraction -> validated Pydantic models
"""

from typing import List, Optional
from models.behavior import (
    ExtractionResult, 
    BehaviorSegment, 
    ExtractedBehavior, 
    StoredBehavior, 
    SimilarityClassification,
    ConflictAnalysisType,
    ConflictType,
    BehaviorState,
    CanonicalBehavior,
    BehaviorFlowAction,
    BehaviorFlowInfo,
    DetailedExtractionResult
)
from services.openAiClient import extract_behavior, embed_text, analyze_conflict, extract_behavior_with_history
from utils.embedding_utils import get_canonical_embedding, get_text_embedding
from utils.similarity_utils import cosine_distance
from services.credibilityCalculator import calculate_initial_credibility, should_store_behavior, get_decay_rate
from services.behaviorRepository import (
    insert_behavior,
    search_similar_behaviors,
    reinforce_behavior,
    insert_conflict,
    supersede_behavior,
    update_behavior_state,
    update_behavior_access_time,
    insert_co_occurrences_batch,
    insert_directed_pairs_batch,
    get_behavior_ids_by_session,
)
from services.profileSignalExtractor import ProfileSignalExtractor
from datetime import datetime
import time
from config.configurations import (
    DEFAULT_DECAY_RATE,
    SAMPLE_USERID,
    SEMANTIC_RELEVANCE_THRESHOLD,
    DECAY_GRACE_PERIOD_SECONDS,
    PARAPHRASE_DUPLICATE_DISTANCE,
    PARAPHRASE_TARGET_DISTANCE
)

import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Profile Signal Extractor instance for validating GPT-4 profile_signals output
_profile_signal_extractor = ProfileSignalExtractor()

# ==============================================================================
# TARGET NORMALIZATION (for duplicate detection)
# ==============================================================================
# Duplicate detection compares behavior targets with exact string equality,
# which misses paraphrases like "Dark Mode" / "the dark mode" / "dark-mode".
# This normalizer collapses common surface variations so equivalent targets
# compare equal, without softening the polarity / context / intent gates that
# the conflict pipeline relies on.
# ==============================================================================

def _normalize_target(target: Optional[str]) -> str:
    """Lowercase, strip articles/punctuation/separators for duplicate matching."""
    if not target:
        return ""
    t = target.lower().strip()
    t = t.replace("-", " ").replace("_", " ")
    for article in ("the ", "a ", "an "):
        if t.startswith(article):
            t = t[len(article):]
    t = t.rstrip(".,!?;:")
    return " ".join(t.split())


def _targets_are_paraphrases(
    existing_target: Optional[str],
    new_target: Optional[str]
) -> bool:
    """
    Confirm two targets are actual paraphrases before upgrading to DUPLICATE.

    The canonical-sentence distance can be tight purely because intent + context
    + polarity match — e.g. "fastapi" vs "asynchronous support" both render as
    "user POSITIVE PREFERENCE <target> in web framework" and end up clustered.
    This second gate embeds the targets in isolation and requires their cosine
    distance to also be tight, which true paraphrases like "dark mode" /
    "dark theme" pass and disjoint-but-related concepts fail.

    Returns False on any embedding failure (conservative: better to miss a
    paraphrase than to silently merge two distinct behaviors).
    """
    norm_existing = _normalize_target(existing_target)
    norm_new = _normalize_target(new_target)
    if not norm_existing or not norm_new:
        return False
    if norm_existing == norm_new:
        return True

    try:
        emb_existing = get_text_embedding(norm_existing)
        emb_new = get_text_embedding(norm_new)
        target_dist = cosine_distance(emb_existing, emb_new)
    except Exception as exc:
        logger.warning(
            f"Target paraphrase check failed for '{existing_target}' vs "
            f"'{new_target}': {exc}. Skipping paraphrase upgrade."
        )
        return False

    is_paraphrase = target_dist < PARAPHRASE_TARGET_DISTANCE
    logger.info(
        f"Target paraphrase check: '{norm_existing}' vs '{norm_new}' "
        f"target_distance={target_dist:.3f} threshold={PARAPHRASE_TARGET_DISTANCE} "
        f"→ {'paraphrase' if is_paraphrase else 'distinct concepts'}"
    )
    return is_paraphrase


# ==============================================================================
# INTENT CONFLICT RULES
# ==============================================================================
# Defines which intents can potentially conflict with each other
# CONSTRAINT is special: it can conflict with ANY intent
# ==============================================================================

INTENT_CONFLICT_MATRIX = {
    "PREFERENCE": {"PREFERENCE", "CONSTRAINT"},
    "SKILL": {"SKILL", "CONSTRAINT"},
    "HABIT": {"HABIT", "CONSTRAINT"},
    "CONSTRAINT": {"PREFERENCE", "SKILL", "HABIT", "CONSTRAINT", "COMMUNICATION"},  # Conflicts with everything
    "COMMUNICATION": {"COMMUNICATION", "CONSTRAINT"},
}


def can_intents_conflict(intent1: str, intent2: str) -> bool:
    """
    Determine if two intents can potentially conflict.
    
    Rules:
    - CONSTRAINT can conflict with ANY intent
    - Other intents only conflict with same intent type
    
    Args:
        intent1: First intent (from existing behavior)
        intent2: Second intent (from new behavior)
        
    Returns:
        True if the intents can conflict, False otherwise
    """
    if intent1 is None or intent2 is None:
        return False
    
    # Check if intent2 is in the conflict set for intent1
    conflict_set = INTENT_CONFLICT_MATRIX.get(intent1, set())
    return intent2 in conflict_set


class RelationType(str, Enum):
    """Types of relationships between behaviors"""
    DUPLICATE = "DUPLICATE"           # Same intent, target, polarity, context → reinforce
    POLARITY_CONFLICT = "POLARITY_CONFLICT"  # Same intent, target, different polarity
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"  # Same intent, different target, same context
    CROSS_INTENT_CONFLICT = "CROSS_INTENT_CONFLICT"  # CONSTRAINT vs other intent
    RELATED = "RELATED"               # Same intent & target, different context
    COMPATIBLE = "COMPATIBLE"         # No conflict, can coexist


@dataclass
class BehaviorRelation:
    """Represents a relationship between new behavior and existing behavior"""
    existing_behavior: any  # The existing behavior from DB
    relation_type: RelationType
    context_relation: str  # DUPLICATE, GENERALIZATION, SPECIALIZATION, DIFFERENT


def run_behavior_extraction(prompt: str) -> ExtractionResult:
    """
    Extract behaviors from a user prompt and return validated Pydantic models.
    
    Args:
        prompt: User's natural language prompt
        
    Returns:
        ExtractionResult object with segments, behaviors, and metadata
        
    Workflow:
        1. Call GPT-4 via openAiClient.extract_behavior()
        2. Validate response structure
        3. Convert to Pydantic models for type safety
        4. Return ExtractionResult
    """
    raw_response= extract_behavior(prompt)

    if not raw_response.get("success", False):
        return ExtractionResult(
            segments=[],
            success=False,
            error=raw_response.get("error", "Unknown extraction error"),
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    try:
        validated_segments = []
        
        for segment_data in raw_response.get("segments", []):
            segment_text = segment_data["text"]

            # Convert each behavior dict to ExtractedBehavior model
            validated_behaviors = []
            
            for behavior_data in segment_data.get("behaviors", []):
                # Pydantic will validate ranges and types automatically
                validated_behavior = ExtractedBehavior(
                    description=behavior_data["description"],
                    confidence=behavior_data["confidence"],
                    clarity=behavior_data["clarity"],
                    linguistic_strength=behavior_data["linguistic_strength"],
                    extracted_at=datetime.now().isoformat(),
                    # Canonical fields for structured reasoning
                    intent=behavior_data.get("intent"),
                    target=behavior_data.get("target"),
                    context=behavior_data.get("context", "general"),
                    polarity=behavior_data.get("polarity")
                )
                validated_behaviors.append(validated_behavior)
            
            # Create validated segment with validated behaviors only
            validated_segment = BehaviorSegment(
                text=segment_text,
                behaviors=validated_behaviors
            )
            validated_segments.append(validated_segment)
        
        # Validate and extract profile_signals for Profile Service integration
        validated_profile_signals = None
        raw_profile_signals = raw_response.get("profile_signals")
        if raw_profile_signals:
            try:
                validated_profile_signals = _profile_signal_extractor.parse_and_validate(raw_profile_signals)
                logger.info(
                    f"Validated profile_signals: behavior_level={validated_profile_signals.get('behavior_level')}, "
                    f"intents={list(validated_profile_signals.get('intents', {}).keys())}, "
                    f"interests={list(validated_profile_signals.get('interests', {}).keys())}"
                )
            except ValueError as e:
                logger.error(f"Profile signals validation failed: {e}. Raw data: {raw_profile_signals}")
                # Continue without profile_signals - not critical for extraction
        else:
            logger.warning(
                "No profile_signals in GPT response. Profile Service integration will not be triggered. "
                "This may be because the GPT prompt did not generate profile_signals, or the user's prompt "
                "did not contain enough information to generate a behavioral profile."
            )
        
        # Return successful extraction result
        return ExtractionResult(
            segments=validated_segments,
            success=True,
            error=None,
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
            profile_signals=validated_profile_signals
        )
    
    except KeyError as e:
        # Missing required field in response
        logger.error(f"Invalid response structure: missing field {str(e)}")
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Invalid response structure: missing field {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    except Exception as e:
        # Pydantic validation error or other unexpected error
        logger.error(f"Failed to validate extraction result: {str(e)}")
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Failed to validate extraction result: {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )


def run_behavior_extraction_with_history(prompt: str, recent_history: List[dict]) -> ExtractionResult:
    """
    Extract behaviors + retrieval probes from a user prompt with conversation history.

    Single LLM call produces:
      - long-term behaviors extracted from the LATEST PROMPT only
      - 1..3 probes (text + canonical structural form) used for HMBR retrieval
      - query_type tag used by HMBR Pillar 3 to choose fusion weights
      - required_intents hint
      - profile_signals for Profile Service integration
    """
    logger.info("sent for behavior extraction")
    raw_response = extract_behavior_with_history(prompt, recent_history)
    logger.info("sent for behavior extraction")
    
    if not raw_response.get("success", False):
        return ExtractionResult(
            segments=[],
            success=False,
            error=raw_response.get("error", "Unknown extraction error"),
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
        )

    try:
        from models.behavior import ProbeSet, ProbeCanonical

        validated_segments = []
        for segment_data in raw_response.get("segments", []):
            segment_text = segment_data["text"]
            validated_behaviors = []
            for behavior_data in segment_data.get("behaviors", []):
                validated_behavior = ExtractedBehavior(
                    description=behavior_data["description"],
                    confidence=behavior_data["confidence"],
                    clarity=behavior_data["clarity"],
                    linguistic_strength=behavior_data["linguistic_strength"],
                    extracted_at=datetime.now().isoformat(),
                    intent=behavior_data.get("intent"),
                    target=behavior_data.get("target"),
                    context=behavior_data.get("context", "general"),
                    polarity=behavior_data.get("polarity"),
                )
                validated_behaviors.append(validated_behavior)
            validated_segments.append(BehaviorSegment(text=segment_text, behaviors=validated_behaviors))

        # ---------- probes (text + canonical) ---------------------------------
        raw_probes = raw_response.get("probes") or []
        probes: List[ProbeSet] = []
        for p in raw_probes:
            if not isinstance(p, dict):
                continue
            text = p.get("text")
            canonical = p.get("canonical") or {}
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                probes.append(ProbeSet(
                    text=text.strip(),
                    canonical=ProbeCanonical(
                        intent=canonical.get("intent", "PREFERENCE"),
                        target=(canonical.get("target") or text.strip())[:200],
                        context=canonical.get("context", "general") or "general",
                        polarity=canonical.get("polarity", "POSITIVE"),
                    ),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed probe {p}: {e}")

        if not probes:
            logger.error("GPT did not return usable probes — retrieval cannot proceed")
            return ExtractionResult(
                segments=[],
                success=False,
                error="Extraction failed: no usable probes returned",
                extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
            )

        # ---------- query_type ------------------------------------------------
        query_type = raw_response.get("query_type", "BROAD")
        if query_type not in {"NARROW", "BROAD", "EXPLORATORY", "TASK", "RECALL"}:
            query_type = "BROAD"

        # ---------- required_intents ------------------------------------------
        required_intents = raw_response.get("required_intents")
        if not required_intents or not isinstance(required_intents, list):
            required_intents = ["PREFERENCE", "CONSTRAINT"]

        # ---------- profile_signals -------------------------------------------
        validated_profile_signals = None
        raw_profile_signals = raw_response.get("profile_signals")
        if raw_profile_signals:
            try:
                validated_profile_signals = _profile_signal_extractor.parse_and_validate(raw_profile_signals)
            except ValueError as e:
                logger.error(f"Profile signals validation failed: {e}")

        return ExtractionResult(
            segments=validated_segments,
            success=True,
            error=None,
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
            probes=probes,
            query_type=query_type,
            required_intents=required_intents,
            profile_signals=validated_profile_signals,
        )

    except KeyError as e:
        logger.error(f"Invalid response structure: missing field {str(e)}")
        return ExtractionResult(
            segments=[], success=False,
            error=f"Invalid response structure: missing field {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to validate extraction result: {e}")
        return ExtractionResult(
            segments=[], success=False,
            error=f"Failed to validate extraction result: {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0),
        )


def create_canonical_behavior(extracted: ExtractedBehavior) -> Optional[CanonicalBehavior]:
    """
    Create a CanonicalBehavior from ExtractedBehavior for structured reasoning.
    Args: extracted: ExtractedBehavior with canonical fields populated
    Returns: CanonicalBehavior if all required fields present, None otherwise
    """
    # Validate required canonical fields are present
    if not all([extracted.intent, extracted.target, extracted.polarity]):
        logger.warning(
            f"Cannot create CanonicalBehavior - missing required fields: "
            f"intent={extracted.intent}, target={extracted.target}, polarity={extracted.polarity}"
        )
        return None
    
    try:
        return CanonicalBehavior(
            intent=extracted.intent,
            target=extracted.target.lower().strip() if extracted.target else "",
            context=(extracted.context or "general").lower().strip(),
            polarity=extracted.polarity,
            strength=extracted.linguistic_strength
        )
    except Exception as e:
        logger.error(f"Failed to create CanonicalBehavior: {e}")
        return None


def contexts_match(context1: str, context2: str) -> tuple[bool, str]:
    """
    Determine if two contexts represent the same scope or generalization/specialization.
    
    Args:
        context1: First context (from existing behavior)
        context2: Second context (from new behavior)
        
    Returns:
        Tuple of (is_duplicate, relationship_type)
        - is_duplicate: True if same context or general subsumes specific
        - relationship_type: "DUPLICATE", "GENERALIZATION", "SPECIALIZATION", or "DIFFERENT"
    
    Examples:
        contexts_match("general", "IDE") -> (True, "GENERALIZATION")  # general subsumes IDE
        contexts_match("IDE", "general") -> (True, "SPECIALIZATION")  # IDE specializes general  
        contexts_match("IDE", "IDE") -> (True, "DUPLICATE")
        contexts_match("frontend", "backend") -> (False, "DIFFERENT")
    """
    c1, c2 = context1.lower().strip(), context2.lower().strip()
    
    # Exact match
    if c1 == c2:
        return (True, "DUPLICATE")
    
    # General context subsumes all specific contexts
    if c1 == "general":
        return (True, "GENERALIZATION")  # context1 (general) is more general than context2 (specific)
    if c2 == "general":
        return (True, "SPECIALIZATION")  # context2 (general) is more general than context1 (specific)
    
    # Different specific contexts - both should exist
    return (False, "DIFFERENT")


def try_auto_resolve_conflict(
    existing_credibility: float,
    new_credibility: float
) -> tuple[Optional[str], str]:
    """
    Attempt to auto-resolve conflict based on credibility scores.
    
    Returns:
        Tuple of (resolution_type, explanation)
        - resolution_type: "SUPERSEDE_EXISTING" | "IGNORE_NEW" | "NEEDS_LLM" | None
        - explanation: Human-readable explanation of the decision
    
    Resolution Rules:
        1. New wins if new_cred > existing_cred AND existing_cred < 0.5
        2. Existing wins if existing_cred > new_cred AND new_cred < 0.5
        3. LLM decides if both >= 0.5 OR both < 0.5
    """
    CREDIBILITY_THRESHOLD = 0.5
    
    # Both high confidence → LLM decides
    if existing_credibility >= CREDIBILITY_THRESHOLD and new_credibility >= CREDIBILITY_THRESHOLD:
        return (
            "NEEDS_LLM",
            f"Both behaviors have high credibility (existing={existing_credibility:.2f}, new={new_credibility:.2f}) → LLM analysis required"
        )
    
    # Both low confidence → LLM decides
    if existing_credibility < CREDIBILITY_THRESHOLD and new_credibility < CREDIBILITY_THRESHOLD:
        return (
            "NEEDS_LLM",
            f"Both behaviors have low credibility (existing={existing_credibility:.2f}, new={new_credibility:.2f}) → LLM analysis required"
        )
    
    # New behavior has higher credibility AND existing is low confidence → supersede
    if new_credibility > existing_credibility and existing_credibility < CREDIBILITY_THRESHOLD:
        return (
            "SUPERSEDE_EXISTING",
            f"New behavior has higher credibility ({new_credibility:.2f}) and existing is low confidence ({existing_credibility:.2f}) → superseding"
        )
    
    # Existing behavior has higher credibility AND new is low confidence → ignore new
    if existing_credibility > new_credibility and new_credibility < CREDIBILITY_THRESHOLD:
        return (
            "IGNORE_NEW",
            f"Existing behavior has higher credibility ({existing_credibility:.2f}) and new is low confidence ({new_credibility:.2f}) → ignoring new"
        )
    
    # Edge case: shouldn't reach here, but default to LLM
    return (
        "NEEDS_LLM",
        f"Ambiguous credibility relationship (existing={existing_credibility:.2f}, new={new_credibility:.2f}) → LLM analysis required"
    )


def _create_stored_behavior(
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    embedding_vector: List[float],
    canonical_embedding_vector: List[float],
    canonical: CanonicalBehavior,
    session_id: str = "default",
    extraction_confidence: Optional[float] = None,
    clarity_score: Optional[float] = None,
    linguistic_strength: Optional[float] = None,
) -> StoredBehavior:
    """Create a StoredBehavior object from extraction data with intent-based decay rate."""
    decay_rate = get_decay_rate(intent=canonical.intent)
    current_time = int(time.time())
    decay_starts_at = current_time + DECAY_GRACE_PERIOD_SECONDS

    return StoredBehavior(
        user_id=user_id,
        session_id=session_id,
        behavior_text=behavior_description,
        intent=canonical.intent,
        target=canonical.target,
        context=canonical.context,
        polarity=canonical.polarity,
        credibility=initial_credibility,
        decay_rate=decay_rate,
        embedding=embedding_vector,
        canonical_embedding=canonical_embedding_vector,
        created_at=current_time,
        last_seen_at=current_time,
        last_decay_applied_at=decay_starts_at,
        extraction_confidence=extraction_confidence,
        clarity_score=clarity_score,
        linguistic_strength=linguistic_strength,
    )


def _flag_and_create_conflict(
    existing_behavior_id: str,
    user_id: str,
    stored: StoredBehavior,
    similarity_distance: float,
    llm_explanation: str,
    old_polarity: Optional[str] = None,
    new_polarity: Optional[str] = None,
    old_target: Optional[str] = None,
    new_target: Optional[str] = None
) -> None:
    """Flag both behaviors and create a conflict record.
    
    Args:
        existing_behavior_id: ID of the existing behavior
        user_id: User ID
        stored: New StoredBehavior to insert
        similarity_distance: Distance between behaviors
        llm_explanation: LLM analysis of the conflict
        old_polarity: Polarity of existing behavior (for drift detection)
        new_polarity: Polarity of new behavior (for drift detection)
        old_target: Target of existing behavior (for drift detection)
        new_target: Target of new behavior (for drift detection)
    """
    update_behavior_state(
        behavior_id=existing_behavior_id,
        user_id=user_id,
        new_state=BehaviorState.FLAGGED
    )
    
    payload = stored.model_dump()
    payload["behavior_state"] = BehaviorState.FLAGGED.value
    insert_behavior(payload)
    
    insert_conflict(
        user_id=user_id,
        behavior_id_1=existing_behavior_id,
        behavior_id_2=stored.behavior_id,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=similarity_distance,
        llm_analysis=llm_explanation,
        old_polarity=old_polarity,
        new_polarity=new_polarity,
        old_target=old_target,
        new_target=new_target
    )


def _supersede_existing_behavior(
    existing_behavior_id: str,
    user_id: str,
    stored: StoredBehavior
) -> None:
    """Insert new behavior as ACTIVE and supersede the existing one."""
    insert_behavior(stored.model_dump())
    supersede_behavior(
        old_behavior_id=existing_behavior_id,
        new_behavior_id=stored.behavior_id,
        user_id=user_id
    )


def _handle_llm_conflict_analysis(
    existing,
    behavior_description: str,
    user_id: str,
    initial_credibility: float,
    embedding_vector: List[float],
    canonical_embedding_vector: List[float],
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior],
    session_id: str = "default",
    conflict_subtype: str = "POLARITY_CONFLICT",
    extraction_confidence: Optional[float] = None,
    clarity_score: Optional[float] = None,
    linguistic_strength: Optional[float] = None,
) -> tuple[bool, bool]:
    """
    Handle LLM conflict analysis for ambiguous credibility scenarios.

    Retries the LLM call once on failure. If both attempts fail, the behavior
    is flagged as CONTEXT_DEPENDENT (fail-safe: prefer flagging over silently
    inserting a potential duplicate/conflict).

    Args:
        conflict_subtype: Label prepended to llm_analysis for UI disambiguation
                          ("POLARITY_CONFLICT" or "CROSS_INTENT_CONFLICT").

    Returns:
        Tuple of (decision_taken, should_break)
    """
    conflict_analysis = None
    for attempt in range(2):
        try:
            conflict_analysis = analyze_conflict(
                behavior_1_text=existing.behavior_text,
                behavior_2_text=behavior_description,
                distance=existing.distance
            )
            break
        except Exception as e:
            logger.warning(
                f"Conflict analysis attempt {attempt + 1}/2 failed: {e}"
            )

    if conflict_analysis is None:
        # Both attempts failed — fail-safe: flag both behaviors for user review
        # rather than silently treating an unknown conflict as COMPATIBLE.
        logger.error(
            "Conflict analysis failed after retry. "
            "Flagging both behaviors for user review (fail-safe)."
        )
        stored = _create_stored_behavior(
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            embedding_vector=embedding_vector,
            canonical_embedding_vector=canonical_embedding_vector,
            canonical=canonical,
            session_id=session_id,
            extraction_confidence=extraction_confidence,
            clarity_score=clarity_score,
            linguistic_strength=linguistic_strength,
        )
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=f"[{conflict_subtype}] LLM analysis unavailable — flagged for manual review.",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
        logger.info("LLM: compatible → insert new")
        return (False, True)

    stored = _create_stored_behavior(
        user_id=user_id,
        behavior_description=behavior_description,
        initial_credibility=initial_credibility,
        embedding_vector=embedding_vector,
        canonical_embedding_vector=canonical_embedding_vector,
        canonical=canonical,
        session_id=session_id,
        extraction_confidence=extraction_confidence,
        clarity_score=clarity_score,
        linguistic_strength=linguistic_strength,
    )

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
        logger.info("LLM: context-dependent → flagging")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=f"[{conflict_subtype}] {conflict_analysis.explanation}",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONFLICT:
        logger.warning("LLM: CONFLICT confirmed → flagging both for user resolution")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=f"[{conflict_subtype}] {conflict_analysis.explanation}",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    return (False, False)


def _handle_polarity_conflict(
    existing,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    embedding_vector: List[float],
    canonical_embedding_vector: List[float],
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior],
    session_id: str = "default",
    conflict_subtype: str = "POLARITY_CONFLICT",
    extraction_confidence: Optional[float] = None,
    clarity_score: Optional[float] = None,
    linguistic_strength: Optional[float] = None,
) -> tuple[bool, bool]:
    """
    Handle polarity conflict (same target, different polarity).

    Args:
        conflict_subtype: Label passed to LLM handler for conflict record
                          disambiguation ("POLARITY_CONFLICT" or
                          "CROSS_INTENT_CONFLICT").

    Returns:
        Tuple of (decision_taken, should_break)
    """
    logger.warning(
        f"POLARITY CONFLICT (same target): "
        f"{existing.polarity} vs {canonical.polarity}"
    )

    # Try credibility-based auto-resolution FIRST
    resolution_type, explanation = try_auto_resolve_conflict(
        existing_credibility=existing.credibility,
        new_credibility=initial_credibility
    )
    logger.info(f"Auto-resolution attempt: {explanation}")

    # Case A: New behavior supersedes existing
    if resolution_type == "SUPERSEDE_EXISTING":
        logger.info(
            f"AUTO-RESOLVE: Superseding {existing.behavior_id} "
            f"with new behavior (credibility: {initial_credibility:.2f} > {existing.credibility:.2f})"
        )

        stored = _create_stored_behavior(
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            embedding_vector=embedding_vector,
            canonical_embedding_vector=canonical_embedding_vector,
            canonical=canonical,
            session_id=session_id,
            extraction_confidence=extraction_confidence,
            clarity_score=clarity_score,
            linguistic_strength=linguistic_strength,
        )

        _supersede_existing_behavior(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored
        )

        stored_behaviors.append(stored)
        return (True, True)

    # Case B: Existing behavior wins, ignore new
    elif resolution_type == "IGNORE_NEW":
        logger.info(
            f"AUTO-RESOLVE: Ignoring new behavior "
            f"(credibility: {initial_credibility:.2f} < {existing.credibility:.2f})"
        )
        update_behavior_access_time(existing.behavior_id, user_id)
        return (True, True)

    # Case C: Ambiguous credibilities → LLM analysis needed
    elif resolution_type == "NEEDS_LLM":
        logger.info("AUTO-RESOLVE: Failed → calling LLM for conflict analysis")
        return _handle_llm_conflict_analysis(
            existing=existing,
            behavior_description=behavior_description,
            user_id=user_id,
            initial_credibility=initial_credibility,
            embedding_vector=embedding_vector,
            canonical_embedding_vector=canonical_embedding_vector,
            canonical=canonical,
            stored_behaviors=stored_behaviors,
            session_id=session_id,
            conflict_subtype=conflict_subtype,
            extraction_confidence=extraction_confidence,
            clarity_score=clarity_score,
            linguistic_strength=linguistic_strength,
        )

    return (False, False)


def _handle_potential_conflict(
    existing,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    embedding_vector: List[float],
    canonical_embedding_vector: List[float],
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior],
    session_id: str = "default",
    extraction_confidence: Optional[float] = None,
    clarity_score: Optional[float] = None,
    linguistic_strength: Optional[float] = None,
) -> tuple[bool, bool]:
    """
    Handle potential conflict (different target, same context).
    
    Returns:
        Tuple of (decision_taken, should_break)
    """
    logger.warning(
        f"POTENTIAL CONFLICT: same intent & context, "
        f"different target ({existing.target} vs {canonical.target})"
    )

    # ------------------------------------------------------------------
    # Step 1: Cheap credibility pre-check BEFORE calling the LLM.
    # If one behavior clearly dominates (clear credibility gap straddling
    # the 0.5 threshold), resolve without spending an LLM call.
    # ------------------------------------------------------------------
    resolution_type, resolution_explanation = try_auto_resolve_conflict(
        existing_credibility=existing.credibility,
        new_credibility=initial_credibility
    )
    logger.info(f"Potential conflict credibility pre-check: {resolution_explanation}")

    if resolution_type == "SUPERSEDE_EXISTING":
        logger.info(
            f"POTENTIAL CONFLICT PRE-RESOLVE: Superseding {existing.behavior_id} "
            f"(credibility: {initial_credibility:.2f} > {existing.credibility:.2f})"
        )
        stored = _create_stored_behavior(
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            embedding_vector=embedding_vector,
            canonical_embedding_vector=canonical_embedding_vector,
            canonical=canonical,
            session_id=session_id,
            extraction_confidence=extraction_confidence,
            clarity_score=clarity_score,
            linguistic_strength=linguistic_strength,
        )
        _supersede_existing_behavior(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
        )
        stored_behaviors.append(stored)
        return (True, True)

    if resolution_type == "IGNORE_NEW":
        logger.info(
            f"POTENTIAL CONFLICT PRE-RESOLVE: Ignoring new behavior "
            f"(credibility: {initial_credibility:.2f} < {existing.credibility:.2f})"
        )
        update_behavior_access_time(existing.behavior_id, user_id)
        return (True, True)

    # ------------------------------------------------------------------
    # Step 2: Credibility is ambiguous → call LLM to determine whether
    # the two different targets are truly exclusive choices or compatible.
    # ------------------------------------------------------------------
    logger.info("POTENTIAL CONFLICT: credibility ambiguous → calling LLM")

    conflict_analysis = None
    for attempt in range(2):
        try:
            conflict_analysis = analyze_conflict(
                behavior_1_text=existing.behavior_text,
                behavior_2_text=behavior_description,
                distance=existing.distance
            )
            logger.info(f"Conflict analysis result: {conflict_analysis.conflict_type}")
            break
        except Exception as e:
            logger.warning(f"Conflict analysis attempt {attempt + 1}/2 failed: {e}")

    if conflict_analysis is None:
        # LLM unavailable after retry — fail-safe: flag for user review.
        logger.error(
            "Conflict analysis failed after retry (potential conflict). "
            "Flagging both behaviors for user review (fail-safe)."
        )
        stored = _create_stored_behavior(
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            embedding_vector=embedding_vector,
            canonical_embedding_vector=canonical_embedding_vector,
            canonical=canonical,
            session_id=session_id,
            extraction_confidence=extraction_confidence,
            clarity_score=clarity_score,
            linguistic_strength=linguistic_strength,
        )
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation="[POTENTIAL_CONFLICT] LLM analysis unavailable — flagged for manual review.",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
        logger.info("LLM: compatible → insert new")
        return (False, True)

    stored = _create_stored_behavior(
        user_id=user_id,
        behavior_description=behavior_description,
        initial_credibility=initial_credibility,
        embedding_vector=embedding_vector,
        canonical_embedding_vector=canonical_embedding_vector,
        canonical=canonical,
        session_id=session_id,
        extraction_confidence=extraction_confidence,
        clarity_score=clarity_score,
        linguistic_strength=linguistic_strength,
    )

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
        logger.info("LLM: context-dependent → flagging both")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=f"[POTENTIAL_CONFLICT] {conflict_analysis.explanation}",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONFLICT:
        logger.warning("LLM: CONFLICT confirmed → flagging both for user resolution")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=f"[POTENTIAL_CONFLICT] {conflict_analysis.explanation}",
            old_polarity=existing.polarity,
            new_polarity=canonical.polarity,
            old_target=existing.target,
            new_target=canonical.target
        )
        stored_behaviors.append(stored)
        return (True, True)

    return (False, False)


def classify_relationship(
    existing,
    canonical: CanonicalBehavior
) -> Optional[BehaviorRelation]:
    """
    Classify the relationship between an existing behavior and new canonical behavior.
    
    Uses intent conflict matrix to determine if behaviors can conflict.
    CONSTRAINT behaviors can conflict with ANY other intent.
    
    Args:
        existing: Existing behavior from database
        canonical: New behavior's canonical form
        
    Returns:
        BehaviorRelation if related, None if unrelated
    """
    # Check if intents can conflict using the conflict matrix
    intents_can_conflict = can_intents_conflict(existing.intent, canonical.intent)
    same_intent = existing.intent == canonical.intent
    
    # Get context relationship
    same_context, context_relation = contexts_match(
        existing.context or "general",
        canonical.context
    )
    
    same_target = _normalize_target(existing.target) == _normalize_target(canonical.target)
    same_polarity = existing.polarity == canonical.polarity
    
    # ==================================================================
    # CASE 1: SAME INTENT relationships (original logic preserved)
    # ==================================================================
    if same_intent:
        # SAME TARGET
        if same_target:
            if not same_polarity:
                # Only a genuine conflict when contexts match (same scope, or one
                # is "general" which subsumes all specific contexts).
                # Opposite polarity in *different* specific contexts is valid
                # coexistence — e.g. "likes Python for backend" vs "dislikes
                # Python for frontend" — and should be stored as RELATED.
                if same_context:
                    return BehaviorRelation(
                        existing_behavior=existing,
                        relation_type=RelationType.POLARITY_CONFLICT,
                        context_relation=context_relation
                    )
                # Different specific contexts → valid coexistence, not a conflict
                return BehaviorRelation(
                    existing_behavior=existing,
                    relation_type=RelationType.RELATED,
                    context_relation=context_relation
                )
            # Same polarity + same context = DUPLICATE
            if same_context:
                return BehaviorRelation(
                    existing_behavior=existing,
                    relation_type=RelationType.DUPLICATE,
                    context_relation=context_relation
                )
            # Same polarity + different context = RELATED
            return BehaviorRelation(
                existing_behavior=existing,
                relation_type=RelationType.RELATED,
                context_relation=context_relation
            )
        
        # DIFFERENT TARGET + SAME CONTEXT = POTENTIAL CONFLICT
        if not same_target and same_context:
            return BehaviorRelation(
                existing_behavior=existing,
                relation_type=RelationType.POTENTIAL_CONFLICT,
                context_relation=context_relation
            )
        
        # DIFFERENT TARGET + DIFFERENT CONTEXT = COMPATIBLE
        return BehaviorRelation(
            existing_behavior=existing,
            relation_type=RelationType.COMPATIBLE,
            context_relation=context_relation
        )
    
    # ==================================================================
    # CASE 2: CROSS-INTENT CONFLICTS (new logic for CONSTRAINT)
    # ==================================================================
    if intents_can_conflict and not same_intent:
        # Only check for conflict if there's semantic + target overlap
        # This prevents false conflicts between unrelated CONSTRAINT behaviors
        
        # If same target with opposite polarity → CROSS_INTENT_CONFLICT
        if same_target and not same_polarity:
            logger.info(
                f"CROSS-INTENT CONFLICT detected: "
                f"{existing.intent} vs {canonical.intent} on target '{canonical.target}'"
            )
            return BehaviorRelation(
                existing_behavior=existing,
                relation_type=RelationType.CROSS_INTENT_CONFLICT,
                context_relation=context_relation
            )
        
        # If same target with same polarity in same context → could be reinforcing
        if same_target and same_polarity and same_context:
            # A CONSTRAINT supporting a PREFERENCE is not a conflict
            logger.info(
                f"CROSS-INTENT SUPPORT detected: "
                f"{existing.intent} and {canonical.intent} agree on target '{canonical.target}'"
            )
            return BehaviorRelation(
                existing_behavior=existing,
                relation_type=RelationType.COMPATIBLE,
                context_relation=context_relation
            )
    
    # No meaningful relationship
    return None


def _collect_all_relationships(
    candidates: List,
    canonical: CanonicalBehavior,
    semantic_threshold: float
) -> List[BehaviorRelation]:
    """
    Collect all relationships between new behavior and existing candidates.
    
    This replaces the single-match approach with comprehensive relationship detection.
    
    Args:
        candidates: List of candidate behaviors from semantic search
        canonical: New behavior's canonical form
        semantic_threshold: Maximum distance for semantic relevance
        
    Returns:
        List of all detected relationships, sorted by priority
    """
    relationships: List[BehaviorRelation] = []
    
    for existing in candidates:
        # HARD SEMANTIC GATE
        if existing.distance > semantic_threshold:
            logger.debug(
                f"Skipping candidate {existing.behavior_id} "
                f"(distance={existing.distance:.3f}) — semantically unrelated"
            )
            continue
        
        logger.debug(
            f"Classifying candidate {existing.behavior_id}: "
            f"intent={existing.intent}, target={existing.target}, "
            f"context={existing.context}, polarity={existing.polarity}, "
            f"distance={existing.distance:.3f}"
        )
        
        # Semantic-distance duplicate upgrade: catches paraphrases where exact
        # target strings differ (e.g. "dark mode" vs "dark theme", "VS Code"
        # vs "Visual Studio Code").  Runs before classify_relationship so it
        # takes priority when the embedding distance is very tight.
        #
        # Two-signal gate: the canonical-sentence distance alone is not
        # enough — it can be tight purely because intent/context/polarity
        # match (see "fastapi" vs "asynchronous support" in "web framework").
        # We additionally require the targets themselves to embed close.
        if (
            existing.distance < PARAPHRASE_DUPLICATE_DISTANCE
            and existing.intent == canonical.intent
            and existing.polarity == canonical.polarity
        ):
            _same_ctx, _ctx_rel = contexts_match(
                existing.context or "general",
                canonical.context
            )
            if _same_ctx and _targets_are_paraphrases(existing.target, canonical.target):
                logger.info(
                    f"PARAPHRASE DUPLICATE: '{existing.target}' ~ '{canonical.target}' "
                    f"(distance={existing.distance:.3f}) — upgrading to DUPLICATE"
                )
                relationships.append(BehaviorRelation(
                    existing_behavior=existing,
                    relation_type=RelationType.DUPLICATE,
                    context_relation=_ctx_rel
                ))
                continue

        relation = classify_relationship(existing, canonical)
        if relation is not None:
            logger.info(
                f"Found {relation.relation_type.value} relationship with "
                f"{existing.behavior_id}"
            )
            relationships.append(relation)
        elif can_intents_conflict(existing.intent, canonical.intent):
            # SEMANTIC FALLBACK: canonical string rules found no relationship (targets differ —
            # synonyms, parent/child, or paraphrases like "coffee" vs "caffeine",
            # "social events" vs "social gatherings", "late night" vs "sleep time").
            # The candidate is semantically close (passed the distance gate) AND the intents
            # can potentially conflict, so route to the LLM for a final verdict.
            logger.info(
                f"SEMANTIC FALLBACK: '{canonical.target}' vs '{existing.target}' — "
                f"no canonical rule matched but intents can conflict "
                f"({existing.intent}/{canonical.intent}), distance={existing.distance:.3f}. "
                f"Routing to LLM via POTENTIAL_CONFLICT."
            )
            relationships.append(BehaviorRelation(
                existing_behavior=existing,
                relation_type=RelationType.POTENTIAL_CONFLICT,
                context_relation="DIFFERENT"
            ))
    
    # Sort by priority: DUPLICATE > POLARITY_CONFLICT > CROSS_INTENT > POTENTIAL > RELATED > COMPATIBLE
    priority_order = {
        RelationType.DUPLICATE: 0,
        RelationType.POLARITY_CONFLICT: 1,
        RelationType.CROSS_INTENT_CONFLICT: 2,
        RelationType.POTENTIAL_CONFLICT: 3,
        RelationType.RELATED: 4,
        RelationType.COMPATIBLE: 5,
    }
    relationships.sort(key=lambda r: priority_order.get(r.relation_type, 99))
    
    return relationships


def _flag_existing_for_known_conflict(
    existing_behavior_id: str,
    new_behavior_id: str,
    user_id: str,
    similarity_distance: float,
    conflict_subtype: str,
    old_polarity: Optional[str] = None,
    new_polarity: Optional[str] = None,
    old_target: Optional[str] = None,
    new_target: Optional[str] = None
) -> None:
    """
    Flag an existing behavior against a new behavior already present in the DB.

    Called when multiple conflict candidates share the same incoming behavior —
    after the first conflict handler has already inserted the new behavior, all
    subsequent candidates must reuse that ID instead of creating duplicate copies
    of the incoming behavior.
    """
    update_behavior_state(
        behavior_id=existing_behavior_id,
        user_id=user_id,
        new_state=BehaviorState.FLAGGED
    )
    insert_conflict(
        user_id=user_id,
        behavior_id_1=existing_behavior_id,
        behavior_id_2=new_behavior_id,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=similarity_distance,
        llm_analysis=f"[{conflict_subtype}] Multiple conflicts detected for same incoming behavior.",
        old_polarity=old_polarity,
        new_polarity=new_polarity,
        old_target=old_target,
        new_target=new_target
    )


def _process_relationships(
    relationships: List[BehaviorRelation],
    canonical: CanonicalBehavior,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    embedding_vector: List[float],
    canonical_embedding_vector: List[float],
    stored_behaviors: List[StoredBehavior],
    session_id: str = "default",
    extraction_confidence: Optional[float] = None,
    clarity_score: Optional[float] = None,
    linguistic_strength: Optional[float] = None,
) -> bool:
    """
    Process all collected relationships and take appropriate actions.

    Priority handling:
    1. DUPLICATE → Reinforce and return (definitive action)
    2. POLARITY_CONFLICT → Handle conflict (may supersede, ignore, or flag)
    3. CROSS_INTENT_CONFLICT → Handle cross-intent conflict
    4. POTENTIAL_CONFLICT → Collect all, then handle
    5. RELATED/COMPATIBLE → No action needed

    Returns:
        True if a definitive action was taken (don't insert new behavior),
        False if new behavior should be inserted.
    """
    if not relationships:
        return False

    # Separate by type for comprehensive handling
    duplicates = [r for r in relationships if r.relation_type == RelationType.DUPLICATE]
    polarity_conflicts = [r for r in relationships if r.relation_type == RelationType.POLARITY_CONFLICT]
    cross_intent_conflicts = [r for r in relationships if r.relation_type == RelationType.CROSS_INTENT_CONFLICT]
    potential_conflicts = [r for r in relationships if r.relation_type == RelationType.POTENTIAL_CONFLICT]

    # Sentinel: captures stored_behaviors length before any conflict handler
    # runs.  Checked inside every conflict loop — if len(stored_behaviors) has
    # grown past this point, the new behavior is already in the DB and
    # subsequent handlers must reuse its ID rather than insert again.
    initial_stored_count = len(stored_behaviors)

    # ==================================================================
    # STEP 1: Handle DUPLICATES (highest priority)
    # ==================================================================
    if duplicates:
        if len(duplicates) > 1:
            # Multiple DUPLICATE candidates is an anomaly — the DB should never
            # hold two behaviors with the same (intent, target, context, polarity).
            # This most likely stems from a race condition in the async flow.
            # Reinforce only the highest-credibility representative; supersede the rest.
            dup_ids = [r.existing_behavior.behavior_id for r in duplicates]
            logger.warning(
                f"ANOMALY: {len(duplicates)} duplicate candidates for "
                f"'{behavior_description}'. IDs: {dup_ids}. "
                f"Reinforcing highest-credibility only; superseding the rest."
            )
            duplicates_sorted = sorted(
                duplicates,
                key=lambda r: r.existing_behavior.credibility,
                reverse=True
            )
            winner = duplicates_sorted[0].existing_behavior
            reinforce_behavior(behavior_id=winner.behavior_id, user_id=user_id)
            logger.info(
                f"DUPLICATE ANOMALY: Reinforced winner {winner.behavior_id} "
                f"(credibility={winner.credibility:.3f})"
            )
            for dup_rel in duplicates_sorted[1:]:
                loser = dup_rel.existing_behavior
                logger.warning(
                    f"DUPLICATE ANOMALY: Superseding {loser.behavior_id} "
                    f"(credibility={loser.credibility:.3f}) → winner {winner.behavior_id}"
                )
                supersede_behavior(
                    old_behavior_id=loser.behavior_id,
                    new_behavior_id=winner.behavior_id,
                    user_id=user_id,
                )
        else:
            existing = duplicates[0].existing_behavior
            logger.info(
                f"DUPLICATE ({duplicates[0].context_relation}) → reinforcing {existing.behavior_id}"
            )
            reinforce_behavior(behavior_id=existing.behavior_id, user_id=user_id)
        return True

    # ==================================================================
    # STEP 2: Handle POLARITY CONFLICTS
    # ==================================================================
    if polarity_conflicts:
        all_conflicts_resolved = True

        for conflict in polarity_conflicts:
            existing = conflict.existing_behavior

            # If the new behavior was already inserted by an earlier iteration
            # of this loop, don't create a second copy — flag the remaining
            # conflicting behaviors against the already-inserted one.
            if len(stored_behaviors) > initial_stored_count:
                inserted_id = stored_behaviors[initial_stored_count].behavior_id
                logger.warning(
                    f"POLARITY CONFLICT: new behavior already inserted as "
                    f"{inserted_id}; flagging existing {existing.behavior_id} "
                    f"against it (prevents duplicate insertion)."
                )
                _flag_existing_for_known_conflict(
                    existing_behavior_id=existing.behavior_id,
                    new_behavior_id=inserted_id,
                    user_id=user_id,
                    similarity_distance=existing.distance,
                    conflict_subtype="POLARITY_CONFLICT",
                    old_polarity=existing.polarity,
                    new_polarity=canonical.polarity,
                    old_target=existing.target,
                    new_target=canonical.target
                )
                continue

            decision_taken, _ = _handle_polarity_conflict(
                existing=existing,
                user_id=user_id,
                behavior_description=behavior_description,
                initial_credibility=initial_credibility,
                embedding_vector=embedding_vector,
                canonical_embedding_vector=canonical_embedding_vector,
                canonical=canonical,
                stored_behaviors=stored_behaviors,
                session_id=session_id,
                extraction_confidence=extraction_confidence,
                clarity_score=clarity_score,
                linguistic_strength=linguistic_strength,
            )
            if not decision_taken:
                all_conflicts_resolved = False

        if all_conflicts_resolved:
            return True

    # ==================================================================
    # STEP 3: Handle CROSS-INTENT CONFLICTS (e.g., CONSTRAINT vs PREFERENCE)
    # ==================================================================
    if cross_intent_conflicts:
        logger.warning(
            f"Processing {len(cross_intent_conflicts)} cross-intent conflict(s)"
        )
        all_cross_intent_resolved = True

        for conflict in cross_intent_conflicts:
            existing = conflict.existing_behavior
            logger.warning(
                f"CROSS-INTENT CONFLICT: {existing.intent} ('{existing.behavior_text}') "
                f"vs {canonical.intent} ('{behavior_description}')"
            )

            # Guard: new behavior may already be in DB from polarity or an
            # earlier iteration of this loop.
            if len(stored_behaviors) > initial_stored_count:
                inserted_id = stored_behaviors[initial_stored_count].behavior_id
                logger.warning(
                    f"CROSS-INTENT CONFLICT: new behavior already inserted as "
                    f"{inserted_id}; flagging existing {existing.behavior_id} against it."
                )
                _flag_existing_for_known_conflict(
                    existing_behavior_id=existing.behavior_id,
                    new_behavior_id=inserted_id,
                    user_id=user_id,
                    similarity_distance=existing.distance,
                    conflict_subtype="CROSS_INTENT_CONFLICT",
                    old_polarity=existing.polarity,
                    new_polarity=canonical.polarity,
                    old_target=existing.target,
                    new_target=canonical.target
                )
                continue

            decision_taken, _ = _handle_polarity_conflict(
                existing=existing,
                user_id=user_id,
                behavior_description=behavior_description,
                initial_credibility=initial_credibility,
                embedding_vector=embedding_vector,
                canonical_embedding_vector=canonical_embedding_vector,
                canonical=canonical,
                stored_behaviors=stored_behaviors,
                session_id=session_id,
                conflict_subtype="CROSS_INTENT_CONFLICT",
                extraction_confidence=extraction_confidence,
                clarity_score=clarity_score,
                linguistic_strength=linguistic_strength,
            )
            if not decision_taken:
                all_cross_intent_resolved = False

        if all_cross_intent_resolved:
            return True

    # ==================================================================
    # STEP 4: Handle POTENTIAL CONFLICTS (LLM-based analysis)
    # ==================================================================
    if potential_conflicts:
        logger.info(
            f"Processing {len(potential_conflicts)} potential conflict(s)"
        )
        all_potential_resolved = True

        for conflict in potential_conflicts:
            existing = conflict.existing_behavior

            # Guard: new behavior may already be in DB from an earlier conflict type
            # or earlier iteration of this loop.
            if len(stored_behaviors) > initial_stored_count:
                inserted_id = stored_behaviors[initial_stored_count].behavior_id
                logger.warning(
                    f"POTENTIAL CONFLICT: new behavior already inserted as "
                    f"{inserted_id}; flagging existing {existing.behavior_id} against it."
                )
                _flag_existing_for_known_conflict(
                    existing_behavior_id=existing.behavior_id,
                    new_behavior_id=inserted_id,
                    user_id=user_id,
                    similarity_distance=existing.distance,
                    conflict_subtype="POTENTIAL_CONFLICT",
                    old_polarity=existing.polarity,
                    new_polarity=canonical.polarity,
                    old_target=existing.target,
                    new_target=canonical.target
                )
                continue

            decision_taken, _ = _handle_potential_conflict(
                existing=existing,
                user_id=user_id,
                behavior_description=behavior_description,
                initial_credibility=initial_credibility,
                embedding_vector=embedding_vector,
                canonical_embedding_vector=canonical_embedding_vector,
                canonical=canonical,
                stored_behaviors=stored_behaviors,
                session_id=session_id,
                extraction_confidence=extraction_confidence,
                clarity_score=clarity_score,
                linguistic_strength=linguistic_strength,
            )
            if not decision_taken:
                all_potential_resolved = False

        if all_potential_resolved:
            return True

    # No definitive action taken → new behavior should be inserted
    return False


def store_behavior(
    extraction_result: ExtractionResult,
    user_id: str = SAMPLE_USERID,
    session_id: str = "default"
) -> List[StoredBehavior]:
    """
    Store extracted behaviors using CANONICAL reasoning with
    semantic gating to avoid unrelated conflicts.

    GUARANTEES:
    - At most ONE action per behavior
    - Embeddings are used ONLY for retrieval + relevance gating
    - Canonical rules decide duplicates, conflicts, compatibility
    """

    if not extraction_result.success:
        logger.warning("store_behavior called with failed extraction result")
        return []

    stored_behaviors: List[StoredBehavior] = []
    prompt_behavior_ids: List[str] = []   # All behavior IDs touched in this prompt (for graph edges)

    for segment in extraction_result.segments:
        for behavior in segment.behaviors:
            logger.info(f"--- Processing behavior: '{behavior.description}' ---")

            # 1. Credibility calculation & pruning
            initial_credibility = calculate_initial_credibility(
                confidence=behavior.confidence,
                clarity=behavior.clarity,
                linguistic_strength=behavior.linguistic_strength,
                behavior_text=behavior.description,
            )
            if not should_store_behavior(initial_credibility):
                logger.info(
                    f"PRUNE: '{behavior.description}' "
                    f"(credibility={initial_credibility:.3f})"
                )
                continue

            # 2. Canonical behavior creation
            canonical = create_canonical_behavior(behavior)
            if canonical is None:
                logger.warning(
                    f"SKIP: Missing canonical fields for '{behavior.description}'"
                )
                continue

            logger.info(
                f"CANONICAL: intent={canonical.intent}, target={canonical.target}, "
                f"context={canonical.context}, polarity={canonical.polarity}"
            )

            # 3. Generate dual embeddings
            try:
                canonical_embedding_vector = get_canonical_embedding(canonical)
            except Exception as e:
                logger.error(f"Canonical embedding failed for '{behavior.description}': {e}")
                continue

            try:
                embedding_vector = embed_text(behavior.description)
            except Exception as e:
                logger.error(f"Embedding failed for '{behavior.description}': {e}")
                continue

            # 4. Retrieve candidate behaviors for duplicate / conflict detection
            try:
                candidates = search_similar_behaviors(
                    user_id=user_id,
                    query_embedding=canonical_embedding_vector,
                    session_id=session_id,
                    limit=10,
                )
                logger.info(
                    f"Retrieved {len(candidates)} candidate(s) for '{behavior.description}'"
                )
            except Exception as e:
                logger.error(f"Failed to search similar behaviors: {e}")
                candidates = []

            # 5. Detect all relationships (DUPLICATE / POLARITY_CONFLICT / etc.)
            relationships = _collect_all_relationships(
                candidates=candidates,
                canonical=canonical,
                semantic_threshold=SEMANTIC_RELEVANCE_THRESHOLD,
            )
            logger.info(
                f"Found {len(relationships)} relationship(s) for '{behavior.description}': "
                f"{[r.relation_type.value for r in relationships]}"
            )

            # 6. Process all relationships (DUPLICATE > POLARITY > CROSS_INTENT > POTENTIAL)
            decision_taken = _process_relationships(
                relationships=relationships,
                canonical=canonical,
                user_id=user_id,
                behavior_description=behavior.description,
                initial_credibility=initial_credibility,
                embedding_vector=embedding_vector,
                canonical_embedding_vector=canonical_embedding_vector,
                stored_behaviors=stored_behaviors,
                session_id=session_id,
                extraction_confidence=behavior.confidence,
                clarity_score=behavior.clarity,
                linguistic_strength=behavior.linguistic_strength,
            )

            # 7. Fallback → insert new behavior
            if decision_taken:
                for rel in relationships:
                    if rel.relation_type == RelationType.DUPLICATE:
                        prompt_behavior_ids.append(rel.existing_behavior.behavior_id)
                logger.info("DECISION TAKEN → skipping insertion")
                continue

            logger.info("NO MATCH → inserting new behavior")

            stored = _create_stored_behavior(
                user_id=user_id,
                behavior_description=behavior.description,
                initial_credibility=initial_credibility,
                embedding_vector=embedding_vector,
                canonical_embedding_vector=canonical_embedding_vector,
                canonical=canonical,
                session_id=session_id,
                extraction_confidence=behavior.confidence,
                clarity_score=behavior.clarity,
                linguistic_strength=behavior.linguistic_strength,
            )

            try:
                insert_behavior(stored.model_dump())
                stored_behaviors.append(stored)
                prompt_behavior_ids.append(stored.behavior_id)
            except Exception as e:
                logger.error(f"Failed to insert new behavior: {e}")

    # Collect behavior IDs from conflict-handling paths (new behaviors
    # inserted by _handle_polarity_conflict / _handle_potential_conflict)
    for sb in stored_behaviors:
        if sb.behavior_id not in prompt_behavior_ids:
            prompt_behavior_ids.append(sb.behavior_id)

    # Create CO_PROMPT edges between all behaviors in this prompt
    if len(prompt_behavior_ids) >= 2:
        try:
            insert_co_occurrences_batch(
                behavior_ids=prompt_behavior_ids,
                user_id=user_id,
                session_id=session_id,
                edge_type="CO_PROMPT",
            )
        except Exception as e:
            logger.error(f"[GRAPH] Failed to create CO_PROMPT edges: {e}")

    # Create CO_SESSION edges between new behaviors (anchors) and pre-existing
    # session behaviors (partners).  Only anchor↔partner pairs are emitted —
    # partner↔partner pairs are NOT created here, because those existing
    # behaviors did not co-occur with each other in *this* prompt event.
    if prompt_behavior_ids and session_id:
        try:
            existing_session_ids = get_behavior_ids_by_session(
                user_id=user_id,
                session_id=session_id,
                exclude_ids=prompt_behavior_ids,
            )
            if existing_session_ids:
                insert_directed_pairs_batch(
                    anchor_ids=prompt_behavior_ids,
                    partner_ids=existing_session_ids,
                    user_id=user_id,
                    session_id=session_id,
                    edge_type="CO_SESSION",
                )
        except Exception as e:
            logger.error(f"[GRAPH] Failed to create CO_SESSION edges: {e}")

    logger.info(f"store_behavior complete: {len(stored_behaviors)} behaviors stored")
    return stored_behaviors

# ==============================================================================
# PROFILE SIGNALS DISPATCH (for Profile Service Integration)
# ==============================================================================

async def dispatch_profile_signals(
    user_id: str,
    prompt_id: str,
    profile_signals: Optional[dict]
) -> Optional[dict]:
    """
    Dispatch profile signals to the Profile Service for cold-start profiling.
    
    This function should be called after each extraction to:
    1. Save profile_signals locally (for drift fallback)
    2. Forward to Profile Service if user is in COLD_START mode
    
    Args:
        user_id: Unique user identifier
        prompt_id: Unique prompt/request identifier (e.g., UUID or segment_id)
        profile_signals: Validated profile signals from extraction result
        
    Returns:
        Profile Service response if dispatched, None otherwise
    """
    if not profile_signals:
        logger.debug(f"No profile_signals to dispatch for user={user_id}")
        return None
    
    try:
        from services.coldStartDispatcher import get_cold_start_dispatcher
        
        dispatcher = get_cold_start_dispatcher()
        result = await dispatcher.dispatch(user_id, prompt_id, profile_signals)
        
        if result:
            logger.info(
                f"Profile signals dispatched for user={user_id}: "
                f"status={result.get('status')}"
            )
        return result
        
    except Exception as e:
        logger.error(f"Failed to dispatch profile_signals for user={user_id}: {e}")
        return None


def dispatch_profile_signals_sync(
    user_id: str,
    prompt_id: str,
    profile_signals: Optional[dict]
) -> Optional[dict]:
    """
    Synchronous wrapper for dispatch_profile_signals.
    
    For use in synchronous contexts where async/await is not available.
    """
    import asyncio
    
    if not profile_signals:
        return None
    
    try:
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    dispatch_profile_signals(user_id, prompt_id, profile_signals)
                )
                return future.result(timeout=15)
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            return asyncio.run(
                dispatch_profile_signals(user_id, prompt_id, profile_signals)
            )
    except Exception as e:
        logger.error(f"Sync dispatch failed for user={user_id}: {e}")
        return None


def save_profile_signals_per_behavior(
    user_id: str,
    prompt_id: str,
    profile_signals: Optional[dict],
    stored_behaviors: List
) -> None:
    """
    Save profile signals for each stored behavior.
    
    This function links profile signals to specific behavior IDs, enabling
    the /api/behaviors/by-ids endpoint to return profile signals for
    specific behaviors.
    
    Args:
        user_id: Unique user identifier
        prompt_id: Unique prompt/request identifier
        profile_signals: Validated profile signals from extraction
        stored_behaviors: List of StoredBehavior objects with behavior_ids
    """
    if not profile_signals or not stored_behaviors:
        return
    
    try:
        from services.profileSignalRepository import get_profile_signal_repository
        
        signal_repo = get_profile_signal_repository()
        
        for behavior in stored_behaviors:
            try:
                # Save profile signals with behavior_id link
                signal_repo.save(
                    user_id=user_id,
                    prompt_id=f"{prompt_id}_{behavior.behavior_id}",  # Unique prompt_id per behavior
                    profile_signals=profile_signals,
                    behavior_id=behavior.behavior_id
                )
                logger.debug(
                    f"Saved profile_signals for behavior_id={behavior.behavior_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save profile_signals for behavior_id={behavior.behavior_id}: {e}"
                )
        
        logger.info(
            f"Saved profile_signals for {len(stored_behaviors)} behaviors "
            f"(user={user_id})"
        )
        
    except Exception as e:
        logger.error(
            f"Failed to save profile_signals per behavior for user={user_id}: {e}"
        )

    
