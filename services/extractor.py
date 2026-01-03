"""
Behavior extraction orchestrator.
Handles the workflow: raw prompt -> GPT extraction -> validated Pydantic models
"""

from typing import Dict, Any, List, Optional
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
from services.openAiClient import extract_behavior, embed_text, analyze_conflict
from services.credibilityCalculator import calculate_initial_credibility, should_store_behavior
from services.behaviorRepository import (
    insert_behavior, 
    insert_prompt_segment, 
    search_similar_behaviors, 
    reinforce_behavior,
    insert_conflict,
    supersede_behavior,
    update_behavior_state
)
from datetime import datetime
from config.configurations import (
    DEFAULT_DECAY_RATE,
    SAMPLE_USERID,
    SEMANTIC_RELEVANCE_THRESHOLD
)

import logging
logger = logging.getLogger(__name__)

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
        
        # Return successful extraction result
        return ExtractionResult(
            segments=validated_segments,
            success=True,
            error=None,
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    except KeyError as e:
        # Missing required field in response
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Invalid response structure: missing field {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    except Exception as e:
        # Pydantic validation error or other unexpected error
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Failed to validate extraction result: {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
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
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    canonical: CanonicalBehavior
) -> StoredBehavior:
    """Create a StoredBehavior object from extraction data."""
    return StoredBehavior(
        user_id=user_id,
        behavior_text=behavior_description,
        credibility=initial_credibility,
        clarity_score=clarity,
        extraction_confidence=confidence,
        linguistic_strength=linguistic_strength,
        decay_rate=DEFAULT_DECAY_RATE,
        embedding=embedding_vector,
        prompt_history_ids=[segment_id],
        session_id="default",
        intent=canonical.intent,
        target=canonical.target,
        context=canonical.context,
        polarity=canonical.polarity
    )


def _flag_and_create_conflict(
    existing_behavior_id: str,
    user_id: str,
    stored: StoredBehavior,
    similarity_distance: float,
    llm_explanation: str
) -> None:
    """Flag both behaviors and create a conflict record."""
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
        llm_analysis=llm_explanation
    )


def _supersede_existing_behavior(
    existing_behavior_id: str,
    user_id: str,
    stored: StoredBehavior
) -> None:
    """Insert new behavior as ACTIVE and supersede the existing one."""
    insert_behavior(stored.model_dump())
    supersede_behavior(
        behavior_id=existing_behavior_id,
        user_id=user_id,
        superseded_by_id=stored.behavior_id
    )


def _handle_llm_conflict_analysis(
    existing,
    behavior_description: str,
    user_id: str,
    initial_credibility: float,
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior]
) -> tuple[bool, bool]:
    """
    Handle LLM conflict analysis for ambiguous credibility scenarios.
    
    Returns:
        Tuple of (decision_taken, should_break)
    """
    try:
        conflict_analysis = analyze_conflict(
            behavior_1_text=existing.behavior_text,
            behavior_2_text=behavior_description,
            distance=existing.distance
        )
    except Exception as e:
        logger.error(f"Conflict analysis failed: {e}. Treating as COMPATIBLE.")
        return (False, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
        logger.info("LLM: compatible → insert new")
        return (False, True)

    stored = _create_stored_behavior(
        user_id=user_id,
        behavior_description=behavior_description,
        initial_credibility=initial_credibility,
        clarity=clarity,
        confidence=confidence,
        linguistic_strength=linguistic_strength,
        embedding_vector=embedding_vector,
        segment_id=segment_id,
        canonical=canonical
    )

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
        logger.info("LLM: context-dependent → flagging")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=conflict_analysis.explanation
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
            llm_explanation=conflict_analysis.explanation
        )
        stored_behaviors.append(stored)
        return (True, True)

    return (False, False)


def _handle_polarity_conflict(
    existing,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior]
) -> tuple[bool, bool]:
    """
    Handle polarity conflict (same target, different polarity).
    
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
            clarity=clarity,
            confidence=confidence,
            linguistic_strength=linguistic_strength,
            embedding_vector=embedding_vector,
            segment_id=segment_id,
            canonical=canonical
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
        return (True, True)

    # Case C: Ambiguous credibilities → LLM analysis needed
    elif resolution_type == "NEEDS_LLM":
        logger.info("AUTO-RESOLVE: Failed → calling LLM for conflict analysis")
        return _handle_llm_conflict_analysis(
            existing=existing,
            behavior_description=behavior_description,
            user_id=user_id,
            initial_credibility=initial_credibility,
            clarity=clarity,
            confidence=confidence,
            linguistic_strength=linguistic_strength,
            embedding_vector=embedding_vector,
            segment_id=segment_id,
            canonical=canonical,
            stored_behaviors=stored_behaviors
        )

    return (False, False)


def _handle_potential_conflict(
    existing,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    canonical: CanonicalBehavior,
    stored_behaviors: List[StoredBehavior]
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

    # First, get LLM analysis
    try:
        conflict_analysis = analyze_conflict(
            behavior_1_text=existing.behavior_text,
            behavior_2_text=behavior_description,
            distance=existing.distance
        )
        logger.info(f"Conflict analysis result: {conflict_analysis.conflict_type}")
    except Exception as e:
        logger.error(f"Conflict analysis failed: {e}. Treating as COMPATIBLE.")
        return (False, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
        logger.info("LLM: compatible → insert new")
        return (False, True)

    stored = _create_stored_behavior(
        user_id=user_id,
        behavior_description=behavior_description,
        initial_credibility=initial_credibility,
        clarity=clarity,
        confidence=confidence,
        linguistic_strength=linguistic_strength,
        embedding_vector=embedding_vector,
        segment_id=segment_id,
        canonical=canonical
    )

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
        logger.info("LLM: context-dependent → flagging both")
        _flag_and_create_conflict(
            existing_behavior_id=existing.behavior_id,
            user_id=user_id,
            stored=stored,
            similarity_distance=existing.distance,
            llm_explanation=conflict_analysis.explanation
        )
        stored_behaviors.append(stored)
        return (True, True)

    if conflict_analysis.conflict_type == ConflictAnalysisType.CONFLICT:
        logger.warning("LLM: CONFLICT confirmed → attempting auto-resolution")

        # Try credibility-based auto-resolution
        resolution_type, explanation = try_auto_resolve_conflict(
            existing_credibility=existing.credibility,
            new_credibility=initial_credibility
        )
        logger.info(f"Auto-resolution: {explanation}")

        # Case A: New behavior supersedes existing
        if resolution_type == "SUPERSEDE_EXISTING":
            logger.info(f"AUTO-RESOLVE: Superseding {existing.behavior_id} with new behavior")
            _supersede_existing_behavior(
                existing_behavior_id=existing.behavior_id,
                user_id=user_id,
                stored=stored
            )
            stored_behaviors.append(stored)
            return (True, True)

        # Case B: Existing behavior wins, ignore new
        elif resolution_type == "IGNORE_NEW":
            logger.info("AUTO-RESOLVE: Ignoring new behavior")
            return (True, True)

        # Case C: Ambiguous credibilities → flag for user decision
        elif resolution_type == "NEEDS_LLM":
            logger.warning("AUTO-RESOLVE: Failed → flagging both for user resolution")
            _flag_and_create_conflict(
                existing_behavior_id=existing.behavior_id,
                user_id=user_id,
                stored=stored,
                similarity_distance=existing.distance,
                llm_explanation=conflict_analysis.explanation
            )
            stored_behaviors.append(stored)
            return (True, True)

    return (False, False)


def _process_candidate_behavior(
    existing,
    canonical: CanonicalBehavior,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    stored_behaviors: List[StoredBehavior]
) -> tuple[bool, bool]:
    """
    Process a single candidate behavior against the new behavior.
    
    Returns:
        Tuple of (decision_taken, should_continue_to_next_candidate)
    """
    # Intent filter
    if existing.intent != canonical.intent:
        return (False, True)

    logger.info(f"INTENT MATCH with behavior {existing.behavior_id}")

    # Context relationship
    same_context, context_relation = contexts_match(
        existing.context or "general",
        canonical.context
    )

    # CASE 1: SAME TARGET
    if existing.target == canonical.target:
        # Polarity conflict
        if existing.polarity != canonical.polarity:
            decision_taken, should_break = _handle_polarity_conflict(
                existing=existing,
                user_id=user_id,
                behavior_description=behavior_description,
                initial_credibility=initial_credibility,
                clarity=clarity,
                confidence=confidence,
                linguistic_strength=linguistic_strength,
                embedding_vector=embedding_vector,
                segment_id=segment_id,
                canonical=canonical,
                stored_behaviors=stored_behaviors
            )
            if decision_taken:
                return (True, False)
            if should_break:
                return (False, False)

        # Same polarity → duplicate if context matches
        if same_context:
            logger.info(f"DUPLICATE ({context_relation}) → reinforcing {existing.behavior_id}")
            reinforce_behavior(
                behavior_id=existing.behavior_id,
                user_id=user_id,
                segment_id=segment_id
            )
            return (True, False)

        logger.info("RELATED (same target, different context) → keep both")
        return (False, True)

    # CASE 2: DIFFERENT TARGET + SAME CONTEXT
    if existing.target != canonical.target and same_context:
        decision_taken, should_break = _handle_potential_conflict(
            existing=existing,
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            clarity=clarity,
            confidence=confidence,
            linguistic_strength=linguistic_strength,
            embedding_vector=embedding_vector,
            segment_id=segment_id,
            canonical=canonical,
            stored_behaviors=stored_behaviors
        )
        if decision_taken:
            return (True, False)
        if should_break:
            return (False, False)

    # CASE 3: DIFFERENT TARGET + DIFFERENT CONTEXT
    logger.info("COMPATIBLE: same intent, different target & context")
    return (False, True)


def store_behavior(
    extraction_result: ExtractionResult,
    user_id: str = SAMPLE_USERID
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

    for segment in extraction_result.segments:
        segment_id: Optional[str] = None

        for behavior in segment.behaviors:
            logger.info(f"--- Processing behavior: '{behavior.description}' ---")

            # ==============================================================
            # 1️⃣ Credibility calculation & pruning
            # ==============================================================
            initial_credibility = calculate_initial_credibility(
                confidence=behavior.confidence,
                clarity=behavior.clarity,
                linguistic_strength=behavior.linguistic_strength,
                behavior_text=behavior.description
            )

            if not should_store_behavior(initial_credibility):
                logger.info(
                    f"PRUNE: '{behavior.description}' "
                    f"(credibility={initial_credibility:.3f})"
                )
                continue

            # ==============================================================
            # 2️⃣ Ensure prompt segment exists
            # ==============================================================
            if segment_id is None:
                segment_result = insert_prompt_segment(
                    segment_text=segment.text,
                    user_id=user_id
                )
                if not segment_result.success:
                    logger.error(
                        f"Failed to insert prompt segment: {segment_result.error}"
                    )
                    continue
                segment_id = segment_result.segment_id

            # ==============================================================
            # 3️⃣ Canonical behavior creation
            # ==============================================================
            canonical = create_canonical_behavior(behavior)
            if canonical is None:
                logger.warning(
                    f"SKIP: Missing canonical fields for "
                    f"'{behavior.description}'"
                )
                continue

            logger.info(
                f"CANONICAL: intent={canonical.intent}, "
                f"target={canonical.target}, "
                f"context={canonical.context}, "
                f"polarity={canonical.polarity}"
            )

            # ==============================================================
            # 4️⃣ Embed FULL behavior text (retrieval purpose)
            # ==============================================================
            try:
                embedding_vector = embed_text(behavior.description)
            except Exception as e:
                logger.error(
                    f"Embedding failed for '{behavior.description}': {e}"
                )
                continue

            # ==============================================================
            # 5️⃣ Retrieve candidate behaviors
            # ==============================================================
            try:
                candidates = search_similar_behaviors(
                    user_id=user_id,
                    query_embedding=embedding_vector,
                    limit=5
                )
                logger.info(
                    f"Retrieved {len(candidates)} candidate(s) for "
                    f"'{behavior.description} : {candidates}'"
                )
            except Exception as e:
                logger.error(f"Failed to search similar behaviors: {e}")
                candidates = []

            decision_taken = False

            # ==============================================================
            # 6️⃣ Semantic gate + canonical decision loop
            # ==============================================================
            for existing in candidates:
                # HARD SEMANTIC GATE
                if existing.distance > SEMANTIC_RELEVANCE_THRESHOLD:
                    logger.info(
                        f"Skipping candidate {existing.behavior_id} "
                        f"(distance={existing.distance:.3f}) — semantically unrelated"
                    )
                    continue

                logger.debug(
                    f"Checking candidate {existing.behavior_id}: "
                    f"intent={existing.intent}, target={existing.target}, "
                    f"context={existing.context}, polarity={existing.polarity}, "
                    f"distance={existing.distance:.3f}"
                )

                # Process this candidate
                decision_taken, should_continue = _process_candidate_behavior(
                    existing=existing,
                    canonical=canonical,
                    user_id=user_id,
                    behavior_description=behavior.description,
                    initial_credibility=initial_credibility,
                    clarity=behavior.clarity,
                    confidence=behavior.confidence,
                    linguistic_strength=behavior.linguistic_strength,
                    embedding_vector=embedding_vector,
                    segment_id=segment_id,
                    stored_behaviors=stored_behaviors
                )

                if decision_taken:
                    break
                if not should_continue:
                    break

            # ==============================================================
            # 7️⃣ FALLBACK → INSERT NEW BEHAVIOR
            # ==============================================================
            if decision_taken:
                logger.info("DECISION TAKEN → skipping insertion")
                continue

            logger.info("NO MATCH → inserting new behavior")

            stored = _create_stored_behavior(
                user_id=user_id,
                behavior_description=behavior.description,
                initial_credibility=initial_credibility,
                clarity=behavior.clarity,
                confidence=behavior.confidence,
                linguistic_strength=behavior.linguistic_strength,
                embedding_vector=embedding_vector,
                segment_id=segment_id,
                canonical=canonical
            )

            try:
                insert_behavior(stored.model_dump())
                stored_behaviors.append(stored)
            except Exception as e:
                logger.error(f"Failed to insert new behavior: {e}")

    logger.info(f"store_behavior complete: {len(stored_behaviors)} behaviors stored")
    return stored_behaviors


def store_behavior_with_tracking(
    extraction_result: ExtractionResult,
    user_id: str = SAMPLE_USERID
) -> DetailedExtractionResult:
    """
    Store extracted behaviors with detailed flow tracking for UI display.
    This wraps store_behavior and tracks what happens to each behavior.
    """
    if not extraction_result.success:
        return DetailedExtractionResult(
            extraction_result=extraction_result,
            flow_info=[],
            total_extracted=0,
            total_stored=0,
            total_reinforced=0,
            total_conflicts=0,
            total_pruned=0
        )
    
    flow_tracking: List[BehaviorFlowInfo] = []
    stored_behaviors: List[StoredBehavior] = []
    
    total_extracted = 0
    total_stored = 0
    total_reinforced = 0
    total_conflicts = 0
    total_pruned = 0
    
    for segment in extraction_result.segments:
        segment_id: Optional[str] = None
        
        for behavior in segment.behaviors:
            total_extracted += 1
            logger.info(f"--- Processing behavior: '{behavior.description}' ---")
            
            # 1️⃣ Credibility calculation & pruning
            initial_credibility = calculate_initial_credibility(
                confidence=behavior.confidence,
                clarity=behavior.clarity,
                linguistic_strength=behavior.linguistic_strength,
                behavior_text=behavior.description
            )
            
            if not should_store_behavior(initial_credibility):
                logger.info(f"PRUNE: '{behavior.description}' (credibility={initial_credibility:.3f})")
                flow_tracking.append(BehaviorFlowInfo(
                    behavior_description=behavior.description,
                    action=BehaviorFlowAction.PRUNED,
                    credibility=initial_credibility,
                    canonical={
                        "intent": behavior.intent,
                        "target": behavior.target,
                        "context": behavior.context,
                        "polarity": behavior.polarity
                    } if behavior.intent else None,
                    details=f"Credibility score ({initial_credibility:.3f}) below threshold"
                ))
                total_pruned += 1
                continue
            
            # 2️⃣ Ensure prompt segment exists
            if segment_id is None:
                segment_result = insert_prompt_segment(
                    segment_text=segment.text,
                    user_id=user_id
                )
                if not segment_result.success:
                    logger.error(f"Failed to insert prompt segment: {segment_result.error}")
                    continue
                segment_id = segment_result.segment_id
            
            # 3️⃣ Canonical behavior creation
            canonical = create_canonical_behavior(behavior)
            if canonical is None:
                logger.warning(f"SKIP: Missing canonical fields for '{behavior.description}'")
                flow_tracking.append(BehaviorFlowInfo(
                    behavior_description=behavior.description,
                    action=BehaviorFlowAction.PRUNED,
                    credibility=initial_credibility,
                    canonical=None,
                    details="Missing required canonical fields (intent, target, or polarity)"
                ))
                total_pruned += 1
                continue
            
            logger.info(
                f"CANONICAL: intent={canonical.intent}, target={canonical.target}, "
                f"context={canonical.context}, polarity={canonical.polarity}"
            )
            
            # 4️⃣ Embed FULL behavior text
            try:
                embedding_vector = embed_text(behavior.description)
            except Exception as e:
                logger.error(f"Embedding failed for '{behavior.description}': {e}")
                flow_tracking.append(BehaviorFlowInfo(
                    behavior_description=behavior.description,
                    action=BehaviorFlowAction.PRUNED,
                    credibility=initial_credibility,
                    canonical={
                        "intent": canonical.intent,
                        "target": canonical.target,
                        "context": canonical.context,
                        "polarity": canonical.polarity
                    },
                    details=f"Embedding generation failed: {str(e)}"
                ))
                total_pruned += 1
                continue
            
            # 5️⃣ Retrieve candidate behaviors
            try:
                candidates = search_similar_behaviors(
                    user_id=user_id,
                    query_embedding=embedding_vector,
                    limit=5
                )
                logger.info(f"Retrieved {len(candidates)} candidate(s)")
            except Exception as e:
                logger.error(f"Failed to search similar behaviors: {e}")
                candidates = []
            
            decision_taken = False
            flow_info = None
            
            # 6️⃣ Semantic gate + canonical decision loop
            for existing in candidates:
                # HARD SEMANTIC GATE
                if existing.distance > SEMANTIC_RELEVANCE_THRESHOLD:
                    logger.info(
                        f"Skipping candidate {existing.behavior_id} "
                        f"(distance={existing.distance:.3f}) — semantically unrelated"
                    )
                    continue
                
                logger.debug(
                    f"Checking candidate {existing.behavior_id}: intent={existing.intent}, "
                    f"target={existing.target}, distance={existing.distance:.3f}"
                )
                
                # Process candidate and track what happens
                decision_taken, flow_info = _process_candidate_with_tracking(
                    existing=existing,
                    canonical=canonical,
                    user_id=user_id,
                    behavior_description=behavior.description,
                    initial_credibility=initial_credibility,
                    clarity=behavior.clarity,
                    confidence=behavior.confidence,
                    linguistic_strength=behavior.linguistic_strength,
                    embedding_vector=embedding_vector,
                    segment_id=segment_id,
                    stored_behaviors=stored_behaviors
                )
                
                if decision_taken:
                    if flow_info:
                        flow_tracking.append(flow_info)
                        if flow_info.action == BehaviorFlowAction.DUPLICATE_REINFORCED:
                            total_reinforced += 1
                        elif flow_info.action in [
                            BehaviorFlowAction.CONFLICT_DETECTED,
                            BehaviorFlowAction.CONFLICT_AUTO_RESOLVED
                        ]:
                            total_conflicts += 1
                            if flow_info.stored_behavior_id:
                                total_stored += 1
                        elif flow_info.action == BehaviorFlowAction.SUPERSEDED_EXISTING:
                            total_stored += 1
                    break
            
            # 7️⃣ FALLBACK → INSERT NEW BEHAVIOR
            if not decision_taken:
                logger.info("NO MATCH → inserting new behavior")
                
                stored = _create_stored_behavior(
                    user_id=user_id,
                    behavior_description=behavior.description,
                    initial_credibility=initial_credibility,
                    clarity=behavior.clarity,
                    confidence=behavior.confidence,
                    linguistic_strength=behavior.linguistic_strength,
                    embedding_vector=embedding_vector,
                    segment_id=segment_id,
                    canonical=canonical
                )
                
                try:
                    insert_behavior(stored.model_dump())
                    stored_behaviors.append(stored)
                    total_stored += 1
                    
                    flow_tracking.append(BehaviorFlowInfo(
                        behavior_description=behavior.description,
                        action=BehaviorFlowAction.NEW_BEHAVIOR,
                        credibility=initial_credibility,
                        canonical={
                            "intent": canonical.intent,
                            "target": canonical.target,
                            "context": canonical.context,
                            "polarity": canonical.polarity
                        },
                        stored_behavior_id=stored.behavior_id,
                        details="No matching behavior found, created new entry"
                    ))
                except Exception as e:
                    logger.error(f"Failed to insert new behavior: {e}")
                    flow_tracking.append(BehaviorFlowInfo(
                        behavior_description=behavior.description,
                        action=BehaviorFlowAction.PRUNED,
                        credibility=initial_credibility,
                        canonical={
                            "intent": canonical.intent,
                            "target": canonical.target,
                            "context": canonical.context,
                            "polarity": canonical.polarity
                        },
                        details=f"Failed to insert into database: {str(e)}"
                    ))
                    total_pruned += 1
    
    logger.info(
        f"store_behavior_with_tracking complete: {total_stored} stored, "
        f"{total_reinforced} reinforced, {total_conflicts} conflicts, {total_pruned} pruned"
    )
    
    return DetailedExtractionResult(
        extraction_result=extraction_result,
        flow_info=flow_tracking,
        total_extracted=total_extracted,
        total_stored=total_stored,
        total_reinforced=total_reinforced,
        total_conflicts=total_conflicts,
        total_pruned=total_pruned
    )


def _process_candidate_with_tracking(
    existing,
    canonical: CanonicalBehavior,
    user_id: str,
    behavior_description: str,
    initial_credibility: float,
    clarity: float,
    confidence: float,
    linguistic_strength: float,
    embedding_vector: List[float],
    segment_id: str,
    stored_behaviors: List[StoredBehavior]
) -> tuple[bool, Optional[BehaviorFlowInfo]]:
    """
    Process a single candidate behavior with flow tracking.
    Returns (decision_taken, flow_info)
    """
    # Intent filter
    if existing.intent != canonical.intent:
        return (False, None)
    
    logger.info(f"INTENT MATCH with behavior {existing.behavior_id}")
    
    # Context relationship
    same_context, context_relation = contexts_match(
        existing.context or "general",
        canonical.context
    )
    
    canonical_dict = {
        "intent": canonical.intent,
        "target": canonical.target,
        "context": canonical.context,
        "polarity": canonical.polarity
    }
    
    # CASE 1: SAME TARGET
    if existing.target == canonical.target:
        # Polarity conflict
        if existing.polarity != canonical.polarity:
            resolution_type, explanation = try_auto_resolve_conflict(
                existing_credibility=existing.credibility,
                new_credibility=initial_credibility
            )
            
            if resolution_type == "SUPERSEDE_EXISTING":
                stored = _create_stored_behavior(
                    user_id=user_id,
                    behavior_description=behavior_description,
                    initial_credibility=initial_credibility,
                    clarity=clarity,
                    confidence=confidence,
                    linguistic_strength=linguistic_strength,
                    embedding_vector=embedding_vector,
                    segment_id=segment_id,
                    canonical=canonical
                )
                _supersede_existing_behavior(
                    existing_behavior_id=existing.behavior_id,
                    user_id=user_id,
                    stored=stored
                )
                stored_behaviors.append(stored)
                
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.SUPERSEDED_EXISTING,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "POLARITY_CONFLICT",
                        "existing_polarity": existing.polarity,
                        "new_polarity": canonical.polarity,
                        "resolution": resolution_type,
                        "explanation": explanation
                    },
                    stored_behavior_id=stored.behavior_id,
                    details=f"Superseded existing behavior due to higher credibility ({initial_credibility:.3f} > {existing.credibility:.3f})"
                )
                return (True, flow_info)
            
            elif resolution_type == "IGNORE_NEW":
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.IGNORED_NEW,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "POLARITY_CONFLICT",
                        "existing_polarity": existing.polarity,
                        "new_polarity": canonical.polarity,
                        "resolution": resolution_type,
                        "explanation": explanation
                    },
                    details=f"Ignored new behavior, existing has higher credibility ({existing.credibility:.3f} > {initial_credibility:.3f})"
                )
                return (True, flow_info)
            
            elif resolution_type == "NEEDS_LLM":
                conflict_analysis = analyze_conflict(
                    behavior_1_text=existing.behavior_text,
                    behavior_2_text=behavior_description,
                    distance=existing.distance
                )
                
                stored = _create_stored_behavior(
                    user_id=user_id,
                    behavior_description=behavior_description,
                    initial_credibility=initial_credibility,
                    clarity=clarity,
                    confidence=confidence,
                    linguistic_strength=linguistic_strength,
                    embedding_vector=embedding_vector,
                    segment_id=segment_id,
                    canonical=canonical
                )
                
                _flag_and_create_conflict(
                    existing_behavior_id=existing.behavior_id,
                    user_id=user_id,
                    stored=stored,
                    similarity_distance=existing.distance,
                    llm_explanation=conflict_analysis.explanation
                )
                stored_behaviors.append(stored)
                
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.CONFLICT_DETECTED,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "POLARITY_CONFLICT",
                        "existing_polarity": existing.polarity,
                        "new_polarity": canonical.polarity,
                        "resolution": "USER_DECISION_NEEDED",
                        "llm_analysis": conflict_analysis.explanation,
                        "llm_confidence": conflict_analysis.confidence
                    },
                    stored_behavior_id=stored.behavior_id,
                    details="Conflict flagged for user decision (both behaviors have similar credibility)"
                )
                return (True, flow_info)
        
        # Same polarity → duplicate if context matches
        if same_context:
            logger.info(f"DUPLICATE ({context_relation}) → reinforcing {existing.behavior_id}")
            reinforce_behavior(
                behavior_id=existing.behavior_id,
                user_id=user_id,
                segment_id=segment_id
            )
            
            flow_info = BehaviorFlowInfo(
                behavior_description=behavior_description,
                action=BehaviorFlowAction.DUPLICATE_REINFORCED,
                credibility=initial_credibility,
                canonical=canonical_dict,
                matched_behavior_id=existing.behavior_id,
                matched_behavior_text=existing.behavior_text,
                distance=existing.distance,
                details=f"Reinforced existing behavior (context match: {context_relation})"
            )
            return (True, flow_info)
        
        # Same target, different context
        logger.info("RELATED (same target, different context) → keep both")
        return (False, None)
    
    # CASE 2: DIFFERENT TARGET + SAME CONTEXT
    if existing.target != canonical.target and same_context:
        try:
            conflict_analysis = analyze_conflict(
                behavior_1_text=existing.behavior_text,
                behavior_2_text=behavior_description,
                distance=existing.distance
            )
        except Exception as e:
            logger.error(f"Conflict analysis failed: {e}. Treating as COMPATIBLE.")
            return (False, None)
        
        if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
            logger.info("LLM: compatible → insert new")
            return (False, None)
        
        stored = _create_stored_behavior(
            user_id=user_id,
            behavior_description=behavior_description,
            initial_credibility=initial_credibility,
            clarity=clarity,
            confidence=confidence,
            linguistic_strength=linguistic_strength,
            embedding_vector=embedding_vector,
            segment_id=segment_id,
            canonical=canonical
        )
        
        if conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
            _flag_and_create_conflict(
                existing_behavior_id=existing.behavior_id,
                user_id=user_id,
                stored=stored,
                similarity_distance=existing.distance,
                llm_explanation=conflict_analysis.explanation
            )
            stored_behaviors.append(stored)
            
            flow_info = BehaviorFlowInfo(
                behavior_description=behavior_description,
                action=BehaviorFlowAction.CONFLICT_DETECTED,
                credibility=initial_credibility,
                canonical=canonical_dict,
                matched_behavior_id=existing.behavior_id,
                matched_behavior_text=existing.behavior_text,
                distance=existing.distance,
                conflict_info={
                    "conflict_type": "CONTEXT_DEPENDENT",
                    "llm_analysis": conflict_analysis.explanation,
                    "llm_confidence": conflict_analysis.confidence
                },
                stored_behavior_id=stored.behavior_id,
                details="Context-dependent conflict detected, both behaviors flagged"
            )
            return (True, flow_info)
        
        if conflict_analysis.conflict_type == ConflictAnalysisType.CONFLICT:
            resolution_type, explanation = try_auto_resolve_conflict(
                existing_credibility=existing.credibility,
                new_credibility=initial_credibility
            )
            
            if resolution_type == "SUPERSEDE_EXISTING":
                _supersede_existing_behavior(
                    existing_behavior_id=existing.behavior_id,
                    user_id=user_id,
                    stored=stored
                )
                stored_behaviors.append(stored)
                
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.SUPERSEDED_EXISTING,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "DIFFERENT_TARGET_CONFLICT",
                        "resolution": resolution_type,
                        "explanation": explanation,
                        "llm_analysis": conflict_analysis.explanation
                    },
                    stored_behavior_id=stored.behavior_id,
                    details="Superseded existing behavior due to higher credibility"
                )
                return (True, flow_info)
            
            elif resolution_type == "IGNORE_NEW":
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.IGNORED_NEW,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "DIFFERENT_TARGET_CONFLICT",
                        "resolution": resolution_type,
                        "explanation": explanation
                    },
                    details="Ignored new behavior, existing has higher credibility"
                )
                return (True, flow_info)
            
            elif resolution_type == "NEEDS_LLM":
                _flag_and_create_conflict(
                    existing_behavior_id=existing.behavior_id,
                    user_id=user_id,
                    stored=stored,
                    similarity_distance=existing.distance,
                    llm_explanation=conflict_analysis.explanation
                )
                stored_behaviors.append(stored)
                
                flow_info = BehaviorFlowInfo(
                    behavior_description=behavior_description,
                    action=BehaviorFlowAction.CONFLICT_DETECTED,
                    credibility=initial_credibility,
                    canonical=canonical_dict,
                    matched_behavior_id=existing.behavior_id,
                    matched_behavior_text=existing.behavior_text,
                    distance=existing.distance,
                    conflict_info={
                        "conflict_type": "DIFFERENT_TARGET_CONFLICT",
                        "resolution": "USER_DECISION_NEEDED",
                        "llm_analysis": conflict_analysis.explanation,
                        "llm_confidence": conflict_analysis.confidence
                    },
                    stored_behavior_id=stored.behavior_id,
                    details="Conflict flagged for user decision"
                )
                return (True, flow_info)
    
    # CASE 3: DIFFERENT TARGET + DIFFERENT CONTEXT
    logger.info("COMPATIBLE: same intent, different target & context")
    return (False, None)




    
