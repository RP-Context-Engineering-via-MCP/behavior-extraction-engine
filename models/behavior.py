from pydantic import BaseModel,Field,field_validator
from typing import List, Optional, Literal
from datetime import datetime, timezone
from enum import Enum
import time
import uuid

class BehaviorSegment(BaseModel):
    text: str = Field(
        ..., 
        description="Original prompt segment text")
    behaviors: List['ExtractedBehavior'] = Field(
        default_factory=list,
        description="List of extracted behaviors from the segment")
    


class ExtractedBehavior(BaseModel):
    description: str = Field(
        ...,
         description="Short behavior description")
    confidence: float = Field(
        ..., 
        ge=0.0,
        le=1.0, 
        description="Confidence score (0.0-1.0)")
    clarity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Clarity score (0.0-1.0)")
    linguistic_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Linguistic strength score (0.0-1.0) - how strongly the user expressed the behavior")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of extraction in UTC")
    
    # Canonical fields for structured reasoning
    intent: Optional[Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"]] = Field(
        None,
        description="Behavioral intent category")
    target: Optional[str] = Field(
        None,
        description="Primary object of behavior (concise noun)")
    context: Optional[str] = Field(
        None,
        description="Scope where behavior applies (IDE, frontend, morning, etc.)")
    polarity: Optional[Literal["POSITIVE", "NEGATIVE"]] = Field(
        None,
        description="Behavioral direction (likes vs dislikes)")
    


class ProbeCanonical(BaseModel):
    """
    Canonical structural fields the LLM emits per probe at extraction time.

    These let HMBR retrieval embed a structured form of the probe
    (e.g., "POSITIVE PREFERENCE Python in backend") and search the
    canonical_embedding column with like-for-like geometry.
    """
    intent: Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"] = Field(
        ..., description="Inferred intent of the probe — used for canonical embedding & intent affinity"
    )
    target: str = Field(..., description="Concise canonical noun the probe is about")
    context: str = Field(default="general", description="Scope where the probe applies")
    polarity: Literal["POSITIVE", "NEGATIVE"] = Field(
        ..., description="Whether the probe is asking about positive or negative preference"
    )


class ProbeSet(BaseModel):
    """One probe + its canonical structural form, both produced by the LLM in one call."""
    text: str = Field(..., description="Conversational probe (e.g., 'uses GitHub Actions for CI/CD')")
    canonical: ProbeCanonical = Field(..., description="Structured form for canonical-embedding lane")


class ExtractionResult(BaseModel):
    segments: List[BehaviorSegment] = Field(
        default_factory=list,
        description="Segment prompt with extracted behaviors"
    )
    success: bool = Field(
        ...,
        description="Indicates if extraction was successful"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if extraction failed"
    )
    extraction_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for extraction in milliseconds"
    )
    probes: List[ProbeSet] = Field(
        default_factory=list,
        description=(
            "1..3 retrieval probes, each with a conversational text form and a canonical "
            "structural form.  Replaces the older standalone_query/standalone_queries pair."
        )
    )
    required_intents: Optional[List[str]] = Field(
        default=None,
        description="LLM-predicted intent types relevant to the query (e.g., ['CONSTRAINT', 'PREFERENCE'])"
    )
    query_type: Literal["NARROW", "BROAD", "EXPLORATORY", "TASK", "RECALL"] = Field(
        default="BROAD",
        description=(
            "Coarse query classification used by HMBR Pillar 3 to choose fusion weights:\n"
            " NARROW: one specific topic — weight semantic+lexical heavily\n"
            " BROAD: multi-faceted — balanced fusion\n"
            " EXPLORATORY: 'what do I usually do?' — weight recency+credibility+graph\n"
            " TASK: 'how do I do X?' — weight canonical+intent+graph\n"
            " RECALL: 'did I say...?' — weight lexical+recency"
        )
    )
    profile_signals: Optional[dict] = Field(
        default=None,
        description="Profile signals extracted for Profile Service integration"
    )


class StoredBehavior(BaseModel):
    """Final object to be stored in the behaviors table."""

    behavior_id: str = Field(
        default_factory=lambda: f"beh_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the behavior"
    )
    user_id: str = Field(..., description="User identifier — behaviors are unique per user")
    session_id: str = Field(default="default", description="Session ID or 'default'")
    behavior_text: str = Field(..., description="Extracted behavior text")

    # Canonical structural fields
    intent: Optional[Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"]] = Field(
        None, description="Behavioral intent"
    )
    target: Optional[str] = Field(None, description="Concise target noun")
    context: Optional[str] = Field(None, description="Context scope")
    polarity: Optional[Literal["POSITIVE", "NEGATIVE"]] = Field(None, description="POSITIVE or NEGATIVE")

    # Credibility & lifecycle
    credibility: float = Field(..., description="Credibility score (0..1)")
    reinforcement_count: int = Field(default=1, ge=1)
    decay_rate: float = Field(default=0.015, ge=0.0, le=1.0)
    usefulness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Personalisation feedback signal — bumped up when retrieval led to reinforcement"
    )
    created_at: int = Field(default_factory=lambda: int(time.time()))
    last_seen_at: int = Field(default_factory=lambda: int(time.time()))
    last_decay_applied_at: Optional[int] = Field(default=None)
    last_accessed_at: Optional[int] = Field(default=None)

    # Extraction quality signals (stored for team analytics)
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM's confidence in the extraction (0..1)")
    clarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM's clarity/cleanness rating (0..1)")
    linguistic_strength: Optional[float] = Field(None, ge=0.0, le=1.0, description="Hedging intensity — 1.0 = definitive, 0.0 = very hedged")

    # Search lanes
    embedding: Optional[List[float]] = Field(None, description="Prose-text embedding (384-dim)")
    canonical_embedding: Optional[List[float]] = Field(None, description="Canonical structural embedding (384-dim)")

    @field_validator('embedding')
    def validate_embedding_dimension(cls, v):
        if v is not None and len(v) != 384:
            raise ValueError(f"Embedding must be 384-dimensional, got {len(v)}")
        return v

    @field_validator('canonical_embedding')
    def validate_canonical_embedding_dimension(cls, v):
        if v is not None and len(v) != 384:
            raise ValueError(f"Canonical embedding must be 384-dimensional, got {len(v)}")
        return v
    

class BehaviorState(str, Enum):
    """lifestyle states for behaviors"""
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    FLAGGED = "FLAGGED"
    ARCHIVED = "ARCHIVED"

class SimilarityClassification(str, Enum):
    """Classification of relationship between behaviors"""
    DUPLICATE = "DUPLICATE"           # 0.00-0.05: Exact match
    SIMILAR = "SIMILAR"               # 0.05-0.15: Related variations
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"  # 0.15-0.40: Might conflict
    UNRELATED = "UNRELATED"           # 0.40+: Different domains

class ConflictType(str, Enum):
    """Types of behavior conflicts"""
    RESOLVABLE = "RESOLVABLE"
    USER_DECISION_NEEDED = "USER_DECISION_NEEDED"

class ResolutionStatus(str, Enum):
    """Status of conflict resolution"""
    PENDING = "PENDING"
    AUTO_RESOLVED = "AUTO_RESOLVED"
    USER_RESOLVED = "USER_RESOLVED"
    EXPIRED = "EXPIRED"

class SimilarityResult(BaseModel):
    """Result of write-time similarity search (used for duplicate / conflict detection)."""
    behavior_id: str = Field(..., description="ID of the similar behavior found")
    behavior_text: str = Field(..., description="Text of the similar behavior")
    distance: float = Field(..., ge=0.0, description="Cosine distance (lower = more similar)")
    credibility: float = Field(..., ge=0.0, le=1.0)
    last_seen_at: int = Field(..., description="Timestamp when behavior was last reinforced")
    reinforcement_count: int = Field(..., ge=1)
    intent: Optional[Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"]] = Field(None)
    target: Optional[str] = Field(None)
    context: Optional[str] = Field(None)
    polarity: Optional[Literal["POSITIVE", "NEGATIVE"]] = Field(None)


class RetrievalRelationship(str, Enum):
    """Whether a retrieved behavior agrees or disagrees with the implied query polarity."""
    AGREES = "AGREES"
    DISAGREES = "DISAGREES"
    NEUTRAL = "NEUTRAL"


class RetrievedBehavior(BaseModel):
    """A behavior returned by HMBR retrieval, with all per-signal scores attached."""
    behavior_id: str
    behavior_text: str
    intent: Optional[str] = None
    target: Optional[str] = None
    context: Optional[str] = None
    polarity: Optional[str] = None
    credibility: float
    s_final: float = Field(..., description="Fused score after Pillar 3 weighting")
    s_semantic: float = 0.0
    s_canonical: float = 0.0
    s_lexical: float = 0.0
    s_recency: float = 0.0
    s_credibility: float = 0.0
    s_usefulness: float = 0.0
    s_ppr: float = 0.0
    relationship: RetrievalRelationship = RetrievalRelationship.NEUTRAL
    source: Literal["seed", "graph"] = "seed"


class BehaviorConflict(BaseModel):
    """Represents a detected conflict between two behaviors"""
    conflict_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the conflict"
    )
    user_id: str = Field(..., description="User whose behaviors conflict")
    behavior_id_1: str = Field(..., description="First conflicting behavior ID (existing/old)")
    behavior_id_2: str = Field(..., description="Second conflicting behavior ID (new)")
    conflict_type: ConflictType = Field(..., description="Type of conflict detected")
    similarity_distance: float = Field(..., ge=0.0, description="Embedding distance between behaviors")
    llm_analysis: Optional[str] = Field(None, description="LLM's explanation of the conflict")
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.PENDING,
        description="Current status of conflict resolution"
    )
    resolved_at: Optional[int] = Field(None, description="Timestamp when resolved")
    resolution_choice: Optional[str] = Field(None, description="How conflict was resolved")
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when conflict was detected"
    )
    # Drift detection fields - track polarity/target changes for preference reversal detection
    old_polarity: Optional[str] = Field(None, description="Polarity of existing behavior (POSITIVE/NEGATIVE)")
    new_polarity: Optional[str] = Field(None, description="Polarity of new behavior (POSITIVE/NEGATIVE)")
    old_target: Optional[str] = Field(None, description="Target of existing behavior")
    new_target: Optional[str] = Field(None, description="Target of new behavior")


class ConflictAnalysisType(str, Enum):
    """LLM's assessment of whether behaviors conflict"""
    CONFLICT = "CONFLICT"                 # Behaviors contradict each other
    COMPATIBLE = "COMPATIBLE"             # Behaviors can coexist
    CONTEXT_DEPENDENT = "CONTEXT_DEPENDENT"  # Depends on context


class ConflictAnalysisResult(BaseModel):
    """Result from LLM conflict analysis"""
    conflict_type: ConflictAnalysisType = Field(..., description="Type of relationship between behaviors")
    explanation: str = Field(..., description="Detailed reasoning for the classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM's confidence in this analysis")


class UserConfirmationRequest(BaseModel):
    """Request for user to resolve an ambiguous conflict"""
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request"
    )
    user_id: str = Field(..., description="User being asked to resolve")
    conflict_id: str = Field(..., description="Associated conflict ID")
    question: str = Field(..., description="Question to present to user")
    options: dict = Field(..., description="Available choices for user")
    status: Literal["PENDING", "RESOLVED", "EXPIRED"] = Field(
        default="PENDING",
        description="Status of the request"
    )
    user_response: Optional[str] = Field(None, description="User's choice")
    responded_at: Optional[int] = Field(None, description="Timestamp of user response")
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when request created"
    )
    expires_at: int = Field(
        default_factory=lambda: int(time.time()) + 604800,  # 7 days
        description="Timestamp when request expires"
    )


class ReinforcementResult(BaseModel):
    """Result of reinforcing an existing behavior"""
    success: bool = Field(..., description="Whether reinforcement succeeded")
    behavior_id: str = Field(..., description="ID of reinforced behavior")
    new_credibility: float = Field(..., ge=0.0, le=1.0, description="Updated credibility score")
    new_reinforcement_count: int = Field(..., ge=1, description="Updated reinforcement count")
    credibility_boost: float = Field(..., description="Amount credibility increased")
    error: Optional[str] = Field(None, description="Error message if failed")

class BehaviorResponse(BaseModel):
    """API response for behavior operations."""
    
    success: bool
    message: str
    behavior_id: Optional[str] = None
    data: Optional[dict] = None


class ExtractRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="User's natural language prompt"
    )
    user_id: str = Field(
        ...,
        description="User identifier for behavior extraction and storage"
    )
    session_id: str = Field(
        default="default",
        description="Optional session ID for session-specific behavior grouping within a user"
    )

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('user_id')
    def validate_user_id(cls, v):
        sanitized = v.strip()
        if not sanitized:
            raise ValueError("User ID cannot be empty")
        # Allow only alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in sanitized):
            raise ValueError("User ID can only contain alphanumeric characters, hyphens, and underscores")
        return sanitized
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        sanitized = v.strip()
        if not sanitized:
            return "default"
        # Allow only alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in sanitized):
            raise ValueError("Session ID can only contain alphanumeric characters, hyphens, and underscores")
        return sanitized
    
class HistoryMessage(BaseModel):
    """Represents a message in conversation history"""
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message sender (user or assistant)"
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Content of the message"
    )
    
    @field_validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty or whitespace only")
        return v.strip()


class ExtractRequestWithHistory(BaseModel):
    prompt: str = Field(
        ...,
        description="User's natural language prompt"
    )
    user_id: str = Field(
        ...,
        description="User identifier for behavior extraction and storage"
    )
    session_id: str = Field(
        default="default",
        description="Optional session ID for session-specific behavior grouping within a user"
    )
    recent_history: Optional[List[HistoryMessage]] = Field(
        default=None,
        description="Optional recent conversation history for context"
    )

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('user_id')
    def validate_user_id(cls, v):
        sanitized = v.strip()
        if not sanitized:
            raise ValueError("User ID cannot be empty")
        # Allow only alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in sanitized):
            raise ValueError("User ID can only contain alphanumeric characters, hyphens, and underscores")
        return sanitized
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        sanitized = v.strip()
        if not sanitized:
            return "default"
        # Allow only alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in sanitized):
            raise ValueError("Session ID can only contain alphanumeric characters, hyphens, and underscores")
        return sanitized


class CanonicalBehavior(BaseModel):
    """Normalized representation used for reasoning, not storage."""
    intent: Literal[
    "PREFERENCE",
    "CONSTRAINT",
    "HABIT",
    "SKILL",
    "COMMUNICATION"
    ]
    target: str = Field(..., min_length=1)
    context: Optional[str] = Field(
    default="general",
    description="Scope like IDE, frontend, night, general")
    polarity: Literal["POSITIVE", "NEGATIVE"]
    strength: float = Field(ge=0.0, le=1.0)


class BehaviorFlowAction(str, Enum):
    """Actions taken during behavior processing"""
    NEW_BEHAVIOR = "NEW_BEHAVIOR"
    DUPLICATE_REINFORCED = "DUPLICATE_REINFORCED"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"
    CONFLICT_AUTO_RESOLVED = "CONFLICT_AUTO_RESOLVED"
    SUPERSEDED_EXISTING = "SUPERSEDED_EXISTING"
    IGNORED_NEW = "IGNORED_NEW"
    COMPATIBLE = "COMPATIBLE"
    PRUNED = "PRUNED"


class BehaviorFlowInfo(BaseModel):
    """Detailed information about what happened to a behavior during processing"""
    behavior_description: str = Field(..., description="Original behavior description")
    action: BehaviorFlowAction = Field(..., description="Action taken for this behavior")
    credibility: float = Field(..., description="Calculated credibility score")
    canonical: Optional[dict] = Field(None, description="Canonical fields (intent, target, context, polarity)")
    matched_behavior_id: Optional[str] = Field(None, description="ID of matched existing behavior (if any)")
    matched_behavior_text: Optional[str] = Field(None, description="Text of matched existing behavior")
    distance: Optional[float] = Field(None, description="Semantic distance to matched behavior")
    conflict_info: Optional[dict] = Field(None, description="Conflict details if conflict detected")
    stored_behavior_id: Optional[str] = Field(None, description="ID of stored behavior (if saved)")
    details: Optional[str] = Field(None, description="Additional details about the action taken")


class DetailedExtractionResult(BaseModel):
    """Enhanced extraction result with flow tracking"""
    extraction_result: ExtractionResult = Field(..., description="Original extraction result")
    flow_info: List[BehaviorFlowInfo] = Field(default_factory=list, description="Detailed flow for each behavior")
    total_extracted: int = Field(default=0, description="Total behaviors extracted")
    total_stored: int = Field(default=0, description="Total behaviors stored")
    total_reinforced: int = Field(default=0, description="Total behaviors reinforced")
    total_conflicts: int = Field(default=0, description="Total conflicts detected")
    total_pruned: int = Field(default=0, description="Total behaviors pruned")
