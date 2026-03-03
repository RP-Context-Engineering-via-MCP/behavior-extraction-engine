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
    


class ExtractionResult(BaseModel):
    segments: List[BehaviorSegment] = Field(
        default_factory=list, 
        description="Segment prompt with extracted behaviors"
    )
    success: bool = Field(
        ...,
        description = "Indicates if extraction was successful"
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
    standalone_query: Optional[str] = Field(
        None,
        description="Enriched standalone version of the prompt for similarity search (with conversation context resolved)"
    )
    required_intents: Optional[List[str]] = Field(
        default=None,
        description="LLM-predicted intent types relevant to the query for metadata pre-filtering (e.g., ['CONSTRAINT', 'PREFERENCE'])"
    )
    profile_signals: Optional[dict] = Field(
        default=None,
        description="Profile signals extracted for Profile Service integration (intents, interests, behavior_level, signals, complexity, consistency)"
    )


class StoredBehavior(BaseModel):
    """final object to be stored in the DB"""

    behavior_id: str = Field(
        default_factory=lambda: f"beh_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the behavior"
    )
    user_id: str = Field(
        ...,
        description="user identifier - behavior are unique per user"
    )
    behavior_text: str = Field(
        ..., 
        description="extracted behavior"
    )
    credibility: float = Field(
        description="Credibility score of the behavior"
    )
    reinforcement_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this behavior has been reinforced"
    )
    decay_rate: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Decay rate for the behavior's credibility over time"
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the behavior was created"
    )
    last_seen_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when the behavior was last reinforced"
    )
    last_decay_applied_at: Optional[int] = Field(
        default=None,
        description="Timestamp when decay was last applied or when it should start (created_at + grace period)"
    )
    last_accessed_at: Optional[int] = Field(
        default=None,
        description="Timestamp when behavior was last actively used (context enrichment, reinforcement, conflict resolution)"
    )
    prompt_history_ids: List[str] = Field(
        default_factory=list,
        description="List of prompt IDs that have triggered this behavior"
    )
    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear/unambiguous the behavior was in the prompt"
    )
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="GPT's confidence in extracting this behavior"
    )
    linguistic_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of user's language when expressing this behavior"
    )
    session_id: str = Field(
        default="default",
        description="Session ID or 'default' for general behaviors"
    )
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding of behavior_text for semantic search"
    )
    # Canonical behavior fields (added with canonical behavior refactor)
    intent: Optional[Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"]] = Field(
        None,
        description="Behavioral intent (PREFERENCE, CONSTRAINT, HABIT, SKILL, COMMUNICATION)"
    )
    target: Optional[str] = Field(
        None,
        description="Concise target noun (dark mode, python, etc.)"
    )
    context: Optional[str] = Field(
        None,
        description="Context scope (general, IDE, frontend, morning, etc.)"
    )
    polarity: Optional[Literal["POSITIVE", "NEGATIVE"]] = Field(
        None,
        description="Polarity (POSITIVE, NEGATIVE)"
    )

    @field_validator('embedding')
    def validate_embedding_dimension(cls, v):
        """Ensure embedding has correct dimensions for text-embedding-3-large."""
        if v is not None and len(v) != 3072:
            raise ValueError(f"Embedding must be 3072-dimensional, got {len(v)}")
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
    """Result of similarity search between behaviors"""
    behavior_id: str = Field(..., description="ID of the similar behavior found")
    behavior_text: str = Field(..., description="Text of the similar behavior")
    distance: float = Field(..., ge=0.0, description="Cosine distance (lower = more similar)")
    credibility: float = Field(..., ge=0.0, le=1.0, description="Current credibility of found behavior")
    last_seen_at: int = Field(..., description="Timestamp when behavior was last reinforced")
    reinforcement_count: int = Field(..., ge=1, description="Number of times behavior reinforced")
    # Canonical behavior fields (added with canonical behavior refactor)
    intent: Optional[Literal["PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"]] = Field(None, description="Behavioral intent (PREFERENCE, CONSTRAINT, HABIT, SKILL, COMMUNICATION)")
    target: Optional[str] = Field(None, description="Concise target noun")
    context: Optional[str] = Field(None, description="Context scope (IDE, frontend, etc.)")
    polarity: Optional[Literal["POSITIVE", "NEGATIVE"]] = Field(None, description="Polarity (POSITIVE, NEGATIVE)")


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
    segment_id_added: Optional[str] = Field(None, description="Segment ID added to prompt_history_ids")
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


class PromptSegment(BaseModel):
    """Represents a segment of user prompt to be stored"""
    user_id: str = Field(..., description="User who provided this segment")
    segment_text: str = Field(..., description="The actual segment text from prompt")
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Timestamp when segment was saved"
    )

class SegmentInsertResult(BaseModel):
    """Result of inserting a prompt segment"""
    success: bool
    segment_id: Optional[str] = None
    error: Optional[str] = None

class BehaviorQuery(BaseModel):
    """Input for behavior analysis"""
    behavior_text: str = Field(
        ...,
        min_length=3,
        description="Behavior text to analyze"
    )
    user_id: str = Field(
        ...,
        description="User identifier"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )

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
