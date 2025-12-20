from urllib import response
from pydantic import BaseModel,Field,field_validator
from typing import List, Optional, Literal
from datetime import datetime
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
        default_factory=lambda: datetime.now(datetime.timezone.utc).isoformat(),
        description="Timestamp of extraction in UTC")
    


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

    @field_validator('embedding')
    def validate_embedding_dimension(cls, v):
        """Ensure embedding has correct dimensions for text-embedding-3-large."""
        if v is not None and len(v) != 3072:
            raise ValueError(f"Embedding must be 3072-dimensional, got {len(v)}")
        return v

class BehaviorResponse(BaseModel):
    """API response for behavior operations."""
    
    success: bool
    message: str
    behavior_id: Optional[str] = None
    data: Optional[dict] = None