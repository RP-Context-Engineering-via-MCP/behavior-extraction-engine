"""
Pydantic schemas for the Behavior Extraction API.

Request and response models live here, keeping them out of both the
app factory (app.py) and the domain models (models/behavior.py).

Separation of concerns:
  models/behavior.py  → domain / storage models
  api/schemas.py      → HTTP transport models (request bodies, response envelopes)
"""

from typing import Any, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Generic envelope
# ---------------------------------------------------------------------------

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    """
    Standard response envelope used by all endpoints.

    success  – True on success, False on any error.
    data     – Populated on success, None on error.
    error    – Populated on error, None on success.
    """

    success: bool
    data: Optional[DataT] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ConflictResolutionRequest(BaseModel):
    """Request model for the /resolve-conflict endpoint."""

    conflict_id: str = Field(..., description="UUID of the conflict to resolve")
    user_id: str = Field(..., description="User ID who owns the conflicting behaviors")
    resolution_choice: Literal["OLD_WINS", "NEW_WINS", "BOTH_CORRECT"] = Field(
        ...,
        description=(
            "User's decision: "
            "OLD_WINS (keep existing), "
            "NEW_WINS (replace with new), "
            "BOTH_CORRECT (keep both)"
        ),
    )


class BehaviorsByIdsRequest(BaseModel):
    """Request model for retrieving specific behaviors by IDs"""
    
    user_id: str = Field(..., description="User ID who owns the behaviors")
    behavior_ids: List[str] = Field(..., description="List of behavior IDs to retrieve")


# ---------------------------------------------------------------------------
# Response data payloads
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    service: str


class ExtractionStorageData(BaseModel):
    stored_behaviors: List[Any]
    total_behaviors_stored: int
    behaviors_filtered: int


class ExtractionResponseData(BaseModel):
    extraction: Any
    storage: ExtractionStorageData
    user_id: str


class BehaviorListResponseData(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    total_behaviors: int
    behaviors: List[Any]


class ConflictListResponseData(BaseModel):
    user_id: str
    total_conflicts: int
    conflicts: List[Any]


class V2ExtractionResponseData(BaseModel):
    behaviors: List[str]
