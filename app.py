from fastapi import FastAPI, status, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from services.extractor import run_behavior_extraction, store_behavior, store_behavior_with_tracking
from services.behaviorRepository import insert_behavior, search_similar_behaviors, get_behaviors_by_user, get_user_conflicts, resolve_conflict
from models.behavior import ExtractRequest
from db.connection import close_db_pool, init_db_pool
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
from utils.embedding_utils import get_behavior_embedding
from utils.similarity_utils import calculate_behavior_distance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorSimilarityRequest(BaseModel):
    """Request model for behavior similarity comparison POC"""
    behavior1: str = Field(..., description="First behavior description")
    behavior2: str = Field(..., description="Second behavior description")
    metric: Literal["cosine", "euclidean", "manhattan"] = Field(
        default="cosine",
        description="Distance metric to use for comparison"
    )

class ConflictResolutionRequest(BaseModel):
    """Request model for resolving behavior conflicts"""
    conflict_id: str = Field(..., description="UUID of the conflict to resolve")
    user_id: str = Field(..., description="User ID who owns the conflicting behaviors")
    resolution_choice: Literal["OLD_WINS", "NEW_WINS", "BOTH_CORRECT"] = Field(
        ...,
        description="User's decision: OLD_WINS (keep existing), NEW_WINS (replace with new), BOTH_CORRECT (keep both)"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("initializeing database connection pool")
    init_db_pool()
    logger.info("database connection pool initialized")

    yield

    logger.info("shutting down database connection pool")
    close_db_pool()
    logger.info("database connection pool shut down")

app = FastAPI(
    title="Behavior Extraction API",
    description="Extract and manage user behaviors from natural language prompts",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
    
@app.post(
    "/extract",
    summary="Extract b ehaviors from prompt",
    description="Analyzes a natural language prompt and extracts user behaviors, preferences, and patterns",
    response_description="Extraction result with segmented behaviors"
)
def extract_behaviors(request: ExtractRequest):
    try:
        logger.info(f"Received extraction request for user: {request.user_id}")

        extraction_result = run_behavior_extraction(request.prompt)

        if not extraction_result.success:
            logger.error(f"Extraction failed: {extraction_result.error}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "data": None,
                    "error": extraction_result.error or "Extraction failed"
                }
            )
        
        stored_behaviors = []
        try:
            stored_behaviors = store_behavior(
                extraction_result,
                user_id=request.user_id,
                session_id=request.session_id
            )

            logger.info(f"Stored {len(stored_behaviors)} behaviors for user: {request.user_id}")
        except Exception as e:
            logger.error(f"Failed to store behaviors: {str(e)}")
            # stored_behaviors remains as empty list if error occurs
            
        
        total_behaviors = sum(len(seg.behaviors) for seg in extraction_result.segments)
        logger.info(
            f"Extraction successful: {len(extraction_result.segments)} segments, "
            f"{total_behaviors} behaviors, {extraction_result.extraction_time:.2f}ms"
        )

        # Prepare extracted behaviors data with canonical fields
        segments_data = [
            {
                "text": segment.text,
                "behaviors": [
                    {
                        "description": behavior.description,
                        "confidence": behavior.confidence,
                        "clarity": behavior.clarity,
                        "linguistic_strength": behavior.linguistic_strength,
                        "extracted_at": behavior.extracted_at,
                        # Canonical fields extracted by LLM
                        "canonical": {
                            "intent": behavior.intent,
                            "target": behavior.target,
                            "context": behavior.context,
                            "polarity": behavior.polarity
                        }
                    }
                    for behavior in segment.behaviors
                ]
            }
            for segment in extraction_result.segments
        ]

        # Prepare stored behaviors data with all fields including canonical
        stored_behaviors_data = [
            {
                "behavior_id": stored_behavior.behavior_id,
                "user_id": stored_behavior.user_id,
                "behavior_text": stored_behavior.behavior_text,
                "credibility": stored_behavior.credibility,
                "reinforcement_count": stored_behavior.reinforcement_count,
                "decay_rate": stored_behavior.decay_rate,
                "created_at": stored_behavior.created_at,
                "last_seen_at": stored_behavior.last_seen_at,
                "prompt_history_ids": stored_behavior.prompt_history_ids,
                "clarity_score": stored_behavior.clarity_score,
                "extraction_confidence": stored_behavior.extraction_confidence,
                "linguistic_strength": stored_behavior.linguistic_strength,
                "session_id": stored_behavior.session_id,
                "embedding_dimensions": len(stored_behavior.embedding) if stored_behavior.embedding else 0,
                # Canonical fields (for structured behavior reasoning)
                "canonical": {
                    "intent": stored_behavior.intent,
                    "target": stored_behavior.target,
                    "context": stored_behavior.context,
                    "polarity": stored_behavior.polarity
                }
            }
            for stored_behavior in stored_behaviors
        ]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "extraction": {
                        "segments": segments_data,
                        "extraction_time_ms": extraction_result.extraction_time,
                        "total_segments": len(extraction_result.segments),
                        "total_behaviors_extracted": total_behaviors
                    },
                    "storage": {
                        "stored_behaviors": stored_behaviors_data,
                        "total_behaviors_stored": len(stored_behaviors),
                        "behaviors_filtered": total_behaviors - len(stored_behaviors)
                    },
                    "user_id": request.user_id
                },
                "error": None
            }
        )
    except ValueError as e:
        # Validation error
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}"
            }
        )
    
    except Exception as e:
        # Unexpected error
        logger.exception("Unexpected error during extraction")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )
    
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "behavior_extraction"}


@app.post(
    "/extract-detailed",
    summary="Extract behaviors with detailed flow tracking",
    description="Analyzes a natural language prompt and extracts behaviors with detailed information about what happened to each behavior (duplicate, conflict, new, etc.)",
    response_description="Detailed extraction result with flow tracking for UI display"
)
def extract_behaviors_detailed(request: ExtractRequest):
    """
    Enhanced extraction endpoint that returns detailed flow information for each behavior.
    This is specifically designed for the frontend UI to show the processing path.
    """
    try:
        logger.info(f"Received detailed extraction request for user: {request.user_id}")

        # Run extraction
        extraction_result = run_behavior_extraction(request.prompt)

        if not extraction_result.success:
            logger.error(f"Extraction failed: {extraction_result.error}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "data": None,
                    "error": extraction_result.error or "Extraction failed"
                }
            )
        
        # Store with detailed tracking
        try:
            detailed_result = store_behavior_with_tracking(
                extraction_result,
                user_id=request.user_id,
                session_id=request.session_id
            )

            logger.info(
                f"Processing complete: {detailed_result.total_stored} stored, "
                f"{detailed_result.total_reinforced} reinforced, "
                f"{detailed_result.total_conflicts} conflicts, "
                f"{detailed_result.total_pruned} pruned"
            )
        except Exception as e:
            logger.error(f"Failed to store behaviors: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Storage error: {str(e)}"
                }
            )
        
        # Format flow info for frontend
        flow_info_formatted = []
        for flow in detailed_result.flow_info:
            flow_dict = {
                "behavior_description": flow.behavior_description,
                "action": flow.action.value,
                "credibility": round(flow.credibility, 3),
                "canonical": flow.canonical,
                "matched_behavior_id": flow.matched_behavior_id,
                "matched_behavior_text": flow.matched_behavior_text,
                "distance": round(flow.distance, 4) if flow.distance is not None else None,
                "conflict_info": flow.conflict_info,
                "stored_behavior_id": flow.stored_behavior_id,
                "details": flow.details
            }
            flow_info_formatted.append(flow_dict)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "extraction": {
                        "extraction_time_ms": extraction_result.extraction_time,
                        "total_segments": len(extraction_result.segments),
                    },
                    "processing": {
                        "total_extracted": detailed_result.total_extracted,
                        "total_stored": detailed_result.total_stored,
                        "total_reinforced": detailed_result.total_reinforced,
                        "total_conflicts": detailed_result.total_conflicts,
                        "total_pruned": detailed_result.total_pruned
                    },
                    "flow_info": flow_info_formatted,
                    "user_id": request.user_id
                },
                "error": None
            }
        )
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}"
            }
        )
    except Exception as e:
        logger.exception("Unexpected error during detailed extraction")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )


@app.get(
    "/behaviors/{user_id}",
    summary="Get all behaviors for a user",
    description="Retrieve all stored behaviors for a specific user. Optionally filter by session_id.",
    response_description="List of behaviors with all details"
)
def get_user_behaviors(user_id: str, session_id: str = Query(None, description="Optional session ID to filter behaviors")):
    """Get all behaviors for a specific user. If session_id is provided, only returns behaviors from that session."""
    try:
        behaviors = get_behaviors_by_user(user_id, session_id=session_id)
        
        session_info = f" in session {session_id}" if session_id else " (all sessions)"
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "total_behaviors": len(behaviors),
                    "behaviors": behaviors
                },
                "error": None
            }
        )
    except Exception as e:
        logger.exception(f"Error retrieving behaviors for user {user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Failed to retrieve behaviors: {str(e)}"
            }
        )


@app.get(
    "/conflicts/{user_id}",
    summary="Get all conflicts for a user",
    description="Retrieve all detected conflicts for a specific user",
    response_description="List of conflicts with behavior details"
)
def get_user_conflicts_endpoint(user_id: str):
    """Get all conflicts for a specific user."""
    try:
        conflicts = get_user_conflicts(user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "total_conflicts": len(conflicts),
                    "conflicts": conflicts
                },
                "error": None
            }
        )
    except Exception as e:
        logger.exception(f"Error retrieving conflicts for user {user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Failed to retrieve conflicts: {str(e)}"
            }
        )

@app.post(
    "/similarity",
    summary="Calculate similarity between two behaviors",
    description="Compare two behavior descriptions using embeddings and return their distance/similarity",
    response_description="Similarity analysis with distance metrics"
)
def calculate_similarity(request: BehaviorSimilarityRequest):
    """
    POC endpoint to understand how embeddings and distance metrics work.
    
    Takes two behavior descriptions, generates embeddings for each,
    and calculates the distance between them.
    """
    try:
        logger.info(f"Received similarity request for behaviors")
        logger.info(f"Behavior 1: {request.behavior1[:50]}...")
        logger.info(f"Behavior 2: {request.behavior2[:50]}...")
        logger.info(f"Metric: {request.metric}")
        
        # Generate embeddings for both behaviors
        try:
            embedding1 = get_behavior_embedding(request.behavior1)
            logger.info(f"Generated embedding1: {len(embedding1)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding for behavior1: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to generate embedding for behavior1: {str(e)}"
                }
            )
        
        try:
            embedding2 = get_behavior_embedding(request.behavior2)
            logger.info(f"Generated embedding2: {len(embedding2)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding for behavior2: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to generate embedding for behavior2: {str(e)}"
                }
            )
        
        # Calculate distance
        try:
            result = calculate_behavior_distance(
                behavior1_text=request.behavior1,
                behavior2_text=request.behavior2,
                embedding1=embedding1,
                embedding2=embedding2,
                metric=request.metric
            )
            logger.info(f"Calculated distance: {result['distance']:.4f}")
        except Exception as e:
            logger.error(f"Failed to calculate distance: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to calculate distance: {str(e)}"
                }
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": result,
                "error": None
            }
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}"
            }
        )
    except Exception as e:
        logger.exception("Unexpected error during similarity calculation")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )


@app.post(
    "/resolve-conflict",
    summary="Resolve a behavior conflict",
    description="Allows user to resolve a flagged behavior conflict by choosing which behavior(s) to keep",
    response_description="Resolution result with updated behavior states"
)
def resolve_behavior_conflict(request: ConflictResolutionRequest):
    """
    Resolve a conflict based on user's decision.
    
    Resolution choices:
    - OLD_WINS: Keep existing behavior (reinforced), invalidate new behavior (credibility = 0.0)
    - NEW_WINS: Replace existing behavior (superseded), new behavior becomes ACTIVE
    - BOTH_CORRECT: Keep both behaviors as ACTIVE (both reinforced)
    
    All resolutions update last_accessed_at to reflect active conflict resolution.
    """
    try:
        logger.info(
            f"Received conflict resolution request: "
            f"conflict_id={request.conflict_id}, "
            f"user_id={request.user_id}, "
            f"choice={request.resolution_choice}"
        )
        
        # Call the repository function to handle the resolution
        result = resolve_conflict(
            conflict_id=request.conflict_id,
            user_id=request.user_id,
            resolution_choice=request.resolution_choice
        )
        
        logger.info(
            f"Conflict {request.conflict_id} resolved successfully: "
            f"{request.resolution_choice}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": result,
                "error": None
            }
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error resolving conflict {request.conflict_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )
