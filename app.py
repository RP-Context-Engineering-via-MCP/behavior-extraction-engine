from fastapi import FastAPI, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from services.extractor import run_behavior_extraction, run_behavior_extraction_with_history, store_behavior, store_behavior_with_tracking, dispatch_profile_signals_sync
from services.behaviorRepository import insert_behavior, search_similar_behaviors, search_similar_behavior_3D, persist_retrieval_updates_batch, get_behaviors_by_user, get_user_conflicts, resolve_conflict, get_behaviors_by_ids
from services.profileSignalRepository import get_profile_signal_repository
from models.behavior import ExtractRequest, ExtractRequestWithHistory, HistoryMessage
from db.connection import close_db_pool, init_db_pool
from config.configurations import RELATED_BEHAVIORS_DISTANCE_THRESHOLD, HYBRID_SCORE_THRESHOLD, PROFILE_SIGNALS_DEFAULT_LIMIT, PROFILE_SIGNALS_MAX_LIMIT
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
from utils.embedding_utils import get_behavior_embedding
from utils.similarity_utils import calculate_behavior_distance
import logging
import uuid
import os

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

class BehaviorsByIdsRequest(BaseModel):
    """Request model for retrieving specific behaviors by IDs"""
    user_id: str = Field(..., description="User ID who owns the behaviors")
    behavior_ids: list[str] = Field(..., description="List of behavior IDs to retrieve")

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

# Mount frontend static files (optional - only if directory exists)
if os.path.exists("frontend") and os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
    logger.info("Frontend static files mounted at /frontend")
else:
    logger.warning("Frontend directory not found - API running without frontend")
    
@app.post(
    "/extract",
    summary="Extract behaviors from prompt",
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
        
        # Dispatch profile signals to Profile Service (for cold-start profiling)
        if extraction_result.profile_signals:
            try:
                prompt_id = f"prompt_{uuid.uuid4().hex[:12]}"
                dispatch_profile_signals_sync(
                    user_id=request.user_id,
                    prompt_id=prompt_id,
                    profile_signals=extraction_result.profile_signals
                )
                logger.debug(f"Profile signals dispatched for prompt={prompt_id}")
            except Exception as e:
                logger.error(f"Failed to dispatch profile signals: {e}")
                # Continue - profile signal dispatch failure is not critical
        
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

def _store_behaviors_async(extraction_result, user_id: str, session_id: str):
    """
    Background task for storing behaviors with conflict detection and reinforcement.
    This runs asynchronously after the response has been sent to the client.
    Also dispatches profile signals to Profile Service for cold-start profiling.
    """
    try:
        stored_behaviors = store_behavior(
            extraction_result,
            user_id=user_id,
            session_id=session_id
        )
        logger.info(f"[ASYNC] Successfully stored {len(stored_behaviors)} behaviors for user: {user_id}")
        
        # Dispatch profile signals to Profile Service (for cold-start profiling)
        if extraction_result.profile_signals:
            try:
                prompt_id = f"prompt_{uuid.uuid4().hex[:12]}"
                dispatch_profile_signals_sync(
                    user_id=user_id,
                    prompt_id=prompt_id,
                    profile_signals=extraction_result.profile_signals
                )
                logger.debug(f"[ASYNC] Profile signals dispatched for user={user_id}, prompt={prompt_id}")
            except Exception as e:
                logger.error(f"[ASYNC] Failed to dispatch profile signals for user {user_id}: {e}")
                # Continue - profile signal dispatch failure is not critical
                
    except Exception as e:
        logger.error(f"[ASYNC] Failed to store behaviors for user {user_id}: {str(e)}")


@app.post(
    "/v2/extract",
    summary="Extract behaviors with conversation history and retrieve related behaviors (optimized)",
    description="""
    Optimized endpoint that prioritizes fast response with related behaviors.
    
    Workflow:
    1. Extract behaviors from prompt with conversation history
    2. Enrich prompt to standalone query for similarity search
    3. Search and return related behaviors IMMEDIATELY
    4. Store extracted behaviors asynchronously in background (conflict detection, reinforcement)
    
    This endpoint resolves contextual references (e.g., "it", "that", "the above") 
    using the recent conversation history, making the prompt fully self-contained 
    for better semantic search against stored behaviors.
    
    Response contains ONLY related behaviors from the database, optimized for speed.
    """,
    response_description="Related behaviors from similarity search (fast response)"
)
def extract_behaviors_with_history(request: ExtractRequestWithHistory, background_tasks: BackgroundTasks):
    """
    Extract behaviors from a prompt with conversation history and return related behaviors quickly.
    
    Optimized for speed by:
    - Returning related behaviors immediately after similarity search
    - Running behavior storage (conflict detection, reinforcement) asynchronously
    - Excluding extracted/stored behavior details from response
    
    This endpoint is useful when:
    - The prompt contains references like "it", "that", "those", "the above"
    - The prompt depends on previous conversation context
    - You need a standalone query for similarity search
    - You need the fastest possible response with related behaviors
    
    Example payload:
    {
      "prompt": "among above what is the sweetest food",
      "recent_history": [
         {"role": "user", "text": "I like healthy breakfast options like oatmeal and fruits."},
         {"role": "assistant", "text": "Great! Oatmeal with berries, or a banana smoothie are excellent choices."}
      ],
      "user_id": "sample_user_01",
      "session_id": "test_session_002"
    }
    """
    try:
        logger.info(f"Received extraction request with history for user: {request.user_id}")
        
        # Convert Pydantic HistoryMessage objects to dicts for the extractor
        history_dicts = []
        if request.recent_history:
            history_dicts = [
                {"role": msg.role, "text": msg.text} 
                for msg in request.recent_history
            ]
        
        # STEP 1: Extract behaviors and get standalone query (LLM call - cannot optimize)
        extraction_result = run_behavior_extraction_with_history(
            prompt=request.prompt,
            recent_history=history_dicts
        )

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
        
        logger.info(f"Extraction process is successful. extracted results : {extraction_result.model_dump_json(indent=2)}")

        # STEP 2: Search for related behaviors using 3D hybrid retrieval (FAST - priority)
        related_behaviors = []
        hybrid_response = None
        if extraction_result.standalone_query:
            try:
                from services.openAiClient import embed_text
                
                # Use the enriched standalone query for similarity search
                query_embedding = embed_text(extraction_result.standalone_query)
                
                # 3D Hybrid Search: Dense (semantic) + Sparse (BM25) + Metadata (intent filter)
                hybrid_response = search_similar_behavior_3D(
                    user_id=request.user_id,
                    query_embedding=query_embedding,
                    query_text=extraction_result.standalone_query,
                    session_id=request.session_id,
                    required_intents=extraction_result.required_intents
                )
                logger.info(f"[3D] Found {len(hybrid_response.results)} similar behaviors for standalone query: '{extraction_result.standalone_query}'")
                logger.info(f"[3D] Required intents boost: {extraction_result.required_intents}")

                logger.info(
                    f"[3D] Hybrid search returned {len(hybrid_response.results)} results, "
                    f"{len(hybrid_response.decay_updates)} pending decay updates, "
                    f"{len(hybrid_response.accessed_behavior_ids)} access timestamps to update"
                )
                
                # Filter by relevance threshold
                related_behaviors = [
                    {
                        "behavior_id": b.behavior_id,
                        "behavior_text": b.behavior_text,
                        "distance": b.distance,
                        "intent": b.intent,
                        "target": b.target,
                        "context": b.context,
                        "polarity": b.polarity,
                        "credibility": b.credibility,
                    }
                    for b in hybrid_response.results
                    if b.distance <= RELATED_BEHAVIORS_DISTANCE_THRESHOLD
                ]
                
                logger.info(
                    f"[3D] Returning {len(related_behaviors)} related behaviors "
                    f"(filtered {len(hybrid_response.results) - len(related_behaviors)} below threshold, "
                    f"distance_threshold: {RELATED_BEHAVIORS_DISTANCE_THRESHOLD})"
                )
            except Exception as e:
                logger.error(f"Failed to search related behaviors: {str(e)}")
        
        # STEP 3: Schedule behavior storage in background (ASYNC - non-blocking)
        background_tasks.add_task(
            _store_behaviors_async,
            extraction_result,
            request.user_id,
            request.session_id
        )
        logger.info(f"Scheduled async storage of behaviors for user: {request.user_id}")
        
        # STEP 3b: Schedule retrieval updates in background (decay persistence + last_accessed_at)
        if hybrid_response and (hybrid_response.decay_updates or hybrid_response.accessed_behavior_ids):
            background_tasks.add_task(
                persist_retrieval_updates_batch,
                hybrid_response.decay_updates,
                hybrid_response.accessed_behavior_ids,
                request.user_id
            )
            logger.info(
                f"Scheduled async retrieval updates: "
                f"{len(hybrid_response.decay_updates)} decay updates, "
                f"{len(hybrid_response.accessed_behavior_ids)} access timestamps"
            )

        # STEP 4: Return response IMMEDIATELY with related behaviors only
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "standalone_query": extraction_result.standalone_query,
                    "required_intents": extraction_result.required_intents,
                    "original_prompt": request.prompt,
                    "related_behaviors": related_behaviors,
                    "extraction_time_ms": extraction_result.extraction_time,
                    "user_id": request.user_id
                },
                "error": None
            }
        )
    
    except ValueError as e:
        # Validation error (e.g., invalid input)
        logger.exception("Validation error during extraction with history")
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
        logger.exception("Unexpected error during extraction with history")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}"
            }
        )

# ==============================================================================
# PROFILE SERVICE INTEGRATION ENDPOINTS
# ==============================================================================

@app.post(
    "/api/behaviors/by-ids",
    summary="Get specific behaviors by IDs for a user",
    description="Retrieves specific user behaviors by their behavior IDs. Returns behaviors with all details including canonical structure.",
    response_description="List of behaviors matching the requested IDs"
)
def get_behaviors_by_ids_endpoint(request: BehaviorsByIdsRequest):
    """
    Get specific behaviors by their IDs for a user.
    
    This endpoint retrieves specific user behaviors by their behavior IDs,
    allowing clients to fetch particular behaviors they need.
    
    Args:
        request: BehaviorsByIdsRequest containing user_id and behavior_ids list
        
    Returns:
        JSON with user_id, count, and list of matching behaviors
    """
    try:
        logger.info(f"Fetching behaviors by IDs for user={request.user_id}, IDs={request.behavior_ids}")
        
        behaviors = get_behaviors_by_ids(
            user_id=request.user_id,
            behavior_ids=request.behavior_ids
        )
        
        logger.info(f"Retrieved {len(behaviors)} behaviors for user={request.user_id}")
        
        # Return direct array as per original endpoint format
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=behaviors
        )
        
    except Exception as e:
        logger.exception(f"Error fetching behaviors by IDs for user={request.user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=[]
        )


@app.get(
    "/api/behaviors/{user_id}/signals/count",
    summary="Get profile signal count for a user",
    description="Returns the total count of stored profile signals for a user.",
    response_description="Count of profile signals"
)
def get_profile_signal_count(user_id: str):
    """
    Get the total count of stored profile signals for a user.
    
    Useful for determining if enough signals have been collected
    for profile assignment or drift detection.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        JSON with user_id and total count
    """
    try:
        signal_repo = get_profile_signal_repository()
        count = signal_repo.get_count(user_id=user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "user_id": user_id,
                "count": count
            }
        )
        
    except Exception as e:
        logger.exception(f"Error fetching profile signal count for user={user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "user_id": user_id,
                "count": 0,
                "error": f"Failed to retrieve signal count: {str(e)}"
            }
        )