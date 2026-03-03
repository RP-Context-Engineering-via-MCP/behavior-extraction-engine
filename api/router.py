"""
API route handlers for the Behavior Extraction API.

All business logic stays untouched in the service layer.  This module
is purely concerned with HTTP plumbing: parse request → call service →
format response.

Registered in app.py via:
    app.include_router(router)
"""

from fastapi import APIRouter, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse

from api.schemas import (
    BehaviorSimilarityRequest,
    BehaviorsByIdsRequest,
    ConflictResolutionRequest,
)
from config.configurations import RELATED_BEHAVIORS_DISTANCE_THRESHOLD
from models.behavior import ExtractRequest, ExtractRequestWithHistory
from services.behaviorRepository import (
    get_behaviors_by_ids,
    get_behaviors_by_user,
    get_user_conflicts,
    persist_retrieval_updates_batch,
    resolve_conflict,
    search_similar_behavior_3D,
)
from services.profileSignalRepository import get_profile_signal_repository
from services.extractor import (
    run_behavior_extraction,
    run_behavior_extraction_with_history,
    store_behavior,
    dispatch_profile_signals_sync,
)
from utils.embedding_utils import get_behavior_embedding
from utils.similarity_utils import calculate_behavior_distance

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _store_behaviors_async(extraction_result, user_id: str, session_id: str) -> None:
    """
    Background task: store behaviors with conflict detection and reinforcement.
    Runs after the response has already been sent to the client.
    """
    try:
        stored_behaviors = store_behavior(
            extraction_result,
            user_id=user_id,
            session_id=session_id,
        )
        logger.info(
            f"[ASYNC] Successfully stored {len(stored_behaviors)} behaviors "
            f"for user: {user_id}"
        )
    except Exception as e:
        logger.error(
            f"[ASYNC] Failed to store behaviors for user {user_id}: {str(e)}"
        )
    
    # Dispatch profile signals asynchronously
    if hasattr(extraction_result, 'profile_signals') and extraction_result.profile_signals:
        try:
            dispatch_profile_signals_sync(
                user_id=user_id,
                prompt_id=session_id,
                profile_signals=extraction_result.profile_signals
            )
            logger.info(f"[ASYNC] Profile signals dispatched for user: {user_id}")
        except Exception as e:
            logger.warning(f"[ASYNC] Failed to dispatch profile signals: {str(e)}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/extract",
    summary="Extract behaviors from prompt",
    description=(
        "Analyzes a natural language prompt and extracts user behaviors, "
        "preferences, and patterns"
    ),
    response_description="Extraction result with segmented behaviors",
)
def extract_behaviors(request: ExtractRequest):
    try:
        logger.info(f"Received extraction request for user: {request.user_id}")

        extraction_result = run_behavior_extraction(request.prompt)

        if not extraction_result.success:
            logger.error(f"Extraction failed: {extraction_result.error}")
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "success": False,
                    "data": None,
                    "error": extraction_result.error or "Extraction failed",
                },
            )

        stored_behaviors = []
        try:
            stored_behaviors = store_behavior(
                extraction_result,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            logger.info(
                f"Stored {len(stored_behaviors)} behaviors for user: {request.user_id}"
            )
        except Exception as e:
            logger.error(f"Failed to store behaviors: {str(e)}")
            # stored_behaviors remains empty if storage fails

        # Dispatch profile signals for Profile Service integration
        if extraction_result.profile_signals:
            try:
                dispatch_profile_signals_sync(
                    user_id=request.user_id,
                    prompt_id=request.session_id,
                    profile_signals=extraction_result.profile_signals
                )
                logger.info(f"Profile signals dispatched for user: {request.user_id}")
            except Exception as e:
                logger.warning(f"Failed to dispatch profile signals: {str(e)}")

        total_behaviors = sum(
            len(seg.behaviors) for seg in extraction_result.segments
        )
        logger.info(
            f"Extraction successful: {len(extraction_result.segments)} segments, "
            f"{total_behaviors} behaviors, "
            f"{extraction_result.extraction_time:.2f}ms"
        )

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
                        "canonical": {
                            "intent": behavior.intent,
                            "target": behavior.target,
                            "context": behavior.context,
                            "polarity": behavior.polarity,
                        },
                    }
                    for behavior in segment.behaviors
                ],
            }
            for segment in extraction_result.segments
        ]

        stored_behaviors_data = [
            {
                "behavior_id": sb.behavior_id,
                "user_id": sb.user_id,
                "behavior_text": sb.behavior_text,
                "credibility": sb.credibility,
                "reinforcement_count": sb.reinforcement_count,
                "decay_rate": sb.decay_rate,
                "created_at": sb.created_at,
                "last_seen_at": sb.last_seen_at,
                "prompt_history_ids": sb.prompt_history_ids,
                "clarity_score": sb.clarity_score,
                "extraction_confidence": sb.extraction_confidence,
                "linguistic_strength": sb.linguistic_strength,
                "session_id": sb.session_id,
                "embedding_dimensions": (
                    len(sb.embedding) if sb.embedding else 0
                ),
                "canonical": {
                    "intent": sb.intent,
                    "target": sb.target,
                    "context": sb.context,
                    "polarity": sb.polarity,
                },
            }
            for sb in stored_behaviors
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
                        "total_behaviors_extracted": total_behaviors,
                    },
                    "storage": {
                        "stored_behaviors": stored_behaviors_data,
                        "total_behaviors_stored": len(stored_behaviors),
                        "behaviors_filtered": total_behaviors - len(stored_behaviors),
                    },
                    "user_id": request.user_id,
                },
                "error": None,
            },
        )

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}",
            },
        )


@router.get("/health", summary="Health check")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "behavior_extraction"}


@router.get(
    "/behaviors/{user_id}",
    summary="Get all behaviors for a user",
    description=(
        "Retrieve all stored behaviors for a specific user. "
        "Optionally filter by session_id."
    ),
    response_description="List of behaviors with all details",
)
def get_user_behaviors(
    user_id: str,
    session_id: str = Query(
        None, description="Optional session ID to filter behaviors"
    ),
):
    """
    Get all behaviors for a specific user.
    If session_id is provided, only returns behaviors from that session.
    """
    try:
        behaviors = get_behaviors_by_user(user_id, session_id=session_id)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "total_behaviors": len(behaviors),
                    "behaviors": behaviors,
                },
                "error": None,
            },
        )
    except Exception as e:
        logger.exception(f"Error retrieving behaviors for user {user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Failed to retrieve behaviors: {str(e)}",
            },
        )


@router.get(
    "/conflicts/{user_id}",
    summary="Get all conflicts for a user",
    description="Retrieve all detected conflicts for a specific user",
    response_description="List of conflicts with behavior details",
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
                    "conflicts": conflicts,
                },
                "error": None,
            },
        )
    except Exception as e:
        logger.exception(f"Error retrieving conflicts for user {user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Failed to retrieve conflicts: {str(e)}",
            },
        )


@router.post(
    "/similarity",
    summary="Calculate similarity between two behaviors",
    description=(
        "Compare two behavior descriptions using embeddings and return "
        "their distance/similarity"
    ),
    response_description="Similarity analysis with distance metrics",
)
def calculate_similarity(request: BehaviorSimilarityRequest):
    """
    POC endpoint to understand how embeddings and distance metrics work.

    Takes two behavior descriptions, generates embeddings for each,
    and calculates the distance between them.
    """
    try:
        logger.info("Received similarity request for behaviors")
        logger.info(f"Behavior 1: {request.behavior1[:50]}...")
        logger.info(f"Behavior 2: {request.behavior2[:50]}...")
        logger.info(f"Metric: {request.metric}")

        try:
            embedding1 = get_behavior_embedding(request.behavior1)
            logger.info(f"Generated embedding1: {len(embedding1)} dimensions")
        except Exception as e:
            logger.error(
                f"Failed to generate embedding for behavior1: {str(e)}"
            )
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to generate embedding for behavior1: {str(e)}",
                },
            )

        try:
            embedding2 = get_behavior_embedding(request.behavior2)
            logger.info(f"Generated embedding2: {len(embedding2)} dimensions")
        except Exception as e:
            logger.error(
                f"Failed to generate embedding for behavior2: {str(e)}"
            )
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to generate embedding for behavior2: {str(e)}",
                },
            )

        try:
            result = calculate_behavior_distance(
                behavior1_text=request.behavior1,
                behavior2_text=request.behavior2,
                embedding1=embedding1,
                embedding2=embedding2,
                metric=request.metric,
            )
            logger.info(f"Calculated distance: {result['distance']:.4f}")
        except Exception as e:
            logger.error(f"Failed to calculate distance: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "success": False,
                    "data": None,
                    "error": f"Failed to calculate distance: {str(e)}",
                },
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "data": result, "error": None},
        )

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during similarity calculation")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}",
            },
        )


@router.post(
    "/resolve-conflict",
    summary="Resolve a behavior conflict",
    description=(
        "Allows user to resolve a flagged behavior conflict by choosing "
        "which behavior(s) to keep"
    ),
    response_description="Resolution result with updated behavior states",
)
def resolve_behavior_conflict(request: ConflictResolutionRequest):
    """
    Resolve a conflict based on user's decision.

    Resolution choices:
    - OLD_WINS: Keep existing (reinforced), invalidate new (credibility → 0.0)
    - NEW_WINS: Replace existing (superseded), new becomes ACTIVE
    - BOTH_CORRECT: Keep both as ACTIVE (both reinforced)

    All resolutions update last_accessed_at to reflect active conflict resolution.
    """
    try:
        logger.info(
            f"Received conflict resolution request: "
            f"conflict_id={request.conflict_id}, "
            f"user_id={request.user_id}, "
            f"choice={request.resolution_choice}"
        )

        result = resolve_conflict(
            conflict_id=request.conflict_id,
            user_id=request.user_id,
            resolution_choice=request.resolution_choice,
        )

        logger.info(
            f"Conflict {request.conflict_id} resolved successfully: "
            f"{request.resolution_choice}"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "data": result, "error": None},
        )

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "data": None, "error": str(e)},
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error resolving conflict {request.conflict_id}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}",
            },
        )


@router.post(
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
    response_description="Related behaviors from similarity search (fast response)",
)
def extract_behaviors_with_history(
    request: ExtractRequestWithHistory, background_tasks: BackgroundTasks
):
    """
    Extract behaviors from a prompt with conversation history and return
    related behaviors quickly.

    Optimized for speed by:
    - Returning related behaviors immediately after similarity search
    - Running behavior storage (conflict detection, reinforcement) asynchronously
    - Excluding extracted/stored behavior details from response
    """
    try:
        logger.info(
            f"Received extraction request with history for user: {request.user_id}"
        )

        # Convert Pydantic HistoryMessage objects to dicts for the extractor
        history_dicts = []
        if request.recent_history:
            history_dicts = [
                {"role": msg.role, "text": msg.text}
                for msg in request.recent_history
            ]

        # STEP 1: Extract behaviors and get standalone query (LLM call)
        extraction_result = run_behavior_extraction_with_history(
            prompt=request.prompt,
            recent_history=history_dicts,
        )

        if not extraction_result.success:
            logger.error(f"Extraction failed: {extraction_result.error}")
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "success": False,
                    "data": None,
                    "error": extraction_result.error or "Extraction failed",
                },
            )

        logger.info(
            f"Extraction process is successful. extracted results : "
            f"{extraction_result.model_dump_json(indent=2)}"
        )

        # STEP 2: Search for related behaviors using 3D hybrid retrieval (FAST)
        related_behaviors = []
        hybrid_response = None
        if extraction_result.standalone_query:
            try:
                from services.openAiClient import embed_text

                query_embedding = embed_text(extraction_result.standalone_query)

                # 3D Hybrid Search: Dense + Sparse (BM25) + Metadata (intent filter)
                hybrid_response = search_similar_behavior_3D(
                    user_id=request.user_id,
                    query_embedding=query_embedding,
                    query_text=extraction_result.standalone_query,
                    session_id=request.session_id,
                    required_intents=extraction_result.required_intents,
                )
                logger.info(
                    f"[3D] Found {len(hybrid_response.results)} similar behaviors "
                    f"for standalone query: '{extraction_result.standalone_query}'"
                )
                logger.info(
                    f"[3D] Required intents boost: {extraction_result.required_intents}"
                )
                logger.info(
                    f"[3D] Hybrid search returned {len(hybrid_response.results)} results, "
                    f"{len(hybrid_response.decay_updates)} pending decay updates, "
                    f"{len(hybrid_response.accessed_behavior_ids)} access timestamps to update"
                )

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
                    f"(filtered "
                    f"{len(hybrid_response.results) - len(related_behaviors)} "
                    f"below threshold, "
                    f"distance_threshold: {RELATED_BEHAVIORS_DISTANCE_THRESHOLD})"
                )
            except Exception as e:
                logger.error(f"Failed to search related behaviors: {str(e)}")

        # STEP 3: Schedule behavior storage in background (ASYNC - non-blocking)
        background_tasks.add_task(
            _store_behaviors_async,
            extraction_result,
            request.user_id,
            request.session_id,
        )
        logger.info(
            f"Scheduled async storage of behaviors for user: {request.user_id}"
        )

        # STEP 3b: Schedule retrieval updates in background
        if hybrid_response and (
            hybrid_response.decay_updates or hybrid_response.accessed_behavior_ids
        ):
            background_tasks.add_task(
                persist_retrieval_updates_batch,
                hybrid_response.decay_updates,
                hybrid_response.accessed_behavior_ids,
                request.user_id,
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
                    "user_id": request.user_id,
                },
                "error": None,
            },
        )

    except ValueError as e:
        logger.exception("Validation error during extraction with history")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "data": None,
                "error": f"Validation error: {str(e)}",
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during extraction with history")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "data": None,
                "error": f"Internal server error: {str(e)}",
            },
        )


# ==============================================================================
# PROFILE SERVICE INTEGRATION ENDPOINTS
# ==============================================================================

@router.post(
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


@router.get(
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


@router.get(
    "/api/behaviors/{user_id}/recent",
    summary="Get recent profile signals for a user",
    description="Returns the most recent profile signals for a user. Used by Profile Service during drift fallback.",
    response_description="List of recent profile signals"
)
def get_recent_profile_signals(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of recent signals to return")
):
    """
    Get the most recent profile signals for a user.
    
    This endpoint is called by the Profile Service during drift fallback
    to retrieve historical behavior patterns for profile re-matching.
    
    Args:
        user_id: Unique user identifier
        limit: Maximum number of recent signals (default: 10, max: 50)
        
    Returns:
        JSON with user_id, count, and list of recent profile signals
    """
    try:
        signal_repo = get_profile_signal_repository()
        behaviors = signal_repo.get_recent(user_id=user_id, limit=limit)
        
        logger.info(f"Retrieved {len(behaviors)} recent profile signals for user={user_id}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "user_id": user_id,
                "count": len(behaviors),
                "behaviors": behaviors
            }
        )
        
    except Exception as e:
        logger.exception(f"Error fetching recent profile signals for user={user_id}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "user_id": user_id,
                "count": 0,
                "behaviors": [],
                "error": f"Failed to retrieve recent signals: {str(e)}"
            }
        )
