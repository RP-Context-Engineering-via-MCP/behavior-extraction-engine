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

from api.schemas import BehaviorSimilarityRequest, ConflictResolutionRequest
from config.configurations import RELATED_BEHAVIORS_DISTANCE_THRESHOLD
from models.behavior import ExtractRequest, ExtractRequestWithHistory
from services.behaviorRepository import (
    get_behaviors_by_user,
    get_user_conflicts,
    get_graph_expanded_behaviors,
    persist_retrieval_updates_batch,
    resolve_conflict,
    search_similar_behavior_3D,
)
from services.extractor import (
    run_behavior_extraction,
    run_behavior_extraction_with_history,
    store_behavior,
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

        # STEP 2b: Graph expansion — 1-hop co-occurrence walk from embedding results
        associated_behaviors = []
        if related_behaviors:
            try:
                seed_ids = [b["behavior_id"] for b in related_behaviors]
                associated_behaviors = get_graph_expanded_behaviors(
                    user_id=request.user_id,
                    seed_behavior_ids=seed_ids,
                    limit=10,
                )
                logger.info(
                    f"[GRAPH] {len(seed_ids)} seed(s) → "
                    f"{len(associated_behaviors)} associated behavior(s)"
                )
            except Exception as e:
                logger.error(f"[GRAPH] Failed to expand graph: {str(e)}")

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
                    "associated_behaviors": associated_behaviors,
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
