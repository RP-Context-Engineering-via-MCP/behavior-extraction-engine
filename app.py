from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import insert_behavior, search_similar_behaviors
from models.behavior import ExtractRequest
from db.connection import close_db_pool, init_db_pool
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
@app.post(
    "/extract",
    summary="Extract b ehaviors from prompt",
    description="Analyzes a natural language prompt and extracts user behaviors, preferences, and patterns",
    response_description="Extraction result with segmented behaviors"
)
def extract_behaviors(request: ExtractRequest):
    try:
        logger.info(f"Received extraction request for session: {request.session_id}")

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
                user_id=request.session_id
            )

            logger.info(f"Stored {len(stored_behaviors)} behaviors for session: {request.session_id}")
        except Exception as e:
            logger.error(f"Failed to store behaviors: {str(e)}")
            # stored_behaviors remains as empty list if error occurs
            
        
        total_behaviors = sum(len(seg.behaviors) for seg in extraction_result.segments)
        logger.info(
            f"Extraction successful: {len(extraction_result.segments)} segments, "
            f"{total_behaviors} behaviors, {extraction_result.extraction_time:.2f}ms"
        )

        # Prepare extracted behaviors data
        segments_data = [
            {
                "text": segment.text,
                "behaviors": [
                    {
                        "description": behavior.description,
                        "confidence": behavior.confidence,
                        "clarity": behavior.clarity,
                        "linguistic_strength": behavior.linguistic_strength,
                        "extracted_at": behavior.extracted_at
                    }
                    for behavior in segment.behaviors
                ]
            }
            for segment in extraction_result.segments
        ]

        # Prepare stored behaviors data with all fields
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
                "embedding_dimensions": len(stored_behavior.embedding) if stored_behavior.embedding else 0
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
                    "session_id": request.session_id
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