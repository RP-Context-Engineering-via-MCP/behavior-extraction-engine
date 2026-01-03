from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import insert_behavior, search_similar_behaviors
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
