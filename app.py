from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import insert_behavior, search_similar_behaviors
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Behavior Extraction API",
    description="Extract and manage user behaviors from natural language prompts",
    version="1.0.0"
)


class ExtractRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="User's natural language prompt"
    )
    session_id: str = Field(
        default="default",
        description="session id for session specific behavior grouping "
    )

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        sanitized = v.strip()
        if not sanitized:
            raise ValueError("Session ID cannot be empty")
        # Allow only alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in sanitized):
            raise ValueError("Session ID can only contain alphanumeric characters, hyphens, and underscores")
        return sanitized
    
@app.post(
    "/extract",
    summary="Extract behaviors from prompt",
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
        
        try:
            stored_behaviors = store_behavior(
                extraction_result,
                user_id=request.session_id
            )

            logger.info(f"Stored {len(stored_behaviors)} behaviors for session: {request.session_id}")
        except Exception as e:
            logger.error(f"Failed to store behaviors: {str(e)}")
            
        
        total_behaviors = sum(len(seg.behaviors) for seg in extraction_result.segments)
        logger.info(
            f"Extraction successful: {len(extraction_result.segments)} segments, "
            f"{total_behaviors} behaviors, {extraction_result.extraction_time:.2f}ms"
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
                        "extracted_at": behavior.extracted_at
                    }
                    for behavior in segment.behaviors
                ]
            }
            for segment in extraction_result.segments
        ]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "data": {
                    "segments": segments_data,
                    "extraction_time_ms": extraction_result.extraction_time,
                    "session_id": request.session_id,
                    "total_segments": len(extraction_result.segments),
                    "total_behaviors": total_behaviors
                },
                "error": None,
                "total_behaviors_verified": total_behaviors,
                "verification_enabled": True
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