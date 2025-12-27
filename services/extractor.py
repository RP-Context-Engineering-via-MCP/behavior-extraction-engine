"""
Behavior extraction orchestrator.
Handles the workflow: raw prompt -> GPT extraction -> validated Pydantic models
"""

from typing import Dict, Any, List
from models.behavior import ExtractionResult, BehaviorSegment, ExtractedBehavior, StoredBehavior, SimilarityClassification
from services.openAiClient import extract_behavior, embed_text
from services.credibilityCalculator import calculate_initial_credibility, should_store_behavior
from services.behaviorRepository import insert_behavior, insert_prompt_segment, search_similar_behaviors, reinforce_behavior
from datetime import datetime
from config.configurations import DEFAULT_DECAY_RATE,SAMPLE_USERID

import logging
logger = logging.getLogger(__name__)

def run_behavior_extraction(prompt: str) -> ExtractionResult:
    """
    Extract behaviors from a user prompt and return validated Pydantic models.
    
    Args:
        prompt: User's natural language prompt
        
    Returns:
        ExtractionResult object with segments, behaviors, and metadata
        
    Workflow:
        1. Call GPT-4 via openAiClient.extract_behavior()
        2. Validate response structure
        3. Convert to Pydantic models for type safety
        4. Return ExtractionResult
    """
    raw_response= extract_behavior(prompt)

    if not raw_response.get("success", False):
        return ExtractionResult(
            segments=[],
            success=False,
            error=raw_response.get("error", "Unknown extraction error"),
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    try:
        validated_segments = []
        
        for segment_data in raw_response.get("segments", []):
            segment_text = segment_data["text"]

            # Convert each behavior dict to ExtractedBehavior model
            validated_behaviors = []
            
            for behavior_data in segment_data.get("behaviors", []):
                # Pydantic will validate ranges and types automatically
                validated_behavior = ExtractedBehavior(
                    description=behavior_data["description"],
                    confidence=behavior_data["confidence"],
                    clarity=behavior_data["clarity"],
                    linguistic_strength=behavior_data["linguistic_strength"],
                    extracted_at=datetime.now().isoformat()  # Use ISO format for consistency
                )
                validated_behaviors.append(validated_behavior)
            
            # Create validated segment with validated behaviors only
            validated_segment = BehaviorSegment(
                text=segment_text,
                behaviors=validated_behaviors
            )
            validated_segments.append(validated_segment)
        
        # Return successful extraction result
        return ExtractionResult(
            segments=validated_segments,
            success=True,
            error=None,
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    except KeyError as e:
        # Missing required field in response
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Invalid response structure: missing field {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )
    
    except Exception as e:
        # Pydantic validation error or other unexpected error
        return ExtractionResult(
            segments=[],
            success=False,
            error=f"Failed to validate extraction result: {str(e)}",
            extraction_time=raw_response.get("metadata", {}).get("extraction_time_ms", 0.0)
        )


def store_behavior(
        extraction_result: ExtractionResult,
        user_id: str = SAMPLE_USERID
) -> List[StoredBehavior]:

        """
    Convert extracted behaviors to database-ready StoredBehavior objects.
    
    This function:
    1. Calculates initial credibility for each behavior
    2. Filters out low-quality behaviors (below threshold)
    3. Generates embeddings for behavior text
    4. Creates StoredBehavior instances ready for DB insertion
    
    Args:
        extraction_result: ExtractionResult from run_behavior_extraction()
        
    Returns:
        List of StoredBehavior objects ready for database insertion
        
    Example:
        >>> result = run_behavior_extraction("I prefer dark mode")
        >>> behaviors_to_store = prepare_behaviors_for_storage(result)
        >>> # behaviors_to_store is now ready for database insertion
    """
        if not extraction_result.success:
             logger.warning("Cannot prepare behaviors from failed extraction result")
             return []
        
        stored_behaviors = []

        for segment in extraction_result.segments:
             segment_id = None

             for behavior in segment.behaviors:
                #   calculate initial credibility
                initial_credibility = calculate_initial_credibility(
                    confidence=behavior.confidence,
                    clarity=behavior.clarity,
                    linguistic_strength=behavior.linguistic_strength,
                    behavior_text=behavior.description
                )

                # filter low quality behaviors
                if not should_store_behavior(initial_credibility):
                    logger.info(
                        f"Pruning behavior due to low credibility: "
                        f"'{behavior.description}' (credibility={initial_credibility})"
                    )
                    continue

                if segment_id is None:
                    segment_result = insert_prompt_segment(
                        segment_text=segment.text,
                        user_id=user_id
                    )
                    if segment_result.success:
                        segment_id = segment_result.segment_id
                    else:
                        logger.error(
                            f"Failed to insert prompt segment into database: {segment_result.error}"
                        )
                        continue

                # generate embedding for behavior text
                try:
                    embedding_vector = embed_text(behavior.description)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for behavior: {str(e)}")
                    continue

                # PHASE 1: Duplicate detection and reinforcement
                # Search for similar behaviors before inserting
                try:
                    similar_behaviors = search_similar_behaviors(
                        user_id=user_id,
                        query_embedding=embedding_vector,
                        limit=5  # Only check top 5 for performance
                    )
                    
                    # Check if duplicate found (distance < 0.05)
                    if similar_behaviors and similar_behaviors[0].classification == SimilarityClassification.DUPLICATE:
                        duplicate = similar_behaviors[0]
                        logger.info(
                            f"Duplicate detected: '{behavior.description}' matches existing behavior "
                            f"'{duplicate.behavior_text}' (distance={duplicate.distance:.4f})"
                        )
                        
                        # Reinforce existing behavior instead of inserting new one
                        reinforce_result = reinforce_behavior(
                            behavior_id=duplicate.behavior_id,
                            user_id=user_id,
                            segment_id=segment_id
                        )
                        
                        if reinforce_result.success:
                            logger.info(
                                f"Successfully reinforced behavior {duplicate.behavior_id}: "
                                f"credibility {duplicate.credibility:.3f} → "
                                f"{reinforce_result.new_credibility:.3f}, "
                                f"count {duplicate.reinforcement_count} → {reinforce_result.new_reinforcement_count}"
                            )
                            # Skip insertion - behavior already exists and reinforced
                            continue
                        else:
                            logger.error(
                                f"Failed to reinforce duplicate behavior: {reinforce_result.error}. "
                                f"Proceeding with insertion as fallback."
                            )
                            # Fallback: insert as new behavior if reinforcement fails
                    
                    # Log similar/conflict cases for Phase 2 implementation
                    elif similar_behaviors:
                        top_match = similar_behaviors[0]
                        if top_match.classification == SimilarityClassification.SIMILAR:
                            logger.info(
                                f"Similar behavior detected (distance={top_match.distance:.4f}): "
                                f"'{behavior.description}' ~ '{top_match.behavior_text}'. "
                                f"Inserting both for Phase 1 (Phase 2 will handle merge logic)."
                            )
                        elif top_match.classification == SimilarityClassification.POTENTIAL_CONFLICT:
                            logger.warning(
                                f"Potential conflict detected (distance={top_match.distance:.4f}): "
                                f"'{behavior.description}' vs '{top_match.behavior_text}'. "
                                f"Inserting for Phase 1 (Phase 2 will add LLM conflict analysis)."
                            )
                
                except Exception as e:
                    logger.error(
                        f"Similarity search failed for '{behavior.description}': {str(e)}. "
                        f"Proceeding with insertion without duplicate check."
                    )
                    # Continue with insertion if similarity search fails
                
                # Create stored behavior instance (if not duplicate)
                stored_behavior = StoredBehavior(
                    user_id=user_id,
                    behavior_text=behavior.description,
                    credibility=initial_credibility,
                    clarity_score=behavior.clarity,
                    extraction_confidence=behavior.confidence,
                    linguistic_strength=behavior.linguistic_strength,
                    decay_rate=DEFAULT_DECAY_RATE,  # 0.015 from config
                    embedding=embedding_vector,
                    # prompt_history_ids will be populated later when we handle segment storage
                    prompt_history_ids=[segment_id],
                    # session_id will be passed from API layer later
                    session_id="default"
                )

                stored_behaviors.append(stored_behavior)

                try:
                    payload = stored_behavior.model_dump()
                    insert_behavior(payload)
                    logger.info(f"Successfully saved behavior to database: {stored_behavior.behavior_id}")
                except Exception as e:
                    logger.error(f"Failed to insert behavior into database: {str(e)}")

        logger.info(
            f"Prepared {len(stored_behaviors)} behaviors for storage "
            f"from {sum(len(seg.behaviors) for seg in extraction_result.segments)} extracted"
        )        
        return stored_behaviors
    
