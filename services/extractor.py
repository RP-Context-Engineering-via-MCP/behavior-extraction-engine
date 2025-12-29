"""
Behavior extraction orchestrator.
Handles the workflow: raw prompt -> GPT extraction -> validated Pydantic models
"""

from typing import Dict, Any, List
from models.behavior import (
    ExtractionResult, 
    BehaviorSegment, 
    ExtractedBehavior, 
    StoredBehavior, 
    SimilarityClassification,
    ConflictAnalysisType,
    ConflictType,
    BehaviorState
)
from services.openAiClient import extract_behavior, embed_text, analyze_conflict
from services.credibilityCalculator import calculate_initial_credibility, should_store_behavior
from services.behaviorRepository import (
    insert_behavior, 
    insert_prompt_segment, 
    search_similar_behaviors, 
    reinforce_behavior,
    insert_conflict,
    supersede_behavior,
    update_behavior_state
)
from datetime import datetime
from config.configurations import (
    DEFAULT_DECAY_RATE,
    SAMPLE_USERID,
    CREDIBILITY_DIFFERENCE_THRESHOLD
)

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
                                f"Inserting both (Phase 1 behavior - variations allowed)."
                            )
                        
                        # PHASE 2: Conflict detection and resolution
                        elif top_match.classification == SimilarityClassification.POTENTIAL_CONFLICT:
                            logger.info(
                                f"Potential conflict detected (distance={top_match.distance:.4f}): "
                                f"'{behavior.description}' vs '{top_match.behavior_text}'. "
                                f"Initiating LLM conflict analysis..."
                            )
                            
                            try:
                                # Call GPT-4 to analyze if behaviors actually conflict
                                conflict_analysis = analyze_conflict(
                                    behavior_1_text=top_match.behavior_text,
                                    behavior_2_text=behavior.description,
                                    distance=top_match.distance
                                )
                                
                                logger.info(
                                    f"LLM analysis: {conflict_analysis.conflict_type.value} "
                                    f"(confidence: {conflict_analysis.confidence:.2f}) - "
                                    f"{conflict_analysis.explanation[:100]}..."
                                )
                                
                                # Handle based on conflict analysis
                                if conflict_analysis.conflict_type == ConflictAnalysisType.COMPATIBLE:
                                    # Behaviors can coexist - insert both
                                    logger.info(
                                        f"Behaviors are compatible. Inserting new behavior alongside existing one."
                                    )
                                    # Continue to insertion below
                                
                                elif conflict_analysis.conflict_type == ConflictAnalysisType.CONTEXT_DEPENDENT:
                                    # Depends on context - insert both with note
                                    logger.info(
                                        f"Behaviors are context-dependent. Inserting both with context notes."
                                    )
                                    
                                    # Create and insert new behavior first
                                    stored_behavior = StoredBehavior(
                                        user_id=user_id,
                                        behavior_text=behavior.description,
                                        credibility=initial_credibility,
                                        clarity_score=behavior.clarity,
                                        extraction_confidence=behavior.confidence,
                                        linguistic_strength=behavior.linguistic_strength,
                                        decay_rate=DEFAULT_DECAY_RATE,
                                        embedding=embedding_vector,
                                        prompt_history_ids=[segment_id],
                                        session_id="default"
                                    )
                                    new_behavior_id = stored_behavior.behavior_id
                                    
                                    stored_behaviors.append(stored_behavior)
                                    try:
                                        payload = stored_behavior.model_dump()
                                        insert_behavior(payload)
                                        logger.info(f"Inserted context-dependent behavior: {new_behavior_id}")
                                        
                                        # Now store conflict with both IDs
                                        conflict_id = insert_conflict(
                                            user_id=user_id,
                                            behavior_id_1=top_match.behavior_id,
                                            behavior_id_2=new_behavior_id,
                                            conflict_type=ConflictType.USER_DECISION_NEEDED,
                                            similarity_distance=top_match.distance,
                                            llm_analysis=f"CONTEXT_DEPENDENT: {conflict_analysis.explanation}"
                                        )
                                        logger.info(f"Logged context-dependent conflict: {conflict_id}")
                                    except Exception as e:
                                        logger.error(f"Failed to handle context-dependent conflict: {e}")
                                    
                                    # Skip normal insertion - already handled
                                    continue
                                
                                elif conflict_analysis.conflict_type == ConflictAnalysisType.CONFLICT:
                                    # Actual conflict detected - apply resolution strategy
                                    logger.warning(
                                        f"CONFLICT confirmed by LLM. Applying auto-resolution logic..."
                                    )
                                    
                                    # Compare credibility scores
                                    existing_credibility = top_match.credibility
                                    new_credibility = initial_credibility
                                    credibility_diff = abs(new_credibility - existing_credibility)
                                    
                                    logger.info(
                                        f"Credibility comparison: existing={existing_credibility:.3f}, "
                                        f"new={new_credibility:.3f}, diff={credibility_diff:.3f}"
                                    )
                                    
                                    # AUTO-RESOLUTION: Clear winner (diff > 0.3)
                                    if credibility_diff > CREDIBILITY_DIFFERENCE_THRESHOLD:
                                        if new_credibility > existing_credibility:
                                            # New behavior wins - supersede old one
                                            logger.info(
                                                f"New behavior has significantly higher credibility. "
                                                f"Superseding old behavior {top_match.behavior_id}."
                                            )
                                            
                                            # Create stored behavior first to get new behavior_id
                                            stored_behavior = StoredBehavior(
                                                user_id=user_id,
                                                behavior_text=behavior.description,
                                                credibility=initial_credibility,
                                                clarity_score=behavior.clarity,
                                                extraction_confidence=behavior.confidence,
                                                linguistic_strength=behavior.linguistic_strength,
                                                decay_rate=DEFAULT_DECAY_RATE,
                                                embedding=embedding_vector,
                                                prompt_history_ids=[segment_id],
                                                session_id="default"
                                            )
                                            new_behavior_id = stored_behavior.behavior_id
                                            
                                            # Mark old behavior as superseded
                                            supersede_behavior(
                                                old_behavior_id=top_match.behavior_id,
                                                new_behavior_id=new_behavior_id,
                                                user_id=user_id
                                            )
                                            
                                            # Store conflict record
                                            conflict_id = insert_conflict(
                                                user_id=user_id,
                                                behavior_id_1=top_match.behavior_id,
                                                behavior_id_2=new_behavior_id,
                                                conflict_type=ConflictType.RESOLVABLE,
                                                similarity_distance=top_match.distance,
                                                llm_analysis=f"AUTO-RESOLVED (new wins): {conflict_analysis.explanation}"
                                            )
                                            
                                            logger.info(
                                                f"Auto-resolved conflict {conflict_id}: "
                                                f"new behavior {new_behavior_id} superseded old {top_match.behavior_id}"
                                            )
                                            
                                            # Insert new behavior and add to list
                                            stored_behaviors.append(stored_behavior)
                                            try:
                                                payload = stored_behavior.model_dump()
                                                insert_behavior(payload)
                                                logger.info(f"Inserted winning behavior: {new_behavior_id}")
                                            except Exception as e:
                                                logger.error(f"Failed to insert winning behavior: {e}")
                                            
                                            # Skip normal insertion flow - already handled
                                            continue
                                        
                                        else:
                                            # Existing behavior wins - skip new insertion
                                            logger.info(
                                                f"Existing behavior has significantly higher credibility. "
                                                f"Skipping insertion of new behavior."
                                            )
                                            
                                            # Store conflict record showing existing won
                                            conflict_id = insert_conflict(
                                                user_id=user_id,
                                                behavior_id_1=top_match.behavior_id,
                                                behavior_id_2="",  # New behavior not inserted
                                                conflict_type=ConflictType.RESOLVABLE,
                                                similarity_distance=top_match.distance,
                                                llm_analysis=f"AUTO-RESOLVED (existing wins): {conflict_analysis.explanation}"
                                            )
                                            
                                            logger.info(
                                                f"Auto-resolved conflict {conflict_id}: "
                                                f"existing behavior {top_match.behavior_id} kept, new behavior rejected"
                                            )
                                            
                                            # Skip insertion
                                            continue
                                    
                                    # FLAGGED: No clear winner - needs user resolution (Phase 3)
                                    else:
                                        logger.warning(
                                            f"Credibility too close ({credibility_diff:.3f} <= {CREDIBILITY_DIFFERENCE_THRESHOLD}). "
                                            f"Flagging both behaviors for user resolution (Phase 3)."
                                        )
                                        
                                        # Mark existing behavior as FLAGGED
                                        update_behavior_state(
                                            behavior_id=top_match.behavior_id,
                                            user_id=user_id,
                                            new_state=BehaviorState.FLAGGED
                                        )
                                        
                                        # Create new behavior with FLAGGED state
                                        stored_behavior = StoredBehavior(
                                            user_id=user_id,
                                            behavior_text=behavior.description,
                                            credibility=initial_credibility,
                                            clarity_score=behavior.clarity,
                                            extraction_confidence=behavior.confidence,
                                            linguistic_strength=behavior.linguistic_strength,
                                            decay_rate=DEFAULT_DECAY_RATE,
                                            embedding=embedding_vector,
                                            prompt_history_ids=[segment_id],
                                            session_id="default"
                                        )
                                        new_behavior_id = stored_behavior.behavior_id
                                        
                                        # Insert new behavior FIRST
                                        stored_behaviors.append(stored_behavior)
                                        try:
                                            payload = stored_behavior.model_dump()
                                            payload['behavior_state'] = BehaviorState.FLAGGED.value
                                            insert_behavior(payload)
                                            logger.info(f"Inserted flagged behavior: {new_behavior_id}")
                                            
                                            # Now store conflict with both IDs existing in database
                                            conflict_id = insert_conflict(
                                                user_id=user_id,
                                                behavior_id_1=top_match.behavior_id,
                                                behavior_id_2=new_behavior_id,
                                                conflict_type=ConflictType.USER_DECISION_NEEDED,
                                                similarity_distance=top_match.distance,
                                                llm_analysis=f"FLAGGED FOR USER: {conflict_analysis.explanation}"
                                            )
                                        
                                            logger.info(
                                                f"Created conflict {conflict_id} requiring user decision. "
                                                f"Both behaviors flagged."
                                            )
                                        except Exception as e:
                                            logger.error(f"Failed to insert flagged behavior or conflict: {e}")
                                        
                                        # Skip normal insertion - already handled
                                        continue
                            
                            except Exception as e:
                                logger.error(
                                    f"Conflict analysis failed: {str(e)}. "
                                    f"Defaulting to conservative behavior: inserting both."
                                )
                                # On error, default to safe behavior: insert both

                
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
    
