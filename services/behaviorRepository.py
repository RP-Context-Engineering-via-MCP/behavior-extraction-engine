from db.connection import get_db_connection, get_db_pool_connection
from datetime import datetime
from typing import List, Tuple, Optional
from models.behavior import (
    PromptSegment, 
    SegmentInsertResult, 
    SimilarityClassification, 
    SimilarityResult, 
    ReinforcementResult,
    ConflictType,
    ResolutionStatus,
    BehaviorState
)
from services.credibilityCalculator import calculate_reinforcement_boost
from config.configurations import (
    DUPLICATE_THRESHOLD,
    SIMILAR_THRESHOLD,
    CONFLICT_THRESHOLD_MIN,
    CONFLICT_THRESHOLD_MAX
)
import time
import uuid
import logging
logger = logging.getLogger(__name__)

def insert_behavior(payload: dict):
    """
    Insert a new behavior into the database.
    
    The payload should include behavior_state (defaults to 'ACTIVE' if not provided).
    All new behaviors start in ACTIVE state unless explicitly specified otherwise.
    
    Args:
        payload: Dictionary containing all behavior fields including:
            - behavior_id, user_id, behavior_text, embedding
            - credibility, extraction metrics, timestamps
            - behavior_state (optional, defaults to 'ACTIVE')
    """
    # Ensure behavior_state is set (default to ACTIVE for new behaviors)
    if 'behavior_state' not in payload:
        payload['behavior_state'] = BehaviorState.ACTIVE.value
    
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO behaviors (
                    behavior_id,
                    user_id,
                    behavior_text,
                    embedding,
                    credibility,
                    extraction_confidence,
                    clarity_score,
                    linguistic_strength,
                    decay_rate,
                    reinforcement_count,
                    created_at,
                    last_seen_at,
                    session_id,
                    prompt_history_ids,
                    behavior_state
                )
                VALUES (
                    %(behavior_id)s,
                    %(user_id)s,
                    %(behavior_text)s,
                    %(embedding)s,
                    %(credibility)s,
                    %(extraction_confidence)s,
                    %(clarity_score)s,
                    %(linguistic_strength)s,
                    %(decay_rate)s,
                    %(reinforcement_count)s,
                    %(created_at)s,
                    %(last_seen_at)s,
                    %(session_id)s,
                    %(prompt_history_ids)s,
                    %(behavior_state)s
                )
                """
            , payload
            )
        conn.commit()

def insert_prompt_segment(segment_text: str, user_id: str) -> SegmentInsertResult:
    """
    Insert a prompt segment into prompt_segments table
    
    Args:
        segment_text: The segment text to store
        user_id: User identifier
        
    Returns:
        SegmentInsertResult with success, segment_id, and error fields
    """
    try:
        # Create PromptSegment model instance (generates ID and timestamp)
        segment = PromptSegment(
            user_id=user_id,
            segment_text=segment_text
        )
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompt_segments (
                        user_id,
                        segment_text,
                        created_at
                    )
                    VALUES (%s, %s, %s)
                    RETURNING segment_id;
                    """,
                    (segment.user_id, segment.segment_text, segment.created_at)
                )
                result = cur.fetchone()
                conn.commit()
                
                returned_id = result[0] if result else None
                # Convert UUID to string for Pydantic validation
                segment_id_str = str(returned_id) if returned_id else None
                logger.info(f"Inserted prompt segment: {segment_id_str} for user: {user_id}")
                
                return SegmentInsertResult(
                    success=True,
                    segment_id=segment_id_str,
                    error=None
                )
                
    except Exception as e:
        logger.error(f"Failed to insert prompt segment: {str(e)}")
        return SegmentInsertResult(
            success=False,
            segment_id=None,
            error=str(e)
        )


def reinforce_behavior(
    behavior_id: str,
    user_id: str,
    segment_id: Optional[str] = None
) -> ReinforcementResult:
    """
    Reinforce an existing behavior by incrementing reinforcement count,
    boosting credibility, updating timestamp, and optionally adding segment reference.
    
    This is called when a duplicate or highly similar behavior is detected,
    instead of inserting a new behavior.
    
    Args:
        behavior_id: ID of the behavior to reinforce
        user_id: User identifier (for validation)
        segment_id: Optional segment ID to add to prompt_history_ids
        
    Returns:
        ReinforcementResult with success status and updated values
        
    Process:
        1. Fetch current behavior data
        2. Calculate credibility boost (diminishing returns)
        3. Increment reinforcement_count
        4. Update last_seen_at timestamp
        5. Add segment_id to prompt_history_ids (if provided)
        6. Commit all changes
    """
    try:
        current_timestamp = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Step 1: Fetch current behavior data
                cur.execute(
                    """
                    SELECT 
                        credibility,
                        reinforcement_count,
                        prompt_history_ids
                    FROM behaviors
                    WHERE behavior_id = %s AND user_id = %s;
                    """,
                    (behavior_id, user_id)
                )
                
                result = cur.fetchone()
                
                if not result:
                    logger.error(f"Behavior {behavior_id} not found for user {user_id}")
                    return ReinforcementResult(
                        success=False,
                        behavior_id=behavior_id,
                        new_credibility=0.0,
                        new_reinforcement_count=0,
                        credibility_boost=0.0,
                        segment_id_added=None,
                        error=f"Behavior not found: {behavior_id}"
                    )
                
                current_credibility, current_count, prompt_history_ids = result
                
                # Step 2: Calculate credibility boost with diminishing returns
                boost = calculate_reinforcement_boost(
                    float(current_credibility),
                    int(current_count)
                )
                new_credibility = min(1.0, float(current_credibility) + boost)
                new_count = int(current_count) + 1
                
                # Step 3: Prepare updated prompt_history_ids
                updated_history_ids = list(prompt_history_ids) if prompt_history_ids else []
                if segment_id and segment_id not in updated_history_ids:
                    updated_history_ids.append(segment_id)
                
                # Step 4: Update behavior in database
                cur.execute(
                    """
                    UPDATE behaviors
                    SET 
                        credibility = %s,
                        reinforcement_count = %s,
                        last_seen_at = %s,
                        prompt_history_ids = %s
                    WHERE behavior_id = %s AND user_id = %s;
                    """,
                    (
                        new_credibility,
                        new_count,
                        current_timestamp,
                        updated_history_ids,
                        behavior_id,
                        user_id
                    )
                )
                
                conn.commit()
                
                logger.info(
                    f"Reinforced behavior {behavior_id}: "
                    f"credibility {current_credibility:.3f} → {new_credibility:.3f}, "
                    f"count {current_count} → {new_count}"
                )
                
                return ReinforcementResult(
                    success=True,
                    behavior_id=behavior_id,
                    new_credibility=new_credibility,
                    new_reinforcement_count=new_count,
                    credibility_boost=boost,
                    segment_id_added=segment_id if segment_id else None,
                    error=None
                )
                
    except Exception as e:
        logger.error(f"Failed to reinforce behavior {behavior_id}: {str(e)}")
        return ReinforcementResult(
            success=False,
            behavior_id=behavior_id,
            new_credibility=0.0,
            new_reinforcement_count=0,
            credibility_boost=0.0,
            segment_id_added=None,
            error=str(e)
        )


def classify_similarity(distance: float) -> SimilarityClassification:
    """
    Classify the relationship between two behaviors based on embedding distance.
    
    Uses cosine distance where lower values indicate higher similarity:
    - 0.00-DUPLICATE_THRESHOLD: DUPLICATE (exact match, different wording)
    - DUPLICATE_THRESHOLD-SIMILAR_THRESHOLD: SIMILAR (related variations)
    - SIMILAR_THRESHOLD-CONFLICT_THRESHOLD_MAX: POTENTIAL_CONFLICT (might be opposing)
    - CONFLICT_THRESHOLD_MAX+: UNRELATED (different domains)
    
    Note: SIMILAR_THRESHOLD should equal CONFLICT_THRESHOLD_MIN to avoid gaps
    
    Args:
        distance: Cosine distance between behavior embeddings (0.0-2.0)
        
    Returns:
        SimilarityClassification enum value
        
    Example:
        >>> classify_similarity(0.03)
        SimilarityClassification.DUPLICATE
        >>> classify_similarity(0.25)
        SimilarityClassification.POTENTIAL_CONFLICT
    """
    if distance < DUPLICATE_THRESHOLD:
        return SimilarityClassification.DUPLICATE
    elif distance < SIMILAR_THRESHOLD:
        return SimilarityClassification.SIMILAR
    elif distance < CONFLICT_THRESHOLD_MAX:
        return SimilarityClassification.POTENTIAL_CONFLICT
    else:
        return SimilarityClassification.UNRELATED


def search_similar_behaviors(
        user_id: str, 
        query_embedding: List[float], 
        limit: int = 5
) -> List[SimilarityResult]:
    """
    Find behaviors similar to query embedding using cosine similarity.
    
    Args:
        user_id: User identifier
        query_embedding: Vector embedding (3072 dimensions) as list of floats
        limit: Maximum number of results to return
        
    Returns:
        List of SimilarityResult objects with classification and metadata
        Sorted by similarity (lowest distance first)
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        behavior_id, 
                        behavior_text,
                        embedding <=> %s::vector AS distance,
                        credibility,
                        last_seen_at,
                        reinforcement_count
                    FROM behaviors
                    WHERE user_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW')
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding, user_id, limit)
                )
                
                results = cur.fetchall()
                
                # Convert to SimilarityResult objects with classification
                similarity_results = []
                for row in results:
                    behavior_id, behavior_text, distance, credibility, last_seen_at, reinforcement_count = row
                    
                    similarity_results.append(SimilarityResult(
                        behavior_id=behavior_id,
                        behavior_text=behavior_text,
                        distance=float(distance),
                        classification=classify_similarity(float(distance)),
                        credibility=float(credibility),
                        last_seen_at=int(last_seen_at),
                        reinforcement_count=int(reinforcement_count)
                    ))
                
                logger.debug(f"Found {len(similarity_results)} similar behaviors for user {user_id}")
                return similarity_results
                
    except Exception as e:
        logger.error(f"Failed to search similar behaviors: {str(e)}")
        return []


def search_similar_behaviors_raw(
        user_id: str, 
        query_embedding: List[float], 
        limit: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Find behaviors similar to query embedding using cosine similarity.
    
    Args:
        user_id: User identifier
        query_embedding: Vector embedding (3072 dimensions) as list of floats
        limit: Maximum number of results to return
        
    Returns:
        List of tuples: (behavior_id, behavior_text, distance)
        Lower distance = more similar
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT behavior_id, behavior_text,
                    embedding <=> %s AS distance
                FROM behaviors
                WHERE user_id = %s
                ORDER BY distance
                LIMIT 10;
                """,
                (query_embedding, user_id)
            )
            return cur.fetchall()
        

def insert_behavior_batch(payloads: List[dict]):
    "insert multiple behaviors in one single transaction"
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for payload in payloads:
                cur.execute(
                    """INSERT INTO behaviors (...) VALUES (...)"""
                , payload
                )
        conn.commit()
    logger.info(f"Inserted batch of {len(payloads)} behaviors into database")


def insert_conflict(
    user_id: str,
    behavior_id_1: str,
    behavior_id_2: str,
    conflict_type: ConflictType,
    similarity_distance: float,
    llm_analysis: Optional[str] = None
) -> str:
    """
    Store a detected conflict between two behaviors in the database.
    
    This creates an audit trail of all conflict detections and their resolutions.
    The conflict record tracks:
    - Which behaviors conflict
    - Type of conflict (resolvable vs needs user input)
    - LLM's analysis and reasoning
    - Resolution status and outcome
    
    Args:
        user_id: User whose behaviors conflict
        behavior_id_1: First behavior ID
        behavior_id_2: Second behavior ID
        conflict_type: RESOLVABLE or USER_DECISION_NEEDED
        similarity_distance: Embedding distance between behaviors
        llm_analysis: Optional LLM explanation of the conflict
        
    Returns:
        conflict_id: UUID of the created conflict record
        
    Raises:
        Exception: If database insertion fails
        
    Example:
        >>> conflict_id = insert_conflict(
        ...     "user123", "beh_abc", "beh_xyz",
        ...     ConflictType.RESOLVABLE, 0.22,
        ...     "Behaviors contradict in same domain"
        ... )
    """
    try:
        conflict_id = str(uuid.uuid4())
        current_timestamp = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO behavior_conflicts (
                        conflict_id,
                        user_id,
                        behavior_id_1,
                        behavior_id_2,
                        conflict_type,
                        similarity_distance,
                        llm_analysis,
                        resolution_status,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conflict_id,
                        user_id,
                        behavior_id_1,
                        behavior_id_2,
                        conflict_type.value,
                        similarity_distance,
                        llm_analysis,
                        ResolutionStatus.PENDING.value,
                        current_timestamp
                    )
                )
                conn.commit()
        
        logger.info(
            f"Stored conflict {conflict_id}: {behavior_id_1} <-> {behavior_id_2} "
            f"(distance: {similarity_distance:.3f}, type: {conflict_type.value})"
        )
        
        return conflict_id
        
    except Exception as e:
        logger.error(f"Failed to insert conflict: {str(e)}")
        raise Exception(f"Database error storing conflict: {str(e)}")


def update_behavior_state(
    behavior_id: str,
    user_id: str,
    new_state: BehaviorState
) -> bool:
    """
    Update the lifecycle state of a behavior.
    
    Behavior states track the lifecycle:
    - NEW: Just created (within 24 hours)
    - ACTIVE: Normal state, used for personalization
    - SUPERSEDED: Replaced by newer conflicting behavior
    - FLAGGED: Needs user resolution
    - ARCHIVED: Credibility decayed below threshold
    
    Args:
        behavior_id: Behavior to update
        user_id: User ID (required for partitioned table)
        new_state: Target state from BehaviorState enum
        
    Returns:
        True if update succeeded, False otherwise
        
    Raises:
        Exception: If database update fails
        
    Example:
        >>> update_behavior_state("beh_abc123", "user_xyz", BehaviorState.SUPERSEDED)
        True
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE behaviors
                    SET behavior_state = %s
                    WHERE behavior_id = %s AND user_id = %s
                    """,
                    (new_state.value, behavior_id, user_id)
                )
                
                rows_affected = cur.rowcount
                conn.commit()
                
                if rows_affected == 0:
                    logger.warning(
                        f"No behavior found to update: {behavior_id} for user {user_id}"
                    )
                    return False
                
                logger.info(
                    f"Updated behavior {behavior_id} state to {new_state.value}"
                )
                return True
                
    except Exception as e:
        logger.error(f"Failed to update behavior state: {str(e)}")
        raise Exception(f"Database error updating behavior state: {str(e)}")


def supersede_behavior(
    old_behavior_id: str,
    new_behavior_id: str,
    user_id: str
) -> bool:
    """
    Mark an old behavior as SUPERSEDED by a new one.
    
    This creates a link between the old and new behaviors, preserving history
    while indicating which behavior is currently active. The old behavior:
    - State changed to SUPERSEDED
    - superseded_by_id set to new behavior ID
    - No longer used for personalization
    - Kept in database for historical analysis
    
    Args:
        old_behavior_id: Behavior being superseded
        new_behavior_id: Behavior that supersedes it
        user_id: User ID (required for partitioned table)
        
    Returns:
        True if update succeeded, False otherwise
        
    Raises:
        Exception: If database update fails
        
    Example:
        >>> supersede_behavior("beh_old123", "beh_new456", "user_xyz")
        True
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Update old behavior to SUPERSEDED state and link to new one
                cur.execute(
                    """
                    UPDATE behaviors
                    SET 
                        behavior_state = %s,
                        superseded_by_id = %s
                    WHERE behavior_id = %s AND user_id = %s
                    """,
                    (
                        BehaviorState.SUPERSEDED.value,
                        new_behavior_id,
                        old_behavior_id,
                        user_id
                    )
                )
                
                rows_affected = cur.rowcount
                conn.commit()
                
                if rows_affected == 0:
                    logger.warning(
                        f"No behavior found to supersede: {old_behavior_id} for user {user_id}"
                    )
                    return False
                
                logger.info(
                    f"Superseded behavior {old_behavior_id} with {new_behavior_id}"
                )
                return True
                
    except Exception as e:
        logger.error(f"Failed to supersede behavior: {str(e)}")
        raise Exception(f"Database error superseding behavior: {str(e)}")

