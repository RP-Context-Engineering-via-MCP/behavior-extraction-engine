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
from services.credibilityCalculator import calculate_reinforcement_boost, apply_lazy_decay
from config.configurations import DECAY_GRACE_PERIOD_SECONDS
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
                    last_decay_applied_at,
                    last_accessed_at,
                    session_id,
                    prompt_history_ids,
                    behavior_state,
                    intent,
                    target,
                    context,
                    polarity
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
                    %(last_decay_applied_at)s,
                    %(last_accessed_at)s,
                    %(session_id)s,
                    %(prompt_history_ids)s,
                    %(behavior_state)s,
                    %(intent)s,
                    %(target)s,
                    %(context)s,
                    %(polarity)s
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
                # Reset last_decay_applied_at to current time when reinforced
                # Set last_accessed_at to mark this behavior as actively used
                cur.execute(
                    """
                    UPDATE behaviors
                    SET 
                        credibility = %s,
                        reinforcement_count = %s,
                        last_seen_at = %s,
                        last_decay_applied_at = %s,
                        last_accessed_at = %s,
                        prompt_history_ids = %s
                    WHERE behavior_id = %s AND user_id = %s;
                    """,
                    (
                        new_credibility,
                        new_count,
                        current_timestamp,
                        current_timestamp,
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



def get_user_behaviors(user_id: str, include_states: List[str] = None) -> List[dict]:
    """
    Get all behaviors for a user
    
    Args:
        user_id: User identifier
        include_states: List of behavior states to include (defaults to ['ACTIVE', 'NEW'])
        
    Returns:
        List of behavior dictionaries with all fields
    """
    if include_states is None:
        include_states = ['ACTIVE', 'NEW']
    
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        behavior_id,
                        user_id,
                        behavior_text,
                        credibility,
                        clarity_score,
                        extraction_confidence,
                        linguistic_strength,
                        reinforcement_count,
                        last_seen_at,
                        created_at,
                        behavior_state,
                        intent,
                        target,
                        context,
                        polarity
                    FROM behaviors
                    WHERE user_id = %s
                    AND behavior_state = ANY(%s)
                    ORDER BY created_at DESC;
                    """,
                    (user_id, include_states)
                )
                
                results = cur.fetchall()
                
                behaviors = []
                for row in results:
                    behaviors.append({
                        'behavior_id': row[0],
                        'user_id': row[1],
                        'behavior_text': row[2],
                        'credibility': float(row[3]),
                        'clarity_score': float(row[4]) if row[4] else None,
                        'extraction_confidence': float(row[5]) if row[5] else None,
                        'linguistic_strength': float(row[6]) if row[6] else None,
                        'reinforcement_count': int(row[7]),
                        'last_seen_at': int(row[8]),
                        'created_at': int(row[9]),
                        'behavior_state': row[10],
                        'intent': row[11],
                        'target': row[12],
                        'context': row[13],
                        'polarity': row[14]
                    })
                
                logger.debug(f"Found {len(behaviors)} behaviors for user {user_id}")
                return behaviors
                
    except Exception as e:
        logger.error(f"Failed to get user behaviors: {str(e)}")
        return []


def search_similar_behaviors(
        user_id: str, 
        query_embedding: List[float], 
        limit: int = 5
) -> List[SimilarityResult]:
    """
    Find behaviors similar to query embedding using cosine similarity.
    
    Applies lazy decay to credibility on-the-fly when retrieving behaviors.
    If decay is applied, updates the behavior in the database with new credibility.
    
    Returns:
        List of SimilarityResult objects with classification and metadata
        Sorted by similarity (lowest distance first)
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Fetch behaviors with decay-related fields
                cur.execute(
                    """
                    SELECT 
                        behavior_id, 
                        behavior_text,
                        embedding <=> %s::vector AS distance,
                        credibility,
                        last_seen_at,
                        reinforcement_count,
                        intent,
                        target,
                        context,
                        polarity,
                        decay_rate,
                        last_decay_applied_at
                    FROM behaviors
                    WHERE user_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW', 'FLAGGED')
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding, user_id, limit)
                )
                
                results = cur.fetchall()
                current_time = int(time.time())
                
                # Convert to SimilarityResult objects and apply lazy decay
                similarity_results = []
                behaviors_to_update = []
                
                for row in results:
                    behavior_id, behavior_text, distance, stored_credibility, last_seen_at, reinforcement_count, intent, target, context, polarity, decay_rate, last_decay_applied_at = row
                    
                    # Apply lazy decay to credibility
                    new_credibility, decay_applied, days_elapsed = apply_lazy_decay(
                        stored_credibility=float(stored_credibility),
                        decay_rate=float(decay_rate),
                        last_decay_applied_at=last_decay_applied_at,
                        current_time=current_time
                    )
                    
                    # Track behaviors that need credibility update in DB
                    if decay_applied:
                        behaviors_to_update.append((
                            new_credibility,
                            current_time,
                            behavior_id,
                            user_id
                        ))
                        logger.debug(
                            f"Lazy decay applied to {behavior_id}: "
                            f"{stored_credibility:.4f} → {new_credibility:.4f} "
                            f"({days_elapsed} days)"
                        )
                    
                    similarity_results.append(SimilarityResult(
                        behavior_id=behavior_id,
                        behavior_text=behavior_text,
                        distance=float(distance),
                        credibility=new_credibility,  # Use decayed credibility
                        last_seen_at=int(last_seen_at),
                        reinforcement_count=int(reinforcement_count),
                        intent=intent,
                        target=target,
                        context=context if context else "general",
                        polarity=polarity
                    ))
                
                # Batch update behaviors with decayed credibility
                if behaviors_to_update:
                    cur.executemany(
                        """
                        UPDATE behaviors
                        SET credibility = %s,
                            last_decay_applied_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        behaviors_to_update
                    )
                    conn.commit()
                    logger.info(
                        f"Updated {len(behaviors_to_update)} behaviors with lazy decay"
                    )
                
                logger.debug(f"Found {len(similarity_results)} similar behaviors for user {user_id}")
                return similarity_results
                
    except Exception as e:
        logger.error(f"Failed to search similar behaviors: {str(e)}")
        return []
        

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
    - last_accessed_at updated (behavior was actively used in conflict resolution)
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
        current_timestamp = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Update old behavior to SUPERSEDED state and link to new one
                # Set last_accessed_at to mark it was actively used in conflict resolution
                cur.execute(
                    """
                    UPDATE behaviors
                    SET 
                        behavior_state = %s,
                        superseded_by_id = %s,
                        last_accessed_at = %s
                    WHERE behavior_id = %s AND user_id = %s
                    """,
                    (
                        BehaviorState.SUPERSEDED.value,
                        new_behavior_id,
                        current_timestamp,
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
                    f"Superseded behavior {old_behavior_id} with {new_behavior_id} "
                    f"(last_accessed_at updated)"
                )
                return True
                
    except Exception as e:
        logger.error(f"Failed to supersede behavior: {str(e)}")
        raise Exception(f"Database error superseding behavior: {str(e)}")


def update_behavior_access_time(
    behavior_id: str,
    user_id: str
) -> bool:
    """
    Update last_accessed_at timestamp for a behavior.
    
    Called when a behavior is actively confirmed or used in system decisions:
    - When existing behavior wins in conflict resolution (IGNORE_NEW)
    - When behavior is selected by user in conflict resolution
    - When behavior is used for context enrichment
    
    Args:
        behavior_id: Behavior that was accessed
        user_id: User ID (required for partitioned table)
        
    Returns:
        True if update succeeded, False otherwise
        
    Example:
        >>> update_behavior_access_time("beh_abc123", "user_xyz")
        True
    """
    try:
        current_timestamp = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE behaviors
                    SET last_accessed_at = %s
                    WHERE behavior_id = %s AND user_id = %s
                    """,
                    (current_timestamp, behavior_id, user_id)
                )
                
                rows_affected = cur.rowcount
                conn.commit()
                
                if rows_affected == 0:
                    logger.warning(
                        f"No behavior found to update access time: {behavior_id} for user {user_id}"
                    )
                    return False
                
                logger.debug(
                    f"Updated last_accessed_at for behavior {behavior_id}"
                )
                return True
                
    except Exception as e:
        logger.error(f"Failed to update behavior access time: {str(e)}")
        return False


def get_behaviors_by_user(user_id: str) -> List[dict]:
    """
    Get all behaviors for a specific user.
    
    Applies lazy decay to credibility on-the-fly when retrieving behaviors.
    If decay is applied, updates the behavior in the database with new credibility.
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of behavior dictionaries with all fields (credibility reflects decay)
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        behavior_id,
                        user_id,
                        behavior_text,
                        credibility,
                        reinforcement_count,
                        decay_rate,
                        created_at,
                        last_seen_at,
                        prompt_history_ids,
                        clarity_score,
                        extraction_confidence,
                        linguistic_strength,
                        session_id,
                        behavior_state,
                        intent,
                        target,
                        context,
                        polarity,
                        last_decay_applied_at,
                        last_accessed_at
                    FROM behaviors
                    WHERE user_id = %s
                    ORDER BY last_seen_at DESC
                    """,
                    (user_id,)
                )
                
                current_time = int(time.time())
                behaviors = []
                behaviors_to_update = []
                
                for row in cur.fetchall():
                    behavior_id = row[0]
                    stored_credibility = row[3]
                    decay_rate = row[5]
                    last_decay_applied_at = row[18]
                    last_accessed_at = row[19]
                    
                    # Apply lazy decay to credibility
                    new_credibility, decay_applied, days_elapsed = apply_lazy_decay(
                        stored_credibility=float(stored_credibility),
                        decay_rate=float(decay_rate),
                        last_decay_applied_at=last_decay_applied_at,
                        current_time=current_time
                    )
                    
                    # Track behaviors that need credibility update in DB
                    if decay_applied:
                        behaviors_to_update.append((
                            new_credibility,
                            current_time,
                            behavior_id,
                            user_id
                        ))
                        logger.debug(
                            f"Lazy decay applied to {behavior_id}: "
                            f"{stored_credibility:.4f} → {new_credibility:.4f} "
                            f"({days_elapsed} days)"
                        )
                    
                    behaviors.append({
                        "behavior_id": behavior_id,
                        "user_id": row[1],
                        "behavior_text": row[2],
                        "credibility": new_credibility,  # Use decayed credibility
                        "reinforcement_count": row[4],
                        "decay_rate": decay_rate,
                        "created_at": row[6],
                        "last_seen_at": row[7],
                        "prompt_history_ids": row[8],
                        "clarity_score": row[9],
                        "extraction_confidence": row[10],
                        "linguistic_strength": row[11],
                        "session_id": row[12],
                        "behavior_state": row[13],
                        "intent": row[14],
                        "target": row[15],
                        "context": row[16],
                        "polarity": row[17],
                        "last_accessed_at": last_accessed_at
                    })
                
                # Batch update behaviors with decayed credibility
                if behaviors_to_update:
                    cur.executemany(
                        """
                        UPDATE behaviors
                        SET credibility = %s,
                            last_decay_applied_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        behaviors_to_update
                    )
                    conn.commit()
                    logger.info(
                        f"Updated {len(behaviors_to_update)} behaviors with lazy decay"
                    )
                
                return behaviors
                
    except Exception as e:
        logger.error(f"Failed to get behaviors for user {user_id}: {str(e)}")
        raise Exception(f"Database error retrieving behaviors: {str(e)}")


def get_user_conflicts(user_id: str) -> List[dict]:
    """
    Get all conflicts for a specific user.
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of conflict dictionaries with behavior details
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        c.conflict_id,
                        c.user_id,
                        c.behavior_id_1,
                        c.behavior_id_2,
                        c.conflict_type,
                        c.similarity_distance,
                        c.llm_analysis,
                        c.resolution_status,
                        c.resolved_at,
                        c.resolution_choice,
                        c.created_at,
                        b1.behavior_text as behavior_1_text,
                        b2.behavior_text as behavior_2_text,
                        b1.credibility as behavior_1_credibility,
                        b2.credibility as behavior_2_credibility,
                        b1.behavior_state as behavior_1_state,
                        b2.behavior_state as behavior_2_state
                    FROM behavior_conflicts c
                    LEFT JOIN behaviors b1 ON c.behavior_id_1 = b1.behavior_id
                    LEFT JOIN behaviors b2 ON c.behavior_id_2 = b2.behavior_id
                    WHERE c.user_id = %s
                    ORDER BY c.created_at DESC
                    """,
                    (user_id,)
                )
                
                conflicts = []
                for row in cur.fetchall():
                    conflicts.append({
                        "conflict_id": str(row[0]) if row[0] else None,  # Convert UUID to string
                        "user_id": row[1],
                        "behavior_id_1": row[2],
                        "behavior_id_2": row[3],
                        "conflict_type": row[4],
                        "similarity_distance": row[5],
                        "llm_analysis": row[6],
                        "resolution_status": row[7],
                        "resolved_at": row[8],
                        "resolution_choice": row[9],
                        "created_at": row[10],
                        "behavior_1_text": row[11],
                        "behavior_2_text": row[12],
                        "behavior_1_credibility": row[13],
                        "behavior_2_credibility": row[14],
                        "behavior_1_state": row[15],
                        "behavior_2_state": row[16]
                    })
                
                return conflicts
                
    except Exception as e:
        logger.error(f"Failed to get conflicts for user {user_id}: {str(e)}")
        raise Exception(f"Database error retrieving conflicts: {str(e)}")


def resolve_conflict(
    conflict_id: str,
    user_id: str,
    resolution_choice: str
) -> dict:
    """
    Resolve a conflict based on user's decision.
    
    Handles three resolution scenarios:
    - OLD_WINS: Reinforce existing behavior, invalidate new (credibility = 0.0 for pruning)
    - NEW_WINS: Set new behavior to ACTIVE, supersede old behavior
    - BOTH_CORRECT: Reinforce both behaviors, set both to ACTIVE
    
    In all cases:
    - Updates last_accessed_at (behavior was actively used in conflict resolution)
    - Updates conflict record with resolution status and choice
    - Maintains audit trail
    
    Args:
        conflict_id: UUID of the conflict to resolve
        user_id: User ID (for validation)
        resolution_choice: One of "OLD_WINS", "NEW_WINS", "BOTH_CORRECT"
        
    Returns:
        Dictionary with resolution details and updated behavior states
        
    Raises:
        ValueError: If resolution_choice is invalid or conflict not found
        Exception: If database operations fail
        
    Example:
        >>> result = resolve_conflict(
        ...     "conf_abc123",
        ...     "user_xyz",
        ...     "OLD_WINS"
        ... )
        >>> print(result["resolution_status"])
        "USER_RESOLVED"
    """
    try:
        # Validate resolution choice
        valid_choices = ["OLD_WINS", "NEW_WINS", "BOTH_CORRECT"]
        if resolution_choice not in valid_choices:
            raise ValueError(
                f"Invalid resolution_choice: {resolution_choice}. "
                f"Must be one of {valid_choices}"
            )
        
        current_timestamp = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Step 1: Fetch conflict details
                cur.execute(
                    """
                    SELECT 
                        conflict_id,
                        user_id,
                        behavior_id_1,
                        behavior_id_2,
                        resolution_status
                    FROM behavior_conflicts
                    WHERE conflict_id = %s AND user_id = %s
                    """,
                    (conflict_id, user_id)
                )
                
                conflict = cur.fetchone()
                
                if not conflict:
                    raise ValueError(
                        f"Conflict {conflict_id} not found for user {user_id}"
                    )
                
                if conflict[4] != ResolutionStatus.PENDING.value:
                    raise ValueError(
                        f"Conflict {conflict_id} already resolved with status: {conflict[4]}"
                    )
                
                behavior_id_1 = conflict[2]  # Old/existing behavior
                behavior_id_2 = conflict[3]  # New behavior
                
                logger.info(
                    f"Resolving conflict {conflict_id}: "
                    f"{behavior_id_1} vs {behavior_id_2} -> {resolution_choice}"
                )
                
                # Step 2: Handle resolution based on user's choice
                if resolution_choice == "OLD_WINS":
                    # Reinforce old behavior (increases credibility, updates timestamps)
                    reinforce_result = reinforce_behavior(
                        behavior_id=behavior_id_1,
                        user_id=user_id,
                        segment_id=None
                    )
                    
                    if not reinforce_result.success:
                        raise Exception(f"Failed to reinforce old behavior: {reinforce_result.error}")
                    
                    # Update old behavior state to ACTIVE and last_accessed_at
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET 
                            behavior_state = %s,
                            last_accessed_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (BehaviorState.ACTIVE.value, current_timestamp, behavior_id_1, user_id)
                    )
                    
                    # Invalidate new behavior by setting credibility to 0.0
                    # User confirmed new behavior is incorrect, so mark it for pruning
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET 
                            credibility = 0.0,
                            last_accessed_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (current_timestamp, behavior_id_2, user_id)
                    )
                    
                    logger.info(
                        f"OLD_WINS: Reinforced {behavior_id_1} (set to ACTIVE), "
                        f"invalidated {behavior_id_2} (credibility set to 0.0 for pruning)"
                    )
                
                elif resolution_choice == "NEW_WINS":
                    # Set new behavior to ACTIVE
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET 
                            behavior_state = %s,
                            last_accessed_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (BehaviorState.ACTIVE.value, current_timestamp, behavior_id_2, user_id)
                    )
                    
                    # Supersede old behavior (sets state to SUPERSEDED, links to new, updates last_accessed_at)
                    supersede_success = supersede_behavior(
                        old_behavior_id=behavior_id_1,
                        new_behavior_id=behavior_id_2,
                        user_id=user_id
                    )
                    
                    if not supersede_success:
                        raise Exception(f"Failed to supersede old behavior {behavior_id_1}")
                    
                    logger.info(
                        f"NEW_WINS: Set {behavior_id_2} to ACTIVE, "
                        f"superseded {behavior_id_1}"
                    )
                
                elif resolution_choice == "BOTH_CORRECT":
                    # Reinforce both behaviors
                    reinforce_result_1 = reinforce_behavior(
                        behavior_id=behavior_id_1,
                        user_id=user_id,
                        segment_id=None
                    )
                    
                    if not reinforce_result_1.success:
                        raise Exception(f"Failed to reinforce behavior 1: {reinforce_result_1.error}")
                    
                    reinforce_result_2 = reinforce_behavior(
                        behavior_id=behavior_id_2,
                        user_id=user_id,
                        segment_id=None
                    )
                    
                    if not reinforce_result_2.success:
                        raise Exception(f"Failed to reinforce behavior 2: {reinforce_result_2.error}")
                    
                    # Set both to ACTIVE and update last_accessed_at
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET 
                            behavior_state = %s,
                            last_accessed_at = %s
                        WHERE behavior_id IN (%s, %s) AND user_id = %s
                        """,
                        (
                            BehaviorState.ACTIVE.value,
                            current_timestamp,
                            behavior_id_1,
                            behavior_id_2,
                            user_id
                        )
                    )
                    
                    logger.info(
                        f"BOTH_CORRECT: Reinforced both {behavior_id_1} and {behavior_id_2}, "
                        f"set both to ACTIVE"
                    )
                
                # Step 3: Update conflict record
                cur.execute(
                    """
                    UPDATE behavior_conflicts
                    SET 
                        resolution_status = %s,
                        resolution_choice = %s,
                        resolved_at = %s
                    WHERE conflict_id = %s AND user_id = %s
                    """,
                    (
                        ResolutionStatus.USER_RESOLVED.value,
                        resolution_choice,
                        current_timestamp,
                        conflict_id,
                        user_id
                    )
                )
                
                # Step 4: Fetch updated behavior states
                cur.execute(
                    """
                    SELECT 
                        behavior_id,
                        behavior_text,
                        behavior_state,
                        credibility,
                        last_accessed_at
                    FROM behaviors
                    WHERE behavior_id IN (%s, %s) AND user_id = %s
                    """,
                    (behavior_id_1, behavior_id_2, user_id)
                )
                
                behaviors = []
                for row in cur.fetchall():
                    behaviors.append({
                        "behavior_id": row[0],
                        "behavior_text": row[1],
                        "behavior_state": row[2],
                        "credibility": row[3],
                        "last_accessed_at": row[4]
                    })
                
                conn.commit()
                
                logger.info(
                    f"Conflict {conflict_id} resolved successfully: {resolution_choice}"
                )
                
                return {
                    "success": True,
                    "conflict_id": conflict_id,
                    "resolution_status": ResolutionStatus.USER_RESOLVED.value,
                    "resolution_choice": resolution_choice,
                    "resolved_at": current_timestamp,
                    "behaviors": behaviors
                }
                
    except ValueError as e:
        logger.error(f"Validation error resolving conflict {conflict_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict {conflict_id}: {str(e)}")
        raise Exception(f"Database error resolving conflict: {str(e)}")

