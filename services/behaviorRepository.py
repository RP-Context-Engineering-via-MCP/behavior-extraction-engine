from db.connection import get_db_connection, get_db_pool_connection
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
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
from services.eventPublisher import get_event_publisher
from config.configurations import DECAY_GRACE_PERIOD_SECONDS
import time
import uuid
import logging
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResponse:
    """
    Response from search_similar_behavior_3D.
    
    Contains both the search results (returned immediately to the client)
    and pending DB updates (processed asynchronously after response is sent).
    
    Attributes:
        results: List of SimilarityResult objects sorted by hybrid_score
        decay_updates: Tuples of (new_credibility, timestamp, behavior_id, user_id)
                       for behaviors where lazy decay was applied in-memory
        accessed_behavior_ids: All behavior_ids that were returned in results,
                               used to update last_accessed_at timestamps
    """
    results: List[SimilarityResult] = field(default_factory=list)
    decay_updates: List[tuple] = field(default_factory=list)
    accessed_behavior_ids: List[str] = field(default_factory=list)

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
    
    # Build enriched search text for tsvector (behavior_text + target + context)
    # Passed as a single parameter to avoid PostgreSQL type inference conflicts
    # (varchar column params vs text in to_tsvector)
    search_text = payload.get('behavior_text', '')
    target_val = payload.get('target') or ''
    context_val = payload.get('context') or ''
    payload['search_text'] = f"{search_text} {target_val} {context_val}".strip()
    
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO behaviors (
                    behavior_id,
                    user_id,
                    behavior_text,
                    embedding,
                    canonical_embedding,
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
                    polarity,
                    search_vector
                )
                VALUES (
                    %(behavior_id)s,
                    %(user_id)s,
                    %(behavior_text)s,
                    %(embedding)s,
                    %(canonical_embedding)s,
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
                    %(polarity)s,
                    to_tsvector('english', %(search_text)s)
                )
                """
            , payload
            )
        conn.commit()
    
    # Publish behavior.created event for drift detection
    try:
        publisher = get_event_publisher()
        publisher.publish_behavior_created(
            user_id=payload.get('user_id'),
            behavior_id=payload.get('behavior_id'),
            target=payload.get('target') or '',
            intent=payload.get('intent') or '',
            context=payload.get('context') or '',
            polarity=payload.get('polarity') or '',
            credibility=payload.get('credibility', 0.0),
            reinforcement_count=payload.get('reinforcement_count', 1),
            state=payload.get('behavior_state', 'ACTIVE'),
            created_at=payload.get('created_at', int(time.time())),
            last_seen_at=payload.get('last_seen_at', int(time.time()))
        )
    except Exception as e:
        logger.warning(f"Failed to publish behavior.created event: {e}")

def insert_prompt_segment(segment_text: str, user_id: str) -> SegmentInsertResult:
    """
    Insert a prompt segment into prompt_segments table, or return the existing
    segment_id if the same (user_id, segment_text) pair was already stored.

    Idempotent: re-submitting the same prompt (e.g. network retry or duplicate
    API call) returns the original segment_id rather than creating a second row,
    preventing duplicate entries in behaviors.prompt_history_ids.

    Args:
        segment_text: The segment text to store
        user_id: User identifier

    Returns:
        SegmentInsertResult with success, segment_id, and error fields
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Check whether this exact segment already exists for this user.
                # Using an exact text match — identical prompts from retries or
                # duplicate calls should reuse the same segment record.
                cur.execute(
                    """
                    SELECT segment_id
                    FROM prompt_segments
                    WHERE user_id = %s AND segment_text = %s
                    LIMIT 1;
                    """,
                    (user_id, segment_text)
                )
                existing = cur.fetchone()
                if existing:
                    segment_id_str = str(existing[0])
                    logger.info(
                        f"Reusing existing prompt segment: {segment_id_str} for user: {user_id}"
                    )
                    return SegmentInsertResult(
                        success=True,
                        segment_id=segment_id_str,
                        error=None
                    )

                # New segment — create a PromptSegment model instance (generates ID and timestamp)
                segment = PromptSegment(
                    user_id=user_id,
                    segment_text=segment_text
                )
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


def _reinforce_behavior_on_cursor(
    cur,
    behavior_id: str,
    user_id: str,
    segment_id: Optional[str] = None,
    current_timestamp: Optional[int] = None,
) -> ReinforcementResult:
    """
    Internal helper — execute all reinforcement SQL on an **existing** cursor.

    Does NOT commit; the caller is responsible for committing (or rolling back).
    This allows the reinforcement to participate in the caller's transaction.

    Args:
        cur: An open psycopg cursor (must be inside an active connection).
        behavior_id: ID of the behavior to reinforce.
        user_id: User identifier (for the WHERE clause).
        segment_id: Optional segment ID to append to prompt_history_ids.
        current_timestamp: Unix timestamp to use; defaults to now.

    Returns:
        ReinforcementResult with success status and updated values.
    """
    if current_timestamp is None:
        current_timestamp = int(time.time())

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
    # Reset last_decay_applied_at to current time when reinforced.
    # Set last_accessed_at to mark this behavior as actively used.
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
                reinforce_result = _reinforce_behavior_on_cursor(
                    cur, behavior_id, user_id, segment_id, current_timestamp
                )
                conn.commit()

        # After reinforcement is committed, check if this behavior is
        # involved in any PENDING conflict that can now be auto-resolved.
        # Runs in its own transaction — failures are logged, never raised.
        if reinforce_result.success:
            _check_and_auto_resolve_conflicts(behavior_id, user_id)

            # Publish behavior.reinforced event for drift detection
            try:
                publisher = get_event_publisher()
                publisher.publish_behavior_reinforced(
                    user_id=user_id,
                    behavior_id=behavior_id,
                    reinforcement_count=reinforce_result.new_reinforcement_count,
                    credibility=reinforce_result.new_credibility,
                    last_seen_at=current_timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to publish behavior.reinforced event: {e}")
                
        return reinforce_result

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


# ---------------------------------------------------------------------------
# Reinforcement-Divergence Auto-Resolution
# ---------------------------------------------------------------------------

def _check_and_auto_resolve_conflicts(
    behavior_id: str,
    user_id: str,
) -> None:
    """
    Check if a just-reinforced behavior is involved in any PENDING conflict
    and attempt automatic resolution based on reinforcement divergence.

    Called after every successful reinforcement.  Uses the existing
    ``idx_conflicts_behaviors`` B-tree index on (behavior_id_1, behavior_id_2)
    for a sub-millisecond lookup — no Redis or external cache needed.

    Resolution rules (either condition triggers resolution):
        reinforcement_gap ≥ AUTO_RESOLVE_MIN_REINFORCEMENT_GAP   (default 3)
        OR credibility_gap ≥ AUTO_RESOLVE_MIN_CREDIBILITY_GAP    (default 0.15)
        ⇒ stronger behaviour wins  (OLD_WINS or NEW_WINS)

    Expiration fallback:
        conflict_age > CONFLICT_EXPIRY_SECONDS   (default 30 days)
        ⇒ resolve as BOTH_CORRECT (user implicitly accepts both)

    All DB work runs in a single transaction so the resolution is atomic.
    Errors are logged but never propagated — reinforcement must not fail
    because of an auto-resolution edge case.
    """
    from config.configurations import (
        AUTO_RESOLVE_MIN_REINFORCEMENT_GAP,
        AUTO_RESOLVE_MIN_CREDIBILITY_GAP,
        CONFLICT_EXPIRY_SECONDS,
    )

    try:
        current_timestamp = int(time.time())

        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # ---------------------------------------------------------
                # 1. Find all PENDING conflicts involving this behavior.
                #    The OR covers both positions (behavior_id_1, _2).
                # ---------------------------------------------------------
                cur.execute(
                    """
                    SELECT
                        c.conflict_id,
                        c.behavior_id_1,
                        c.behavior_id_2,
                        c.created_at,
                        b1.credibility              AS cred_1,
                        b1.reinforcement_count       AS rc_1,
                        b1.decay_rate               AS decay_rate_1,
                        b1.last_decay_applied_at    AS last_decay_1,
                        b2.credibility              AS cred_2,
                        b2.reinforcement_count       AS rc_2,
                        b2.decay_rate               AS decay_rate_2,
                        b2.last_decay_applied_at    AS last_decay_2
                    FROM behavior_conflicts c
                    JOIN behaviors b1
                      ON b1.behavior_id = c.behavior_id_1 AND b1.user_id = c.user_id
                    JOIN behaviors b2
                      ON b2.behavior_id = c.behavior_id_2 AND b2.user_id = c.user_id
                    WHERE c.user_id = %s
                      AND c.resolution_status = 'PENDING'
                      AND (c.behavior_id_1 = %s OR c.behavior_id_2 = %s)
                    """,
                    (user_id, behavior_id, behavior_id),
                )

                pending_conflicts = cur.fetchall()

                if not pending_conflicts:
                    return  # fast path — nothing to do

                logger.info(
                    f"[AUTO-RESOLVE-FLAGGED] Checking {len(pending_conflicts)} pending "
                    f"conflict(s) for behavior {behavior_id}"
                )

                for row in pending_conflicts:
                    (
                        conflict_id,
                        bid_1, bid_2,
                        created_at,
                        stored_cred_1, rc_1, decay_rate_1, last_decay_1,
                        stored_cred_2, rc_2, decay_rate_2, last_decay_2,
                    ) = row

                    # Apply lazy decay to stored credibilities before comparing.
                    # Without this, a behavior that hasn't been accessed in months
                    # retains its stored (inflated) credibility and can incorrectly
                    # win auto-resolution over a fresher, genuinely stronger behavior.
                    cred_1, _, _ = apply_lazy_decay(
                        stored_credibility=float(stored_cred_1),
                        decay_rate=float(decay_rate_1),
                        last_decay_applied_at=last_decay_1,
                        current_time=current_timestamp,
                    )
                    cred_2, _, _ = apply_lazy_decay(
                        stored_credibility=float(stored_cred_2),
                        decay_rate=float(decay_rate_2),
                        last_decay_applied_at=last_decay_2,
                        current_time=current_timestamp,
                    )

                    conflict_age = current_timestamp - created_at
                    reinforcement_gap = abs(int(rc_1) - int(rc_2))
                    credibility_gap = abs(float(cred_1) - float(cred_2))

                    # -------------------------------------------------
                    # Rule 1: Divergence threshold met → winner takes all
                    #
                    # Either condition alone is sufficient evidence:
                    #   - reinforcement_gap ≥ 3 means the user has
                    #     repeatedly expressed one behavior but not the
                    #     other — clear user intent signal.
                    #   - credibility_gap ≥ 0.15 means quality scores
                    #     have diverged enough (possible when one
                    #     started with low credibility).
                    #
                    # Using OR avoids the case where both behaviors
                    # start with high credibility (~0.88) making the
                    # gap physically unreachable (capped at 1.0).
                    # -------------------------------------------------
                    if (
                        reinforcement_gap >= AUTO_RESOLVE_MIN_REINFORCEMENT_GAP
                        or credibility_gap >= AUTO_RESOLVE_MIN_CREDIBILITY_GAP
                    ):
                        # Determine winner (behavior_id_1 = old, _2 = new)
                        if float(cred_1) > float(cred_2):
                            resolution_choice = "OLD_WINS"
                            winner_id, loser_id = bid_1, bid_2
                        else:
                            resolution_choice = "NEW_WINS"
                            winner_id, loser_id = bid_2, bid_1

                        # Execute resolution atomically on this cursor
                        _apply_auto_resolution(
                            cur,
                            conflict_id=conflict_id,
                            user_id=user_id,
                            resolution_choice=resolution_choice,
                            winner_id=winner_id,
                            loser_id=loser_id,
                            current_timestamp=current_timestamp,
                        )

                        logger.info(
                            f"[AUTO-RESOLVE] Conflict {conflict_id} → {resolution_choice} "
                            f"(gap: rc={reinforcement_gap}, cred={credibility_gap:.3f})"
                        )
                        continue

                    # -------------------------------------------------
                    # Rule 2: Expiration → BOTH_CORRECT
                    # -------------------------------------------------
                    if conflict_age >= CONFLICT_EXPIRY_SECONDS:
                        _apply_auto_resolution_both_correct(
                            cur,
                            conflict_id=conflict_id,
                            user_id=user_id,
                            bid_1=bid_1,
                            bid_2=bid_2,
                            current_timestamp=current_timestamp,
                        )

                        logger.info(
                            f"[AUTO-RESOLVE] Conflict {conflict_id} expired after "
                            f"{conflict_age // 86400} days → BOTH_CORRECT"
                        )
                        continue

                    # Neither threshold met — leave as PENDING
                    logger.info(
                        f"[AUTO-RESOLVE] Conflict {conflict_id} not yet resolvable "
                        f"(rc_gap={reinforcement_gap} [need≥{AUTO_RESOLVE_MIN_REINFORCEMENT_GAP}], "
                        f"cred_gap={credibility_gap:.3f} [need≥{AUTO_RESOLVE_MIN_CREDIBILITY_GAP}], "
                        f"age={conflict_age // 86400}d [expire≥{CONFLICT_EXPIRY_SECONDS // 86400}d])"
                    )

                conn.commit()

    except Exception as e:
        # Never propagate — reinforcement should not fail because of this.
        logger.error(
            f"[AUTO-RESOLVE] Error checking conflicts for behavior "
            f"{behavior_id}: {str(e)}"
        )


def _apply_auto_resolution(
    cur,
    conflict_id: str,
    user_id: str,
    resolution_choice: str,
    winner_id: str,
    loser_id: str,
    current_timestamp: int,
) -> None:
    """
    Apply OLD_WINS or NEW_WINS auto-resolution on an existing cursor.

    Winner  → ACTIVE, reinforced once.
    Loser   → SUPERSEDED, credibility → 0.0 (queued for pruning).
    Conflict → resolution_status = AUTO_RESOLVED.

    Does NOT commit — caller is responsible.
    """
    # Reinforce winner
    _reinforce_behavior_on_cursor(
        cur, winner_id, user_id,
        segment_id=None,
        current_timestamp=current_timestamp,
    )

    # Ensure winner is ACTIVE
    cur.execute(
        """
        UPDATE behaviors
        SET behavior_state = %s,
            last_accessed_at = %s
        WHERE behavior_id = %s AND user_id = %s
        """,
        (BehaviorState.ACTIVE.value, current_timestamp, winner_id, user_id),
    )

    # Supersede loser
    cur.execute(
        """
        UPDATE behaviors
        SET credibility = 0.0,
            behavior_state = %s,
            superseded_by_id = %s,
            last_accessed_at = %s
        WHERE behavior_id = %s AND user_id = %s
        """,
        (
            BehaviorState.SUPERSEDED.value,
            winner_id,
            current_timestamp,
            loser_id,
            user_id,
        ),
    )

    # Inherit loser's graph edges → winner (0.5× weight)
    _inherit_edges_on_cursor(cur, winner_id, loser_id, user_id, current_timestamp)

    # Mark conflict as AUTO_RESOLVED
    cur.execute(
        """
        UPDATE behavior_conflicts
        SET resolution_status = %s,
            resolution_choice = %s,
            resolved_at = %s
        WHERE conflict_id = %s AND user_id = %s
        """,
        (
            ResolutionStatus.AUTO_RESOLVED.value,
            resolution_choice,
            current_timestamp,
            conflict_id,
            user_id,
        ),
    )


def _apply_auto_resolution_both_correct(
    cur,
    conflict_id: str,
    user_id: str,
    bid_1: str,
    bid_2: str,
    current_timestamp: int,
) -> None:
    """
    Expire a stale conflict as BOTH_CORRECT on an existing cursor.

    Both behaviours → ACTIVE (reinforced once each).
    Conflict → resolution_status = EXPIRED, resolution_choice = BOTH_CORRECT.

    Does NOT commit — caller is responsible.
    """
    # Reinforce both behaviours
    _reinforce_behavior_on_cursor(
        cur, bid_1, user_id,
        segment_id=None,
        current_timestamp=current_timestamp,
    )
    _reinforce_behavior_on_cursor(
        cur, bid_2, user_id,
        segment_id=None,
        current_timestamp=current_timestamp,
    )

    # Ensure both are ACTIVE
    cur.execute(
        """
        UPDATE behaviors
        SET behavior_state = %s,
            last_accessed_at = %s
        WHERE behavior_id IN (%s, %s) AND user_id = %s
        """,
        (
            BehaviorState.ACTIVE.value,
            current_timestamp,
            bid_1,
            bid_2,
            user_id,
        ),
    )

    # Mark conflict as EXPIRED / BOTH_CORRECT
    cur.execute(
        """
        UPDATE behavior_conflicts
        SET resolution_status = %s,
            resolution_choice = %s,
            resolved_at = %s
        WHERE conflict_id = %s AND user_id = %s
        """,
        (
            ResolutionStatus.EXPIRED.value,
            "BOTH_CORRECT",
            current_timestamp,
            conflict_id,
            user_id,
        ),
    )


def get_user_behaviors(user_id: str, session_id: Optional[str] = None, include_states: List[str] = None) -> List[dict]:
    """
    Get all behaviors for a user
    
    SESSION ISOLATION: If session_id is provided, only returns behaviors from that session.
    If session_id is None, returns behaviors from all sessions.
    
    Args:
        user_id: User identifier
        session_id: Optional session identifier for filtering (None = all sessions)
        include_states: List of behavior states to include (defaults to ['ACTIVE', 'NEW'])
        
    Returns:
        List of behavior dictionaries with all fields
    """
    if include_states is None:
        include_states = ['ACTIVE', 'NEW']
    
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Build query with optional session_id filter
                query = """
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
                        polarity,
                        session_id
                    FROM behaviors
                    WHERE user_id = %s
                    AND behavior_state = ANY(%s)
                """
                params = [user_id, include_states]
                
                # Add session_id filter if provided
                if session_id is not None:
                    query += " AND session_id = %s"
                    params.append(session_id)
                
                query += " ORDER BY created_at DESC;"
                
                cur.execute(query, params)
                
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
                        'polarity': row[14],
                        'session_id': row[15]
                    })
                
                session_info = f" in session {session_id}" if session_id else " (all sessions)"
                logger.debug(f"Found {len(behaviors)} behaviors for user {user_id}{session_info}")
                return behaviors
                
    except Exception as e:
        logger.error(f"Failed to get user behaviors: {str(e)}")
        return []


def search_similar_behaviors(
        user_id: str, 
        query_embedding: List[float], 
        session_id: str = "default",
        limit: int = 10
) -> List[SimilarityResult]:
    """
    Find behaviors similar to query embedding using cosine similarity.
    
    SESSION ISOLATION: Only searches behaviors within the same session_id.
    This ensures behaviors from different contexts (work, personal, etc.) don't interfere.
    
    Applies lazy decay to credibility on-the-fly when retrieving behaviors.
    If decay is applied, updates the behavior in the database with new credibility.
    
    Args:
        user_id: User identifier
        query_embedding: Vector embedding to search against
        session_id: Session identifier for isolation (defaults to "default")
        limit: Maximum number of results to return
    
    Returns:
        List of SimilarityResult objects with classification and metadata
        Sorted by similarity (lowest distance first)
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Fetch behaviors with decay-related fields
                # SESSION ISOLATION: Only search within the same session_id
                cur.execute(
                    """
                    SELECT
                        behavior_id,
                        behavior_text,
                        canonical_embedding <=> %s::vector AS distance,
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
                    AND session_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW', 'FLAGGED')
                    AND canonical_embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding, user_id, session_id, limit)
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
                
                logger.debug(f"Found {len(similarity_results)} similar behaviors for user {user_id} in session {session_id}")
                return similarity_results
                
    except Exception as e:
        logger.error(f"Failed to search similar behaviors: {str(e)}")
        return []


def search_similar_behavior_3D(
    user_id: str,
    query_embedding: Optional[List[float]] = None,
    query_text: str = "",
    session_id: str = "default",
    required_intents: Optional[List[str]] = None,
    limit: int = None,
    query_embeddings: Optional[List[List[float]]] = None,
) -> HybridSearchResponse:
    """
    Layered Retrieval Architecture (LRA) — 3-stage behavior search.

    Replaces the previous single-pass additive hybrid (TGHR) with a
    retrieve-and-rerank pipeline that avoids BM25 score incompatibility
    and lexical sparsity issues on micro-behaviors (<7 words).

    Pipeline stages:
      Stage 1 — Coarse Semantic Retrieval (Recall Layer):
          Pure cosine-distance query via pgvector.  Fetches the top K
          candidates purely by semantic meaning, maximising recall.

      Stage 2 — Multiplicative Intent Re-ranking (Precision Layer):
          Neuro-symbolic fusion: each candidate's base semantic score is
          modulated by a multiplicative intent-affinity scalar.
              S_base  = 1 - D_cosine
              M       = 1 + (α · A)          (α = INTENT_RERANK_ALPHA)
              S_final = S_base · M
          This ensures intent alignment *amplifies* strong semantic
          matches but cannot rescue irrelevant behaviors.

      Stage 3 — Dynamic Thresholding & Relevance Gap Cutoff:
          a) Absolute Semantic Floor — discard any S_final < τ_min.
          b) Relevance Gap — T_dynamic = S_max · (1 - ρ).  Any result
             below T_dynamic is cut, preventing "tail noise" from
             diluting the LLM's attention.
          c) Soft cap — hard limit on returned results.

    This method is READ-ONLY at query time.  It collects pending updates
    (lazy decay + last_accessed_at) which the caller should persist
    asynchronously via persist_retrieval_updates_batch().

    Args:
        user_id: User identifier
        query_embedding: (legacy) Dense vector embedding of a single probe.
                         Used only when query_embeddings is not provided.
        query_text: Plain text of the (first) probe — kept for logging
        session_id: Session identifier for isolation (defaults to "default")
        required_intents: Optional list of intent types predicted by the LLM
                          (e.g., ["CONSTRAINT", "PREFERENCE"]).  If None or
                          empty, no intent re-ranking is applied.
        limit: Maximum candidates fetched per probe from DB (defaults to
               HYBRID_SEARCH_LIMIT)
        query_embeddings: 1..N dense vector embeddings (multi-probe HyDE).
                          Each probe runs an independent cosine-distance
                          fetch; per-candidate `s_base` uses the BEST
                          (minimum) cosine distance across probes, and a
                          small multi-probe agreement boost (10% per extra
                          probe matched) is applied to candidates that
                          appeared in multiple probes' top-K.

    Returns:
        HybridSearchResponse containing:
          - results: List of SimilarityResult objects (sorted by S_final desc)
          - decay_updates: Pending credibility updates for async persistence
          - accessed_behavior_ids: IDs of all returned behaviors for
            last_accessed_at update
    """
    from config.configurations import (
        HYBRID_SEARCH_LIMIT,
        INTENT_RERANK_ALPHA,
        SEMANTIC_FLOOR_THRESHOLD,
        SEMANTIC_FLOOR_FALLBACK,
        MAX_FALLBACK_RESULTS,
        RELEVANCE_GAP_DROP_RATIO,
        MAX_RETRIEVAL_RESULTS,
        ALL_INTENT_TYPES,
        INTENT_AFFINITY,
    )

    if limit is None:
        limit = HYBRID_SEARCH_LIMIT

    # Normalise input: build the probe list from whichever arg the caller used.
    probes: List[List[float]]
    if query_embeddings and len(query_embeddings) > 0:
        probes = [p for p in query_embeddings if p]
    elif query_embedding is not None:
        probes = [query_embedding]
    else:
        probes = []

    if not probes:
        logger.warning("[LRA] No query embeddings provided — returning empty result")
        return HybridSearchResponse()

    # Multi-probe agreement boost: candidates matched by N>1 probes get a
    # consensus multiplier.  10% per additional probe — bounded so that a
    # single strong-semantic match cannot be drowned out by weak duplicates.
    AGREEMENT_BOOST_PER_PROBE: float = 0.10

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # ==========================================================
                # STAGE 1 — Coarse Semantic Retrieval (Recall Layer)
                # Pure dense vector search via pgvector cosine distance.
                # Run ONCE PER PROBE; merge candidates by behavior_id keeping
                # the BEST (smallest) cosine distance across probes, and
                # tracking how many probes matched each candidate.
                # ==========================================================
                base_conditions = """
                    user_id = %s
                    AND session_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW')
                """

                query = f"""
                    SELECT
                        behavior_id,
                        behavior_text,
                        embedding <=> %s::vector AS cosine_distance,
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
                    WHERE {base_conditions}
                    ORDER BY cosine_distance ASC
                    LIMIT %s;
                """

                # Map: behavior_id → merged row dict (best distance + match count)
                merged: dict[str, dict] = {}
                for probe_idx, probe in enumerate(probes):
                    cur.execute(
                        query,
                        [probe, user_id, session_id, limit],
                    )
                    for row in cur.fetchall():
                        (
                            behavior_id, behavior_text, cosine_distance,
                            stored_credibility, last_seen_at, reinforcement_count,
                            intent, target, context, polarity,
                            decay_rate, last_decay_applied_at,
                        ) = row
                        cd = float(cosine_distance)
                        if behavior_id in merged:
                            entry = merged[behavior_id]
                            if cd < entry["cosine_distance"]:
                                entry["cosine_distance"] = cd
                                entry["best_probe_idx"] = probe_idx
                            entry["probe_match_count"] += 1
                        else:
                            merged[behavior_id] = {
                                "behavior_id": behavior_id,
                                "behavior_text": behavior_text,
                                "cosine_distance": cd,
                                "best_probe_idx": probe_idx,
                                "probe_match_count": 1,
                                "stored_credibility": stored_credibility,
                                "last_seen_at": last_seen_at,
                                "reinforcement_count": reinforcement_count,
                                "intent": intent,
                                "target": target,
                                "context": context,
                                "polarity": polarity,
                                "decay_rate": decay_rate,
                                "last_decay_applied_at": last_decay_applied_at,
                            }

                rows = list(merged.values())
                current_time = int(time.time())

                # ==========================================================
                # STAGE 2 — Multiplicative Intent Re-ranking (Precision Layer)
                #
                # Build per-intent affinity map, then compute:
                #   S_base  = 1 - D_cosine         (bounded 0..1)
                #   M       = 1 + (α · A)          (bounded 1..1+α)
                #   S_final = S_base · M
                #
                # This is a neuro-symbolic operation: neural (embedding
                # distance) modulated by symbolic (intent taxonomy).
                # Multiplication guarantees that intent alone cannot rescue
                # a semantically poor match.
                # ==========================================================

                # Pre-compute affinity map: intent_type → best affinity
                use_intent_rerank = required_intents and len(required_intents) > 0
                affinity_map: dict[str, float] = {}
                if use_intent_rerank:
                    for intent_type in ALL_INTENT_TYPES:
                        if intent_type in required_intents:
                            affinity_map[intent_type] = 1.0
                        else:
                            best = 0.0
                            for req in required_intents:
                                key = frozenset({intent_type, req})
                                best = max(best, INTENT_AFFINITY.get(key, 0.0))
                            affinity_map[intent_type] = best
                    logger.debug(f"[LRA] Intent affinity map: {affinity_map}")

                # Process rows: compute S_final, apply lazy decay, collect
                scored_candidates = []
                decay_updates = []

                for row in rows:
                    behavior_id = row["behavior_id"]
                    behavior_text = row["behavior_text"]
                    cosine_distance = row["cosine_distance"]
                    stored_credibility = row["stored_credibility"]
                    last_seen_at = row["last_seen_at"]
                    reinforcement_count = row["reinforcement_count"]
                    intent = row["intent"]
                    target = row["target"]
                    context = row["context"]
                    polarity = row["polarity"]
                    decay_rate = row["decay_rate"]
                    last_decay_applied_at = row["last_decay_applied_at"]
                    probe_match_count = row["probe_match_count"]

                    # S_base: convert (best) cosine distance → similarity (0..1)
                    s_base = max(0.0, 1.0 - float(cosine_distance))

                    # Multiplicative intent scalar
                    if use_intent_rerank:
                        affinity = affinity_map.get(intent, 0.0)
                        intent_multiplier = 1.0 + (INTENT_RERANK_ALPHA * affinity)
                    else:
                        affinity = 0.0
                        intent_multiplier = 1.0

                    # Multi-probe agreement multiplier: a candidate that
                    # matched in K>1 probes' top-K gets a consensus boost.
                    # Single-probe queries reduce this to 1.0 (no-op).
                    agreement_multiplier = 1.0 + (
                        AGREEMENT_BOOST_PER_PROBE * max(0, probe_match_count - 1)
                    )

                    multiplier = intent_multiplier * agreement_multiplier
                    s_final = s_base * multiplier

                    # Apply lazy decay in-memory (read-only, no DB update)
                    new_credibility, decay_applied, days_elapsed = apply_lazy_decay(
                        stored_credibility=float(stored_credibility),
                        decay_rate=float(decay_rate),
                        last_decay_applied_at=last_decay_applied_at,
                        current_time=current_time
                    )

                    if decay_applied:
                        logger.debug(
                            f"[LRA] Lazy decay (in-memory) for {behavior_id}: "
                            f"{stored_credibility:.4f} → {new_credibility:.4f} "
                            f"({days_elapsed} days)"
                        )
                        decay_updates.append((
                            new_credibility,
                            current_time,
                            behavior_id,
                            user_id
                        ))

                    scored_candidates.append({
                        "behavior_id": behavior_id,
                        "behavior_text": behavior_text,
                        "cosine_distance": float(cosine_distance),
                        "s_base": s_base,
                        "affinity": affinity,
                        "multiplier": multiplier,
                        "intent_multiplier": intent_multiplier,
                        "agreement_multiplier": agreement_multiplier,
                        "probe_match_count": probe_match_count,
                        "s_final": s_final,
                        "credibility": new_credibility,
                        "last_seen_at": int(last_seen_at),
                        "reinforcement_count": int(reinforcement_count),
                        "intent": intent,
                        "target": target,
                        "context": context if context else "general",
                        "polarity": polarity,
                    })

                # Sort by S_final descending (re-ranking may reorder)
                scored_candidates.sort(key=lambda c: c["s_final"], reverse=True)

                # ==========================================================
                # STAGE 3a — Absolute Semantic Floor (Hard Cutoff)
                # Discard any candidate with S_final < τ_min.
                # Prevents injecting weakly-related behaviors that could
                # cause hallucination or knowledge fragmentation in the LLM.
                # ==========================================================
                # Save sorted candidates before floor for potential fallback
                candidates_before_floor = list(scored_candidates)

                before_floor = len(scored_candidates)
                scored_candidates = [
                    c for c in scored_candidates
                    if c["s_final"] >= SEMANTIC_FLOOR_THRESHOLD
                ]
                dropped_by_floor = before_floor - len(scored_candidates)
                if dropped_by_floor > 0:
                    logger.info(
                        f"[LRA] Semantic floor: dropped {dropped_by_floor} candidates "
                        f"below τ_min={SEMANTIC_FLOOR_THRESHOLD:.2f}"
                    )

                # ==========================================================
                # STAGE 3a-FALLBACK — Soft floor for zero-result recovery
                # If the primary floor drops ALL candidates, try a relaxed
                # threshold (τ_fallback).  Only fires when the strict floor
                # returns nothing — cannot affect queries that already pass.
                # Capped at MAX_FALLBACK_RESULTS to limit noise.
                # ==========================================================
                if not scored_candidates and candidates_before_floor:
                    scored_candidates = [
                        c for c in candidates_before_floor
                        if c["s_final"] >= SEMANTIC_FLOOR_FALLBACK
                    ][:MAX_FALLBACK_RESULTS]
                    if scored_candidates:
                        logger.info(
                            f"[LRA] Soft fallback: recovered {len(scored_candidates)} "
                            f"candidate(s) above τ_fallback={SEMANTIC_FLOOR_FALLBACK:.2f} "
                            f"(primary floor returned 0)"
                        )

                # ==========================================================
                # STAGE 3b — Relevance Gap (Dynamic Context Truncation)
                # T_dynamic = S_ref · (1 - ρ)
                # Detects the "semantic cliff" between relevant results and
                # tail noise.  Prevents weak matches from diluting the
                # LLM's attention mechanism.
                #
                # Exact-match guard: when the top candidate is a near-exact
                # match (S_base ≥ 0.95, i.e. distance ≈ 0), the gap between
                # it and every other candidate is artificially large.  In
                # that case, compute T_dynamic from the second-best candidate
                # so the gap cutoff evaluates the real cluster of results.
                # ==========================================================
                if scored_candidates:
                    # Check if top candidate is a near-exact match (S_base ≥ 0.95)
                    top_s_base = scored_candidates[0].get("s_base", 0.0)
                    if top_s_base >= 0.95 and len(scored_candidates) >= 2:
                        # Use second-best for gap reference — always keep the exact match
                        s_ref = scored_candidates[1]["s_final"]
                        t_dynamic = s_ref * (1.0 - RELEVANCE_GAP_DROP_RATIO)
                        gap_filtered = [scored_candidates[0]]  # exact match always kept
                        for c in scored_candidates[1:]:
                            if c["s_final"] < t_dynamic:
                                break
                            gap_filtered.append(c)
                    else:
                        s_ref = scored_candidates[0]["s_final"]
                        # Cap at 1.0 — intent boost can push S_final > 1.0
                        s_ref = min(s_ref, 1.0)
                        t_dynamic = s_ref * (1.0 - RELEVANCE_GAP_DROP_RATIO)
                        gap_filtered = []
                        for c in scored_candidates:
                            if c["s_final"] < t_dynamic:
                                break
                            gap_filtered.append(c)

                    dropped_by_gap = len(scored_candidates) - len(gap_filtered)
                    if dropped_by_gap > 0:
                        logger.info(
                            f"[LRA] Relevance gap cutoff: kept {len(gap_filtered)}, "
                            f"dropped {dropped_by_gap} "
                            f"(S_ref={s_ref:.4f}, T_dynamic={t_dynamic:.4f}, "
                            f"ρ={RELEVANCE_GAP_DROP_RATIO}, "
                            f"exact_match_guard={top_s_base >= 0.95})"
                        )
                    scored_candidates = gap_filtered

                # ==========================================================
                # STAGE 3c — Soft Cap
                # Hard limit on returned results to prevent over-retrieval
                # for broad/vague queries.
                # ==========================================================
                if len(scored_candidates) > MAX_RETRIEVAL_RESULTS:
                    dropped_by_cap = len(scored_candidates) - MAX_RETRIEVAL_RESULTS
                    scored_candidates = scored_candidates[:MAX_RETRIEVAL_RESULTS]
                    logger.info(
                        f"[LRA] Soft cap applied: kept top {MAX_RETRIEVAL_RESULTS}, "
                        f"dropped {dropped_by_cap} excess results"
                    )

                # ==========================================================
                # Build final response objects
                # ==========================================================
                similarity_results = []
                accessed_behavior_ids = []

                for i, c in enumerate(scored_candidates):
                    # distance = 1 - S_final  (lower = better, API consistency)
                    effective_distance = max(0.0, 1.0 - c["s_final"])

                    similarity_results.append(SimilarityResult(
                        behavior_id=c["behavior_id"],
                        behavior_text=c["behavior_text"],
                        distance=effective_distance,
                        credibility=c["credibility"],
                        last_seen_at=c["last_seen_at"],
                        reinforcement_count=c["reinforcement_count"],
                        intent=c["intent"],
                        target=c["target"],
                        context=c["context"],
                        polarity=c["polarity"],
                    ))
                    accessed_behavior_ids.append(c["behavior_id"])

                    logger.info(
                        f"[LRA] MATCH #{i + 1}: "
                        f"id={c['behavior_id']} | intent={c['intent']} | "
                        f"S_base={c['s_base']:.4f} | "
                        f"affinity={c['affinity']:.2f} | "
                        f"M_intent={c['intent_multiplier']:.4f} | "
                        f"M_agree={c['agreement_multiplier']:.4f} | "
                        f"probes={c['probe_match_count']}/{len(probes)} | "
                        f"S_final={c['s_final']:.4f} | "
                        f"credibility={c['credibility']:.4f} | "
                        f"text='{c['behavior_text'][:80]}...'"
                    )

                # Trim decay_updates to only include returned behavior IDs
                returned_ids = set(accessed_behavior_ids)
                decay_updates = [d for d in decay_updates if d[2] in returned_ids]

                logger.info(
                    f"[LRA] Search for user {user_id} in session {session_id}: "
                    f"{len(similarity_results)} results "
                    f"(probes={len(probes)}, "
                    f"unique_candidates={len(rows)}, "
                    f"α={INTENT_RERANK_ALPHA}, "
                    f"τ_min={SEMANTIC_FLOOR_THRESHOLD}, "
                    f"ρ={RELEVANCE_GAP_DROP_RATIO}, "
                    f"intents={required_intents or 'NONE'}, "
                    f"decay_pending={len(decay_updates)}, "
                    f"query='{query_text[:60]}...')"
                )
                return HybridSearchResponse(
                    results=similarity_results,
                    decay_updates=decay_updates,
                    accessed_behavior_ids=accessed_behavior_ids
                )

    except Exception as e:
        logger.error(f"[LRA] Failed to search similar behaviors: {str(e)}")
        return HybridSearchResponse()


def persist_retrieval_updates_batch(
    decay_updates: List[tuple],
    accessed_behavior_ids: List[str],
    user_id: str
) -> None:
    """
    Persist pending updates from a hybrid search in a single batch transaction.
    
    Called asynchronously (via BackgroundTasks) after the v2/extract response
    has already been sent to the client. Performs two operations:
    
    1. Credibility decay persistence — for behaviors where lazy decay was applied
       in-memory during retrieval, persist the new credibility + last_decay_applied_at.
    2. Access timestamp update — for ALL behaviors returned in the search results,
       update last_accessed_at to mark they were used for prompt enrichment.
    
    Both operations run in a single transaction for efficiency.
    
    Args:
        decay_updates: List of (new_credibility, timestamp, behavior_id, user_id)
                       tuples from HybridSearchResponse.decay_updates
        accessed_behavior_ids: List of behavior_id strings from
                               HybridSearchResponse.accessed_behavior_ids
        user_id: User identifier (for logging and WHERE clause)
    """
    if not decay_updates and not accessed_behavior_ids:
        logger.debug("[3D-ASYNC] No pending updates to persist, skipping")
        return

    try:
        current_time = int(time.time())
        
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # ---------------------------------------------------------
                # 1. Batch update: credibility decay persistence
                # ---------------------------------------------------------
                if decay_updates:
                    cur.executemany(
                        """
                        UPDATE behaviors
                        SET credibility = %s,
                            last_decay_applied_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        decay_updates
                    )
                    logger.info(
                        f"[3D-ASYNC] Persisted lazy decay for "
                        f"{len(decay_updates)} behaviors (user: {user_id})"
                    )

                # ---------------------------------------------------------
                # 2. Batch update: last_accessed_at for all returned behaviors
                # ---------------------------------------------------------
                if accessed_behavior_ids:
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET last_accessed_at = %s
                        WHERE user_id = %s
                        AND behavior_id = ANY(%s)
                        """,
                        (current_time, user_id, accessed_behavior_ids)
                    )
                    logger.info(
                        f"[3D-ASYNC] Updated last_accessed_at for "
                        f"{len(accessed_behavior_ids)} behaviors (user: {user_id})"
                    )

                conn.commit()

    except Exception as e:
        logger.error(
            f"[3D-ASYNC] Failed to persist retrieval updates for user {user_id}: {str(e)}"
        )
        

def insert_conflict(
    user_id: str,
    behavior_id_1: str,
    behavior_id_2: str,
    conflict_type: ConflictType,
    similarity_distance: float,
    llm_analysis: Optional[str] = None,
    old_polarity: Optional[str] = None,
    new_polarity: Optional[str] = None,
    old_target: Optional[str] = None,
    new_target: Optional[str] = None
) -> str:
    """
    Store a detected conflict between two behaviors in the database.
    
    This creates an audit trail of all conflict detections and their resolutions.
    The conflict record tracks:
    - Which behaviors conflict
    - Type of conflict (resolvable vs needs user input)
    - LLM's analysis and reasoning
    - Resolution status and outcome
    - Polarity and target information for drift detection
    
    Args:
        user_id: User whose behaviors conflict
        behavior_id_1: First behavior ID (existing/old behavior)
        behavior_id_2: Second behavior ID (new behavior)
        conflict_type: RESOLVABLE or USER_DECISION_NEEDED
        similarity_distance: Embedding distance between behaviors
        llm_analysis: Optional LLM explanation of the conflict
        old_polarity: Polarity of existing behavior (POSITIVE/NEGATIVE)
        new_polarity: Polarity of new behavior (POSITIVE/NEGATIVE)
        old_target: Target of existing behavior
        new_target: Target of new behavior
        
    Returns:
        conflict_id: UUID of the created conflict record
        
    Raises:
        Exception: If database insertion fails
        
    Example:
        >>> conflict_id = insert_conflict(
        ...     "user123", "beh_abc", "beh_xyz",
        ...     ConflictType.RESOLVABLE, 0.22,
        ...     "Behaviors contradict in same domain",
        ...     old_polarity="POSITIVE", new_polarity="NEGATIVE",
        ...     old_target="python", new_target="python"
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
                        created_at,
                        old_polarity,
                        new_polarity,
                        old_target,
                        new_target
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        current_timestamp,
                        old_polarity,
                        new_polarity,
                        old_target,
                        new_target
                    )
                )
                conn.commit()
        
        logger.info(
            f"Stored conflict {conflict_id}: {behavior_id_1} <-> {behavior_id_2} "
            f"(distance: {similarity_distance:.3f}, type: {conflict_type.value}, "
            f"polarity: {old_polarity}→{new_polarity}, target: {old_target}→{new_target})"
        )
        
        # Publish behavior.conflict.resolved event for drift detection
        try:
            publisher = get_event_publisher()
            publisher.publish_conflict_resolved(
                user_id=user_id,
                conflict_id=conflict_id,
                behavior_id_1=behavior_id_1,
                behavior_id_2=behavior_id_2,
                conflict_type=conflict_type.value,
                resolution_status=ResolutionStatus.PENDING.value,
                old_polarity=old_polarity,
                new_polarity=new_polarity,
                old_target=old_target,
                new_target=new_target,
                created_at=current_timestamp
            )
        except Exception as pub_error:
            logger.warning(f"Failed to publish behavior.conflict.resolved event: {pub_error}")
        
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


def _supersede_behavior_on_cursor(
    cur,
    old_behavior_id: str,
    new_behavior_id: str,
    user_id: str,
    current_timestamp: Optional[int] = None,
) -> bool:
    """
    Internal helper — execute the supersede UPDATE on an **existing** cursor.

    Does NOT commit; the caller is responsible for committing (or rolling back).
    This allows the update to participate in the caller's transaction.

    Args:
        cur: An open psycopg cursor (must be inside an active connection).
        old_behavior_id: Behavior being superseded.
        new_behavior_id: Behavior that supersedes it.
        user_id: User ID (required for the partitioned table WHERE clause).
        current_timestamp: Unix timestamp to use; defaults to now.

    Returns:
        True if the UPDATE touched at least one row, False otherwise.
    """
    if current_timestamp is None:
        current_timestamp = int(time.time())

    # Update old behavior to SUPERSEDED state and link to new one.
    # Set last_accessed_at to mark it was actively used in conflict resolution.
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
                success = _supersede_behavior_on_cursor(
                    cur, old_behavior_id, new_behavior_id, user_id, current_timestamp
                )
                conn.commit()
                
                # Publish behavior.superseded event for drift detection
                if success:
                    try:
                        publisher = get_event_publisher()
                        publisher.publish_behavior_superseded(
                            user_id=user_id,
                            behavior_id=old_behavior_id,
                            superseded_by=new_behavior_id
                        )
                    except Exception as e:
                        logger.warning(f"Failed to publish behavior.superseded event: {e}")
                
                return success

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


def get_behaviors_by_user(user_id: str, session_id: Optional[str] = None) -> List[dict]:
    """
    Get all behaviors for a specific user.
    
    SESSION ISOLATION: If session_id is provided, only returns behaviors from that session.
    If session_id is None, returns behaviors from all sessions.
    
    Applies lazy decay to credibility on-the-fly when retrieving behaviors.
    If decay is applied, updates the behavior in the database with new credibility.
    
    Args:
        user_id: The user identifier
        session_id: Optional session identifier for filtering (None = all sessions)
        
    Returns:
        List of behavior dictionaries with all fields (credibility reflects decay)
    """
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Build query with optional session_id filter
                query = """
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
                """
                params = [user_id]
                
                # Add session_id filter if provided
                if session_id is not None:
                    query += " AND session_id = %s"
                    params.append(session_id)
                
                query += " ORDER BY last_seen_at DESC"
                
                cur.execute(query, params)
                
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
                
                # ------------------------------------------------------------------
                # Step 2: Handle resolution — all SQL runs on the same cursor so
                # everything commits or rolls back as a single unit of work.
                # ------------------------------------------------------------------
                if resolution_choice == "OLD_WINS":
                    # Reinforce old behavior within this transaction
                    reinforce_result = _reinforce_behavior_on_cursor(
                        cur, behavior_id_1, user_id, segment_id=None,
                        current_timestamp=current_timestamp
                    )

                    if not reinforce_result.success:
                        raise Exception(f"Failed to reinforce old behavior: {reinforce_result.error}")

                    # Ensure old behavior is ACTIVE and refresh last_accessed_at
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

                    # Invalidate new behavior — credibility → 0.0 so it is pruned
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET 
                            credibility = 0.0,
                            behavior_state = %s,
                            last_accessed_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (BehaviorState.SUPERSEDED.value, current_timestamp, behavior_id_2, user_id)
                    )

                    # Inherit loser's graph edges → winner (0.5× weight)
                    _inherit_edges_on_cursor(
                        cur, behavior_id_1, behavior_id_2, user_id, current_timestamp
                    )

                    logger.info(
                        f"OLD_WINS: Reinforced {behavior_id_1} (set to ACTIVE), "
                        f"invalidated {behavior_id_2} (credibility set to 0.0 for pruning)"
                    )

                elif resolution_choice == "NEW_WINS":
                    # Set new behavior to ACTIVE first
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

                    # Supersede old behavior within this same transaction
                    supersede_success = _supersede_behavior_on_cursor(
                        cur, behavior_id_1, behavior_id_2, user_id,
                        current_timestamp=current_timestamp
                    )

                    if not supersede_success:
                        raise Exception(f"Failed to supersede old behavior {behavior_id_1}")

                    # Inherit loser's graph edges → winner (0.5× weight)
                    _inherit_edges_on_cursor(
                        cur, behavior_id_2, behavior_id_1, user_id, current_timestamp
                    )

                    logger.info(
                        f"NEW_WINS: Set {behavior_id_2} to ACTIVE, "
                        f"superseded {behavior_id_1}"
                    )

                elif resolution_choice == "BOTH_CORRECT":
                    # Reinforce both behaviors within this transaction
                    reinforce_result_1 = _reinforce_behavior_on_cursor(
                        cur, behavior_id_1, user_id, segment_id=None,
                        current_timestamp=current_timestamp
                    )

                    if not reinforce_result_1.success:
                        raise Exception(f"Failed to reinforce behavior 1: {reinforce_result_1.error}")

                    reinforce_result_2 = _reinforce_behavior_on_cursor(
                        cur, behavior_id_2, user_id, segment_id=None,
                        current_timestamp=current_timestamp
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


def get_behaviors_by_ids(user_id: str, behavior_ids: List[str]) -> List[dict]:
    """
    Retrieve specific behaviors by their IDs for a given user.
    
    Args:
        user_id: The user ID who owns the behaviors
        behavior_ids: List of behavior IDs to retrieve
        
    Returns:
        List of behavior dictionaries with all fields including canonical structure
    """
    if not behavior_ids:
        return []
    
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Build parameterized query for multiple behavior IDs
                placeholders = ','.join(['%s'] * len(behavior_ids))
                
                cur.execute(
                    f"""
                    SELECT 
                        behavior_id,
                        user_id,
                        session_id,
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
                        behavior_state,
                        superseded_by_id,
                        related_behaviors,
                        last_decay_applied_at,
                        context_notes,
                        last_accessed_at,
                        intent,
                        target,
                        context,
                        polarity
                    FROM behaviors
                    WHERE user_id = %s 
                    AND behavior_id IN ({placeholders})
                    ORDER BY last_seen_at DESC
                    """,
                    [user_id] + behavior_ids
                )
                
                rows = cur.fetchall()
        
        behaviors = []
        for row in rows:
            behavior = {
                "behavior_id": row[0],
                "user_id": row[1],
                "session_id": row[2],
                "behavior_text": row[3],
                "credibility": float(row[4]) if row[4] is not None else 0.0,
                "reinforcement_count": row[5] or 0,
                "decay_rate": float(row[6]) if row[6] is not None else 0.0,
                "created_at": row[7],
                "last_seen_at": row[8],
                "prompt_history_ids": row[9] or [],
                "clarity_score": float(row[10]) if row[10] is not None else 0.0,
                "extraction_confidence": float(row[11]) if row[11] is not None else 0.0,
                "linguistic_strength": float(row[12]) if row[12] is not None else 0.0,
                "behavior_state": row[13],
                "superseded_by_id": row[14],
                "related_behaviors": row[15] or [],
                "last_decay_applied_at": row[16],
                "context_notes": row[17],
                "last_accessed_at": row[18],
                # Canonical fields
                "canonical": {
                    "intent": row[19],
                    "target": row[20],
                    "context": row[21],
                    "polarity": row[22]
                }
            }
            behaviors.append(behavior)
        
        logger.info(f"Retrieved {len(behaviors)} behaviors for user={user_id} with IDs={behavior_ids}")
        return behaviors
        
    except Exception as e:
        logger.error(f"Failed to retrieve behaviors by IDs for user={user_id}: {str(e)}")
        raise


# ===========================================================================
# Co-Occurrence Graph — edge creation, expansion, inheritance, cleanup
# ===========================================================================

def get_behavior_ids_by_session(
    user_id: str,
    session_id: str,
    exclude_ids: List[str] = None,
) -> List[str]:
    """
    Return behavior IDs for a given user+session, excluding the provided IDs.
    Used to identify pre-existing session behaviors when creating CO_SESSION edges.
    """
    exclude_ids = exclude_ids or []
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT behavior_id FROM behaviors
                    WHERE user_id = %s
                      AND session_id = %s
                      AND behavior_state IN ('ACTIVE', 'NEW')
                      AND behavior_id != ALL(%s)
                    """,
                    (user_id, session_id, exclude_ids),
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"[GRAPH] get_behavior_ids_by_session failed: {e}")
        return []


def insert_co_occurrences_batch(
    behavior_ids: List[str],
    user_id: str,
    edge_type: str,
    session_id: str = "default",
) -> int:
    """
    Create pairwise co-occurrence edges for a list of behavior IDs.

    Generates all unique (a, b) pairs (a < b lexicographically to avoid
    duplicate reversed edges) and upserts them.  On conflict the edge
    weight is incremented by 0.5 (diminishing reinforcement signal).

    Args:
        behavior_ids: List of behavior IDs that co-occurred.
        user_id: The user who owns these behaviors.
        edge_type: 'CO_PROMPT' or 'CO_SESSION'.
        session_id: Session in which these behaviors were extracted.

    Returns:
        Number of edges written (inserted or updated).
    """
    if len(behavior_ids) < 2:
        return 0

    current_timestamp = int(time.time())

    # Build all unique pairs (sorted to guarantee canonical order)
    pairs = []
    sorted_ids = sorted(set(behavior_ids))
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            pairs.append((sorted_ids[i], sorted_ids[j]))

    if not pairs:
        return 0

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Use executemany with UPSERT — ON CONFLICT bumps weight
                cur.executemany(
                    """
                    INSERT INTO behavior_co_occurrences
                        (behavior_id_1, behavior_id_2, user_id, session_id, edge_type, weight, created_at)
                    VALUES (%s, %s, %s, %s, %s, 1.0, %s)
                    ON CONFLICT (behavior_id_1, behavior_id_2, edge_type)
                    DO UPDATE SET
                        weight     = behavior_co_occurrences.weight + 0.5,
                        created_at = EXCLUDED.created_at
                    """,
                    [
                        (a, b, user_id, session_id, edge_type, current_timestamp)
                        for a, b in pairs
                    ],
                )
                conn.commit()

        logger.info(
            f"[GRAPH] Upserted {len(pairs)} {edge_type} edge(s) for user {user_id} session {session_id}"
        )
        return len(pairs)

    except Exception as e:
        logger.error(f"[GRAPH] Failed to insert co-occurrence edges: {str(e)}")
        return 0


def get_graph_expanded_behaviors(
    user_id: str,
    seed_behavior_ids: List[str],
    session_id: str = "default",
    limit: int = 10,
    query_embeddings: Optional[List[List[float]]] = None,
) -> List[dict]:
    """
    1-hop graph expansion from seed behaviors, gated by query relevance.

    Given a set of behavior IDs returned by embedding search, walk
    the co-occurrence graph one hop to find associated behaviors.
    Results are ranked by ``edge_weight * decayed_credibility`` so
    that strongly-associated, high-credibility behaviors bubble up.

    Only returns behaviors in ACTIVE / NEW state — SUPERSEDED, ARCHIVED,
    and FLAGGED behaviors are excluded.  Already-retrieved seed IDs are
    also excluded to avoid duplicates.  Session isolation is enforced:
    only neighbors belonging to the same session are returned.

    RELEVANCE GATE (when query_embeddings is provided):
    Each candidate neighbor's stored prose embedding is compared against
    every probe; the BEST cosine distance must be ≤
    GRAPH_EXPANSION_DISTANCE_THRESHOLD.  This prevents cross-domain
    co-occurrence noise (e.g., a user mentions Python and cooking in the
    same prompt → write-time edge → otherwise gets replayed at read time
    on a Python-only query).  When query_embeddings is None or empty, the
    gate is skipped (legacy behaviour).

    Lazy decay is applied in Python after fetch (same pattern as LRA),
    with a ``limit * 2`` pre-fetch from SQL to allow re-ranking post-decay
    while still bounding the data transferred from the database.

    Args:
        user_id: User identifier.
        seed_behavior_ids: Behavior IDs from the embedding search.
        session_id: Session identifier for isolation.
        limit: Maximum neighbors to return (default 10).
        query_embeddings: Optional probe embeddings used for the relevance
                          gate.  When provided, neighbors farther than
                          GRAPH_EXPANSION_DISTANCE_THRESHOLD from every
                          probe are filtered out.

    Returns:
        List of dicts, each containing behavior details + edge metadata.
        Empty list if no graph neighbors exist.
    """
    if not seed_behavior_ids:
        return []

    from config.configurations import GRAPH_EXPANSION_DISTANCE_THRESHOLD

    sql_limit = limit * 2  # pre-fetch extra candidates to allow decay re-ranking

    # Build the optional relevance-gate SQL fragment.  Computes the MIN
    # cosine distance from the neighbor's prose embedding to ANY probe;
    # rows whose min-distance exceeds the threshold are dropped.
    use_relevance_gate = bool(query_embeddings) and len(query_embeddings) > 0
    if use_relevance_gate:
        # LEAST(b.embedding <=> %s::vector, b.embedding <=> %s::vector, ...)
        distance_terms = ", ".join(
            ["b.embedding <=> %s::vector"] * len(query_embeddings)
        )
        if len(query_embeddings) == 1:
            min_dist_expr = distance_terms  # LEAST() of one is invalid; use the term directly
        else:
            min_dist_expr = f"LEAST({distance_terms})"
        relevance_select = f", {min_dist_expr} AS min_query_distance"
        relevance_where = f" AND {min_dist_expr} <= %s"
        # Probe vectors appear twice: once in SELECT, once in WHERE
        relevance_params_select = list(query_embeddings)
        relevance_params_where = list(query_embeddings) + [GRAPH_EXPANSION_DISTANCE_THRESHOLD]
    else:
        relevance_select = ""
        relevance_where = ""
        relevance_params_select = []
        relevance_params_where = []

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH edges AS (
                        -- Forward edges: seed is behavior_id_1
                        SELECT behavior_id_2 AS neighbor_id, edge_type, weight
                        FROM behavior_co_occurrences
                        WHERE user_id = %s
                          AND session_id = %s
                          AND behavior_id_1 = ANY(%s)

                        UNION ALL

                        -- Reverse edges: seed is behavior_id_2
                        SELECT behavior_id_1 AS neighbor_id, edge_type, weight
                        FROM behavior_co_occurrences
                        WHERE user_id = %s
                          AND session_id = %s
                          AND behavior_id_2 = ANY(%s)
                    ),
                    ranked AS (
                        SELECT DISTINCT ON (e.neighbor_id)
                            e.neighbor_id,
                            b.behavior_text,
                            b.credibility,
                            b.intent,
                            b.target,
                            b.context,
                            b.polarity,
                            b.decay_rate,
                            b.last_decay_applied_at,
                            e.edge_type,
                            e.weight,
                            (e.weight * b.credibility) AS rank_score
                            {relevance_select}
                        FROM edges e
                        JOIN behaviors b
                          ON b.behavior_id = e.neighbor_id
                         AND b.user_id = %s
                         AND b.session_id = %s
                        WHERE b.behavior_state IN ('ACTIVE', 'NEW')
                          AND e.neighbor_id != ALL(%s)
                          {relevance_where}
                        ORDER BY e.neighbor_id, rank_score DESC
                    )
                    SELECT * FROM ranked ORDER BY rank_score DESC LIMIT %s;
                    """,
                    (
                        user_id, session_id, seed_behavior_ids,
                        user_id, session_id, seed_behavior_ids,
                        *relevance_params_select,
                        user_id, session_id, seed_behavior_ids,
                        *relevance_params_where,
                        sql_limit,
                    ),
                )
                rows = cur.fetchall()

        if not rows:
            logger.info(
                f"[GRAPH] No neighbors found for {len(seed_behavior_ids)} seed(s), "
                f"user {user_id}, session {session_id}"
            )
            return []

        # Apply lazy decay in Python and recompute rank_score before final sort.
        # Row layout: 12 fixed columns, then an optional min_query_distance
        # column when the relevance gate was active.
        current_time = int(time.time())
        decayed_rows = []
        for row in rows:
            neighbor_id = row[0]
            behavior_text = row[1]
            stored_credibility = row[2]
            intent = row[3]
            target = row[4]
            context = row[5]
            polarity = row[6]
            decay_rate = row[7]
            last_decay_applied_at = row[8]
            edge_type = row[9]
            edge_weight = row[10]
            # row[11] is rank_score from SQL (replaced below), row[12] is optional min_query_distance

            decayed_credibility, _, _ = apply_lazy_decay(
                stored_credibility=float(stored_credibility),
                decay_rate=float(decay_rate),
                last_decay_applied_at=last_decay_applied_at,
                current_time=current_time,
            )
            decayed_rank_score = float(edge_weight) * decayed_credibility

            decayed_rows.append((
                neighbor_id, behavior_text, decayed_credibility, intent,
                target, context, polarity, edge_type, float(edge_weight),
                decayed_rank_score,
            ))

        # Re-sort by decayed rank_score and apply final limit
        decayed_rows.sort(key=lambda r: r[9], reverse=True)
        decayed_rows = decayed_rows[:limit]

        results = []
        for row in decayed_rows:
            results.append({
                "behavior_id": row[0],
                "behavior_text": row[1],
                "credibility": row[2],
                "intent": row[3],
                "target": row[4],
                "context": row[5],
                "polarity": row[6],
                "edge_type": row[7],
                "edge_weight": row[8],
                "source": "graph",
            })

        gate_info = (
            f", relevance_gate=ON (τ={GRAPH_EXPANSION_DISTANCE_THRESHOLD:.2f}, "
            f"probes={len(query_embeddings) if query_embeddings else 0})"
            if use_relevance_gate else ", relevance_gate=OFF"
        )
        logger.info(
            f"[GRAPH] Expanded {len(seed_behavior_ids)} seed(s) → "
            f"{len(results)} associated behavior(s) for user {user_id}, session {session_id}"
            f"{gate_info}"
        )
        return results

    except Exception as e:
        logger.error(f"[GRAPH] Failed to expand graph: {str(e)}")
        return []


def _inherit_edges_on_cursor(
    cur,
    winner_id: str,
    loser_id: str,
    user_id: str,
    current_timestamp: int,
) -> int:
    """
    Transfer co-occurrence edges from the loser to the winner during
    conflict resolution.  Runs on an existing cursor (no commit).

    For every edge the loser has, create an equivalent edge pointing
    at the winner — at half the original weight (inherited association
    is weaker than direct co-occurrence).

    Skips edges where:
    - The neighbor IS the winner (self-loop)
    - An edge between the winner and that neighbor already exists
      with the same edge_type (ON CONFLICT DO NOTHING)

    Args:
        cur: Open psycopg cursor in an active transaction.
        winner_id: Behavior that survived the conflict.
        loser_id: Behavior that was superseded.
        user_id: User identifier.
        current_timestamp: Unix epoch seconds.

    Returns:
        Number of edges inherited.
    """
    # Step 1: Collect all of the loser's neighbors (both directions)
    cur.execute(
        """
        SELECT neighbor_id, edge_type, weight FROM (
            SELECT behavior_id_2 AS neighbor_id, edge_type, weight
            FROM behavior_co_occurrences
            WHERE user_id = %s AND behavior_id_1 = %s

            UNION ALL

            SELECT behavior_id_1 AS neighbor_id, edge_type, weight
            FROM behavior_co_occurrences
            WHERE user_id = %s AND behavior_id_2 = %s
        ) sub
        WHERE neighbor_id != %s
        """,
        (user_id, loser_id, user_id, loser_id, winner_id),
    )

    loser_edges = cur.fetchall()
    if not loser_edges:
        return 0

    # Step 2: Insert inherited edges (canonical order: min < max)
    inherited = 0
    for neighbor_id, edge_type, weight in loser_edges:
        a, b = (min(winner_id, neighbor_id), max(winner_id, neighbor_id))
        cur.execute(
            """
            INSERT INTO behavior_co_occurrences
                (behavior_id_1, behavior_id_2, user_id, edge_type, weight, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (behavior_id_1, behavior_id_2, edge_type)
            DO NOTHING
            """,
            (a, b, user_id, edge_type, float(weight) * 0.5, current_timestamp),
        )
        inherited += cur.rowcount  # 1 if inserted, 0 if conflict

    logger.info(
        f"[GRAPH] Inherited {inherited} edge(s) from {loser_id} → {winner_id}"
    )
    return inherited
