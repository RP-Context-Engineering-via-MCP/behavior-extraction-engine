from db.connection import get_db_connection, get_db_pool_connection
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from models.behavior import (
    SimilarityClassification,
    SimilarityResult,
    ReinforcementResult,
    ConflictType,
    ResolutionStatus,
    BehaviorState,
    RetrievedBehavior,
    RetrievalRelationship,
)
from services.credibilityCalculator import calculate_reinforcement_boost, apply_lazy_decay
from services.eventPublisher import get_event_publisher
from config.configurations import DECAY_GRACE_PERIOD_SECONDS
import time
import uuid
import math
import logging
logger = logging.getLogger(__name__)


@dataclass
class HMBRResponse:
    """
    Result of HMBR retrieval — synchronous results plus pending async updates.

    Attributes:
        results:               Final fused/ranked RetrievedBehavior objects
                               sorted by S_final descending.
        decay_updates:         (new_credibility, timestamp, behavior_id, user_id)
                               tuples for behaviors that had lazy-decay applied
                               in-memory during retrieval.  Persisted async.
        accessed_behavior_ids: All behavior_ids returned in `results`; used by
                               persist_retrieval_updates_batch to bump
                               last_accessed_at.
    """
    results: List[RetrievedBehavior] = field(default_factory=list)
    decay_updates: List[tuple] = field(default_factory=list)
    accessed_behavior_ids: List[str] = field(default_factory=list)

def insert_behavior(payload: dict):
    """
    Insert a new behavior into the database.

    The payload is the dict form of a StoredBehavior model (see models/behavior.py).
    Defaults behavior_state to 'ACTIVE' and usefulness_score to 0.5 if not provided.

    Side-effect: kicks off SEMANTIC_SIMILAR edge generation for the new behavior
    so that HMBR's graph-expansion stage can find it without waiting on the cron.
    Failures here are logged but do not block the insert.
    """
    payload.setdefault('behavior_state', BehaviorState.ACTIVE.value)
    payload.setdefault('usefulness_score', 0.5)

    # tsvector content for the lexical retrieval lane: behavior_text + target + context
    search_text = payload.get('behavior_text', '')
    target_val = payload.get('target') or ''
    context_val = payload.get('context') or ''
    payload['search_text'] = f"{search_text} {target_val} {context_val}".strip()

    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO behaviors (
                    behavior_id, user_id, session_id, behavior_text,
                    intent, target, context, polarity,
                    credibility, decay_rate, reinforcement_count,
                    usefulness_score, behavior_state,
                    created_at, last_seen_at, last_accessed_at, last_decay_applied_at,
                    embedding, canonical_embedding, search_vector,
                    extraction_confidence, clarity_score, linguistic_strength
                )
                VALUES (
                    %(behavior_id)s, %(user_id)s, %(session_id)s, %(behavior_text)s,
                    %(intent)s, %(target)s, %(context)s, %(polarity)s,
                    %(credibility)s, %(decay_rate)s, %(reinforcement_count)s,
                    %(usefulness_score)s, %(behavior_state)s,
                    %(created_at)s, %(last_seen_at)s, %(last_accessed_at)s, %(last_decay_applied_at)s,
                    %(embedding)s, %(canonical_embedding)s,
                    to_tsvector('english', %(search_text)s),
                    %(extraction_confidence)s, %(clarity_score)s, %(linguistic_strength)s
                )
                """,
                payload,
            )
        conn.commit()

    # Compute SEMANTIC_SIMILAR edges so the new behavior is reachable in PPR
    # graph walks even when it never co-occurred with any other behavior.
    try:
        if payload.get('canonical_embedding') is not None:
            compute_semantic_similar_edges_for_behavior(
                behavior_id=payload['behavior_id'],
                user_id=payload['user_id'],
                session_id=payload['session_id'],
                canonical_embedding=payload['canonical_embedding'],
            )
    except Exception as e:
        logger.warning(f"Failed to compute SEMANTIC_SIMILAR edges for new behavior: {e}")

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
            last_seen_at=payload.get('last_seen_at', int(time.time())),
        )
    except Exception as e:
        logger.warning(f"Failed to publish behavior.created event: {e}")

def _reinforce_behavior_on_cursor(
    cur,
    behavior_id: str,
    user_id: str,
    current_timestamp: Optional[int] = None,
) -> ReinforcementResult:
    """
    Reinforce a behavior on an *existing* open cursor (caller commits).

    Increments reinforcement_count, applies a diminishing-returns credibility
    boost, refreshes last_seen_at / last_accessed_at / last_decay_applied_at,
    and bumps usefulness_score upward — reinforcement is implicit positive
    feedback that any prior retrieval result was useful.
    """
    if current_timestamp is None:
        current_timestamp = int(time.time())

    cur.execute(
        """
        SELECT credibility, reinforcement_count
        FROM behaviors
        WHERE behavior_id = %s AND user_id = %s;
        """,
        (behavior_id, user_id),
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
            error=f"Behavior not found: {behavior_id}",
        )

    current_credibility, current_count = result
    boost = calculate_reinforcement_boost(float(current_credibility), int(current_count))
    new_credibility = min(1.0, float(current_credibility) + boost)
    new_count = int(current_count) + 1

    cur.execute(
        """
        UPDATE behaviors
        SET credibility = %s,
            reinforcement_count = %s,
            last_seen_at = %s,
            last_decay_applied_at = %s,
            last_accessed_at = %s,
            usefulness_score = LEAST(1.0, usefulness_score + 0.05)
        WHERE behavior_id = %s AND user_id = %s;
        """,
        (
            new_credibility,
            new_count,
            current_timestamp,
            current_timestamp,
            current_timestamp,
            behavior_id,
            user_id,
        ),
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
        error=None,
    )


def reinforce_behavior(
    behavior_id: str,
    user_id: str,
) -> ReinforcementResult:
    """
    Reinforce an existing behavior by incrementing reinforcement count,
    boosting credibility, updating timestamps, and bumping usefulness_score.

    Called when a duplicate behavior is detected (instead of inserting a new one)
    or when retrieval feedback indicates the behavior is being actively reused.

    Returns:
        ReinforcementResult with success status and updated values
    """
    try:
        current_timestamp = int(time.time())

        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                reinforce_result = _reinforce_behavior_on_cursor(
                    cur, behavior_id, user_id, current_timestamp
                )
                conn.commit()

        # After reinforcement is committed, check if this behavior is
        # involved in any PENDING conflict that can now be auto-resolved.
        if reinforce_result.success:
            _check_and_auto_resolve_conflicts(behavior_id, user_id)

            try:
                publisher = get_event_publisher()
                publisher.publish_behavior_reinforced(
                    user_id=user_id,
                    behavior_id=behavior_id,
                    reinforcement_count=reinforce_result.new_reinforcement_count,
                    credibility=reinforce_result.new_credibility,
                    last_seen_at=current_timestamp,
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
            error=str(e),
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
                query = """
                    SELECT
                        behavior_id, user_id, session_id, behavior_text,
                        intent, target, context, polarity,
                        credibility, reinforcement_count, usefulness_score,
                        last_seen_at, created_at, behavior_state
                    FROM behaviors
                    WHERE user_id = %s
                      AND behavior_state = ANY(%s)
                """
                params = [user_id, include_states]

                if session_id is not None:
                    query += " AND session_id = %s"
                    params.append(session_id)

                query += " ORDER BY created_at DESC;"

                cur.execute(query, params)
                results = cur.fetchall()

                behaviors = [
                    {
                        'behavior_id': row[0],
                        'user_id': row[1],
                        'session_id': row[2],
                        'behavior_text': row[3],
                        'intent': row[4],
                        'target': row[5],
                        'context': row[6],
                        'polarity': row[7],
                        'credibility': float(row[8]) if row[8] is not None else 0.0,
                        'reinforcement_count': int(row[9]) if row[9] is not None else 0,
                        'usefulness_score': float(row[10]) if row[10] is not None else 0.5,
                        'last_seen_at': int(row[11]) if row[11] is not None else 0,
                        'created_at': int(row[12]) if row[12] is not None else 0,
                        'behavior_state': row[13],
                    }
                    for row in results
                ]
                logger.debug(
                    f"Found {len(behaviors)} behaviors for user {user_id} "
                    f"(session={session_id or 'ALL'})"
                )
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
                query = """
                    SELECT
                        behavior_id, user_id, session_id, behavior_text,
                        intent, target, context, polarity,
                        credibility, reinforcement_count, usefulness_score,
                        decay_rate, last_decay_applied_at,
                        created_at, last_seen_at, last_accessed_at,
                        behavior_state
                    FROM behaviors
                    WHERE user_id = %s
                """
                params = [user_id]
                if session_id is not None:
                    query += " AND session_id = %s"
                    params.append(session_id)
                query += " ORDER BY last_seen_at DESC"

                cur.execute(query, params)

                current_time = int(time.time())
                behaviors = []
                behaviors_to_update = []

                for row in cur.fetchall():
                    (
                        behavior_id, _user_id, b_session_id, behavior_text,
                        intent, target, context, polarity,
                        stored_credibility, reinforcement_count, usefulness_score,
                        decay_rate, last_decay_applied_at,
                        created_at, last_seen_at, last_accessed_at,
                        behavior_state,
                    ) = row

                    new_credibility, decay_applied, _days = apply_lazy_decay(
                        stored_credibility=float(stored_credibility),
                        decay_rate=float(decay_rate),
                        last_decay_applied_at=last_decay_applied_at,
                        current_time=current_time,
                    )
                    if decay_applied:
                        behaviors_to_update.append(
                            (new_credibility, current_time, behavior_id, user_id)
                        )

                    behaviors.append({
                        "behavior_id": behavior_id,
                        "user_id": _user_id,
                        "session_id": b_session_id,
                        "behavior_text": behavior_text,
                        "intent": intent,
                        "target": target,
                        "context": context,
                        "polarity": polarity,
                        "credibility": new_credibility,
                        "reinforcement_count": reinforcement_count,
                        "usefulness_score": float(usefulness_score) if usefulness_score is not None else 0.5,
                        "decay_rate": float(decay_rate) if decay_rate is not None else 0.0,
                        "created_at": created_at,
                        "last_seen_at": last_seen_at,
                        "last_accessed_at": last_accessed_at,
                        "behavior_state": behavior_state,
                    })

                if behaviors_to_update:
                    cur.executemany(
                        """
                        UPDATE behaviors
                        SET credibility = %s, last_decay_applied_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        behaviors_to_update,
                    )
                    conn.commit()

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
                placeholders = ','.join(['%s'] * len(behavior_ids))
                cur.execute(
                    f"""
                    SELECT
                        behavior_id, user_id, session_id, behavior_text,
                        intent, target, context, polarity,
                        credibility, reinforcement_count, usefulness_score,
                        decay_rate, last_decay_applied_at,
                        created_at, last_seen_at, last_accessed_at,
                        behavior_state, superseded_by_id
                    FROM behaviors
                    WHERE user_id = %s AND behavior_id IN ({placeholders})
                    ORDER BY last_seen_at DESC
                    """,
                    [user_id] + behavior_ids,
                )
                rows = cur.fetchall()

        behaviors = []
        for row in rows:
            behaviors.append({
                "behavior_id": row[0],
                "user_id": row[1],
                "session_id": row[2],
                "behavior_text": row[3],
                "canonical": {
                    "intent": row[4],
                    "target": row[5],
                    "context": row[6],
                    "polarity": row[7],
                },
                "credibility": float(row[8]) if row[8] is not None else 0.0,
                "reinforcement_count": row[9] or 0,
                "usefulness_score": float(row[10]) if row[10] is not None else 0.5,
                "decay_rate": float(row[11]) if row[11] is not None else 0.0,
                "last_decay_applied_at": row[12],
                "created_at": row[13],
                "last_seen_at": row[14],
                "last_accessed_at": row[15],
                "behavior_state": row[16],
                "superseded_by_id": row[17],
            })

        logger.info(f"Retrieved {len(behaviors)} behaviors for user={user_id}")
        return behaviors

    except Exception as e:
        logger.error(f"Failed to retrieve behaviors by IDs for user={user_id}: {str(e)}")
        raise


# ===========================================================================
# HMBR — Hybrid Multi-Signal Behavior Retrieval
# ===========================================================================

def compute_semantic_similar_edges_for_behavior(
    behavior_id: str,
    user_id: str,
    session_id: str,
    canonical_embedding: List[float],
) -> int:
    """
    Find top-N existing behaviors closest to this one by canonical_embedding
    and create SEMANTIC_SIMILAR edges.  Called right after insert_behavior so
    the new node is reachable in PPR walks immediately.
    """
    from config.configurations import (
        HMBR_SEMANTIC_SIMILAR_TOP_N,
        HMBR_SEMANTIC_SIMILAR_MAX_DISTANCE,
    )
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT behavior_id
                    FROM behaviors
                    WHERE user_id = %s
                      AND behavior_id != %s
                      AND behavior_state IN ('ACTIVE', 'NEW')
                      AND canonical_embedding IS NOT NULL
                      AND (canonical_embedding <=> %s::vector) <= %s
                    ORDER BY canonical_embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        user_id,
                        behavior_id,
                        canonical_embedding,
                        HMBR_SEMANTIC_SIMILAR_MAX_DISTANCE,
                        canonical_embedding,
                        HMBR_SEMANTIC_SIMILAR_TOP_N,
                    ),
                )
                neighbors = [row[0] for row in cur.fetchall()]

        if not neighbors:
            return 0

        pairs = [
            (min(behavior_id, n), max(behavior_id, n))
            for n in neighbors
        ]
        created = _upsert_co_occurrence_pairs(pairs, user_id, "SEMANTIC_SIMILAR", session_id)
        logger.info(
            f"[GRAPH] SEMANTIC_SIMILAR: {created} edge(s) created for behavior={behavior_id}"
        )
        return created
    except Exception as e:
        logger.warning(f"[GRAPH] compute_semantic_similar_edges_for_behavior failed: {e}")
        return 0


def retrieve_behaviors(
    probes: list,
    query_type: str,
    required_intents: Optional[List[str]],
    user_id: str,
    session_id: str,
) -> "HMBRResponse":
    """
    Hybrid Multi-Signal Behavior Retrieval (HMBR).

    Pillar 1 — multi-signal candidate pool:
      Runs prose-cosine, canonical-cosine, and BM25 queries for each probe,
      merges results, and computes per-signal scores for each candidate.

    Pillar 2 — Personalised PageRank graph expansion:
      Loads the user's co-occurrence subgraph, seeds mass proportional to each
      candidate's Pillar-1 scores, and runs PPR to surface graph neighbors.

    Pillar 3 — adaptive fusion + elbow selection:
      Combines all signals with query_type-specific weights, applies a
      same-session boost, cuts at the score elbow, enforces a floor, and caps
      results at HMBR_MAX_RESULTS.
    """
    from config.configurations import (
        HMBR_PER_LANE_TOP_K,
        HMBR_LEXICAL_MIN_RANK,
        HMBR_RECENCY_TAU_DAYS,
        HMBR_SESSION_BOOST,
        HMBR_MAX_RESULTS,
        HMBR_MIN_FINAL_SCORE,
        HMBR_PPR_ALPHA,
        HMBR_PPR_ITERATIONS,
        HMBR_EDGE_WEIGHT_CO_PROMPT,
        HMBR_EDGE_WEIGHT_SEMANTIC_SIMILAR,
        HMBR_EDGE_WEIGHT_CO_SESSION,
        HMBR_GRAPH_MAX_NODES,
        HMBR_FUSION_WEIGHTS,
        INTENT_AFFINITY,
        DECAY_GRACE_PERIOD_SECONDS,
    )
    from services.openAiClient import embed_batch

    fusion_weights = HMBR_FUSION_WEIGHTS.get(query_type, HMBR_FUSION_WEIGHTS["BROAD"])
    current_ts = int(time.time())

    # -----------------------------------------------------------------------
    # Pre-compute ALL embeddings in ONE batch call before touching the DB.
    #
    # Previously each probe called embed_text() twice (prose + canonical),
    # meaning 3 probes × 2 = 6 sequential model.encode() calls at ~2–3s each
    # = 12–18s just on embeddings.  embed_batch() does a single forward pass
    # over all texts — one call regardless of how many probes there are.
    # -----------------------------------------------------------------------
    _texts_to_embed: list[str] = []
    _embed_plan: list[tuple[int, str]] = []  # (probe_index, 'prose'|'canonical')

    for i, probe in enumerate(probes):
        probe_text = probe.text if hasattr(probe, "text") else str(probe)
        _texts_to_embed.append(probe_text)
        _embed_plan.append((i, "prose"))
        if hasattr(probe, "canonical") and probe.canonical is not None:
            canon = probe.canonical
            _texts_to_embed.append(
                f"{canon.polarity} {canon.intent} {canon.target} {canon.context}"
            )
            _embed_plan.append((i, "canonical"))

    # Single model.encode() — the core performance fix
    _probe_embs: dict[int, dict[str, list]] = {}
    try:
        _all_embs = embed_batch(_texts_to_embed)
        for idx, (probe_idx, lane) in enumerate(_embed_plan):
            _probe_embs.setdefault(probe_idx, {})[lane] = _all_embs[idx]
    except Exception as e:
        logger.error(f"[HMBR] Batch embedding failed: {e}")
        return HMBRResponse()

    # -----------------------------------------------------------------------
    # Pillar 1 — build candidate pool
    #
    # Performance: instead of 1 execute() per probe per lane (up to 9 round-trips
    # for 3 probes), we build one UNION ALL per lane and send 3 round-trips total.
    # Each subquery uses (ORDER BY ... LIMIT k) inside parentheses so PostgreSQL
    # applies the top-K cut before combining results.
    # -----------------------------------------------------------------------
    candidates: dict[str, dict] = {}

    def _merge(bid: str, row_data: dict) -> None:
        if bid not in candidates:
            candidates[bid] = row_data.copy()
        else:
            existing = candidates[bid]
            for sig in ("s_semantic", "s_canonical", "s_lexical"):
                if row_data.get(sig, 0.0) > existing.get(sig, 0.0):
                    existing[sig] = row_data[sig]

    # Column layout shared by all three lanes (signal value is always col 16)
    _BASE_COLS = """
        behavior_id, behavior_text, intent, target, context,
        polarity, credibility, reinforcement_count, usefulness_score,
        decay_rate, last_decay_applied_at, created_at, last_seen_at,
        last_accessed_at, behavior_state, session_id
    """

    def _unpack(row, signal_name: str, signal_val: float) -> dict:
        sigs = {"s_semantic": 0.0, "s_canonical": 0.0, "s_lexical": 0.0}
        sigs[signal_name] = signal_val
        return {
            "behavior_text": row[1], "intent": row[2],
            "target": row[3], "context": row[4], "polarity": row[5],
            "credibility": float(row[6]) if row[6] else 0.0,
            "reinforcement_count": row[7] or 1,
            "usefulness_score": float(row[8]) if row[8] else 0.5,
            "decay_rate": float(row[9]) if row[9] else 0.015,
            "last_decay_applied_at": row[10],
            "created_at": row[11] or 0, "last_seen_at": row[12] or 0,
            "last_accessed_at": row[13], "behavior_state": row[14],
            "session_id": row[15],
            **sigs,
        }

    # Collect embeddings keyed by probe index
    prose_embs  = [(i, d["prose"])     for i, d in _probe_embs.items() if "prose"     in d]
    canon_embs  = [(i, d["canonical"]) for i, d in _probe_embs.items() if "canonical" in d]
    probe_texts = [
        (i, probe.text if hasattr(probe, "text") else str(probe))
        for i, probe in enumerate(probes)
    ]

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:

                # --- Prose cosine lane: one UNION ALL for all probes (1 round-trip) ---
                if prose_embs:
                    try:
                        subs, params = [], []
                        for _, emb in prose_embs:
                            subs.append(f"""
                                (SELECT {_BASE_COLS},
                                        (embedding <=> %s::vector) AS sig
                                 FROM behaviors
                                 WHERE user_id = %s
                                   AND behavior_state IN ('ACTIVE', 'NEW')
                                   AND embedding IS NOT NULL
                                 ORDER BY sig LIMIT %s)
                            """)
                            params.extend([emb, user_id, HMBR_PER_LANE_TOP_K])
                        cur.execute(" UNION ALL ".join(subs), params)
                        for row in cur.fetchall():
                            dist = float(row[16]) if row[16] is not None else 1.0
                            _merge(row[0], _unpack(row, "s_semantic", max(0.0, 1.0 - dist)))
                    except Exception as e:
                        logger.warning(f"[HMBR] prose UNION ALL failed: {e}")

                # --- Canonical cosine lane: one UNION ALL (1 round-trip) ---
                if canon_embs:
                    try:
                        subs, params = [], []
                        for _, emb in canon_embs:
                            subs.append(f"""
                                (SELECT {_BASE_COLS},
                                        (canonical_embedding <=> %s::vector) AS sig
                                 FROM behaviors
                                 WHERE user_id = %s
                                   AND behavior_state IN ('ACTIVE', 'NEW')
                                   AND canonical_embedding IS NOT NULL
                                 ORDER BY sig LIMIT %s)
                            """)
                            params.extend([emb, user_id, HMBR_PER_LANE_TOP_K])
                        cur.execute(" UNION ALL ".join(subs), params)
                        for row in cur.fetchall():
                            dist = float(row[16]) if row[16] is not None else 1.0
                            _merge(row[0], _unpack(row, "s_canonical", max(0.0, 1.0 - dist)))
                    except Exception as e:
                        logger.warning(f"[HMBR] canonical UNION ALL failed: {e}")

                # --- Lexical BM25 lane: one UNION ALL (1 round-trip) ---
                if probe_texts:
                    try:
                        subs, params = [], []
                        for _, pt in probe_texts:
                            subs.append(f"""
                                (SELECT {_BASE_COLS},
                                        ts_rank(search_vector, plainto_tsquery('english', %s)) AS sig
                                 FROM behaviors
                                 WHERE user_id = %s
                                   AND behavior_state IN ('ACTIVE', 'NEW')
                                   AND search_vector @@ plainto_tsquery('english', %s)
                                 ORDER BY sig DESC LIMIT %s)
                            """)
                            params.extend([pt, user_id, pt, HMBR_PER_LANE_TOP_K])
                        cur.execute(" UNION ALL ".join(subs), params)
                        for row in cur.fetchall():
                            rank = float(row[16]) if row[16] is not None else 0.0
                            if rank >= HMBR_LEXICAL_MIN_RANK:
                                _merge(row[0], _unpack(row, "s_lexical", min(1.0, rank)))
                    except Exception as e:
                        logger.warning(f"[HMBR] lexical UNION ALL failed: {e}")

    except Exception as e:
        logger.error(f"[HMBR] Pillar 1 DB error: {e}")
        return HMBRResponse()

    if not candidates:
        logger.info(f"[HMBR] No candidates found for user={user_id}")
        return HMBRResponse()

    # -----------------------------------------------------------------------
    # Per-candidate signal computation (recency, credibility, usefulness,
    # intent affinity, lazy decay)
    # -----------------------------------------------------------------------
    decay_updates: list[tuple] = []
    tau_seconds = HMBR_RECENCY_TAU_DAYS * 86400.0

    for bid, c in candidates.items():
        # Lazy decay
        credibility = c["credibility"]
        last_decay = c.get("last_decay_applied_at")
        age_since_grace = current_ts - (c["created_at"] + DECAY_GRACE_PERIOD_SECONDS)
        if age_since_grace > 0 and last_decay is not None:
            try:
                new_cred, applied, _ = apply_lazy_decay(
                    credibility, c["decay_rate"], last_decay, current_ts
                )
                if applied:
                    credibility = new_cred
                    decay_updates.append((new_cred, current_ts, bid, user_id))
            except Exception:
                pass
        c["credibility"] = credibility

        # Recency: exp(-Δt / τ), Δt = seconds since last_seen_at
        last_seen = c.get("last_seen_at") or c.get("created_at") or current_ts
        delta_t = max(0.0, float(current_ts - last_seen))
        c["s_recency"] = math.exp(-delta_t / tau_seconds)

        c["s_credibility"] = float(credibility)
        c["s_usefulness"] = float(c.get("usefulness_score", 0.5))

        # Intent affinity multiplier on semantic signals
        b_intent = c.get("intent")
        if required_intents and b_intent:
            best_affinity = max(
                (
                    1.0 if b_intent == ri else
                    INTENT_AFFINITY.get(frozenset({b_intent, ri}), 0.0)
                    for ri in required_intents
                ),
                default=0.3,
            )
            affinity_mult = 0.5 + 0.5 * best_affinity  # 0.5 … 1.0
            c["s_semantic"] = c.get("s_semantic", 0.0) * affinity_mult
            c["s_canonical"] = c.get("s_canonical", 0.0) * affinity_mult

    # -----------------------------------------------------------------------
    # Pillar 2 — Personalised PageRank
    # -----------------------------------------------------------------------
    candidate_ids = set(candidates.keys())
    ppr_scores: dict[str, float] = {bid: 0.0 for bid in candidate_ids}

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Load subgraph edges among (or adjacent to) candidates
                edge_weight_map = {
                    "CO_PROMPT": HMBR_EDGE_WEIGHT_CO_PROMPT,
                    "SEMANTIC_SIMILAR": HMBR_EDGE_WEIGHT_SEMANTIC_SIMILAR,
                    "CO_SESSION": HMBR_EDGE_WEIGHT_CO_SESSION,
                }
                cur.execute(
                    """
                    SELECT behavior_id_1, behavior_id_2, edge_type, weight
                    FROM behavior_co_occurrences
                    WHERE user_id = %s
                      AND (behavior_id_1 = ANY(%s) OR behavior_id_2 = ANY(%s))
                    LIMIT %s
                    """,
                    (user_id, list(candidate_ids), list(candidate_ids),
                     HMBR_GRAPH_MAX_NODES * 2),
                )
                edges_raw = cur.fetchall()

        # Build adjacency: node → [(neighbor, weight)]
        adjacency: dict[str, list[tuple[str, float]]] = {}
        all_nodes: set[str] = set(candidate_ids)
        for b1, b2, etype, w in edges_raw:
            base_w = edge_weight_map.get(etype, 0.1) * float(w)
            adjacency.setdefault(b1, []).append((b2, base_w))
            adjacency.setdefault(b2, []).append((b1, base_w))
            all_nodes.add(b1)
            all_nodes.add(b2)

        # Cap graph size
        if len(all_nodes) > HMBR_GRAPH_MAX_NODES:
            # keep all candidates; trim extra nodes by connection to seeds
            extra = all_nodes - candidate_ids
            conn_count = {n: 0 for n in extra}
            for n in extra:
                for nbr, _ in adjacency.get(n, []):
                    if nbr in candidate_ids:
                        conn_count[n] += 1
            keep_extra = sorted(extra, key=lambda n: -conn_count[n])[
                : HMBR_GRAPH_MAX_NODES - len(candidate_ids)
            ]
            all_nodes = candidate_ids | set(keep_extra)

        node_list = list(all_nodes)
        idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)

        # Build row-normalised transition matrix as sparse dict
        # T[i] = {j: w} where sum(T[i].values()) = 1
        out_weights: dict[int, float] = {}
        trans: dict[int, dict[int, float]] = {}
        for node in node_list:
            i = idx[node]
            nbrs = adjacency.get(node, [])
            nbrs_in_graph = [(idx[nb], w) for nb, w in nbrs if nb in idx]
            if not nbrs_in_graph:
                continue
            total = sum(w for _, w in nbrs_in_graph)
            trans[i] = {j: w / total for j, w in nbrs_in_graph}
            out_weights[i] = total

        # Seed distribution — proportional to pre-fusion seed score
        seed_dist: dict[int, float] = {}
        for bid in candidate_ids:
            c = candidates[bid]
            s_seed = (
                c.get("s_semantic", 0.0) * 0.4
                + c.get("s_canonical", 0.0) * 0.3
                + c.get("s_lexical", 0.0) * 0.15
                + c.get("s_credibility", 0.0) * 0.15
            )
            seed_dist[idx[bid]] = max(0.0, s_seed)

        seed_total = sum(seed_dist.values())
        if seed_total > 0:
            for i in seed_dist:
                seed_dist[i] /= seed_total

        # PPR power iterations: r = α·seed + (1-α)·Tᵀ·r
        r: dict[int, float] = dict(seed_dist)
        alpha = HMBR_PPR_ALPHA
        for _ in range(HMBR_PPR_ITERATIONS):
            new_r: dict[int, float] = {i: alpha * seed_dist.get(i, 0.0) for i in range(N)}
            for i, nbrs in trans.items():
                ri = r.get(i, 0.0)
                for j, w in nbrs.items():
                    new_r[j] = new_r.get(j, 0.0) + (1.0 - alpha) * ri * w
            r = new_r

        for bid in candidate_ids:
            ppr_scores[bid] = r.get(idx[bid], 0.0)

        # Also surface graph neighbors not already in candidates
        for node in node_list:
            if node not in candidate_ids:
                ppr_scores[node] = r.get(idx[node], 0.0)

    except Exception as e:
        logger.warning(f"[HMBR] PPR failed (using zero scores): {e}")

    # Normalise PPR scores to [0, 1]
    max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0
    if max_ppr > 0:
        ppr_scores = {k: v / max_ppr for k, v in ppr_scores.items()}

    # Fetch metadata for any graph-only nodes (not already in candidates)
    graph_only_ids = [n for n in ppr_scores if n not in candidates and ppr_scores[n] > 0]
    if graph_only_ids:
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT behavior_id, behavior_text, intent, target, context,
                               polarity, credibility, reinforcement_count, usefulness_score,
                               decay_rate, last_decay_applied_at, created_at, last_seen_at,
                               last_accessed_at, behavior_state, session_id
                        FROM behaviors
                        WHERE user_id = %s
                          AND behavior_id = ANY(%s)
                          AND behavior_state IN ('ACTIVE', 'NEW')
                        """,
                        (user_id, graph_only_ids),
                    )
                    for row in cur.fetchall():
                        bid = row[0]
                        last_seen = row[12] or row[11] or current_ts
                        delta_t = max(0.0, float(current_ts - last_seen))
                        s_recency = math.exp(-delta_t / tau_seconds)
                        candidates[bid] = {
                            "behavior_text": row[1], "intent": row[2],
                            "target": row[3], "context": row[4], "polarity": row[5],
                            "credibility": float(row[6]) if row[6] else 0.0,
                            "reinforcement_count": row[7] or 1,
                            "usefulness_score": float(row[8]) if row[8] else 0.5,
                            "decay_rate": float(row[9]) if row[9] else 0.015,
                            "last_decay_applied_at": row[10],
                            "created_at": row[11] or 0,
                            "last_seen_at": row[12] or 0,
                            "last_accessed_at": row[13],
                            "behavior_state": row[14],
                            "session_id": row[15],
                            "s_semantic": 0.0,
                            "s_canonical": 0.0,
                            "s_lexical": 0.0,
                            "s_recency": s_recency,
                            "s_credibility": float(row[6]) if row[6] else 0.0,
                            "s_usefulness": float(row[8]) if row[8] else 0.5,
                        }
        except Exception as e:
            logger.warning(f"[HMBR] Failed to fetch graph-only nodes: {e}")

    # -----------------------------------------------------------------------
    # Pillar 3 — adaptive fusion + elbow selection
    # -----------------------------------------------------------------------
    w = fusion_weights  # shorthand
    fused: list[tuple[str, float, dict]] = []

    for bid, c in candidates.items():
        s_ppr = ppr_scores.get(bid, 0.0)
        s_final = (
            w["sem"]    * c.get("s_semantic", 0.0)
            + w["canon"]  * c.get("s_canonical", 0.0)
            + w["lex"]    * c.get("s_lexical", 0.0)
            + w["rec"]    * c.get("s_recency", 0.0)
            + w["cred"]   * c.get("s_credibility", 0.0)
            + w["useful"] * c.get("s_usefulness", 0.0)
            + w["ppr"]    * s_ppr
        )

        # Same-session boost
        if c.get("session_id") == session_id:
            s_final = min(1.0, s_final + HMBR_SESSION_BOOST)

        c["s_ppr"] = s_ppr
        c["s_final"] = s_final
        fused.append((bid, s_final, c))

    # Sort descending by s_final
    fused.sort(key=lambda x: -x[1])

    # Elbow detection — find the biggest score gap in top-20 candidates
    top = fused[:max(HMBR_MAX_RESULTS * 2, 20)]
    cut_idx = len(top)
    if len(top) > 2:
        gaps = [(top[i][1] - top[i + 1][1], i + 1) for i in range(len(top) - 1)]
        max_gap, gap_pos = max(gaps, key=lambda g: g[0])
        # Only cut at the gap if it's meaningfully large (> 0.05)
        if max_gap > 0.05:
            cut_idx = gap_pos

    # Apply cuts: elbow, floor, cap
    results_raw = [
        (bid, s, c) for bid, s, c in fused[:cut_idx]
        if s >= HMBR_MIN_FINAL_SCORE
    ][:HMBR_MAX_RESULTS]

    # Polarity tagging — compare behavior polarity vs. primary probe polarity
    primary_polarity = None
    if probes:
        first = probes[0]
        if hasattr(first, "canonical") and first.canonical:
            primary_polarity = first.canonical.polarity

    retrieved: list[RetrievedBehavior] = []
    accessed_ids: list[str] = []
    for bid, s_final, c in results_raw:
        b_polarity = c.get("polarity")
        if primary_polarity and b_polarity:
            if b_polarity == primary_polarity:
                rel = RetrievalRelationship.AGREES
            else:
                rel = RetrievalRelationship.DISAGREES
        else:
            rel = RetrievalRelationship.NEUTRAL

        source = "seed" if bid in candidate_ids else "graph"
        retrieved.append(RetrievedBehavior(
            behavior_id=bid,
            behavior_text=c["behavior_text"],
            intent=c.get("intent"),
            target=c.get("target"),
            context=c.get("context"),
            polarity=b_polarity,
            credibility=c["credibility"],
            s_final=round(s_final, 4),
            s_semantic=round(c.get("s_semantic", 0.0), 4),
            s_canonical=round(c.get("s_canonical", 0.0), 4),
            s_lexical=round(c.get("s_lexical", 0.0), 4),
            s_recency=round(c.get("s_recency", 0.0), 4),
            s_credibility=round(c.get("s_credibility", 0.0), 4),
            s_usefulness=round(c.get("s_usefulness", 0.0), 4),
            s_ppr=round(c.get("s_ppr", 0.0), 4),
            relationship=rel,
            source=source,
        ))
        accessed_ids.append(bid)

    logger.info(
        f"[HMBR] user={user_id} query_type={query_type} "
        f"candidates={len(candidates)} returned={len(retrieved)}"
    )
    return HMBRResponse(
        results=retrieved,
        decay_updates=decay_updates,
        accessed_behavior_ids=accessed_ids,
    )


def persist_retrieval_updates_batch(
    decay_updates: list,
    accessed_behavior_ids: list,
    user_id: str,
) -> None:
    """
    Persist lazy-decay credibility updates and last_accessed_at timestamps
    for behaviors that were returned by HMBR retrieval.

    Called as a background task so it never blocks the HTTP response.
    """
    if not decay_updates and not accessed_behavior_ids:
        return

    current_ts = int(time.time())
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # Apply credibility updates from lazy decay
                for new_cred, decay_ts, bid, uid in decay_updates:
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET credibility = %s,
                            last_decay_applied_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (new_cred, decay_ts, bid, uid),
                    )

                # Bump last_accessed_at for all returned behaviors
                if accessed_behavior_ids:
                    cur.execute(
                        """
                        UPDATE behaviors
                        SET last_accessed_at = %s
                        WHERE behavior_id = ANY(%s) AND user_id = %s
                        """,
                        (current_ts, accessed_behavior_ids, user_id),
                    )
            conn.commit()
        logger.debug(
            f"[HMBR] persist_retrieval_updates_batch: "
            f"{len(decay_updates)} decay updates, "
            f"{len(accessed_behavior_ids)} access timestamps for user={user_id}"
        )
    except Exception as e:
        logger.error(f"[HMBR] persist_retrieval_updates_batch failed: {e}")


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


def _upsert_co_occurrence_pairs(
    pairs: List[Tuple[str, str]],
    user_id: str,
    edge_type: str,
    session_id: str,
) -> int:
    """
    Internal helper: UPSERT a list of canonical (a, b) pairs as edges.

    Each pair is assumed to already be in canonical order (a < b lexicographically)
    and free of self-loops.  On conflict, weight is bumped by +0.5 (diminishing
    reinforcement signal) and created_at is refreshed.
    """
    if not pairs:
        return 0

    current_timestamp = int(time.time())

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
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
            f"[GRAPH] Upserted {len(pairs)} {edge_type} edge(s) "
            f"for user {user_id} session {session_id}"
        )
        return len(pairs)

    except Exception as e:
        logger.error(f"[GRAPH] Failed to insert co-occurrence edges: {str(e)}")
        return 0


def insert_co_occurrences_batch(
    behavior_ids: List[str],
    user_id: str,
    edge_type: str,
    session_id: str = "default",
) -> int:
    """
    Create pairwise co-occurrence edges for a list of behavior IDs.

    Generates all unique (a, b) pairs (a < b lexicographically to avoid
    duplicate reversed edges) and upserts them.  Use this for CO_PROMPT
    edges where every behavior in the list genuinely co-occurred (e.g. all
    behaviors extracted from the same prompt).

    For "anchor + partners" semantics — i.e. when only a subset of the IDs
    are anchors that co-occurred with a separate partner set, but the
    partners did NOT co-occur with each other in this event — use
    insert_directed_pairs_batch instead.  Calling this function with an
    anchor + partners list would inflate edge weights between unrelated
    partner pairs.

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

    sorted_ids = sorted(set(behavior_ids))
    pairs = [
        (sorted_ids[i], sorted_ids[j])
        for i in range(len(sorted_ids))
        for j in range(i + 1, len(sorted_ids))
    ]
    return _upsert_co_occurrence_pairs(pairs, user_id, edge_type, session_id)


def insert_directed_pairs_batch(
    anchor_ids: List[str],
    partner_ids: List[str],
    user_id: str,
    edge_type: str,
    session_id: str = "default",
) -> int:
    """
    Create co-occurrence edges between every (anchor, partner) pair only.

    Unlike ``insert_co_occurrences_batch`` (which generates ALL pairwise
    combinations of a single id list), this helper emits edges strictly
    between the anchor set and the partner set.  Anchor↔anchor and
    partner↔partner pairs are NOT created — those did not co-occur in
    this event.

    Use case: CO_SESSION edges from newly inserted / reinforced behaviors
    (anchors) to pre-existing session behaviors (partners).  Earlier
    versions of this code passed ``[new_id] + existing_session_ids`` to
    ``insert_co_occurrences_batch`` once per new id, which inserted
    spurious existing↔existing edges and bumped their weights by +0.5
    on every prompt.  This helper avoids that quadratic noise.

    Edges are stored in canonical order (min, max).  Self-loops and
    duplicates are skipped.
    """
    if not anchor_ids or not partner_ids:
        return 0

    anchor_set = set(anchor_ids)
    partner_set = set(partner_ids)

    seen: set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []
    for a in anchor_set:
        for p in partner_set:
            if a == p:
                continue  # self-loop
            lo, hi = (a, p) if a < p else (p, a)
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            pairs.append((lo, hi))

    return _upsert_co_occurrence_pairs(pairs, user_id, edge_type, session_id)



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
