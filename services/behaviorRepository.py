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
        limit: int = 5
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
                    AND session_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW', 'FLAGGED')
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
    query_embedding: List[float],
    query_text: str,
    session_id: str = "default",
    required_intents: Optional[List[str]] = None,
    limit: int = None
) -> HybridSearchResponse:
    """
    Tuple-Guided Hybrid Retrieval (TGHR) — 3-dimensional behavior search.
    
    Combines three retrieval signals in a single database query:
      1. Dense retrieval  — cosine similarity via pgvector (semantic meaning)
      2. Sparse retrieval — BM25 via PostgreSQL tsvector (exact keyword matching)
      3. Metadata pre-filtering — intent-based filtering predicted by the LLM
    
    The dense and sparse scores are fused using a weighted linear combination:
        hybrid_score = DENSE_WEIGHT * (1 - cosine_distance) + SPARSE_WEIGHT * bm25_rank
    
    This method is READ-ONLY at query time. It collects pending updates
    (lazy decay + last_accessed_at) which the caller should persist
    asynchronously via persist_retrieval_updates_batch().
    
    Args:
        user_id: User identifier
        query_embedding: Dense vector embedding of the standalone query
        query_text: Plain text of the standalone query (used for BM25 sparse search)
        session_id: Session identifier for isolation (defaults to "default")
        required_intents: Optional list of intent types predicted by the LLM
                          (e.g., ["CONSTRAINT", "PREFERENCE"]). If None or empty,
                          no intent filtering is applied.
        limit: Maximum number of results to return (defaults to HYBRID_SEARCH_LIMIT)
    
    Returns:
        HybridSearchResponse containing:
          - results: List of SimilarityResult objects (sorted by hybrid_score desc)
          - decay_updates: Pending credibility updates for async persistence
          - accessed_behavior_ids: IDs of all returned behaviors for last_accessed_at update
    """
    from config.configurations import (
        HYBRID_DENSE_WEIGHT,
        HYBRID_SPARSE_WEIGHT,
        HYBRID_INTENT_BOOST_WEIGHT,
        HYBRID_SEARCH_LIMIT,
        HYBRID_SCORE_THRESHOLD,
        RELEVANCE_GAP_DROP_RATIO,
        MAX_RETRIEVAL_RESULTS,
        ALL_INTENT_TYPES,
        INTENT_AFFINITY
    )
    
    if limit is None:
        limit = HYBRID_SEARCH_LIMIT

    # ----------------------------------------------------------
    # Build OR-based tsquery from query text.
    # plainto_tsquery uses AND between all terms, which produces 0
    # when short behavior texts only partially overlap with long queries.
    # OR-based matching gives partial credit for any keyword match.
    # ----------------------------------------------------------
    def _build_or_tsquery(text: str) -> str:
        """Convert natural language text to OR-based tsquery string.
        
        'What foods should I eat before my morning run'
        → 'food | eat | morn | run'  (after PostgreSQL stemming)
        
        We send the raw words joined by | and let to_tsquery('english', ...)
        handle stemming. Stop words are kept but to_tsquery ignores them.
        """
        import re
        # Extract alphanumeric words, skip very short ones (likely stop words)
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        if not words:
            return text  # fallback: let PostgreSQL handle it
        return ' | '.join(words)

    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                # ----------------------------------------------------------
                # Build the WHERE clause dynamically based on intent filter
                # ----------------------------------------------------------
                base_conditions = """
                    user_id = %s
                    AND session_id = %s
                    AND behavior_state IN ('ACTIVE', 'NEW', 'FLAGGED')
                """
                where_params: list = [user_id, session_id]

                # ----------------------------------------------------------
                # Hybrid scoring query — 3 signals combined:
                #   Signal 1: Dense score  = 1 - cosine_distance  (semantic)
                #   Signal 2: Sparse score = ts_rank_cd / BM25    (keyword)
                #   Signal 3: Intent boost = graduated affinity-based boost
                #
                # Intent is a GRADUATED SOFT BOOST via affinity matrix.
                # Exact intent matches get full boost, related intents get
                # partial boost (e.g., HABIT→CONSTRAINT = 0.50), unrelated
                # intents get 0. This prevents "intent blind spots".
                #
                # IMPORTANT: params must be ordered to match %s appearance:
                #   SELECT clause params → WHERE clause params → LIMIT
                # ----------------------------------------------------------
                
                # Build intent boost SQL fragment (graduated via affinity matrix)
                # Instead of binary 1.0/0.0, related intents get partial boost.
                # This prevents "intent blind spots" where HABIT behaviors about
                # health are invisible to CONSTRAINT/PREFERENCE queries.
                use_intent_boost = required_intents and len(required_intents) > 0
                if use_intent_boost:
                    boost_map = {}
                    for intent_type in ALL_INTENT_TYPES:
                        if intent_type in required_intents:
                            boost_map[intent_type] = 1.0
                        else:
                            max_affinity = 0.0
                            for req in required_intents:
                                key = frozenset({intent_type, req})
                                affinity = INTENT_AFFINITY.get(key, 0.0)
                                max_affinity = max(max_affinity, affinity)
                            boost_map[intent_type] = max_affinity

                    # Build graduated CASE — intent names are from ALL_INTENT_TYPES
                    # (known enum, safe to interpolate). Only weight is parameterized.
                    case_parts = []
                    for intent_type, boost_val in boost_map.items():
                        if boost_val > 0.0:
                            case_parts.append(f"WHEN intent = '{intent_type}' THEN {boost_val:.4f}")

                    if case_parts:
                        case_sql = f"CASE {' '.join(case_parts)} ELSE 0.0 END"
                        intent_boost_sql = f"%s * ({case_sql})"
                    else:
                        intent_boost_sql = "0.0"
                        use_intent_boost = False

                    logger.debug(f"[3D] Intent affinity boosts: {boost_map}")
                else:
                    intent_boost_sql = "0.0"

                # Build OR-based tsquery string for BM25 sparse matching
                or_tsquery_str = _build_or_tsquery(query_text)
                logger.debug(f"[3D] OR tsquery: '{or_tsquery_str}'")

                query = f"""
                    SELECT
                        behavior_id,
                        behavior_text,
                        embedding <=> %s::vector AS cosine_distance,
                        ts_rank_cd(
                            COALESCE(search_vector, to_tsvector('english', behavior_text || ' ' || COALESCE(target, '') || ' ' || COALESCE(context, ''))),
                            to_tsquery('english', %s)
                        ) AS bm25_score,
                        credibility,
                        last_seen_at,
                        reinforcement_count,
                        intent,
                        target,
                        context,
                        polarity,
                        decay_rate,
                        last_decay_applied_at,
                        (
                            %s * (1.0 - (embedding <=> %s::vector))
                            +
                            %s * ts_rank_cd(
                                COALESCE(search_vector, to_tsvector('english', behavior_text || ' ' || COALESCE(target, '') || ' ' || COALESCE(context, ''))),
                                to_tsquery('english', %s)
                            )
                            +
                            {intent_boost_sql}
                        ) AS hybrid_score
                    FROM behaviors
                    WHERE {base_conditions}
                    ORDER BY hybrid_score DESC
                    LIMIT %s;
                """
                
                # Build params in SQL %s appearance order: SELECT → WHERE → LIMIT
                select_params = [
                    query_embedding,          # embedding <=> %s::vector (cosine_distance)
                    or_tsquery_str,           # to_tsquery('english', %s) (bm25_score)
                    HYBRID_DENSE_WEIGHT,      # %s * (1.0 - ...) (dense weight)
                    query_embedding,          # embedding <=> %s::vector (hybrid dense)
                    HYBRID_SPARSE_WEIGHT,     # %s * ts_rank_cd(...) (sparse weight)
                    or_tsquery_str,           # to_tsquery('english', %s) (hybrid sparse)
                ]
                # Add intent boost weight param if applicable
                # (boost values per intent are baked into CASE — only weight is parameterized)
                if use_intent_boost:
                    select_params.append(HYBRID_INTENT_BOOST_WEIGHT)  # %s * (CASE ...)
                
                params = select_params + where_params + [limit]

                cur.execute(query, params)
                results = cur.fetchall()
                current_time = int(time.time())

                # ----------------------------------------------------------
                # Build SimilarityResult objects with in-memory lazy decay
                # Collect pending updates for async persistence
                # ----------------------------------------------------------
                similarity_results = []
                decay_updates = []
                accessed_behavior_ids = []
                
                for row in results:
                    (
                        behavior_id, behavior_text, cosine_distance, bm25_score,
                        stored_credibility, last_seen_at, reinforcement_count,
                        intent, target, context, polarity,
                        decay_rate, last_decay_applied_at, hybrid_score
                    ) = row

                    # Skip results below the hybrid score threshold
                    if float(hybrid_score) < HYBRID_SCORE_THRESHOLD:
                        continue

                    # Apply lazy decay in-memory (read-only, no DB update)
                    new_credibility, decay_applied, days_elapsed = apply_lazy_decay(
                        stored_credibility=float(stored_credibility),
                        decay_rate=float(decay_rate),
                        last_decay_applied_at=last_decay_applied_at,
                        current_time=current_time
                    )

                    if decay_applied:
                        logger.debug(
                            f"[3D] Lazy decay (in-memory) for {behavior_id}: "
                            f"{stored_credibility:.4f} → {new_credibility:.4f} "
                            f"({days_elapsed} days)"
                        )
                        # Collect for async batch update
                        decay_updates.append((
                            new_credibility,
                            current_time,
                            behavior_id,
                            user_id
                        ))

                    # Map hybrid_score → distance so lower = better (API consistency)
                    # hybrid_score is 0..~1, so distance = 1 - hybrid_score
                    effective_distance = max(0.0, 1.0 - float(hybrid_score))

                    similarity_results.append(SimilarityResult(
                        behavior_id=behavior_id,
                        behavior_text=behavior_text,
                        distance=effective_distance,
                        credibility=new_credibility,
                        last_seen_at=int(last_seen_at),
                        reinforcement_count=int(reinforcement_count),
                        intent=intent,
                        target=target,
                        context=context if context else "general",
                        polarity=polarity
                    ))
                    
                    # Track all returned behavior IDs for last_accessed_at update
                    accessed_behavior_ids.append(behavior_id)

                    # Log each matched behavior with scoring breakdown
                    logger.info(
                        f"[3D] MATCH #{len(similarity_results)}: "
                        f"id={behavior_id} | intent={intent} | "
                        f"dense={1.0 - float(cosine_distance):.4f} | "
                        f"sparse={float(bm25_score):.4f} | "
                        f"hybrid={float(hybrid_score):.4f} | "
                        f"credibility={new_credibility:.4f} | "
                        f"text='{behavior_text[:80]}...'"
                    )

                # ----------------------------------------------------------
                # Apply relevance gap cutoff:
                # If a result's score drops more than RELEVANCE_GAP_DROP_RATIO
                # below the top result, stop including further results.
                # This prevents low-quality tail noise from being returned.
                # ----------------------------------------------------------
                if similarity_results:
                    top_score = float(1.0 - similarity_results[0].distance)  # convert back to hybrid score
                    cutoff_score = top_score * (1.0 - RELEVANCE_GAP_DROP_RATIO)
                    
                    filtered_results = []
                    filtered_decay = []
                    filtered_ids = []
                    
                    for i, sr in enumerate(similarity_results):
                        sr_score = 1.0 - sr.distance
                        if sr_score < cutoff_score:
                            break
                        filtered_results.append(sr)
                        # Keep corresponding decay update if it exists
                        # decay_updates and accessed_behavior_ids align with similarity_results
                        if sr.behavior_id in [d[2] for d in decay_updates]:
                            for d in decay_updates:
                                if d[2] == sr.behavior_id:
                                    filtered_decay.append(d)
                                    break
                        filtered_ids.append(sr.behavior_id)
                    
                    dropped = len(similarity_results) - len(filtered_results)
                    if dropped > 0:
                        logger.info(
                            f"[3D] Relevance gap cutoff: kept {len(filtered_results)}, "
                            f"dropped {dropped} (top_score={top_score:.4f}, "
                            f"cutoff={cutoff_score:.4f})"
                        )
                    
                    similarity_results = filtered_results
                    decay_updates = filtered_decay
                    accessed_behavior_ids = filtered_ids

                # ----------------------------------------------------------
                # Soft cap: limit maximum results to prevent over-retrieval
                # for broad/vague queries where many behaviors cluster in a
                # similar score range and gap cutoff alone can't separate them.
                # ----------------------------------------------------------
                if len(similarity_results) > MAX_RETRIEVAL_RESULTS:
                    dropped_by_cap = len(similarity_results) - MAX_RETRIEVAL_RESULTS
                    similarity_results = similarity_results[:MAX_RETRIEVAL_RESULTS]
                    accessed_behavior_ids = accessed_behavior_ids[:MAX_RETRIEVAL_RESULTS]
                    remaining_ids = set(accessed_behavior_ids)
                    decay_updates = [d for d in decay_updates if d[2] in remaining_ids]
                    logger.info(
                        f"[3D] Soft cap applied: kept top {MAX_RETRIEVAL_RESULTS}, "
                        f"dropped {dropped_by_cap} excess results"
                    )

                logger.info(
                    f"[3D] Hybrid search for user {user_id} in session {session_id}: "
                    f"{len(similarity_results)} results "
                    f"(dense_w={HYBRID_DENSE_WEIGHT}, sparse_w={HYBRID_SPARSE_WEIGHT}, "
                    f"intent_boost_w={HYBRID_INTENT_BOOST_WEIGHT}, "
                    f"intents_boost={required_intents or 'NONE'}, "
                    f"decay_pending={len(decay_updates)}, "
                    f"query='{query_text[:60]}...')"
                )
                return HybridSearchResponse(
                    results=similarity_results,
                    decay_updates=decay_updates,
                    accessed_behavior_ids=accessed_behavior_ids
                )

    except Exception as e:
        logger.error(f"[3D] Failed to search similar behaviors: {str(e)}")
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
                            behavior_state = %s,
                            last_accessed_at = %s
                        WHERE behavior_id = %s AND user_id = %s
                        """,
                        (BehaviorState.SUPERSEDED.value, current_timestamp, behavior_id_2, user_id)
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

