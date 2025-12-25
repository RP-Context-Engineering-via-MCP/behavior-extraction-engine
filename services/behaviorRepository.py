from db.connection import get_db_connection, get_db_pool_connection
from datetime import datetime
from typing import List, Tuple
from models.behavior import PromptSegment, SegmentInsertResult
import logging
logger = logging.getLogger(__name__)

def insert_behavior(payload: dict):
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
                    prompt_history_ids
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
                    %(prompt_history_ids)s
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


def search_similar_behaviors(
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