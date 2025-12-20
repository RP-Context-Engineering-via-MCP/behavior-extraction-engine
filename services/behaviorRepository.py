from db.connection import get_db_connection
from datetime import datetime
from typing import List, Tuple

def insert_behavior(payload: dict):
    with get_db_connection() as conn:
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