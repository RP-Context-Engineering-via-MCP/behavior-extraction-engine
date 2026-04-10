"""
Profile Signal Repository

Persists and retrieves profile_signals per prompt per user.
Enables the /recent endpoint to serve historical signals to the
Profile Service during drift fallback.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional

from db.connection import get_db_pool_connection

logger = logging.getLogger(__name__)


class ProfileSignalRepository:
    """
    Repository for user_profile_signals table operations.
    
    Stores profile_signals extracted from each prompt, enabling:
    1. Cold-start profiling (signals sent to Profile Service per prompt)
    2. Drift fallback (Profile Service fetches recent signals via API)
    """
    
    def save(
        self, 
        user_id: str, 
        prompt_id: str, 
        profile_signals: Dict[str, Any],
        behavior_id: Optional[str] = None
    ) -> None:
        """
        Save or update profile signals for a user's prompt.
        
        Uses UPSERT semantics: if (user_id, prompt_id) already exists,
        the profile_signals and extracted_at are updated.
        
        Args:
            user_id: Unique user identifier
            prompt_id: Unique prompt/request identifier
            profile_signals: Validated profile signals dict
            behavior_id: Optional behavior ID to link these signals to a specific behavior
            
        Raises:
            ValueError: If profile_signals has invalid format (e.g., canonical behavior format)
        """
        # Validate profile_signals format before saving
        required_fields = {'intents', 'interests', 'behavior_level'}
        if not required_fields.issubset(set(profile_signals.keys())):
            raise ValueError(
                f"Invalid profile_signals format. Missing required fields: "
                f"{required_fields - set(profile_signals.keys())}. "
                f"Got keys: {profile_signals.keys()}"
            )
        
        # Check for canonical behavior format (data corruption prevention)
        canonical_fields = {'intent', 'target', 'context', 'polarity'}
        if canonical_fields.issubset(set(profile_signals.keys())):
            raise ValueError(
                f"Canonical behavior format detected! Cannot save canonical behaviors "
                f"as profile signals. This indicates a programming error in the caller."
            )
        
        extracted_at = int(time.time())
        signals_json = json.dumps(profile_signals)
        
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    if behavior_id:
                        # When behavior_id is provided, insert a new row (no UPSERT)
                        # This allows multiple behaviors from the same prompt to have separate records
                        cur.execute(
                            """
                            INSERT INTO user_profile_signals
                                (user_id, prompt_id, profile_signals, extracted_at, behavior_id)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (user_id, prompt_id) DO UPDATE
                                SET profile_signals = EXCLUDED.profile_signals,
                                    extracted_at    = EXCLUDED.extracted_at,
                                    behavior_id     = EXCLUDED.behavior_id
                            """,
                            (user_id, prompt_id, signals_json, extracted_at, behavior_id)
                        )
                    else:
                        # Legacy path: no behavior_id (for backward compatibility)
                        cur.execute(
                            """
                            INSERT INTO user_profile_signals
                                (user_id, prompt_id, profile_signals, extracted_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (user_id, prompt_id) DO UPDATE
                                SET profile_signals = EXCLUDED.profile_signals,
                                    extracted_at    = EXCLUDED.extracted_at
                            """,
                            (user_id, prompt_id, signals_json, extracted_at)
                        )
                conn.commit()
            
            logger.info(
                f"Saved profile_signals for user={user_id}, prompt={prompt_id}"
                + (f", behavior={behavior_id}" if behavior_id else "")
            )
        except Exception as e:
            logger.error(
                f"Failed to save profile_signals for user={user_id}: {e}"
            )
            raise
    
    def get_recent(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent profile signals for a user.
        
        Used by the Profile Service during drift fallback to get
        historical behavior patterns for profile re-matching.
        
        Args:
            user_id: Unique user identifier
            limit: Maximum number of recent signals to return
            
        Returns:
            List of profile_signals dicts, ordered by most recent first
        """
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT profile_signals 
                        FROM user_profile_signals
                        WHERE user_id = %s
                        ORDER BY extracted_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                    rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Handle both string JSON and already-parsed dict
                signals = row[0]
                if isinstance(signals, str):
                    signals = json.loads(signals)
                results.append(signals)
            
            logger.debug(
                f"Retrieved {len(results)} recent profile_signals for user={user_id}"
            )
            return results
            
        except Exception as e:
            logger.error(
                f"Failed to get recent profile_signals for user={user_id}: {e}"
            )
            raise
    
    def get_count(self, user_id: str) -> int:
        """
        Get the total count of stored profile signals for a user.
        
        Useful for determining if enough signals have been collected
        for profile assignment or drift detection.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Total count of profile signals for the user
        """
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) 
                        FROM user_profile_signals 
                        WHERE user_id = %s
                        """,
                        (user_id,)
                    )
                    result = cur.fetchone()
            
            count = result[0] if result else 0
            logger.debug(f"Profile signal count for user={user_id}: {count}")
            return count
            
        except Exception as e:
            logger.error(
                f"Failed to get profile_signals count for user={user_id}: {e}"
            )
            raise
    
    def get_by_prompt_id(
        self, 
        user_id: str, 
        prompt_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve profile signals for a specific prompt.
        
        Args:
            user_id: Unique user identifier
            prompt_id: Unique prompt identifier
            
        Returns:
            Profile signals dict if found, None otherwise
        """
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT profile_signals 
                        FROM user_profile_signals
                        WHERE user_id = %s AND prompt_id = %s
                        """,
                        (user_id, prompt_id)
                    )
                    row = cur.fetchone()
            
            if row:
                signals = row[0]
                if isinstance(signals, str):
                    signals = json.loads(signals)
                return signals
            return None
            
        except Exception as e:
            logger.error(
                f"Failed to get profile_signals for prompt={prompt_id}: {e}"
            )
            raise
    
    def delete_old_signals(
        self, 
        user_id: str, 
        keep_count: int = 100
    ) -> int:
        """
        Delete old profile signals, keeping only the most recent ones.
        
        Useful for data retention and storage management.
        
        Args:
            user_id: Unique user identifier
            keep_count: Number of most recent signals to keep
            
        Returns:
            Number of deleted records
        """
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    # Delete signals older than the Nth most recent
                    cur.execute(
                        """
                        DELETE FROM user_profile_signals
                        WHERE user_id = %s
                        AND id NOT IN (
                            SELECT id 
                            FROM user_profile_signals
                            WHERE user_id = %s
                            ORDER BY extracted_at DESC
                            LIMIT %s
                        )
                        """,
                        (user_id, user_id, keep_count)
                    )
                    deleted_count = cur.rowcount
                conn.commit()
            
            logger.info(
                f"Deleted {deleted_count} old profile_signals for user={user_id}"
            )
            return deleted_count
            
        except Exception as e:
            logger.error(
                f"Failed to delete old profile_signals for user={user_id}: {e}"
            )
            raise
    
    def get_by_behavior_ids(
        self,
        user_id: str,
        behavior_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve profile signals for specific behavior IDs.
        
        This method is used by the /api/behaviors/by-ids endpoint to return
        profile signals associated with specific behaviors.
        
        Args:
            user_id: Unique user identifier
            behavior_ids: List of behavior IDs to retrieve signals for
            
        Returns:
            List of profile_signals dicts with behavior_id included
        """
        if not behavior_ids:
            return []
        
        try:
            with get_db_pool_connection() as conn:
                with conn.cursor() as cur:
                    # Use ANY for PostgreSQL array comparison
                    cur.execute(
                        """
                        SELECT profile_signals, behavior_id, extracted_at
                        FROM user_profile_signals
                        WHERE user_id = %s AND behavior_id = ANY(%s)
                        ORDER BY extracted_at DESC
                        """,
                        (user_id, behavior_ids)
                    )
                    rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Handle both string JSON and already-parsed dict
                signals = row[0]
                if isinstance(signals, str):
                    signals = json.loads(signals)
                
                # Add behavior_id to the result
                result = signals.copy()
                result['behavior_id'] = row[1]
                result['extracted_at'] = row[2]
                results.append(result)
            
            logger.info(
                f"Retrieved {len(results)} profile_signals for user={user_id} "
                f"with {len(behavior_ids)} behavior_ids"
            )
            return results
            
        except Exception as e:
            logger.error(
                f"Failed to get profile_signals by behavior_ids for user={user_id}: {e}"
            )
            raise


# Singleton instance for convenience
_repository_instance: Optional[ProfileSignalRepository] = None


def get_profile_signal_repository() -> ProfileSignalRepository:
    """Get or create the singleton ProfileSignalRepository instance."""
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = ProfileSignalRepository()
    return _repository_instance
