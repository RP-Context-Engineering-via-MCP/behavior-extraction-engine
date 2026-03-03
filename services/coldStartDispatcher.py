"""
Cold Start Dispatcher

Orchestrates the cold-start profiling flow after each extraction:
1. Always saves profile_signals (needed for drift fallback regardless of mode)
2. Checks if user is still in COLD_START mode via Profile Service
3. If COLD_START, sends profile_signals to Profile Service for assignment
"""

import logging
from typing import Dict, Any, Optional

from services.profileServiceClient import (
    ProfileServiceClient, 
    get_profile_service_client
)
from services.profileSignalRepository import (
    ProfileSignalRepository,
    get_profile_signal_repository
)

logger = logging.getLogger(__name__)


class ColdStartDispatcher:
    """
    Dispatcher that handles cold-start profile assignment flow.
    
    This component runs after every extraction to:
    1. Persist profile_signals locally (for drift fallback)
    2. Forward signals to Profile Service if user is in COLD_START mode
    
    The Profile Service is the single source of truth for user mode.
    We check it each time rather than caching, as mode can change
    after any prompt (when profile gets assigned).
    """
    
    def __init__(
        self,
        profile_client: Optional[ProfileServiceClient] = None,
        signal_repo: Optional[ProfileSignalRepository] = None
    ):
        """
        Initialize the dispatcher with its dependencies.
        
        Args:
            profile_client: HTTP client for Profile Service (uses singleton if None)
            signal_repo: Repository for profile signals (uses singleton if None)
        """
        self._profile_client = profile_client or get_profile_service_client()
        self._signal_repo = signal_repo or get_profile_signal_repository()
    
    async def dispatch(
        self, 
        user_id: str, 
        prompt_id: str, 
        profile_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process profile signals after extraction.
        
        This is the main entry point called by the extraction pipeline.
        It always saves signals locally, then conditionally forwards to
        Profile Service based on user's current mode.
        
        Args:
            user_id: Unique user identifier
            prompt_id: Unique prompt/request identifier
            profile_signals: Validated profile signals from extraction
            
        Returns:
            Profile Service response if called, None otherwise
        """
        # 1. Always save locally - needed for drift fallback regardless of mode
        try:
            self._signal_repo.save(user_id, prompt_id, profile_signals)
            logger.debug(
                f"Saved profile_signals for user={user_id}, prompt={prompt_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save profile_signals for user={user_id}: {e}"
            )
            # Continue even if save fails - try to call Profile Service
        
        # 2. Check if user still needs cold-start profiling
        is_cold_start = await self._is_cold_start_user(user_id)
        
        if not is_cold_start:
            logger.debug(
                f"User {user_id} not in COLD_START mode, skipping Profile Service call"
            )
            return None
        
        # 3. Call Profile Service for assignment
        result = await self._profile_client.assign_profile(user_id, profile_signals)
        
        if result:
            status = result.get("status")
            profile_id = result.get("assigned_profile_id")
            
            logger.info(
                f"Cold-start dispatch for user={user_id}: "
                f"status={status}, profile={profile_id}"
            )
            
            # Log when profile is actually assigned
            if status == "ASSIGNED" and profile_id:
                logger.info(
                    f"Profile assigned for user={user_id}: {profile_id} "
                    f"(confidence={result.get('confidence', 'N/A')})"
                )
        else:
            logger.warning(
                f"Cold-start dispatch failed for user={user_id} - "
                f"Profile Service unavailable or returned error"
            )
        
        return result
    
    async def _is_cold_start_user(self, user_id: str) -> bool:
        """
        Check if user is currently in COLD_START mode.
        
        Queries the Profile Service (single source of truth) to determine
        if the user still needs cold-start profiling. Returns True for:
        - New users (404 from Profile Service)
        - Users with user_mode == "COLD_START" and no assigned_profile_id
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if user needs cold-start profiling, False otherwise
        """
        try:
            status = await self._profile_client.get_user_profile_status(user_id)
            
            # New user - not yet in Profile Service
            if status is None:
                logger.debug(f"User {user_id} is new, needs cold-start profiling")
                return True
            
            # Already has an assigned profile - cold start is complete
            if status.get("assigned_profile_id"):
                logger.debug(
                    f"User {user_id} already has profile "
                    f"{status.get('assigned_profile_id')}, cold start complete"
                )
                return False
            
            # Check explicit user_mode
            user_mode = status.get("user_mode")
            is_cold_start = user_mode == "COLD_START"
            
            logger.debug(
                f"User {user_id} mode={user_mode}, is_cold_start={is_cold_start}"
            )
            return is_cold_start
            
        except Exception as e:
            logger.error(
                f"Could not check cold-start mode for user={user_id}: {e}"
            )
            # Fail safe: don't call Profile Service if we can't verify mode
            # This prevents duplicate calls for already-profiled users
            return False
    
    async def dispatch_sync_wrapper(
        self,
        user_id: str,
        prompt_id: str,
        profile_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous-compatible wrapper for dispatch.
        
        For use in synchronous contexts where the extraction pipeline
        needs to call dispatch without async/await.
        
        Note: This still runs async internally - it's just a convenience
        wrapper for contexts that can't directly await.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                return await self.dispatch(user_id, prompt_id, profile_signals)
            else:
                # Run in new event loop
                return loop.run_until_complete(
                    self.dispatch(user_id, prompt_id, profile_signals)
                )
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(
                self.dispatch(user_id, prompt_id, profile_signals)
            )


# Singleton instance for convenience
_dispatcher_instance: Optional[ColdStartDispatcher] = None


def get_cold_start_dispatcher() -> ColdStartDispatcher:
    """Get or create the singleton ColdStartDispatcher instance."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = ColdStartDispatcher()
    return _dispatcher_instance
