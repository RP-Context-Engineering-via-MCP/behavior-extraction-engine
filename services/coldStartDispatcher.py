"""
Cold Start Dispatcher

Dispatcher that saves profile signals locally and publishes events for users in COLD_START mode.

This dispatcher:
1. Saves profile_signals locally for drift fallback
2. Checks user's profile_mode from User Management Service
3. If profile_mode is COLD_START, publishes profile_signals event to Redis
4. Maintains backward compatibility with existing extraction pipeline
"""

import logging
from typing import Dict, Any, Optional

from services.profileSignalRepository import (
    ProfileSignalRepository,
    get_profile_signal_repository
)
from services.profileServiceClient import get_user_management_client
from services.eventPublisher import get_event_publisher

logger = logging.getLogger(__name__)


class ColdStartDispatcher:
    """
    Simplified dispatcher that persists profile signals locally.
    
    This component runs after every extraction to persist profile_signals
    for drift detection and fallback scenarios. Profile assignment logic
    has been decoupled from this service.
    """
    
    def __init__(
        self,
        signal_repo: Optional[ProfileSignalRepository] = None
    ):
        """
        Initialize the dispatcher with signal repository.
        
        Args:
            signal_repo: Repository for profile signals (uses singleton if None)
        """
        self._signal_repo = signal_repo or get_profile_signal_repository()
    
    async def dispatch(
        self, 
        user_id: str, 
        prompt_id: str, 
        profile_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Save profile signals locally and publish event if user is in COLD_START mode.
        
        This is the main entry point called by the extraction pipeline.
        
        Flow:
        1. Save profile_signals locally for drift detection
        2. Check user's profile_mode from User Management Service
        3. If profile_mode == "COLD_START", publish profile_signals event
        
        Args:
            user_id: Unique user identifier
            prompt_id: Unique prompt/request identifier
            profile_signals: Validated profile signals from extraction
            
        Returns:
            None (profile assignment removed)
        """
        # 1. Save profile signals locally for drift detection
        try:
            self._signal_repo.save(user_id, prompt_id, profile_signals)
            logger.debug(
                f"Saved profile_signals for user={user_id}, prompt={prompt_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save profile_signals for user={user_id}: {e}"
            )
        
        # 2. Check if user is in COLD_START mode
        try:
            user_client = get_user_management_client()
            user_data = await user_client.get_user(user_id)
            
            if user_data and user_data.get("profile_mode") == "COLD_START":
                logger.info(f"User {user_id} is in COLD_START mode, publishing profile_signals event")
                
                # 3. Publish profile_signals event to Redis
                event_publisher = get_event_publisher()
                message_id = event_publisher.publish_profile_signals(
                    user_id=user_id,
                    prompt_id=prompt_id,
                    profile_signals=profile_signals
                )
                
                if message_id:
                    logger.info(
                        f"Published profile_signals event for user={user_id} "
                        f"(message_id: {message_id})"
                    )
                else:
                    logger.warning(
                        f"Failed to publish profile_signals event for user={user_id}"
                    )
            else:
                profile_mode = user_data.get("profile_mode", "UNKNOWN") if user_data else "UNKNOWN"
                logger.debug(
                    f"User {user_id} profile_mode={profile_mode}, skipping event publishing"
                )
                
        except Exception as e:
            logger.warning(
                f"Failed to check user profile_mode or publish event for user={user_id}: {e}"
            )
        
        return None
    
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
