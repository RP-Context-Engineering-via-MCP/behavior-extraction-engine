"""
Profile Service Client

HTTP client that communicates with the Profile Service for:
1. Cold-start profile assignment (POST /api/predefined-profiles/assign-profile)
2. Checking user profile status (GET /api/predefined-profiles/user/{user_id})
"""

import httpx
import logging
from typing import Dict, Any, Optional

from config.configurations import PROFILE_SERVICE_BASE_URL

logger = logging.getLogger(__name__)


class ProfileServiceClient:
    """
    HTTP client for Profile Service API communication.
    
    Handles cold-start profile assignment and user status checks.
    All methods are designed to be fault-tolerant - they return None
    on failure rather than raising exceptions, allowing the main
    extraction pipeline to continue even if Profile Service is unavailable.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        """
        Initialize the Profile Service client.
        
        Args:
            base_url: Profile Service base URL (defaults to config value)
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url or PROFILE_SERVICE_BASE_URL
        self.timeout = timeout
        
    async def assign_profile(
        self, 
        user_id: str, 
        profile_signals: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Request profile assignment for a cold-start user.
        
        Calls POST /api/predefined-profiles/assign-profile with the
        extracted profile signals. The Profile Service will either:
        - Return PENDING if more prompts needed for confident assignment
        - Return ASSIGNED with assigned_profile_id if threshold reached
        
        Args:
            user_id: Unique user identifier
            profile_signals: Validated profile signals from extraction
            
        Returns:
            Response dict with status and optional profile assignment,
            or None if request failed
            
        Example response:
            {
                "status": "PENDING",
                "prompts_collected": 3,
                "prompts_required": 5
            }
            or
            {
                "status": "ASSIGNED",
                "assigned_profile_id": "tech_enthusiast_v1",
                "confidence": 0.87
            }
        """
        url = f"{self.base_url}/api/predefined-profiles/assign-profile"
        payload = {
            "user_id": user_id,
            "mode": "COLD_START",
            "extracted_behavior": profile_signals
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                logger.info(
                    f"Profile assignment response for user={user_id}: "
                    f"status={result.get('status')}"
                )
                return result
                
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Profile Service HTTP error for user={user_id}: "
                f"{e.response.status_code} - {e.response.text}"
            )
            return None
            
        except httpx.TimeoutException:
            logger.warning(
                f"Profile Service timeout for user={user_id} after {self.timeout}s"
            )
            return None
            
        except httpx.RequestError as e:
            logger.warning(
                f"Profile Service request error for user={user_id}: {e}"
            )
            return None
            
        except Exception as e:
            logger.error(
                f"Unexpected error calling Profile Service for user={user_id}: {e}"
            )
            return None
    
    async def get_user_profile_status(
        self, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current profile status for a user.
        
        Calls GET /api/predefined-profiles/user/{user_id} to check:
        - Whether user exists in Profile Service
        - Current user_mode (COLD_START, ACTIVE, etc.)
        - Whether a profile has been assigned
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User status dict, or None if user not found or request failed
            
        Example response:
            {
                "user_id": "user_123",
                "user_mode": "COLD_START",
                "assigned_profile_id": null,
                "prompts_collected": 2
            }
        """
        url = f"{self.base_url}/api/predefined-profiles/user/{user_id}"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                
                # 404 means user doesn't exist yet - this is expected for new users
                if response.status_code == 404:
                    logger.debug(f"User {user_id} not found in Profile Service")
                    return None
                    
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Profile Service HTTP error checking user={user_id}: "
                f"{e.response.status_code}"
            )
            return None
            
        except httpx.TimeoutException:
            logger.warning(
                f"Profile Service timeout checking user={user_id}"
            )
            return None
            
        except httpx.RequestError as e:
            logger.warning(
                f"Profile Service request error checking user={user_id}: {e}"
            )
            return None
            
        except Exception as e:
            logger.error(
                f"Unexpected error checking Profile Service for user={user_id}: {e}"
            )
            return None
    
    async def health_check(self) -> bool:
        """
        Check if Profile Service is reachable and healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        url = f"{self.base_url}/health"
        
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False


# Singleton instance for convenience
_client_instance: Optional[ProfileServiceClient] = None


def get_profile_service_client() -> ProfileServiceClient:
    """Get or create the singleton ProfileServiceClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ProfileServiceClient()
    return _client_instance
