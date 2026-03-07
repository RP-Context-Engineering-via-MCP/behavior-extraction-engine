"""
User Management Service Client

HTTP client that communicates with the User Management Service for:
1. Fetching user information (GET /api/users/{user_id})
2. Updating user profiles (PUT /api/users/{user_id})
3. Managing fallback profiles (POST /api/users/{user_id}/fallback/activate|deactivate)

Note: Profile assignment features have been decoupled from this service.
"""

import httpx
import logging
from typing import Dict, Any, Optional

from config.configurations import USER_MANAGEMENT_SERVICE_BASE_URL

logger = logging.getLogger(__name__)


class UserManagementServiceClient:
    """
    HTTP client for User Management Service API communication.
    
    Handles user data retrieval and profile management.
    All methods are designed to be fault-tolerant - they return None
    on failure rather than raising exceptions, allowing the main
    extraction pipeline to continue even if the service is unavailable.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        """
        Initialize the User Management Service client.
        
        Args:
            base_url: User Management Service base URL (defaults to config value)
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url or USER_MANAGEMENT_SERVICE_BASE_URL
        self.timeout = timeout
        
    async def get_user(
        self, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch user information from User Management Service.
        
        Calls GET /api/users/{user_id} to retrieve user profile data.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User data dict, or None if user not found or request failed
            
        Example response:
            {
                "user_id": "user_123",
                "username": "john_doe",
                "email": "john@example.com",
                "profile_data": {...}
            }
        """
        url = f"{self.base_url}/api/users/{user_id}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                
                # 404 means user doesn't exist
                if response.status_code == 404:
                    logger.debug(f"User {user_id} not found in User Management Service")
                    return None
                    
                response.raise_for_status()
                result = response.json()
                
                logger.debug(f"Retrieved user data for user={user_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"User Management Service HTTP error for user={user_id}: "
                f"{e.response.status_code} - {e.response.text}"
            )
            return None
            
        except httpx.TimeoutException:
            logger.warning(
                f"User Management Service timeout for user={user_id} after {self.timeout}s"
            )
            return None
            
        except httpx.RequestError as e:
            logger.warning(
                f"User Management Service request error for user={user_id}: {e}"
            )
            return None
            
        except Exception as e:
            logger.error(
                f"Unexpected error calling User Management Service for user={user_id}: {e}"
            )
            return None
    
    async def activate_fallback_profile(
        self,
        user_id: str
    ) -> bool:
        """
        Activate fallback profile for a user.
        
        Calls POST /api/users/{user_id}/fallback/activate
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/api/users/{user_id}/fallback/activate"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url)
                response.raise_for_status()
                logger.info(f"Activated fallback profile for user={user_id}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to activate fallback profile for user={user_id}: {e}")
            return False
    
    async def deactivate_fallback_profile(
        self,
        user_id: str
    ) -> bool:
        """
        Deactivate fallback profile for a user.
        
        Calls POST /api/users/{user_id}/fallback/deactivate
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/api/users/{user_id}/fallback/deactivate"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url)
                response.raise_for_status()
                logger.info(f"Deactivated fallback profile for user={user_id}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to deactivate fallback profile for user={user_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if User Management Service is reachable and healthy.
        
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
_client_instance: Optional[UserManagementServiceClient] = None


def get_user_management_client() -> UserManagementServiceClient:
    """Get or create the singleton UserManagementServiceClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = UserManagementServiceClient()
    return _client_instance


# Backward compatibility alias
def get_profile_service_client() -> UserManagementServiceClient:
    """Deprecated: Use get_user_management_client() instead."""
    return get_user_management_client()
