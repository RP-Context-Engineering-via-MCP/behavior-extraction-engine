"""
Event Publisher for Drift Detection Service Integration

Publishes behavior events to Redis Streams when behaviors change.
These events are consumed by the Drift Detection Service to maintain
real-time snapshots of user behaviors and detect preference drift.

Events published:
- behavior.created: New behavior inserted
- behavior.reinforced: Existing behavior reinforced (duplicate detected)
- behavior.superseded: Behavior replaced by newer conflicting behavior
- behavior.conflict.resolved: Conflict detected and resolved
"""

import redis
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config.configurations import REDIS_URL, REDIS_STREAM_NAME, REDIS_EVENTS_ENABLED

logger = logging.getLogger(__name__)


class BehaviorEventPublisher:
    """
    Publishes behavior change events to Redis Streams.
    
    Events are consumed by the Drift Detection Service to maintain
    real-time snapshots of user behaviors.
    
    Usage:
        publisher = BehaviorEventPublisher()
        publisher.publish_behavior_created(user_id="user123", ...)
        
    The publisher gracefully handles connection failures - if Redis is
    unavailable, events are logged but the main service continues working.
    """
    
    def __init__(
        self, 
        redis_url: str = REDIS_URL, 
        stream_name: str = REDIS_STREAM_NAME,
        enabled: bool = REDIS_EVENTS_ENABLED
    ):
        """
        Initialize the event publisher.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            stream_name: Name of the Redis Stream (default: "behavior.events")
            enabled: Whether event publishing is enabled
        """
        self.stream_name = stream_name
        self.enabled = enabled
        self.client: Optional[redis.Redis] = None
        
        if not self.enabled:
            logger.info("Event publishing disabled via REDIS_EVENTS_ENABLED=false")
            return
            
        try:
            self.client = redis.Redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.client.ping()
            logger.info(f"✓ Connected to Redis Stream: {stream_name}")
        except Exception as e:
            logger.warning(f"✗ Redis connection failed: {e}. Events will not be published.")
            self.client = None
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"evt_{uuid.uuid4().hex}"
    
    def _get_timestamp(self) -> int:
        """Get current Unix timestamp."""
        return int(datetime.now(timezone.utc).timestamp())
    
    def _publish_event(self, event_type: str, event_id: str, published_at: int, payload: Dict[str, Any]) -> Optional[str]:
        """
        Publish event to Redis Stream.
        
        Args:
            event_type: Type of event (e.g., "behavior.created")
            event_id: Unique event identifier
            published_at: Unix timestamp when event was published
            payload: Event payload dictionary (will be JSON serialized)
            
        Returns:
            Message ID if successful, None if failed or disabled
        """
        if not self.enabled or self.client is None:
            logger.debug(f"Event not published (disabled/no connection): {event_type}")
            return None
            
        try:
            # Structure: event_type, event_id, published_at, payload (JSON string)
            payload_json = json.dumps(payload)
            message_id = self.client.xadd(
                self.stream_name,
                {
                    "event_type": event_type,
                    "event_id": event_id,
                    "published_at": str(published_at),
                    "payload": payload_json
                },
                maxlen=10000  # Keep last 10k events (prevents unbounded growth)
            )
            logger.debug(
                f"Published {event_type} for user {payload.get('user_id', 'unknown')} "
                f"(msg_id: {message_id.decode() if message_id else 'unknown'})"
            )
            return message_id.decode() if message_id else None
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}", exc_info=True)
            return None
    
    def publish_behavior_created(
        self,
        user_id: str,
        behavior_id: str,
        target: str,
        intent: str,
        context: str,
        polarity: str,
        credibility: float,
        reinforcement_count: int,
        state: str,
        created_at: int,
        last_seen_at: int
    ) -> Optional[str]:
        """
        Publish behavior.created event.
        
        Called when a NEW behavior is inserted into the database.
        
        Args:
            user_id: User identifier
            behavior_id: Behavior identifier
            target: Target topic (e.g., "python")
            intent: Intent type (PREFERENCE, CONSTRAINT, etc.)
            context: Context (e.g., "backend", "general")
            polarity: POSITIVE or NEGATIVE
            credibility: Credibility score (0.0-1.0)
            reinforcement_count: Number of reinforcements
            state: Behavior state (ACTIVE, SUPERSEDED, etc.)
            created_at: Creation timestamp
            last_seen_at: Last seen timestamp
        
        Returns:
            Message ID if successful, None if failed
        """
        event_type = "behavior.created"
        event_id = self._generate_event_id()
        published_at = self._get_timestamp()
        payload = {
            "user_id": user_id,
            "behavior_id": behavior_id,
            "target": target,
            "intent": intent,
            "context": context,
            "polarity": polarity,
            "credibility": credibility,
            "reinforcement_count": reinforcement_count,
            "state": state,
            "created_at": created_at,
            "last_seen_at": last_seen_at
        }
        return self._publish_event(event_type, event_id, published_at, payload)
    
    def publish_behavior_reinforced(
        self,
        user_id: str,
        behavior_id: str,
        reinforcement_count: int,
        credibility: float,
        last_seen_at: int
    ) -> Optional[str]:
        """
        Publish behavior.reinforced event.
        
        Called when a duplicate behavior is detected and the existing
        behavior's reinforcement_count is incremented.
        
        Args:
            user_id: User identifier
            behavior_id: Behavior that was reinforced
            reinforcement_count: Updated reinforcement count
            credibility: Updated credibility score
            last_seen_at: Updated last seen timestamp
        
        Returns:
            Message ID if successful, None if failed
        """
        event_type = "behavior.reinforced"
        event_id = self._generate_event_id()
        published_at = self._get_timestamp()
        payload = {
            "user_id": user_id,
            "behavior_id": behavior_id,
            "reinforcement_count": reinforcement_count,
            "credibility": credibility,
            "last_seen_at": last_seen_at
        }
        return self._publish_event(event_type, event_id, published_at, payload)
    
    def publish_behavior_superseded(
        self,
        user_id: str,
        behavior_id: str,
        superseded_by: str
    ) -> Optional[str]:
        """
        Publish behavior.superseded event.
        
        Called when a behavior's state changes from ACTIVE to SUPERSEDED.
        
        Args:
            user_id: User identifier
            behavior_id: Behavior that was superseded
            superseded_by: ID of the new behavior that replaced it
        
        Returns:
            Message ID if successful, None if failed
        """
        event_type = "behavior.superseded"
        event_id = self._generate_event_id()
        published_at = self._get_timestamp()
        payload = {
            "user_id": user_id,
            "behavior_id": behavior_id,
            "superseded_by": superseded_by,
            "state": "SUPERSEDED"
        }
        return self._publish_event(event_type, event_id, published_at, payload)
    
    def publish_conflict_resolved(
        self,
        user_id: str,
        conflict_id: str,
        behavior_id_1: str,
        behavior_id_2: str,
        conflict_type: str,
        resolution_status: str,
        old_polarity: Optional[str] = None,
        new_polarity: Optional[str] = None,
        old_target: Optional[str] = None,
        new_target: Optional[str] = None,
        created_at: Optional[int] = None
    ) -> Optional[str]:
        """
        Publish behavior.conflict.resolved event.
        
        Called when a conflict is detected and resolved (auto or manual).
        
        Args:
            user_id: User identifier
            conflict_id: Conflict identifier
            behavior_id_1: Old behavior ID
            behavior_id_2: New behavior ID
            conflict_type: RESOLVABLE or USER_DECISION_NEEDED
            resolution_status: PENDING, AUTO_RESOLVED, USER_RESOLVED, etc.
            old_polarity: Old behavior polarity
            new_polarity: New behavior polarity
            old_target: Old behavior target
            new_target: New behavior target
            created_at: Conflict creation timestamp
        
        Returns:
            Message ID if successful, None if failed
        """
        event_type = "behavior.conflict.resolved"
        event_id = self._generate_event_id()
        published_at = self._get_timestamp()
        payload = {
            "user_id": user_id,
            "conflict_id": conflict_id,
            "behavior_id_1": behavior_id_1,
            "behavior_id_2": behavior_id_2,
            "conflict_type": conflict_type,
            "resolution_status": resolution_status,
            "old_polarity": old_polarity,
            "new_polarity": new_polarity,
            "old_target": old_target,
            "new_target": new_target,
            "created_at": created_at or self._get_timestamp()
        }
        return self._publish_event(event_type, event_id, published_at, payload)
    
    def publish_profile_signals(
        self,
        user_id: str,
        profile_signals: Dict[str, Any]
    ) -> Optional[str]:
        """
        Publish profile_signals.extracted event.
        
        Called when a user is in COLD_START mode and profile signals are extracted.
        This event is consumed by the Profile Service for cold-start profiling.
        
        Args:
            user_id: User identifier
            profile_signals: Extracted profile signals dictionary
        
        Returns:
            Message ID if successful, None if failed
        """
        event_type = "profile_signals.extracted"
        event_id = self._generate_event_id()
        published_at = self._get_timestamp()
        payload = {
            "user_id": user_id,
            "profile_signals": profile_signals
        }
        return self._publish_event(event_type, event_id, published_at, payload)
    
    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        if not self.enabled or self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("Closed BehaviorEventPublisher connection")
            except:
                pass


# Singleton instance for use across the application
# Initialized lazily to avoid connection attempts during import
_publisher_instance: Optional[BehaviorEventPublisher] = None


def get_event_publisher() -> BehaviorEventPublisher:
    """
    Get the singleton event publisher instance.
    
    Returns:
        BehaviorEventPublisher instance (may be disabled if Redis unavailable)
    """
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = BehaviorEventPublisher()
    return _publisher_instance
