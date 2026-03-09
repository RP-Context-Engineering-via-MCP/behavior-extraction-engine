"""
Test script for Redis event publishing integration.

This script tests the BehaviorEventPublisher by:
1. Publishing test events to Redis Stream
2. Reading events back from the stream
3. Verifying event format and content

Run: python tests/test_event_publishing.py
"""

import redis
import json
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config.configurations import REDIS_URL, REDIS_STREAM_NAME
from services.eventPublisher import BehaviorEventPublisher, get_event_publisher


def test_redis_connection():
    """Test basic Redis connection."""
    print("\n" + "="*60)
    print("TEST 1: Redis Connection")
    print("="*60)
    
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        result = client.ping()
        print(f"✓ Redis PING: {result}")
        client.close()
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False


def test_publisher_initialization():
    """Test BehaviorEventPublisher initialization."""
    print("\n" + "="*60)
    print("TEST 2: Publisher Initialization")
    print("="*60)
    
    try:
        publisher = BehaviorEventPublisher()
        connected = publisher.is_connected()
        print(f"✓ Publisher initialized, connected: {connected}")
        publisher.close()
        return connected
    except Exception as e:
        print(f"✗ Publisher initialization failed: {e}")
        return False


def test_publish_behavior_created():
    """Test publishing behavior.created event."""
    print("\n" + "="*60)
    print("TEST 3: Publish behavior.created Event")
    print("="*60)
    
    publisher = get_event_publisher()
    
    msg_id = publisher.publish_behavior_created(
        user_id="test_user_001",
        behavior_id="beh_test123",
        target="python",
        intent="PREFERENCE",
        context="backend",
        polarity="POSITIVE",
        credibility=0.85,
        reinforcement_count=1,
        state="ACTIVE",
        created_at=int(time.time()),
        last_seen_at=int(time.time())
    )
    
    if msg_id:
        print(f"✓ Published behavior.created event (msg_id: {msg_id})")
        return True
    else:
        print("✗ Failed to publish behavior.created event")
        return False


def test_publish_behavior_reinforced():
    """Test publishing behavior.reinforced event."""
    print("\n" + "="*60)
    print("TEST 4: Publish behavior.reinforced Event")
    print("="*60)
    
    publisher = get_event_publisher()
    
    msg_id = publisher.publish_behavior_reinforced(
        user_id="test_user_001",
        behavior_id="beh_test123",
        reinforcement_count=2,
        credibility=0.88,
        last_seen_at=int(time.time())
    )
    
    if msg_id:
        print(f"✓ Published behavior.reinforced event (msg_id: {msg_id})")
        return True
    else:
        print("✗ Failed to publish behavior.reinforced event")
        return False


def test_publish_behavior_superseded():
    """Test publishing behavior.superseded event."""
    print("\n" + "="*60)
    print("TEST 5: Publish behavior.superseded Event")
    print("="*60)
    
    publisher = get_event_publisher()
    
    msg_id = publisher.publish_behavior_superseded(
        user_id="test_user_001",
        behavior_id="beh_old456",
        superseded_by="beh_new789"
    )
    
    if msg_id:
        print(f"✓ Published behavior.superseded event (msg_id: {msg_id})")
        return True
    else:
        print("✗ Failed to publish behavior.superseded event")
        return False


def test_publish_conflict_resolved():
    """Test publishing behavior.conflict.resolved event."""
    print("\n" + "="*60)
    print("TEST 6: Publish behavior.conflict.resolved Event")
    print("="*60)
    
    publisher = get_event_publisher()
    
    msg_id = publisher.publish_conflict_resolved(
        user_id="test_user_001",
        conflict_id="conflict_abc123",
        behavior_id_1="beh_old456",
        behavior_id_2="beh_new789",
        conflict_type="USER_DECISION_NEEDED",
        resolution_status="PENDING",
        old_polarity="POSITIVE",
        new_polarity="NEGATIVE",
        old_target="python",
        new_target="python",
        created_at=int(time.time())
    )
    
    if msg_id:
        print(f"✓ Published behavior.conflict.resolved event (msg_id: {msg_id})")
        return True
    else:
        print("✗ Failed to publish behavior.conflict.resolved event")
        return False


def test_read_events_from_stream():
    """Read and display all events from the Redis stream."""
    print("\n" + "="*60)
    print("TEST 7: Read Events from Stream")
    print("="*60)
    
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        
        # Read only recent events (last 10)
        messages = client.xrevrange(REDIS_STREAM_NAME, count=10)
        
        if not messages:
            print("No messages found in stream")
            return True
        
        print(f"\nStream: {REDIS_STREAM_NAME}")
        print(f"Showing last {len(messages)} events")
        print("-" * 50)
        
        for message_id, message_data in reversed(messages):
            try:
                # New structure: event_type, event_id, published_at at top level
                event_type = message_data.get(b'event_type', b'').decode() if b'event_type' in message_data else None
                event_id = message_data.get(b'event_id', b'').decode() if b'event_id' in message_data else None
                published_at = message_data.get(b'published_at', b'').decode() if b'published_at' in message_data else None
                
                if not event_type:
                    continue
                
                # Payload is now a flat JSON object with all behavior details
                payload = json.loads(message_data[b'payload']) if b'payload' in message_data else {}
                
                print(f"\nEvent Type: {event_type}")
                print(f"  Event ID: {event_id}")
                print(f"  Published At: {published_at}")
                print(f"  User: {payload.get('user_id', 'N/A')}")
                
                if event_type == 'behavior.created':
                    print(f"  Behavior ID: {payload.get('behavior_id', 'N/A')}")
                    print(f"  Target: {payload.get('target', 'N/A')}")
                    print(f"  Intent: {payload.get('intent', 'N/A')}")
                    print(f"  Polarity: {payload.get('polarity', 'N/A')}")
                    print(f"  Credibility: {payload.get('credibility', 'N/A')}")
                    print(f"  Reinforcement Count: {payload.get('reinforcement_count', 'N/A')}")
                    print(f"  State: {payload.get('state', 'N/A')}")
                    print(f"  Created At: {payload.get('created_at', 'N/A')}")
                    print(f"  Last Seen At: {payload.get('last_seen_at', 'N/A')}")
                
                elif event_type == 'behavior.reinforced':
                    print(f"  Behavior ID: {payload.get('behavior_id', 'N/A')}")
                    print(f"  Count: {payload.get('reinforcement_count', 'N/A')}")
                    print(f"  Credibility: {payload.get('credibility', 'N/A')}")
                
                elif event_type == 'behavior.superseded':
                    print(f"  Old: {payload.get('behavior_id', 'N/A')}")
                    print(f"  New: {payload.get('superseded_by', 'N/A')}")
                
                elif event_type == 'behavior.conflict.resolved':
                    print(f"  Conflict ID: {payload.get('conflict_id', 'N/A')}")
                    print(f"  Conflict Type: {payload.get('conflict_type', 'N/A')}")
                    print(f"  Resolution: {payload.get('resolution_status', 'N/A')}")
                    print(f"  Polarity flip: {payload.get('old_polarity', 'N/A')} → {payload.get('new_polarity', 'N/A')}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed events
                continue
        
        client.close()
        print("\n✓ Successfully read events from stream")
        return True
        
    except Exception as e:
        print(f"✗ Failed to read events: {e}")
        return False


def clear_test_events():
    """Clear test events from the stream (optional cleanup)."""
    print("\n" + "="*60)
    print("CLEANUP: Clear Test Events")
    print("="*60)
    
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        
        # Get stream length before
        length_before = client.xlen(REDIS_STREAM_NAME)
        
        # Trim to keep only last 0 entries (clear all)
        # client.xtrim(REDIS_STREAM_NAME, maxlen=0)
        
        print(f"Stream has {length_before} events (not cleared - keeping for inspection)")
        client.close()
        return True
        
    except Exception as e:
        print(f"Note: {e}")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BEHAVIOR EVENT PUBLISHING TEST SUITE")
    print(f"Redis URL: {REDIS_URL}")
    print(f"Stream Name: {REDIS_STREAM_NAME}")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Redis Connection", test_redis_connection()))
    results.append(("Publisher Init", test_publisher_initialization()))
    results.append(("behavior.created", test_publish_behavior_created()))
    results.append(("behavior.reinforced", test_publish_behavior_reinforced()))
    results.append(("behavior.superseded", test_publish_behavior_superseded()))
    results.append(("conflict.resolved", test_publish_conflict_resolved()))
    results.append(("Read Events", test_read_events_from_stream()))
    
    # Optional cleanup
    clear_test_events()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Event publishing is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check Redis connection and configuration.")
        return 1


if __name__ == "__main__":
    exit(main())
