"""
Test conflict resolution API and workflow.

This test validates that users can resolve flagged behavior conflicts
through the API and that the system correctly updates behavior states,
credibility, and access times based on the user's decision.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.behaviorRepository import (
    insert_behavior,
    insert_conflict,
    resolve_conflict,
    get_user_conflicts
)
from models.behavior import BehaviorState, ConflictType, ResolutionStatus
from db.connection import get_db_pool_connection
import time
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_USER_ID = "test_conflict_resolution_user"


def cleanup_test_data():
    """Clean up test data before and after tests."""
    try:
        with get_db_pool_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM behavior_conflicts WHERE user_id = %s", (TEST_USER_ID,))
                cur.execute("DELETE FROM behaviors WHERE user_id = %s", (TEST_USER_ID,))
            conn.commit()
        logger.info("Test data cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup test data: {e}")


def create_test_behavior(behavior_text: str, credibility: float = 0.75) -> str:
    """Helper to create a test behavior."""
    behavior_id = f"beh_test_{uuid.uuid4().hex[:8]}"
    current_time = int(time.time())
    
    payload = {
        "behavior_id": behavior_id,
        "user_id": TEST_USER_ID,
        "behavior_text": behavior_text,
        "embedding": [0.1] * 3072,  # Dummy embedding
        "credibility": credibility,
        "extraction_confidence": 0.85,
        "clarity_score": 0.80,
        "linguistic_strength": 0.75,
        "decay_rate": 0.01,
        "reinforcement_count": 1,
        "created_at": current_time,
        "last_seen_at": current_time,
        "last_decay_applied_at": current_time,
        "last_accessed_at": None,
        "session_id": "test_session",
        "prompt_history_ids": [],
        "behavior_state": BehaviorState.FLAGGED.value,
        "intent": "PREFERENCE",
        "target": "editor",
        "context": "general",
        "polarity": "POSITIVE"
    }
    
    insert_behavior(payload)
    logger.info(f"Created test behavior: {behavior_id}")
    return behavior_id


def test_resolve_conflict_old_wins():
    """Test resolution when user chooses OLD_WINS."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Resolve Conflict - OLD_WINS")
    logger.info("="*80)
    
    cleanup_test_data()
    
    # Create two conflicting behaviors
    behavior_id_1 = create_test_behavior("I prefer VSCode for Python development", credibility=0.72)
    behavior_id_2 = create_test_behavior("I prefer PyCharm for Python development", credibility=0.68)
    
    # Create conflict
    conflict_id = insert_conflict(
        user_id=TEST_USER_ID,
        behavior_id_1=behavior_id_1,
        behavior_id_2=behavior_id_2,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=0.15,
        llm_analysis="Both behaviors express preference for Python IDE but conflict on which one"
    )
    
    logger.info(f"Created conflict: {conflict_id}")
    logger.info(f"Behavior 1 (OLD): {behavior_id_1}")
    logger.info(f"Behavior 2 (NEW): {behavior_id_2}")
    
    # Resolve with OLD_WINS
    result = resolve_conflict(
        conflict_id=conflict_id,
        user_id=TEST_USER_ID,
        resolution_choice="OLD_WINS"
    )
    
    logger.info(f"Resolution result: {result}")
    
    # Verify results
    assert result["success"] == True
    assert result["resolution_choice"] == "OLD_WINS"
    assert result["resolution_status"] == ResolutionStatus.USER_RESOLVED.value
    
    # Check behaviors
    behaviors = result["behaviors"]
    assert len(behaviors) == 2
    
    old_behavior = next(b for b in behaviors if b["behavior_id"] == behavior_id_1)
    new_behavior = next(b for b in behaviors if b["behavior_id"] == behavior_id_2)
    
    # Old behavior should be ACTIVE and reinforced
    assert old_behavior["behavior_state"] == BehaviorState.ACTIVE.value
    assert old_behavior["last_accessed_at"] is not None
    assert old_behavior["credibility"] > 0.72  # Reinforced from 0.72
    
    # New behavior should be invalidated (credibility = 0.0 for pruning)
    assert new_behavior["credibility"] == 0.0
    assert new_behavior["last_accessed_at"] is not None
    
    logger.info(
        f"Old behavior {old_behavior['behavior_id']}: "
        f"state={old_behavior['behavior_state']}, "
        f"credibility={old_behavior['credibility']:.3f} (reinforced)"
    )
    logger.info(
        f"New behavior {new_behavior['behavior_id']}: "
        f"credibility={new_behavior['credibility']:.3f} (invalidated for pruning)"
    )
    
    logger.info("✅ OLD_WINS test passed!")
    cleanup_test_data()


def test_resolve_conflict_new_wins():
    """Test resolution when user chooses NEW_WINS."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Resolve Conflict - NEW_WINS")
    logger.info("="*80)
    
    cleanup_test_data()
    
    # Create two conflicting behaviors
    behavior_id_1 = create_test_behavior("I prefer light mode", credibility=0.65)
    behavior_id_2 = create_test_behavior("I prefer dark mode", credibility=0.78)
    
    # Create conflict
    conflict_id = insert_conflict(
        user_id=TEST_USER_ID,
        behavior_id_1=behavior_id_1,
        behavior_id_2=behavior_id_2,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=0.12,
        llm_analysis="Conflicting preferences on editor theme"
    )
    
    logger.info(f"Created conflict: {conflict_id}")
    logger.info(f"Behavior 1 (OLD): {behavior_id_1}")
    logger.info(f"Behavior 2 (NEW): {behavior_id_2}")
    
    # Resolve with NEW_WINS
    result = resolve_conflict(
        conflict_id=conflict_id,
        user_id=TEST_USER_ID,
        resolution_choice="NEW_WINS"
    )
    
    logger.info(f"Resolution result: {result}")
    
    # Verify results
    assert result["success"] == True
    assert result["resolution_choice"] == "NEW_WINS"
    
    # Check behaviors
    behaviors = result["behaviors"]
    
    old_behavior = next(b for b in behaviors if b["behavior_id"] == behavior_id_1)
    new_behavior = next(b for b in behaviors if b["behavior_id"] == behavior_id_2)
    
    assert new_behavior["behavior_state"] == BehaviorState.ACTIVE.value
    assert old_behavior["behavior_state"] == BehaviorState.SUPERSEDED.value
    
    logger.info(
        f"Old behavior {old_behavior['behavior_id']}: "
        f"state={old_behavior['behavior_state']}"
    )
    logger.info(
        f"New behavior {new_behavior['behavior_id']}: "
        f"state={new_behavior['behavior_state']}"
    )
    
    logger.info("✅ NEW_WINS test passed!")
    cleanup_test_data()


def test_resolve_conflict_both_correct():
    """Test resolution when user chooses BOTH_CORRECT."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Resolve Conflict - BOTH_CORRECT")
    logger.info("="*80)
    
    cleanup_test_data()
    
    # Create two conflicting behaviors
    behavior_id_1 = create_test_behavior("I prefer tabs for Python", credibility=0.70)
    behavior_id_2 = create_test_behavior("I prefer spaces for JavaScript", credibility=0.70)
    
    # Create conflict
    conflict_id = insert_conflict(
        user_id=TEST_USER_ID,
        behavior_id_1=behavior_id_1,
        behavior_id_2=behavior_id_2,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=0.18,
        llm_analysis="Different indentation preferences for different languages"
    )
    
    logger.info(f"Created conflict: {conflict_id}")
    logger.info(f"Behavior 1: {behavior_id_1}")
    logger.info(f"Behavior 2: {behavior_id_2}")
    
    # Resolve with BOTH_CORRECT
    result = resolve_conflict(
        conflict_id=conflict_id,
        user_id=TEST_USER_ID,
        resolution_choice="BOTH_CORRECT"
    )
    
    logger.info(f"Resolution result: {result}")
    
    # Verify results
    assert result["success"] == True
    assert result["resolution_choice"] == "BOTH_CORRECT"
    
    # Check behaviors - both should be ACTIVE and reinforced
    behaviors = result["behaviors"]
    assert len(behaviors) == 2
    
    for behavior in behaviors:
        assert behavior["behavior_state"] == BehaviorState.ACTIVE.value
        assert behavior["credibility"] > 0.70  # Should be reinforced
        assert behavior["last_accessed_at"] is not None
        logger.info(
            f"Behavior {behavior['behavior_id']}: "
            f"state={behavior['behavior_state']}, "
            f"credibility={behavior['credibility']:.3f}"
        )
    
    logger.info("✅ BOTH_CORRECT test passed!")
    cleanup_test_data()


def test_get_conflicts():
    """Test getting all conflicts for a user."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Get User Conflicts")
    logger.info("="*80)
    
    cleanup_test_data()
    
    # Create behaviors and conflicts
    behavior_id_1 = create_test_behavior("Test behavior 1")
    behavior_id_2 = create_test_behavior("Test behavior 2")
    
    conflict_id = insert_conflict(
        user_id=TEST_USER_ID,
        behavior_id_1=behavior_id_1,
        behavior_id_2=behavior_id_2,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=0.20
    )
    
    # Get conflicts
    conflicts = get_user_conflicts(TEST_USER_ID)
    
    logger.info(f"Found {len(conflicts)} conflict(s)")
    assert len(conflicts) > 0
    
    conflict = conflicts[0]
    assert conflict["conflict_id"] == conflict_id
    assert conflict["user_id"] == TEST_USER_ID
    assert conflict["resolution_status"] == ResolutionStatus.PENDING.value
    
    logger.info(f"Conflict details: {conflict}")
    logger.info("✅ Get conflicts test passed!")
    cleanup_test_data()


def test_invalid_resolution_choice():
    """Test that invalid resolution choices are rejected."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Invalid Resolution Choice")
    logger.info("="*80)
    
    cleanup_test_data()
    
    # Create behaviors and conflict
    behavior_id_1 = create_test_behavior("Test behavior 1")
    behavior_id_2 = create_test_behavior("Test behavior 2")
    
    conflict_id = insert_conflict(
        user_id=TEST_USER_ID,
        behavior_id_1=behavior_id_1,
        behavior_id_2=behavior_id_2,
        conflict_type=ConflictType.USER_DECISION_NEEDED,
        similarity_distance=0.20
    )
    
    # Try invalid resolution choice
    try:
        resolve_conflict(
            conflict_id=conflict_id,
            user_id=TEST_USER_ID,
            resolution_choice="INVALID_CHOICE"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"Correctly rejected invalid choice: {e}")
        assert "Invalid resolution_choice" in str(e)
    
    logger.info("✅ Invalid choice test passed!")
    cleanup_test_data()


if __name__ == "__main__":
    try:
        logger.info("Starting conflict resolution tests...")
        
        test_resolve_conflict_old_wins()
        test_resolve_conflict_new_wins()
        test_resolve_conflict_both_correct()
        test_get_conflicts()
        test_invalid_resolution_choice()
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*80)
        
    except Exception as e:
        logger.exception("Test failed!")
        cleanup_test_data()
        raise
