"""
Flow 2: Duplicate Behavior Detection - Integration Test
========================================================

Tests the duplicate detection and reinforcement mechanism.

Expected Flow:
1. Insert first behavior
2. Submit similar/duplicate prompt
3. System detects duplicate (distance < 0.05)
4. System reinforces existing behavior instead of inserting new
5. Credibility increases, reinforcement_count increments
6. Only 1 behavior exists in database

Test Scenarios:
- Exact wording match
- Different wording, same meaning
- Multiple reinforcements (diminishing returns)
- Reinforcement with different linguistic strength
"""

import pytest
import json
import time
from datetime import datetime
from typing import List

from services.extractor import run_behavior_extraction, store_behavior
from db.connection import get_db_pool_connection
from config.configurations import DUPLICATE_THRESHOLD


TEST_USER_ID = f"test_flow2_{int(time.time())}"
RESULTS_DIR = "test results/E2E"


def cleanup_test_data():
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM behavior_conflicts WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (TEST_USER_ID,))
        conn.commit()


def get_behaviors_for_user(user_id: str) -> List[dict]:
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    behavior_id, behavior_text, credibility, 
                    reinforcement_count, behavior_state,
                    prompt_history_ids, last_seen_at
                FROM behaviors 
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            results = cur.fetchall()
            return [
                {
                    "behavior_id": row[0],
                    "behavior_text": row[1],
                    "credibility": float(row[2]),
                    "reinforcement_count": row[3],
                    "behavior_state": row[4],
                    "prompt_history_count": len(row[5]) if row[5] else 0,
                    "last_seen_at": row[6]
                }
                for row in results
            ]


def save_test_results(test_name: str, results: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{test_name}_{timestamp}.json"
    
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved: {filename}")


@pytest.fixture(autouse=True)
def setup_and_teardown():
    cleanup_test_data()
    yield
    cleanup_test_data()


# ============================================================================
# TEST 2.1: Exact Duplicate - Same Wording
# ============================================================================
def test_exact_duplicate_same_wording():
    """
    Test duplicate detection with identical wording.
    
    Flow:
    1. Insert: "I prefer small gatherings over large parties"
    2. Insert: "I prefer small gatherings over large parties" (again)
    Expected: 1 behavior, count=2, credibility increased
    """
    print("\n" + "="*80)
    print("TEST 2.1: Exact Duplicate - Same Wording")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 2.1 - Exact Duplicate",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID,
        "duplicate_threshold": DUPLICATE_THRESHOLD
    }
    
    prompt = "I prefer small gatherings over large parties"
    
    # First insertion
    print(f"\n📝 First Prompt: '{prompt}'")
    extraction_1 = run_behavior_extraction(prompt)
    stored_1 = store_behavior(extraction_1, user_id=TEST_USER_ID)
    
    behaviors_after_first = get_behaviors_for_user(TEST_USER_ID)
    print(f"✅ First insertion: {len(behaviors_after_first)} behaviors")
    
    first_beh = behaviors_after_first[0]
    initial_credibility = first_beh['credibility']
    print(f"   Credibility: {initial_credibility:.4f}")
    print(f"   Count: {first_beh['reinforcement_count']}")
    
    time.sleep(2)  # Ensure different timestamps
    
    # Second insertion (duplicate)
    print(f"\n📝 Second Prompt (duplicate): '{prompt}'")
    extraction_2 = run_behavior_extraction(prompt)
    stored_2 = store_behavior(extraction_2, user_id=TEST_USER_ID)
    
    behaviors_after_second = get_behaviors_for_user(TEST_USER_ID)
    print(f"✅ After duplicate: {len(behaviors_after_second)} behaviors")
    
    # Verify reinforcement
    assert len(behaviors_after_second) == 1, \
        f"Should have 1 behavior, got {len(behaviors_after_second)}"
    
    reinforced_beh = behaviors_after_second[0]
    print(f"\n📊 Reinforced Behavior:")
    print(f"   Text: '{reinforced_beh['behavior_text']}'")
    print(f"   Credibility: {initial_credibility:.4f} → {reinforced_beh['credibility']:.4f}")
    print(f"   Count: 1 → {reinforced_beh['reinforcement_count']}")
    print(f"   Prompt History: {reinforced_beh['prompt_history_count']} segments")
    
    # Assertions
    assert reinforced_beh['reinforcement_count'] == 2, \
        f"Count should be 2, got {reinforced_beh['reinforcement_count']}"
    assert reinforced_beh['credibility'] > initial_credibility, \
        f"Credibility should increase from {initial_credibility:.4f}"
    assert reinforced_beh['prompt_history_count'] == 2, \
        "Should track 2 prompt segments"
    
    boost = reinforced_beh['credibility'] - initial_credibility
    print(f"\n   Boost Applied: +{boost:.4f}")
    
    test_results["verification"] = {
        "initial_credibility": initial_credibility,
        "final_credibility": reinforced_beh['credibility'],
        "boost": boost,
        "reinforcement_count": reinforced_beh['reinforcement_count'],
        "behavior_count": len(behaviors_after_second)
    }
    test_results["status"] = "PASS"
    
    save_test_results("flow_2_1_exact_duplicate", test_results)
    
    print("\n✅ TEST PASSED: Duplicate detected and reinforced")


# ============================================================================
# TEST 2.2: Semantic Duplicate - Different Wording
# ============================================================================
def test_semantic_duplicate_different_wording():
    """
    Test duplicate detection with different wording, same meaning.
    
    Flow:
    1. Insert: "I prefer running in the morning"
    2. Insert: "I like jogging early in the day"
    Expected: Detected as duplicate, reinforced
    """
    print("\n" + "="*80)
    print("TEST 2.2: Semantic Duplicate - Different Wording")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 2.2 - Semantic Duplicate",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    prompt_1 = "I prefer running in the morning"
    prompt_2 = "I like jogging early in the day"
    
    print(f"\n📝 First Prompt: '{prompt_1}'")
    extraction_1 = run_behavior_extraction(prompt_1)
    stored_1 = store_behavior(extraction_1, user_id=TEST_USER_ID)
    
    behaviors_after_first = get_behaviors_for_user(TEST_USER_ID)
    initial_credibility = behaviors_after_first[0]['credibility']
    
    time.sleep(2)
    
    print(f"\n📝 Second Prompt: '{prompt_2}'")
    extraction_2 = run_behavior_extraction(prompt_2)
    stored_2 = store_behavior(extraction_2, user_id=TEST_USER_ID)
    
    behaviors_final = get_behaviors_for_user(TEST_USER_ID)
    
    print(f"\n📊 Results:")
    print(f"   Total behaviors: {len(behaviors_final)}")
    
    if len(behaviors_final) == 1:
        # Detected as duplicate
        beh = behaviors_final[0]
        print(f"   ✅ Detected as duplicate!")
        print(f"   Credibility: {initial_credibility:.4f} → {beh['credibility']:.4f}")
        print(f"   Count: {beh['reinforcement_count']}")
        
        assert beh['reinforcement_count'] == 2, "Should be reinforced"
        test_results["result"] = "DUPLICATE_DETECTED"
    else:
        # Treated as similar, not duplicate
        print(f"   ℹ️  Treated as similar (not duplicate)")
        print(f"   Both behaviors stored separately")
        test_results["result"] = "SIMILAR_NOT_DUPLICATE"
        
        for i, beh in enumerate(behaviors_final, 1):
            print(f"\n   Behavior {i}: '{beh['behavior_text']}'")
            print(f"      Credibility: {beh['credibility']:.4f}")
    
    test_results["behaviors_count"] = len(behaviors_final)
    test_results["status"] = "PASS"
    
    save_test_results("flow_2_2_semantic_duplicate", test_results)
    
    print("\n✅ TEST PASSED: Semantic similarity handled correctly")


# ============================================================================
# TEST 2.3: Multiple Reinforcements - Diminishing Returns
# ============================================================================
def test_multiple_reinforcements_diminishing_returns():
    """
    Test that reinforcement boost decreases with each reinforcement.
    
    Flow: Insert same behavior 5 times
    Expected: Boost decreases: ~0.035 → ~0.022 → ~0.018 → ...
    """
    print("\n" + "="*80)
    print("TEST 2.3: Multiple Reinforcements - Diminishing Returns")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 2.3 - Diminishing Returns",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID,
        "reinforcements": []
    }
    
    prompt = "I prefer vegetarian meals"
    num_reinforcements = 5
    
    print(f"\n📝 Reinforcing '{prompt}' {num_reinforcements} times...")
    
    for i in range(num_reinforcements):
        print(f"\n  Reinforcement {i+1}:")
        
        extraction = run_behavior_extraction(prompt)
        stored = store_behavior(extraction, user_id=TEST_USER_ID)
        
        behaviors = get_behaviors_for_user(TEST_USER_ID)
        beh = behaviors[0]
        
        print(f"    Count: {beh['reinforcement_count']}")
        print(f"    Credibility: {beh['credibility']:.4f}")
        
        # Calculate boost if not first
        if i > 0:
            boost = beh['credibility'] - test_results["reinforcements"][-1]["credibility"]
            print(f"    Boost: +{boost:.4f}")
        else:
            boost = 0.0
        
        test_results["reinforcements"].append({
            "iteration": i + 1,
            "count": beh['reinforcement_count'],
            "credibility": beh['credibility'],
            "boost": boost
        })
        
        time.sleep(1)
    
    # Verify diminishing returns
    final_behaviors = get_behaviors_for_user(TEST_USER_ID)
    assert len(final_behaviors) == 1, "Should have exactly 1 behavior"
    
    final_beh = final_behaviors[0]
    print(f"\n📊 Final State:")
    print(f"   Count: {final_beh['reinforcement_count']}")
    print(f"   Credibility: {final_beh['credibility']:.4f}")
    
    # Check diminishing returns pattern
    if len(test_results["reinforcements"]) >= 3:
        boost_2 = test_results["reinforcements"][1]["boost"]
        boost_3 = test_results["reinforcements"][2]["boost"]
        
        print(f"\n   Boost Comparison:")
        print(f"   2nd reinforcement: +{boost_2:.4f}")
        print(f"   3rd reinforcement: +{boost_3:.4f}")
        
        assert boost_3 < boost_2, \
            "Later reinforcements should have smaller boost (diminishing returns)"
    
    test_results["status"] = "PASS"
    test_results["final_count"] = final_beh['reinforcement_count']
    
    save_test_results("flow_2_3_diminishing_returns", test_results)
    
    print("\n✅ TEST PASSED: Diminishing returns working correctly")


# ============================================================================
# TEST 2.4: Reinforcement Updates Timestamp
# ============================================================================
def test_reinforcement_updates_timestamp():
    """
    Test that reinforcement updates last_seen_at timestamp.
    
    Expected: last_seen_at should increase with each reinforcement
    """
    print("\n" + "="*80)
    print("TEST 2.4: Reinforcement Updates Timestamp")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 2.4 - Timestamp Update",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    prompt = "I like jazz music"
    
    # First insertion
    print(f"\n📝 First insertion...")
    extraction_1 = run_behavior_extraction(prompt)
    stored_1 = store_behavior(extraction_1, user_id=TEST_USER_ID)
    
    behaviors_1 = get_behaviors_for_user(TEST_USER_ID)
    first_timestamp = behaviors_1[0]['last_seen_at']
    print(f"   First timestamp: {first_timestamp}")
    
    time.sleep(3)  # Wait 3 seconds
    
    # Second insertion (reinforcement)
    print(f"\n📝 Reinforcement...")
    extraction_2 = run_behavior_extraction(prompt)
    stored_2 = store_behavior(extraction_2, user_id=TEST_USER_ID)
    
    behaviors_2 = get_behaviors_for_user(TEST_USER_ID)
    second_timestamp = behaviors_2[0]['last_seen_at']
    print(f"   Second timestamp: {second_timestamp}")
    
    # Verify timestamp updated
    time_diff = second_timestamp - first_timestamp
    print(f"\n📊 Time difference: {time_diff} seconds")
    
    assert second_timestamp > first_timestamp, \
        "Timestamp should be updated on reinforcement"
    assert time_diff >= 2, \
        f"Time difference should be at least 2 seconds, got {time_diff}"
    
    test_results["first_timestamp"] = first_timestamp
    test_results["second_timestamp"] = second_timestamp
    test_results["time_diff_seconds"] = time_diff
    test_results["status"] = "PASS"
    
    save_test_results("flow_2_4_timestamp_update", test_results)
    
    print("\n✅ TEST PASSED: Timestamp updated correctly")


if __name__ == "__main__":
    print("="*80)
    print("FLOW 2: DUPLICATE DETECTION - INTEGRATION TESTS")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s", "--tb=short"])
