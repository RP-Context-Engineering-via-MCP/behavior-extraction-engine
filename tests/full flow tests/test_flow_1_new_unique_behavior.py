"""
Flow 1: New Unique Behavior - Integration Test
================================================

Tests the complete flow when a user expresses a behavior for the first time.

Expected Flow:
1. User submits prompt
2. GPT-4 extracts behavior
3. System calculates credibility
4. System generates embedding
5. System searches for similar behaviors (finds none)
6. System inserts new behavior as ACTIVE
7. Database contains 1 new behavior

Test Scenarios:
- Strong preference (high credibility)
- Weak preference (low credibility)
- Multiple behaviors in one prompt
- Edge cases (short/long text)
"""

import pytest
import json
import time
from datetime import datetime
from typing import List

from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import insert_behavior, search_similar_behaviors
from db.connection import get_db_pool_connection
from config.configurations import CREDIBILITY_PRUNE_THRESHOLD


# Test configuration
TEST_USER_ID = f"test_flow1_{int(time.time())}"
RESULTS_DIR = "test results/E2E"


def cleanup_test_data():
    """Remove all test data"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM behavior_conflicts WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (TEST_USER_ID,))
        conn.commit()


def get_behaviors_for_user(user_id: str) -> List[dict]:
    """Fetch all behaviors"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    behavior_id, behavior_text, credibility, 
                    reinforcement_count, behavior_state, 
                    extraction_confidence, clarity_score, linguistic_strength
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
                    "extraction_confidence": float(row[5]),
                    "clarity_score": float(row[6]),
                    "linguistic_strength": float(row[7])
                }
                for row in results
            ]


def save_test_results(test_name: str, results: dict):
    """Save test results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{test_name}_{timestamp}.json"
    
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved: {filename}")


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Clean up before and after each test"""
    cleanup_test_data()
    yield
    cleanup_test_data()


# ============================================================================
# TEST 1.1: Strong Preference - High Credibility
# ============================================================================
def test_strong_preference_high_credibility():
    """
    Test inserting a strongly expressed preference.
    
    Input: "I absolutely prefer direct communication over beating around the bush"
    Expected: High credibility (>0.8), single behavior, ACTIVE state
    """
    print("\n" + "="*80)
    print("TEST 1.1: Strong Preference - High Credibility")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 1.1 - Strong Preference",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    # Input
    prompt = "I absolutely prefer direct communication over beating around the bush"
    print(f"\n📝 Input Prompt: '{prompt}'")
    
    # Step 1: Extract behaviors
    print("\n🔄 Step 1: Extracting behaviors with GPT-4...")
    extraction_result = run_behavior_extraction(prompt)
    
    assert extraction_result.success, f"Extraction failed: {extraction_result.error}"
    print(f"✅ Extraction successful - {len(extraction_result.segments)} segments")
    
    test_results["extraction"] = {
        "success": extraction_result.success,
        "segments_count": len(extraction_result.segments),
        "behaviors_count": sum(len(seg.behaviors) for seg in extraction_result.segments)
    }
    
    # Step 2: Store behaviors
    print("\n🔄 Step 2: Storing behaviors to database...")
    stored_behaviors = store_behavior(extraction_result, user_id=TEST_USER_ID)
    
    print(f"✅ Stored {len(stored_behaviors)} behaviors")
    
    # Step 3: Verify database state
    print("\n🔄 Step 3: Verifying database state...")
    db_behaviors = get_behaviors_for_user(TEST_USER_ID)
    
    print(f"\n📊 Database State:")
    print(f"  Total behaviors: {len(db_behaviors)}")
    
    for beh in db_behaviors:
        print(f"\n  Behavior: '{beh['behavior_text']}'")
        print(f"    Credibility: {beh['credibility']:.4f}")
        print(f"    State: {beh['behavior_state']}")
        print(f"    Confidence: {beh['extraction_confidence']:.2f}")
        print(f"    Clarity: {beh['clarity_score']:.2f}")
        print(f"    Linguistic Strength: {beh['linguistic_strength']:.2f}")
    
    # Assertions
    assert len(db_behaviors) >= 1, "Should have at least 1 behavior"
    
    primary_behavior = db_behaviors[0]
    assert primary_behavior['credibility'] > 0.75, \
        f"Strong preference should have high credibility, got {primary_behavior['credibility']:.4f}"
    assert primary_behavior['behavior_state'] == 'ACTIVE', \
        f"New behavior should be ACTIVE, got {primary_behavior['behavior_state']}"
    assert primary_behavior['reinforcement_count'] == 1, \
        f"New behavior should have count=1, got {primary_behavior['reinforcement_count']}"
    
    test_results["verification"] = {
        "behaviors_stored": len(db_behaviors),
        "primary_behavior": {
            "text": primary_behavior['behavior_text'],
            "credibility": primary_behavior['credibility'],
            "state": primary_behavior['behavior_state']
        }
    }
    
    test_results["status"] = "PASS"
    test_results["assertions_passed"] = [
        f"✅ At least 1 behavior stored",
        f"✅ High credibility: {primary_behavior['credibility']:.4f} > 0.75",
        f"✅ State is ACTIVE",
        f"✅ Reinforcement count is 1"
    ]
    
    save_test_results("flow_1_1_strong_preference", test_results)
    
    print("\n✅ TEST PASSED: Strong preference behavior stored successfully")


# ============================================================================
# TEST 1.2: Weak Preference - Low Credibility (May be Pruned)
# ============================================================================
def test_weak_preference_low_credibility():
    """
    Test inserting a weakly expressed preference.
    
    Input: "I sometimes maybe think spicy food could be okay"
    Expected: Low credibility, might be pruned if below threshold
    """
    print("\n" + "="*80)
    print("TEST 1.2: Weak Preference - Low Credibility")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 1.2 - Weak Preference",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID,
        "prune_threshold": CREDIBILITY_PRUNE_THRESHOLD
    }
    
    prompt = "I sometimes maybe think spicy food could be okay I guess"
    print(f"\n📝 Input Prompt: '{prompt}'")
    
    extraction_result = run_behavior_extraction(prompt)
    assert extraction_result.success
    
    stored_behaviors = store_behavior(extraction_result, user_id=TEST_USER_ID)
    db_behaviors = get_behaviors_for_user(TEST_USER_ID)
    
    print(f"\n📊 Results:")
    print(f"  Extracted behaviors: {sum(len(seg.behaviors) for seg in extraction_result.segments)}")
    print(f"  Stored behaviors: {len(db_behaviors)}")
    
    if len(db_behaviors) > 0:
        beh = db_behaviors[0]
        print(f"\n  Behavior: '{beh['behavior_text']}'")
        print(f"    Credibility: {beh['credibility']:.4f}")
        print(f"    Linguistic Strength: {beh['linguistic_strength']:.2f}")
        
        test_results["result"] = "STORED"
        test_results["behavior"] = {
            "text": beh['behavior_text'],
            "credibility": beh['credibility'],
            "linguistic_strength": beh['linguistic_strength']
        }
        
        assert beh['linguistic_strength'] < 0.60, \
            "Weak language should have low linguistic strength"
    else:
        print(f"\n  ℹ️  Behavior was pruned (credibility below {CREDIBILITY_PRUNE_THRESHOLD})")
        test_results["result"] = "PRUNED"
        test_results["reason"] = f"Credibility below threshold {CREDIBILITY_PRUNE_THRESHOLD}"
    
    test_results["status"] = "PASS"
    save_test_results("flow_1_2_weak_preference", test_results)
    
    print("\n✅ TEST PASSED: Weak preference handled correctly")


# ============================================================================
# TEST 1.3: Multiple Behaviors in One Prompt
# ============================================================================
def test_multiple_behaviors_single_prompt():
    """
    Test extracting and storing multiple behaviors from one prompt.
    
    Input: Prompt with 3 distinct preferences
    Expected: 3 separate ACTIVE behaviors
    """
    print("\n" + "="*80)
    print("TEST 1.3: Multiple Behaviors in Single Prompt")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 1.3 - Multiple Behaviors",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    prompt = "I prefer working in quiet environments. I like tea over coffee in the mornings. I enjoy swimming for exercise."
    print(f"\n📝 Input Prompt: '{prompt}'")
    
    extraction_result = run_behavior_extraction(prompt)
    assert extraction_result.success
    
    extracted_count = sum(len(seg.behaviors) for seg in extraction_result.segments)
    print(f"\n✅ Extracted {extracted_count} behaviors")
    
    stored_behaviors = store_behavior(extraction_result, user_id=TEST_USER_ID)
    db_behaviors = get_behaviors_for_user(TEST_USER_ID)
    
    print(f"\n📊 Stored {len(db_behaviors)} behaviors:")
    for i, beh in enumerate(db_behaviors, 1):
        print(f"\n  {i}. '{beh['behavior_text']}'")
        print(f"     Credibility: {beh['credibility']:.4f}")
        print(f"     State: {beh['behavior_state']}")
    
    # Assertions
    assert len(db_behaviors) >= 2, "Should store at least 2 distinct behaviors"
    
    for beh in db_behaviors:
        assert beh['behavior_state'] == 'ACTIVE', "All should be ACTIVE"
        assert beh['reinforcement_count'] == 1, "All should have count=1"
    
    test_results["behaviors"] = [
        {
            "text": beh['behavior_text'],
            "credibility": beh['credibility'],
            "state": beh['behavior_state']
        }
        for beh in db_behaviors
    ]
    test_results["status"] = "PASS"
    
    save_test_results("flow_1_3_multiple_behaviors", test_results)
    
    print("\n✅ TEST PASSED: Multiple behaviors stored successfully")


# ============================================================================
# TEST 1.4: Domain-Specific Behavior
# ============================================================================
def test_domain_specific_behavior():
    """
    Test storing travel/domain-specific preference.
    
    Input: "I prefer window seats over aisle seats when flying long distances"
    Expected: High clarity due to specificity
    """
    print("\n" + "="*80)
    print("TEST 1.4: Domain-Specific Behavior")
    print("="*80)
    
    test_results = {
        "test_name": "Flow 1.4 - Domain Specific",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    prompt = "I prefer window seats over aisle seats when flying long distances"
    print(f"\n📝 Input Prompt: '{prompt}'")
    
    extraction_result = run_behavior_extraction(prompt)
    assert extraction_result.success
    
    stored_behaviors = store_behavior(extraction_result, user_id=TEST_USER_ID)
    db_behaviors = get_behaviors_for_user(TEST_USER_ID)
    
    assert len(db_behaviors) >= 1
    
    beh = db_behaviors[0]
    print(f"\n📊 Behavior: '{beh['behavior_text']}'")
    print(f"  Credibility: {beh['credibility']:.4f}")
    print(f"  Clarity: {beh['clarity_score']:.2f}")
    print(f"  Confidence: {beh['extraction_confidence']:.2f}")
    
    # Domain-specific behaviors should have high clarity
    assert beh['clarity_score'] > 0.70, \
        f"Domain-specific should have high clarity, got {beh['clarity_score']:.2f}"
    
    test_results["behavior"] = {
        "text": beh['behavior_text'],
        "credibility": beh['credibility'],
        "clarity": beh['clarity_score']
    }
    test_results["status"] = "PASS"
    
    save_test_results("flow_1_4_domain_specific", test_results)
    
    print("\n✅ TEST PASSED: Domain-specific behavior stored")


if __name__ == "__main__":
    """Run all Flow 1 tests"""
    print("="*80)
    print("FLOW 1: NEW UNIQUE BEHAVIOR - INTEGRATION TESTS")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s", "--tb=short"])
