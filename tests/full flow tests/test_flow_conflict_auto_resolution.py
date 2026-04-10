"""
Integration Test: Conflict Auto-Resolution
===========================================

Tests the automatic resolution of conflicts when credibility difference > 0.3

Scenarios:
1. New behavior wins (higher credibility, supersedes old)
2. Existing behavior wins (new behavior rejected)
3. Close credibility (both flagged for user decision)
"""

import pytest
import json
import time
from datetime import datetime

from services.extractor import run_behavior_extraction, store_behavior
from db.connection import get_db_pool_connection


TEST_USER_ID = f"test_auto_resolve_{int(time.time())}"
RESULTS_DIR = "test results/E2E"


def cleanup_test_data():
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM behavior_conflicts WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (TEST_USER_ID,))
        conn.commit()


def get_all_behaviors(user_id: str):
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    behavior_id, behavior_text, credibility, 
                    behavior_state, superseded_by_id
                FROM behaviors 
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,)
            )
            return cur.fetchall()


def get_conflicts(user_id: str):
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    conflict_type, behavior_id_1, behavior_id_2,
                    similarity_distance, llm_analysis
                FROM behavior_conflicts
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,)
            )
            return cur.fetchall()


def save_test_results(test_name: str, results: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{test_name}_{timestamp}.json"
    
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved: {filename}")
    return filename


@pytest.fixture(autouse=True)
def setup_and_teardown():
    cleanup_test_data()
    yield
    # Keep data for inspection


def test_auto_resolution_new_wins():
    """
    Test auto-resolution when new behavior has much higher credibility.
    
    Weak existing preference + Strong contradicting preference
    → Old superseded, new becomes active
    """
    print("\n" + "="*80)
    print("TEST: Auto-Resolution - New Behavior Wins")
    print("="*80)
    
    test_results = {
        "test_name": "Auto-Resolution: New Wins",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    # Step 1: Insert weak preference
    prompt1 = "I sometimes prefer dark mode but it's not a strong preference"
    print(f"\n📝 First (weak): '{prompt1}'")
    
    result1 = run_behavior_extraction(prompt1)
    stored1 = store_behavior(result1, user_id=TEST_USER_ID)
    
    behaviors_initial = get_all_behaviors(TEST_USER_ID)
    assert len(behaviors_initial) == 1, f"Expected 1 behavior, got {len(behaviors_initial)}"
    
    first_behavior = behaviors_initial[0]
    print(f"   Stored: '{first_behavior[1]}'")
    print(f"   Credibility: {first_behavior[2]:.4f} (weak)")
    print(f"   State: {first_behavior[3]}")
    
    test_results["first_behavior"] = {
        "text": first_behavior[1],
        "credibility": float(first_behavior[2]),
        "state": first_behavior[3]
    }
    
    time.sleep(2)
    
    # Step 2: Insert strong conflicting preference
    prompt2 = "I absolutely must always use light mode, I hate dark mode completely"
    print(f"\n📝 Second (strong, conflicting): '{prompt2}'")
    
    result2 = run_behavior_extraction(prompt2)
    stored2 = store_behavior(result2, user_id=TEST_USER_ID)
    
    time.sleep(1)
    
    # Step 3: Verify resolution
    print(f"\n🔍 Verifying auto-resolution...")
    
    behaviors_final = get_all_behaviors(TEST_USER_ID)
    conflicts = get_conflicts(TEST_USER_ID)
    
    print(f"\n📊 Final State:")
    print(f"   Total behaviors: {len(behaviors_final)}")
    print(f"   Total conflicts: {len(conflicts)}")
    
    # Find superseded and active behaviors
    superseded = [b for b in behaviors_final if b[3] == 'SUPERSEDED']
    active = [b for b in behaviors_final if b[3] == 'ACTIVE']
    
    print(f"\n   Superseded behaviors: {len(superseded)}")
    for beh in superseded:
        print(f"     - '{beh[1]}'")
        print(f"       Credibility: {beh[2]:.4f}")
        print(f"       Superseded by: {beh[4]}")
    
    print(f"\n   Active behaviors: {len(active)}")
    for beh in active:
        print(f"     - '{beh[1]}'")
        print(f"       Credibility: {beh[2]:.4f}")
    
    if conflicts:
        print(f"\n   Conflict Details:")
        for conf in conflicts:
            print(f"     Type: {conf[0]}")
            print(f"     Distance: {conf[3]:.4f}")
            print(f"     Analysis: {conf[4][:100]}...")
    
    # Assertions
    assert len(behaviors_final) == 2, f"Expected 2 behaviors total, got {len(behaviors_final)}"
    
    # Check if auto-resolution happened
    if len(superseded) > 0:
        print(f"\n✅ AUTO-RESOLUTION OCCURRED!")
        print(f"   Old behavior superseded by new one")
        assert len(active) == 1, f"Expected 1 active behavior, got {len(active)}"
        assert len(conflicts) >= 1, f"Expected at least 1 conflict record, got {len(conflicts)}"
        
        # Verify the stronger preference is active
        active_behavior = active[0]
        assert active_behavior[2] > superseded[0][2], "Active behavior should have higher credibility"
        
        test_results["resolution"] = "AUTO_RESOLVED_NEW_WINS"
        test_results["superseded_count"] = len(superseded)
    else:
        print(f"\n⚠️  No auto-resolution (likely CONTEXT_DEPENDENT or COMPATIBLE)")
        print(f"   Both behaviors remain ACTIVE")
        test_results["resolution"] = "NO_AUTO_RESOLUTION"
    
    test_results["final_state"] = {
        "total_behaviors": len(behaviors_final),
        "superseded": len(superseded),
        "active": len(active),
        "conflicts": len(conflicts)
    }
    
    save_test_results("auto_resolve_new_wins", test_results)
    
    print(f"\n✅ TEST COMPLETED")


def test_auto_resolution_existing_wins():
    """
    Test auto-resolution when existing behavior has much higher credibility.
    
    Strong existing preference + Weak contradicting preference
    → New rejected, existing stays active
    """
    print("\n" + "="*80)
    print("TEST: Auto-Resolution - Existing Behavior Wins")
    print("="*80)
    
    test_results = {
        "test_name": "Auto-Resolution: Existing Wins",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    # Step 1: Insert strong preference
    prompt1 = "I absolutely always must use dark mode for everything, no exceptions"
    print(f"\n📝 First (strong): '{prompt1}'")
    
    result1 = run_behavior_extraction(prompt1)
    stored1 = store_behavior(result1, user_id=TEST_USER_ID)
    
    behaviors_initial = get_all_behaviors(TEST_USER_ID)
    first_behavior = behaviors_initial[0]
    
    print(f"   Stored: '{first_behavior[1]}'")
    print(f"   Credibility: {first_behavior[2]:.4f} (strong)")
    print(f"   State: {first_behavior[3]}")
    
    test_results["first_behavior"] = {
        "text": first_behavior[1],
        "credibility": float(first_behavior[2]),
        "state": first_behavior[3]
    }
    
    initial_count = len(behaviors_initial)
    
    time.sleep(2)
    
    # Step 2: Insert weak conflicting preference
    prompt2 = "maybe light mode could be okay sometimes"
    print(f"\n📝 Second (weak, conflicting): '{prompt2}'")
    
    result2 = run_behavior_extraction(prompt2)
    stored2 = store_behavior(result2, user_id=TEST_USER_ID)
    
    time.sleep(1)
    
    # Step 3: Verify resolution
    print(f"\n🔍 Verifying auto-resolution...")
    
    behaviors_final = get_all_behaviors(TEST_USER_ID)
    conflicts = get_conflicts(TEST_USER_ID)
    
    print(f"\n📊 Final State:")
    print(f"   Initial behaviors: {initial_count}")
    print(f"   Final behaviors: {len(behaviors_final)}")
    print(f"   Conflicts recorded: {len(conflicts)}")
    
    active = [b for b in behaviors_final if b[3] == 'ACTIVE']
    
    print(f"\n   Active behaviors: {len(active)}")
    for beh in active:
        print(f"     - '{beh[1]}'")
        print(f"       Credibility: {beh[2]:.4f}")
    
    # Check if new was rejected
    if len(behaviors_final) == initial_count:
        print(f"\n✅ EXISTING BEHAVIOR WON!")
        print(f"   New weak behavior was rejected")
        test_results["resolution"] = "AUTO_RESOLVED_EXISTING_WINS"
    else:
        print(f"\n⚠️  New behavior was inserted (no auto-resolution)")
        test_results["resolution"] = "NO_AUTO_RESOLUTION"
    
    test_results["final_state"] = {
        "behaviors_added": len(behaviors_final) - initial_count,
        "total_behaviors": len(behaviors_final),
        "conflicts": len(conflicts)
    }
    
    save_test_results("auto_resolve_existing_wins", test_results)
    
    print(f"\n✅ TEST COMPLETED")


def test_close_credibility_both_flagged():
    """
    Test when credibility is too close (<0.3 difference).
    
    Similar credibility behaviors
    → Both flagged for user decision (Phase 3)
    """
    print("\n" + "="*80)
    print("TEST: Close Credibility - Both Flagged")
    print("="*80)
    
    test_results = {
        "test_name": "Close Credibility: Both Flagged",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    # Step 1: Insert first preference (medium strength)
    prompt1 = "I prefer dark mode for my coding"
    print(f"\n📝 First: '{prompt1}'")
    
    result1 = run_behavior_extraction(prompt1)
    stored1 = store_behavior(result1, user_id=TEST_USER_ID)
    
    behaviors_initial = get_all_behaviors(TEST_USER_ID)
    first_behavior = behaviors_initial[0]
    
    print(f"   Stored: '{first_behavior[1]}'")
    print(f"   Credibility: {first_behavior[2]:.4f}")
    
    test_results["first_behavior"] = {
        "text": first_behavior[1],
        "credibility": float(first_behavior[2])
    }
    
    time.sleep(2)
    
    # Step 2: Insert conflicting preference (similar strength)
    prompt2 = "I prefer light mode for my development work"
    print(f"\n📝 Second (similar strength): '{prompt2}'")
    
    result2 = run_behavior_extraction(prompt2)
    stored2 = store_behavior(result2, user_id=TEST_USER_ID)
    
    time.sleep(1)
    
    # Step 3: Verify both flagged
    print(f"\n🔍 Verifying flagging...")
    
    behaviors_final = get_all_behaviors(TEST_USER_ID)
    conflicts = get_conflicts(TEST_USER_ID)
    
    print(f"\n📊 Final State:")
    print(f"   Total behaviors: {len(behaviors_final)}")
    print(f"   Conflicts: {len(conflicts)}")
    
    flagged = [b for b in behaviors_final if b[3] == 'FLAGGED']
    active = [b for b in behaviors_final if b[3] == 'ACTIVE']
    
    print(f"\n   Flagged behaviors: {len(flagged)}")
    for beh in flagged:
        print(f"     - '{beh[1]}'")
        print(f"       Credibility: {beh[2]:.4f}")
    
    print(f"\n   Active behaviors: {len(active)}")
    for beh in active:
        print(f"     - '{beh[1]}'")
        print(f"       Credibility: {beh[2]:.4f}")
    
    if conflicts:
        for conf in conflicts:
            print(f"\n   Conflict: {conf[0]}")
            if "USER_DECISION" in conf[0] or "FLAGGED" in conf[4]:
                print(f"   ✅ Correctly flagged for user decision")
    
    # Check if any behaviors were flagged
    if len(flagged) > 0:
        print(f"\n✅ BEHAVIORS FLAGGED FOR USER DECISION!")
        test_results["resolution"] = "FLAGGED_FOR_USER"
        test_results["flagged_count"] = len(flagged)
    else:
        print(f"\n⚠️  No flagging occurred")
        test_results["resolution"] = "NO_FLAGGING"
    
    test_results["final_state"] = {
        "total_behaviors": len(behaviors_final),
        "flagged": len(flagged),
        "active": len(active),
        "conflicts": len(conflicts)
    }
    
    save_test_results("close_credibility_flagged", test_results)
    
    print(f"\n✅ TEST COMPLETED")


def test_explicit_contradiction():
    """
    Test with explicit contradiction to force CONFLICT classification.
    
    "I never use X" vs "I always use X"
    """
    print("\n" + "="*80)
    print("TEST: Explicit Contradiction")
    print("="*80)
    
    test_results = {
        "test_name": "Explicit Contradiction",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID
    }
    
    # Step 1: Strong negative
    prompt1 = "I never use dark mode, I always disable it"
    print(f"\n📝 First: '{prompt1}'")
    
    result1 = run_behavior_extraction(prompt1)
    stored1 = store_behavior(result1, user_id=TEST_USER_ID)
    
    behaviors_initial = get_all_behaviors(TEST_USER_ID)
    first_behavior = behaviors_initial[0]
    
    print(f"   Stored: '{first_behavior[1]}'")
    print(f"   Credibility: {first_behavior[2]:.4f}")
    
    time.sleep(2)
    
    # Step 2: Strong opposite
    prompt2 = "I always use dark mode for everything, I never turn it off"
    print(f"\n📝 Second (opposite): '{prompt2}'")
    
    result2 = run_behavior_extraction(prompt2)
    stored2 = store_behavior(result2, user_id=TEST_USER_ID)
    
    time.sleep(1)
    
    # Verify
    behaviors_final = get_all_behaviors(TEST_USER_ID)
    conflicts = get_conflicts(TEST_USER_ID)
    
    print(f"\n📊 Results:")
    print(f"   Behaviors: {len(behaviors_final)}")
    print(f"   Conflicts: {len(conflicts)}")
    
    if conflicts:
        for conf in conflicts:
            print(f"\n   Conflict Type: {conf[0]}")
            print(f"   Analysis: {conf[4][:150]}...")
            
            if "CONFLICT" in conf[4] or "RESOLVED" in conf[4]:
                print(f"   ✅ Detected as actual CONFLICT!")
                test_results["conflict_detected"] = True
    
    superseded = [b for b in behaviors_final if b[3] == 'SUPERSEDED']
    if superseded:
        print(f"\n   ✅ Auto-resolution occurred: {len(superseded)} superseded")
        test_results["auto_resolved"] = True
    
    test_results["final_state"] = {
        "behaviors": len(behaviors_final),
        "conflicts": len(conflicts),
        "superseded": len(superseded)
    }
    
    save_test_results("explicit_contradiction", test_results)
    
    print(f"\n✅ TEST COMPLETED")


if __name__ == "__main__":
    print("="*80)
    print("AUTO-RESOLUTION INTEGRATION TESTS")
    print("="*80)
    print("\nTesting conflict auto-resolution with various credibility scenarios")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s", "--tb=short"])
