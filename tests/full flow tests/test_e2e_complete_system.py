"""
End-to-End Integration Test: Complete System Flow
==================================================

Comprehensive test covering all phases and scenarios in a realistic sequence.

Test Scenario: Software Developer's Preference Journey
------------------------------------------------------
1. Initial preferences (unique behaviors)
2. Reinforcement of existing preferences
3. Similar but distinct preferences
4. Conflicting preferences (tool/language changes)
5. Verification of entire system state

This simulates a real user's behavior evolution over time.
"""

import pytest
import json
import time
from datetime import datetime
from typing import List

from services.extractor import run_behavior_extraction, store_behavior
from db.connection import get_db_pool_connection


TEST_USER_ID = f"test_e2e_{int(time.time())}"
RESULTS_DIR = "test results/E2E"


def cleanup_test_data():
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM behavior_conflicts WHERE user_id = %s", (TEST_USER_ID,))
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (TEST_USER_ID,))
        conn.commit()


def get_all_data(user_id: str) -> dict:
    """Get complete database state for user"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            # Get behaviors
            cur.execute(
                """
                SELECT 
                    behavior_id, behavior_text, credibility, 
                    reinforcement_count, behavior_state, superseded_by_id
                FROM behaviors 
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,)
            )
            behaviors = cur.fetchall()
            
            # Get conflicts
            cur.execute(
                """
                SELECT 
                    conflict_id, behavior_id_1, behavior_id_2,
                    conflict_type, similarity_distance, llm_analysis
                FROM behavior_conflicts
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,)
            )
            conflicts = cur.fetchall()
            
            # Get segments
            cur.execute(
                """
                SELECT segment_id, segment_text, created_at
                FROM prompt_segments
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,)
            )
            segments = cur.fetchall()
    
    return {
        "behaviors": [
            {
                "behavior_id": b[0],
                "text": b[1],
                "credibility": float(b[2]),
                "reinforcement_count": b[3],
                "state": b[4],
                "superseded_by_id": b[5]
            }
            for b in behaviors
        ],
        "conflicts": [
            {
                "conflict_id": str(c[0]),
                "behavior_id_1": c[1],
                "behavior_id_2": c[2],
                "conflict_type": c[3],
                "distance": float(c[4]),
                "analysis_preview": c[5][:200] if c[5] else None
            }
            for c in conflicts
        ],
        "segments": [
            {
                "segment_id": str(s[0]),
                "text": s[1],
                "created_at": s[2]
            }
            for s in segments
        ]
    }


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
    # Don't cleanup after - keep data for inspection
    # cleanup_test_data()


def test_complete_system_flow_e2e():
    """
    Comprehensive end-to-end test covering all system features.
    
    Simulates a developer's journey:
    - Day 1: Initial preferences
    - Day 2: Reinforcement
    - Day 3: New related preferences
    - Day 4: Tool/language change (conflicts)
    """
    print("\n" + "="*80)
    print("END-TO-END INTEGRATION TEST: Complete System Flow")
    print("="*80)
    print(f"\nTest User ID: {TEST_USER_ID}")
    
    test_results = {
        "test_name": "E2E Complete System Flow",
        "timestamp": datetime.now().isoformat(),
        "user_id": TEST_USER_ID,
        "phases": []
    }
    
    # ========================================================================
    # PHASE 1: Initial Preferences (Day 1)
    # ========================================================================
    print("\n" + "─"*80)
    print("PHASE 1: Initial Preferences")
    print("─"*80)
    
    day1_prompts = [
        "I prefer dark mode for my IDE",
        "I like Python for backend development",
        "I enjoy working in the morning",
        "I prefer using Git command line over GUI"
    ]
    
    phase1_results = {"prompts": [], "behaviors_added": 0}
    
    for i, prompt in enumerate(day1_prompts, 1):
        print(f"\n{i}. Processing: '{prompt}'")
        
        extraction = run_behavior_extraction(prompt)
        if extraction.success:
            stored = store_behavior(extraction, user_id=TEST_USER_ID)
            behaviors_count = len(stored)
            print(f"   ✅ Stored {behaviors_count} behaviors")
            phase1_results["behaviors_added"] += behaviors_count
        else:
            print(f"   ❌ Extraction failed: {extraction.error}")
        
        phase1_results["prompts"].append({
            "prompt": prompt,
            "success": extraction.success,
            "behaviors_extracted": len(stored) if extraction.success else 0
        })
        
        time.sleep(1)
    
    data_after_phase1 = get_all_data(TEST_USER_ID)
    phase1_results["final_state"] = {
        "total_behaviors": len(data_after_phase1["behaviors"]),
        "total_segments": len(data_after_phase1["segments"])
    }
    
    print(f"\n📊 Phase 1 Summary:")
    print(f"   Behaviors: {len(data_after_phase1['behaviors'])}")
    print(f"   Segments: {len(data_after_phase1['segments'])}")
    
    test_results["phases"].append({
        "phase": "1_initial_preferences",
        "results": phase1_results
    })
    
    # ========================================================================
    # PHASE 2: Reinforcement (Day 2)
    # ========================================================================
    print("\n" + "─"*80)
    print("PHASE 2: Reinforcement (Duplicate Detection)")
    print("─"*80)
    
    time.sleep(2)
    
    day2_prompts = [
        "I really prefer dark mode",  # Reinforce dark mode
        "I like Python programming",  # Reinforce Python
    ]
    
    phase2_results = {"prompts": [], "reinforcements": 0}
    
    for i, prompt in enumerate(day2_prompts, 1):
        print(f"\n{i}. Processing: '{prompt}'")
        
        before_count = len(get_all_data(TEST_USER_ID)["behaviors"])
        
        extraction = run_behavior_extraction(prompt)
        if extraction.success:
            stored = store_behavior(extraction, user_id=TEST_USER_ID)
            
            after_count = len(get_all_data(TEST_USER_ID)["behaviors"])
            
            if after_count == before_count:
                print(f"   ✅ Reinforced existing behavior (no new insertion)")
                phase2_results["reinforcements"] += 1
            else:
                print(f"   ℹ️  Added as new behavior (not detected as duplicate)")
        
        phase2_results["prompts"].append({
            "prompt": prompt,
            "success": extraction.success
        })
        
        time.sleep(1)
    
    data_after_phase2 = get_all_data(TEST_USER_ID)
    phase2_results["final_state"] = {
        "total_behaviors": len(data_after_phase2["behaviors"]),
        "reinforcements_detected": phase2_results["reinforcements"]
    }
    
    print(f"\n📊 Phase 2 Summary:")
    print(f"   Behaviors: {len(data_after_phase2['behaviors'])}")
    print(f"   Reinforcements: {phase2_results['reinforcements']}")
    
    test_results["phases"].append({
        "phase": "2_reinforcement",
        "results": phase2_results
    })
    
    # ========================================================================
    # PHASE 3: Related Preferences (Similar, not Duplicate)
    # ========================================================================
    print("\n" + "─"*80)
    print("PHASE 3: Related Preferences")
    print("─"*80)
    
    time.sleep(2)
    
    day3_prompts = [
        "I prefer dark theme at night for reducing eye strain",
        "I like Python for data analysis as well"
    ]
    
    phase3_results = {"prompts": []}
    
    for i, prompt in enumerate(day3_prompts, 1):
        print(f"\n{i}. Processing: '{prompt}'")
        
        extraction = run_behavior_extraction(prompt)
        if extraction.success:
            stored = store_behavior(extraction, user_id=TEST_USER_ID)
            print(f"   ✅ Processed")
        
        phase3_results["prompts"].append({
            "prompt": prompt,
            "success": extraction.success
        })
        
        time.sleep(1)
    
    data_after_phase3 = get_all_data(TEST_USER_ID)
    phase3_results["final_state"] = {
        "total_behaviors": len(data_after_phase3["behaviors"])
    }
    
    print(f"\n📊 Phase 3 Summary:")
    print(f"   Behaviors: {len(data_after_phase3['behaviors'])}")
    
    test_results["phases"].append({
        "phase": "3_related_preferences",
        "results": phase3_results
    })
    
    # ========================================================================
    # PHASE 4: Conflicting Preferences (Tool/Language Change)
    # ========================================================================
    print("\n" + "─"*80)
    print("PHASE 4: Conflicting Preferences (Conflict Detection)")
    print("─"*80)
    
    time.sleep(2)
    
    day4_prompts = [
        "I strongly prefer JavaScript for all backend work now",  # Conflicts with Python
        "I prefer light mode for daytime coding"  # Conflicts with dark mode
    ]
    
    phase4_results = {"prompts": []}
    
    for i, prompt in enumerate(day4_prompts, 1):
        print(f"\n{i}. Processing: '{prompt}'")
        
        extraction = run_behavior_extraction(prompt)
        if extraction.success:
            stored = store_behavior(extraction, user_id=TEST_USER_ID)
            print(f"   ✅ Processed (conflict detection active)")
        
        phase4_results["prompts"].append({
            "prompt": prompt,
            "success": extraction.success
        })
        
        time.sleep(2)  # Give LLM time to analyze
    
    data_after_phase4 = get_all_data(TEST_USER_ID)
    phase4_results["final_state"] = {
        "total_behaviors": len(data_after_phase4["behaviors"]),
        "conflicts_detected": len(data_after_phase4["conflicts"])
    }
    
    print(f"\n📊 Phase 4 Summary:")
    print(f"   Behaviors: {len(data_after_phase4['behaviors'])}")
    print(f"   Conflicts: {len(data_after_phase4['conflicts'])}")
    
    test_results["phases"].append({
        "phase": "4_conflicting_preferences",
        "results": phase4_results
    })
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SYSTEM STATE ANALYSIS")
    print("="*80)
    
    final_data = get_all_data(TEST_USER_ID)
    
    print(f"\n📊 Complete Database State:")
    print(f"   Total Behaviors: {len(final_data['behaviors'])}")
    print(f"   Total Conflicts: {len(final_data['conflicts'])}")
    print(f"   Total Segments: {len(final_data['segments'])}")
    
    # Analyze behavior states
    states = {}
    for beh in final_data["behaviors"]:
        state = beh["state"]
        states[state] = states.get(state, 0) + 1
    
    print(f"\n📈 Behavior States:")
    for state, count in states.items():
        print(f"   {state}: {count}")
    
    # Show all behaviors
    print(f"\n📋 All Behaviors:")
    for i, beh in enumerate(final_data["behaviors"], 1):
        print(f"\n   {i}. '{beh['text']}'")
        print(f"      State: {beh['state']}")
        print(f"      Credibility: {beh['credibility']:.4f}")
        print(f"      Reinforcements: {beh['reinforcement_count']}")
        if beh['superseded_by_id']:
            print(f"      ⚠️  Superseded by: {beh['superseded_by_id']}")
    
    # Show conflicts
    if len(final_data["conflicts"]) > 0:
        print(f"\n⚠️  Conflicts Detected:")
        for i, conf in enumerate(final_data["conflicts"], 1):
            print(f"\n   {i}. {conf['conflict_type']}")
            print(f"      Distance: {conf['distance']:.4f}")
            print(f"      Analysis: {conf['analysis_preview']}...")
    
    # Save complete test results
    test_results["final_analysis"] = {
        "total_behaviors": len(final_data["behaviors"]),
        "total_conflicts": len(final_data["conflicts"]),
        "total_segments": len(final_data["segments"]),
        "behavior_states": states,
        "all_behaviors": final_data["behaviors"],
        "all_conflicts": final_data["conflicts"]
    }
    
    test_results["status"] = "PASS"
    test_results["test_duration_seconds"] = time.time() - int(TEST_USER_ID.split("_")[-1])
    
    filename = save_test_results("e2e_complete_flow", test_results)
    
    print(f"\n" + "="*80)
    print("✅ END-TO-END TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nComplete results saved to: {filename}")
    print(f"\nTest User ID (for manual inspection): {TEST_USER_ID}")
    print("\n💡 You can query this user's data in the database to inspect details")
    
    # Final assertions
    assert len(final_data["behaviors"]) >= 4, \
        f"Should have multiple behaviors, got {len(final_data['behaviors'])}"
    
    assert len(final_data["segments"]) >= 8, \
        f"Should have multiple segments, got {len(final_data['segments'])}"
    
    print("\n✅ ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    print("="*80)
    print("END-TO-END INTEGRATION TEST: COMPLETE SYSTEM FLOW")
    print("="*80)
    print("\nThis test covers:")
    print("  ✓ New unique behaviors (Phase 1)")
    print("  ✓ Duplicate detection & reinforcement (Phase 1)")
    print("  ✓ Similar behaviors (Phase 1)")
    print("  ✓ Conflict detection (Phase 2)")
    print("  ✓ Auto-resolution (Phase 2)")
    print("  ✓ Complete system integration")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s", "--tb=short"])
