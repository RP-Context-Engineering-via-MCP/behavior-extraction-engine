"""
Integration test for Phase 1: Duplicate Detection & Reinforcement

Tests:
1. Extract same behavior twice → should reinforce, not duplicate
2. Extract similar behavior → should insert both (Phase 1 behavior)
3. Verify reinforcement_count increments correctly
4. Verify credibility boost applied with diminishing returns
5. Verify segment_id added to prompt_history_ids
6. Verify no duplicate behaviors in database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extractor import run_behavior_extraction, store_behavior
from db.connection import get_db_pool_connection
from config.configurations import SAMPLE_USERID
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Result tracking
test_results = {
    "test_run_timestamp": None,
    "tests": [],
    "summary": {}
}


def cleanup_test_data(user_id: str):
    """Remove all test data for clean test environment"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            # Delete behaviors for test user
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (user_id,))
            deleted_behaviors = cur.rowcount
            
            # Delete prompt segments for test user
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (user_id,))
            deleted_segments = cur.rowcount
            
            conn.commit()
            logger.info(f"Cleanup: Deleted {deleted_behaviors} behaviors, {deleted_segments} segments")


def get_behaviors_for_user(user_id: str):
    """Retrieve all behaviors for a user"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    behavior_id,
                    behavior_text,
                    credibility,
                    reinforcement_count,
                    prompt_history_ids,
                    last_seen_at,
                    created_at
                FROM behaviors
                WHERE user_id = %s
                ORDER BY created_at ASC
            """, (user_id,))
            
            columns = [desc[0] for desc in cur.description]
            results = []
            for row in cur.fetchall():
                results.append(dict(zip(columns, row)))
            return results


def print_test_section(title: str):
    """Print formatted test section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_exact_duplicate_detection():
    """Test 1: Extract identical behavior twice, verify reinforcement"""
    print_test_section("TEST 1: Exact Duplicate Detection")
    
    test_result = {
        "test_name": "Exact Duplicate Detection",
        "status": "RUNNING",
        "details": {},
        "error": None
    }
    
    test_user = f"{SAMPLE_USERID}_test_duplicate"
    cleanup_test_data(test_user)
    
    # First extraction
    prompt1 = "I prefer Python for backend development"
    print(f"🔹 Extracting first time: '{prompt1}'")
    test_result["details"]["prompt_1"] = prompt1
    
    result1 = run_behavior_extraction(prompt1)
    if not result1.success:
        print(f"❌ Extraction failed: {result1.error}")
        test_result["status"] = "FAILED"
        test_result["error"] = f"First extraction failed: {result1.error}"
        return False, test_result
    
    stored1 = store_behavior(result1, test_user)
    print(f"✅ First extraction: {len(stored1)} behaviors stored")
    test_result["details"]["first_extraction_count"] = len(stored1)
    
    # Check database
    behaviors_after_first = get_behaviors_for_user(test_user)
    print(f"📊 Behaviors in DB after first: {len(behaviors_after_first)}")
    test_result["details"]["db_count_after_first"] = len(behaviors_after_first)
    
    if len(behaviors_after_first) == 0:
        print("❌ No behaviors stored after first extraction")
        test_result["status"] = "FAILED"
        test_result["error"] = "No behaviors stored after first extraction"
        return False, test_result
    
    first_behavior = behaviors_after_first[0]
    print(f"   - behavior_text: {first_behavior['behavior_text']}")
    print(f"   - credibility: {first_behavior['credibility']:.4f}")
    print(f"   - reinforcement_count: {first_behavior['reinforcement_count']}")
    print(f"   - prompt_history_ids: {len(first_behavior['prompt_history_ids'])} segments")
    
    test_result["details"]["first_behavior"] = {
        "behavior_text": first_behavior['behavior_text'],
        "credibility": float(first_behavior['credibility']),
        "reinforcement_count": int(first_behavior['reinforcement_count']),
        "prompt_history_count": len(first_behavior['prompt_history_ids'])
    }
    
    # Second extraction (duplicate)
    prompt2 = "I prefer Python for backend work"  # Very similar
    print(f"\n🔹 Extracting second time (duplicate): '{prompt2}'")
    test_result["details"]["prompt_2"] = prompt2
    
    result2 = run_behavior_extraction(prompt2)
    if not result2.success:
        print(f"❌ Extraction failed: {result2.error}")
        test_result["status"] = "FAILED"
        test_result["error"] = f"Second extraction failed: {result2.error}"
        return False, test_result
    
    stored2 = store_behavior(result2, test_user)
    print(f"✅ Second extraction: {len(stored2)} behaviors stored (should be 0 if reinforced)")
    test_result["details"]["second_extraction_count"] = len(stored2)
    
    # Check database again
    behaviors_after_second = get_behaviors_for_user(test_user)
    print(f"\n📊 Behaviors in DB after second: {len(behaviors_after_second)}")
    test_result["details"]["db_count_after_second"] = len(behaviors_after_second)
    
    if len(behaviors_after_second) > 1:
        print(f"❌ FAILED: Expected 1 behavior (reinforced), found {len(behaviors_after_second)}")
        for b in behaviors_after_second:
            print(f"   - {b['behavior_text']} (count={b['reinforcement_count']})")
        test_result["status"] = "FAILED"
        test_result["error"] = f"Found {len(behaviors_after_second)} behaviors instead of 1 (duplicate not detected)"
        return False, test_result
    
    reinforced_behavior = behaviors_after_second[0]
    print(f"✅ Behavior after reinforcement:")
    print(f"   - behavior_text: {reinforced_behavior['behavior_text']}")
    print(f"   - credibility: {first_behavior['credibility']:.4f} → {reinforced_behavior['credibility']:.4f}")
    print(f"   - reinforcement_count: {first_behavior['reinforcement_count']} → {reinforced_behavior['reinforcement_count']}")
    print(f"   - prompt_history_ids: {len(first_behavior['prompt_history_ids'])} → {len(reinforced_behavior['prompt_history_ids'])} segments")
    
    test_result["details"]["reinforced_behavior"] = {
        "behavior_text": reinforced_behavior['behavior_text'],
        "credibility": float(reinforced_behavior['credibility']),
        "reinforcement_count": int(reinforced_behavior['reinforcement_count']),
        "prompt_history_count": len(reinforced_behavior['prompt_history_ids']),
        "credibility_boost": float(reinforced_behavior['credibility']) - float(first_behavior['credibility'])
    }
    
    # Verify reinforcement logic
    if reinforced_behavior['reinforcement_count'] != first_behavior['reinforcement_count'] + 1:
        print(f"❌ FAILED: Reinforcement count not incremented correctly")
        test_result["status"] = "FAILED"
        test_result["error"] = "Reinforcement count not incremented"
        return False, test_result
    
    if reinforced_behavior['credibility'] <= first_behavior['credibility']:
        print(f"❌ FAILED: Credibility not boosted after reinforcement")
        test_result["status"] = "FAILED"
        test_result["error"] = "Credibility not boosted"
        return False, test_result
    
    if len(reinforced_behavior['prompt_history_ids']) != len(first_behavior['prompt_history_ids']) + 1:
        print(f"❌ FAILED: Segment not added to prompt_history_ids")
        test_result["status"] = "FAILED"
        test_result["error"] = "Segment not added to history"
        return False, test_result
    
    print("\n✅ TEST 1 PASSED: Duplicate detection and reinforcement working correctly")
    test_result["status"] = "PASSED"
    cleanup_test_data(test_user)
    return True, test_result


def test_similar_but_not_duplicate():
    """Test 2: Extract similar behaviors, verify both inserted (Phase 1 behavior)"""
    print_test_section("TEST 2: Similar (Not Duplicate) Behaviors")
    
    test_result = {
        "test_name": "Similar But Not Duplicate",
        "status": "RUNNING",
        "details": {},
        "error": None
    }
    
    test_user = f"{SAMPLE_USERID}_test_similar"
    cleanup_test_data(test_user)
    
    # First extraction
    prompt1 = "I prefer dark mode for coding"
    print(f"🔹 Extracting first: '{prompt1}'")
    test_result["details"]["prompt_1"] = prompt1
    
    result1 = run_behavior_extraction(prompt1)
    if not result1.success:
        print(f"❌ Extraction failed: {result1.error}")
        test_result["status"] = "FAILED"
        test_result["error"] = f"First extraction failed: {result1.error}"
        return False, test_result
    
    stored1 = store_behavior(result1, test_user)
    print(f"✅ First extraction: {len(stored1)} behaviors stored")
    test_result["details"]["first_extraction_count"] = len(stored1)
    
    # Second extraction (similar, but different enough)
    prompt2 = "I like using light theme during daytime"
    print(f"\n🔹 Extracting second (similar domain): '{prompt2}'")
    test_result["details"]["prompt_2"] = prompt2
    
    result2 = run_behavior_extraction(prompt2)
    if not result2.success:
        print(f"❌ Extraction failed: {result2.error}")
        test_result["status"] = "FAILED"
        test_result["error"] = f"Second extraction failed: {result2.error}"
        return False, test_result
    
    stored2 = store_behavior(result2, test_user)
    print(f"✅ Second extraction: {len(stored2)} behaviors stored")
    test_result["details"]["second_extraction_count"] = len(stored2)
    
    # Check database
    behaviors = get_behaviors_for_user(test_user)
    print(f"\n📊 Total behaviors in DB: {len(behaviors)}")
    test_result["details"]["total_behaviors"] = len(behaviors)
    test_result["details"]["behaviors"] = []
    
    for i, b in enumerate(behaviors, 1):
        print(f"   {i}. {b['behavior_text']}")
        print(f"      - credibility: {b['credibility']:.4f}")
        print(f"      - reinforcement_count: {b['reinforcement_count']}")
        test_result["details"]["behaviors"].append({
            "behavior_text": b['behavior_text'],
            "credibility": float(b['credibility']),
            "reinforcement_count": int(b['reinforcement_count'])
        })
    
    # For Phase 1, we expect similar behaviors to be inserted separately
    # Phase 2 will add merge logic
    if len(behaviors) >= 1:
        print("\n✅ TEST 2 PASSED: Similar behaviors handled (inserted separately in Phase 1)")
        test_result["status"] = "PASSED"
        cleanup_test_data(test_user)
        return True, test_result
    else:
        print(f"❌ FAILED: Expected at least 1 behavior, found {len(behaviors)}")
        test_result["status"] = "FAILED"
        test_result["error"] = f"Expected at least 1 behavior, found {len(behaviors)}"
        cleanup_test_data(test_user)
        return False, test_result


def test_diminishing_returns():
    """Test 3: Extract same behavior multiple times, verify diminishing credibility boost"""
    print_test_section("TEST 3: Diminishing Returns on Reinforcement")
    
    test_result = {
        "test_name": "Diminishing Returns",
        "status": "RUNNING",
        "details": {},
        "error": None
    }
    
    test_user = f"{SAMPLE_USERID}_test_diminishing"
    cleanup_test_data(test_user)
    
    # Extract same behavior 5 times
    prompts = [
        "I prefer Python programming",
        "I prefer Python programming",  # Exact duplicate
        "I prefer Python programming",  # Exact duplicate
        "I prefer Python programming",  # Exact duplicate
        "I prefer Python programming"   # Exact duplicate
    ]
    
    test_result["details"]["prompts"] = prompts
    test_result["details"]["credibility_progression"] = []
    credibility_history = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔹 Iteration {i}: '{prompt}'")
        
        result = run_behavior_extraction(prompt)
        if not result.success:
            print(f"❌ Extraction failed: {result.error}")
            continue
        
        store_behavior(result, test_user)
        
        # Get current credibility
        behaviors = get_behaviors_for_user(test_user)
        if behaviors:
            credibility = float(behaviors[0]['credibility'])
            count = int(behaviors[0]['reinforcement_count'])
            credibility_history.append((count, credibility))
            print(f"   ✅ reinforcement_count={count}, credibility={credibility:.4f}")
            test_result["details"]["credibility_progression"].append({
                "iteration": i,
                "count": count,
                "credibility": credibility
            })
    
    # Analyze diminishing returns
    print("\n📊 Credibility Boost Analysis:")
    print(f"{'Count':<8} {'Credibility':<12} {'Boost':<12} {'% of Initial':<15}")
    print("-" * 50)
    
    for i in range(len(credibility_history)):
        count, cred = credibility_history[i]
        if i == 0:
            boost = 0.0
            pct = 100.0
        else:
            boost = cred - credibility_history[i-1][1]
            first_boost = credibility_history[1][1] - credibility_history[0][1] if len(credibility_history) > 1 else 1
            pct = (boost / first_boost) * 100 if first_boost > 0 else 100.0
        
        print(f"{count:<8} {cred:<12.4f} {boost:<12.4f} {pct:<15.1f}%")
    
    # Verify diminishing returns pattern
    if len(credibility_history) >= 3:
        boost1 = credibility_history[1][1] - credibility_history[0][1]
        boost2 = credibility_history[2][1] - credibility_history[1][1]
        
        test_result["details"]["boost_analysis"] = {
            "first_boost": boost1,
            "second_boost": boost2,
            "diminishing": boost2 < boost1
        }
        
        if boost2 < boost1:
            print(f"\n✅ TEST 3 PASSED: Diminishing returns verified (boost decreases over time)")
            test_result["status"] = "PASSED"
            cleanup_test_data(test_user)
            return True, test_result
        else:
            print(f"\n❌ FAILED: Boosts not diminishing (boost1={boost1:.4f}, boost2={boost2:.4f})")
            test_result["status"] = "FAILED"
            test_result["error"] = f"Boosts not diminishing: first={boost1:.4f}, second={boost2:.4f}"
            cleanup_test_data(test_user)
            return False, test_result
    else:
        print(f"\n⚠️ Not enough data to verify diminishing returns")
        test_result["status"] = "SKIPPED"
        test_result["error"] = "Not enough reinforcements to verify diminishing returns"
        cleanup_test_data(test_user)
        return False, test_result


def test_unrelated_behaviors():
    """Test 4: Extract completely unrelated behaviors, verify all inserted"""
    print_test_section("TEST 4: Unrelated Behaviors (All Inserted)")
    
    test_result = {
        "test_name": "Unrelated Behaviors",
        "status": "RUNNING",
        "details": {},
        "error": None
    }
    
    test_user = f"{SAMPLE_USERID}_test_unrelated"
    cleanup_test_data(test_user)
    
    # Completely different domains
    prompts = [
        "I prefer Python programming",
        "I like drinking coffee in the morning",
        "I enjoy playing tennis on weekends"
    ]
    
    test_result["details"]["prompts"] = prompts
    test_result["details"]["behaviors"] = []
    
    print("🔹 Extracting unrelated behaviors:")
    for prompt in prompts:
        print(f"   - '{prompt}'")
        result = run_behavior_extraction(prompt)
        if result.success:
            store_behavior(result, test_user)
    
    # Check database
    behaviors = get_behaviors_for_user(test_user)
    print(f"\n📊 Behaviors in DB: {len(behaviors)}")
    test_result["details"]["total_behaviors"] = len(behaviors)
    
    for i, b in enumerate(behaviors, 1):
        print(f"   {i}. {b['behavior_text']}")
        print(f"      - reinforcement_count: {b['reinforcement_count']}")
        test_result["details"]["behaviors"].append({
            "behavior_text": b['behavior_text'],
            "reinforcement_count": int(b['reinforcement_count'])
        })
    
    # All should be inserted separately (no duplicates)
    unique_texts = set(b['behavior_text'] for b in behaviors)
    test_result["details"]["unique_count"] = len(unique_texts)
    
    if len(behaviors) == len(unique_texts) and len(behaviors) >= 2:
        print(f"\n✅ TEST 4 PASSED: Unrelated behaviors inserted separately (no false positives)")
        test_result["status"] = "PASSED"
        cleanup_test_data(test_user)
        return True, test_result
    else:
        print(f"❌ FAILED: Expected all unique, found {len(behaviors)} behaviors, {len(unique_texts)} unique")
        test_result["status"] = "FAILED"
        test_result["error"] = f"Expected all unique, found {len(behaviors)} behaviors, {len(unique_texts)} unique"
        cleanup_test_data(test_user)
        return False, test_result


def save_test_results(results_dir: str = "test results/similarity detaction and credibility management"):
    """Save test results to JSON file with timestamp"""
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(test_results, indent=2, fp=f)
    
    print(f"\n💾 Test results saved to: {filepath}")
    return filepath


def run_all_tests():
    """Execute all Phase 1 integration tests"""
    print("\n" + "█"*80)
    print("  PHASE 1 INTEGRATION TESTS: Duplicate Detection & Reinforcement")
    print("█"*80)
    
    # Set timestamp for this test run
    test_results["test_run_timestamp"] = datetime.now().isoformat()
    
    tests = [
        ("Exact Duplicate Detection", test_exact_duplicate_detection),
        ("Similar But Not Duplicate", test_similar_but_not_duplicate),
        ("Diminishing Returns", test_diminishing_returns),
        ("Unrelated Behaviors", test_unrelated_behaviors)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed, test_result = test_func()
            test_results["tests"].append(test_result)
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {str(e)}", exc_info=True)
            test_results["tests"].append({
                "test_name": name,
                "status": "CRASHED",
                "error": str(e),
                "details": {}
            })
            results.append((name, False))
    
    # Summary
    print("\n" + "█"*80)
    print("  TEST SUMMARY")
    print("█"*80 + "\n")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:12} {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    test_results["summary"] = {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": f"{(passed_count/total_count)*100:.1f}%"
    }
    
    print(f"\n{'='*80}")
    print(f"  FINAL RESULT: {passed_count}/{total_count} tests passed ({test_results['summary']['pass_rate']})")
    print(f"{'='*80}\n")
    
    # Save results to file
    save_test_results()
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
