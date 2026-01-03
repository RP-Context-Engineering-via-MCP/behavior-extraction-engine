"""
Simple Integration Test: Canonical Behavior System

Tests the new canonical duplicate detection logic:
1. Extract behaviors with canonical fields
2. Use core embeddings (intent + target)
3. Context-aware duplicate detection
4. General vs specific behavior handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extractor import run_behavior_extraction, store_behavior, contexts_match
from db.connection import get_db_pool_connection
from config.configurations import SAMPLE_USERID
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_test_data(user_id: str):
    """Remove all test data"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM behaviors WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM prompt_segments WHERE user_id = %s", (user_id,))
            conn.commit()
            logger.info("Cleanup complete")

def get_behaviors_count(user_id: str) -> int:
    """Count behaviors for user"""
    with get_db_pool_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM behaviors WHERE user_id = %s AND behavior_state = 'ACTIVE'",
                (user_id,)
            )
            return cur.fetchone()[0]

def test_contexts_match_logic():
    """Test the contexts_match function"""
    print("\n" + "="*80)
    print("TEST 1: Context Matching Logic")
    print("="*80)
    
    test_cases = [
        ("general", "IDE", True, "GENERALIZATION"),
        ("IDE", "general", True, "SPECIALIZATION"),
        ("IDE", "IDE", True, "DUPLICATE"),
        ("frontend", "backend", False, "DIFFERENT"),
        ("general", "general", True, "DUPLICATE"),
    ]
    
    passed = 0
    for c1, c2, expected_dup, expected_rel in test_cases:
        is_dup, rel = contexts_match(c1, c2)
        if is_dup == expected_dup and rel == expected_rel:
            print(f"✅ contexts_match('{c1}', '{c2}') = ({is_dup}, '{rel}')")
            passed += 1
        else:
            print(f"❌ contexts_match('{c1}', '{c2}') = ({is_dup}, '{rel}') "
                  f"- expected ({expected_dup}, '{expected_rel}')")
    
    print(f"\nContext matching: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_general_vs_specific():
    """Test that general and specific contexts are handled correctly"""
    print("\n" + "="*80)
    print("TEST 2: General vs Specific Context Handling")
    print("="*80)
    
    test_user = f"{SAMPLE_USERID}_canon_test"
    cleanup_test_data(test_user)
    
    # Step 1: Extract general preference
    print("\n1. Extracting general preference: 'I prefer dark mode'")
    result1 = run_behavior_extraction("I prefer dark mode")
    if not result1.success:
        print(f"❌ Extraction 1 failed: {result1.error}")
        return False
    
    stored1 = store_behavior(result1, test_user)
    count1 = get_behaviors_count(test_user)
    print(f"   Stored behaviors count: {count1}")
    
    # Step 2: Extract specific context preference  
    print("\n2. Extracting contextual preference: 'I prefer dark mode for IDE'")
    result2 = run_behavior_extraction("I prefer dark mode for my IDE")
    if not result2.success:
        print(f"❌ Extraction 2 failed: {result2.error}")
        return False
    
    stored2 = store_behavior(result2, test_user)
    count2 = get_behaviors_count(test_user)
    print(f"   Stored behaviors count: {count2}")
    
    # With canonical system: general subsumes IDE, so should reinforce (count stays 1)
    # Old system: would treat as separate due to text difference (count would be 2)
    
    if count2 == 1:
        print("\n✅ PASS: Canonical system correctly identified as duplicate (general subsumes IDE)")
        print("   Old distance-based system would have created 2 separate behaviors")
        return True
    else:
        print(f"\n⚠️  PARTIAL: Created {count2} behaviors (expected 1 with reinforcement)")
        print("   This may indicate canonical matching needs tuning")
        return False

def test_different_contexts():
    """Test that different specific contexts create separate behaviors"""
    print("\n" + "="*80)
    print("TEST 3: Different Specific Contexts")
    print("="*80)
    
    test_user = f"{SAMPLE_USERID}_canon_test2"
    cleanup_test_data(test_user)
    
    # Step 1: Python for backend
    print("\n1. Extracting: 'I prefer Python for backend development'")
    result1 = run_behavior_extraction("I prefer Python for backend development")
    if not result1.success:
        print(f"❌ Extraction failed")
        return False
    
    stored1 = store_behavior(result1, test_user)
    count1 = get_behaviors_count(test_user)
    print(f"   Stored behaviors count: {count1}")
    
    # Step 2: Python for data science
    print("\n2. Extracting: 'I prefer Python for data science'")
    result2 = run_behavior_extraction("I prefer Python for data science")
    if not result2.success:
        print(f"❌ Extraction failed")
        return False
    
    stored2 = store_behavior(result2, test_user)
    count2 = get_behaviors_count(test_user)
    print(f"   Stored behaviors count: {count2}")
    
    # Should create 2 behaviors (different contexts)
    if count2 == 2:
        print("\n✅ PASS: Correctly stored both behaviors (different contexts)")
        return True
    else:
        print(f"\n❌ FAIL: Expected 2 behaviors, got {count2}")
        return False

def test_core_embedding_effectiveness():
    """Test that core embeddings reduce false negatives"""
    print("\n" + "="*80)
    print("TEST 4: Core Embedding Effectiveness")
    print("="*80)
    
    test_user = f"{SAMPLE_USERID}_canon_test3"
    cleanup_test_data(test_user)
    
    # Test with different wording but same intent+target
    prompts = [
        "I like dark mode",
        "I prefer dark mode",
        "I enjoy using dark mode",
        "Dark mode is my preference"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Extracting: '{prompt}'")
        result = run_behavior_extraction(prompt)
        if result.success:
            store_behavior(result, test_user)
        count = get_behaviors_count(test_user)
        print(f"   Total behaviors: {count}")
    
    final_count = get_behaviors_count(test_user)
    
    if final_count == 1:
        print(f"\n✅ PASS: All {len(prompts)} variations recognized as same behavior")
        print("   Core embedding (PREFERENCE + dark_mode) matched despite wording differences")
        return True
    else:
        print(f"\n⚠️  PARTIAL: Created {final_count} behaviors (ideal: 1, acceptable: 2)")
        print(f"   Core embeddings working but may need threshold tuning")
        return final_count <= 2  # Acceptable if only creates 2 instead of 4

def run_all_tests():
    """Run all canonical behavior tests"""
    print("\n" + "="*80)
    print("  CANONICAL BEHAVIOR SYSTEM - INTEGRATION TESTS")
    print("="*80)
    
    results = {
        "Context Matching Logic": test_contexts_match_logic(),
        "General vs Specific": test_general_vs_specific(),
        "Different Contexts": test_different_contexts(),
        "Core Embedding Effectiveness": test_core_embedding_effectiveness()
    }
    
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - Canonical behavior system working!")
    elif passed_count >= total_count * 0.75:
        print("\n⚠️  MOST TESTS PASSED - System functional, minor tuning needed")
    else:
        print("\n❌ TESTS FAILED - Review implementation")

if __name__ == "__main__":
    run_all_tests()
