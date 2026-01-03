"""
Focused Demonstration Test - Canonical Behavior System

This test demonstrates the key improvements of the canonical behavior system:
1. Perfect duplicate detection using core embeddings (intent + target)
2. Context-aware reasoning (general vs specific)
3. Multi-domain behavior extraction
4. Polarity-based conflict detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import get_user_behaviors
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_duplicate_detection():
    """
    Test 1: Perfect Duplicate Detection via Core Embedding
    
    Scenario: User expresses same preference with different wording
    Expected: All 4 variations recognized as same behavior (distance=0.0000)
    """
    user_id = f"demo_user_dup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("\n" + "="*80)
    logger.info("TEST 1: PERFECT DUPLICATE DETECTION")
    logger.info("="*80)
    logger.info("\nPrompts:")
    logger.info("  1. 'I prefer dark mode'")
    logger.info("  2. 'I like dark mode'")
    logger.info("  3. 'I enjoy dark mode'")
    logger.info("  4. 'Dark mode is my preference'\n")
    
    prompts = [
        "I prefer dark mode",
        "I like dark mode",
        "I enjoy dark mode",
        "Dark mode is my preference"
    ]
    
    for idx, prompt in enumerate(prompts, 1):
        result = run_behavior_extraction(prompt)
        if result.success:
            store_behavior(result, user_id)
            logger.info(f"  ✓ Processed prompt {idx}")
    
    behaviors = get_user_behaviors(user_id)
    
    logger.info(f"\n📊 RESULT:")
    logger.info(f"  Behaviors stored: {len(behaviors)}")
    logger.info(f"  Expected: 1 (all variations should reinforce same behavior)")
    
    if len(behaviors) == 1:
        behavior = behaviors[0]
        logger.info(f"\n  ✅ SUCCESS - Perfect duplicate detection!")
        logger.info(f"  Behavior: {behavior['behavior_text']}")
        logger.info(f"  Intent: {behavior['intent']}")
        logger.info(f"  Target: {behavior['target']}")
        logger.info(f"  Context: {behavior['context']}")
        logger.info(f"  Reinforcement count: {behavior['reinforcement_count']}")
        logger.info(f"  Credibility: {behavior['credibility']:.3f}")
        return True
    else:
        logger.info(f"\n  ❌ FAILED - Expected 1 behavior, got {len(behaviors)}")
        for b in behaviors:
            logger.info(f"    - {b['behavior_text']}")
        return False


def test_context_reasoning():
    """
    Test 2: Context-Aware Reasoning (General vs Specific)
    
    Scenario: User states general preference, then specific context
    Expected: Specific context reinforces general (GENERALIZATION detected)
    """
    user_id = f"demo_user_ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("\n" + "="*80)
    logger.info("TEST 2: CONTEXT-AWARE REASONING")
    logger.info("="*80)
    logger.info("\nPrompts:")
    logger.info("  1. 'I prefer Python'  (general)")
    logger.info("  2. 'I prefer Python for data science'  (specific)\n")
    
    # First: general preference
    result1 = run_behavior_extraction("I prefer Python")
    if result1.success:
        store_behavior(result1, user_id)
        logger.info("  ✓ Stored general preference")
    
    # Second: specific context
    result2 = run_behavior_extraction("I prefer Python for data science")
    if result2.success:
        store_behavior(result2, user_id)
        logger.info("  ✓ Processed specific context")
    
    behaviors = get_user_behaviors(user_id)
    
    logger.info(f"\n📊 RESULT:")
    logger.info(f"  Behaviors stored: {len(behaviors)}")
    logger.info(f"  Expected: 1 (specific should reinforce general)")
    
    if len(behaviors) == 1:
        behavior = behaviors[0]
        logger.info(f"\n  ✅ SUCCESS - Context reasoning working!")
        logger.info(f"  Behavior: {behavior['behavior_text']}")
        logger.info(f"  Context: {behavior['context']}")
        logger.info(f"  Reinforcement count: {behavior['reinforcement_count']}")
        logger.info(f"  (General context correctly subsumed specific)")
        return True
    else:
        logger.info(f"\n  ❌ FAILED - Expected 1 behavior, got {len(behaviors)}")
        return False


def test_multi_domain_extraction():
    """
    Test 3: Multi-Domain Behavior Extraction
    
    Scenario: Extract behaviors from programming, food, and health domains
    Expected: All intents, targets, contexts correctly extracted
    """
    user_id = f"demo_user_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("\n" + "="*80)
    logger.info("TEST 3: MULTI-DOMAIN EXTRACTION")
    logger.info("="*80)
    logger.info("\nPrompt:")
    prompt = "I prefer dark mode for coding. I cannot eat dairy products. I usually exercise in the morning."
    logger.info(f"  '{prompt}'\n")
    
    result = run_behavior_extraction(prompt)
    if result.success:
        store_behavior(result, user_id)
    
    behaviors = get_user_behaviors(user_id)
    
    logger.info(f"📊 RESULT:")
    logger.info(f"  Behaviors extracted: {len(behaviors)}\n")
    
    expected_intents = {"PREFERENCE", "CONSTRAINT", "HABIT"}
    actual_intents = {b['intent'] for b in behaviors if b['intent']}
    
    success = len(behaviors) == 3 and expected_intents == actual_intents
    
    for idx, behavior in enumerate(behaviors, 1):
        logger.info(f"  {idx}. [{behavior['intent']}] {behavior['target']}")
        logger.info(f"     Context: {behavior['context']}")
        logger.info(f"     Polarity: {behavior['polarity']}")
        logger.info(f"     Text: {behavior['behavior_text'][:60]}...")
    
    if success:
        logger.info(f"\n  ✅ SUCCESS - All domains extracted correctly!")
        logger.info(f"     Programming (PREFERENCE): dark mode")
        logger.info(f"     Food (CONSTRAINT): dairy")
        logger.info(f"     Health (HABIT): exercise")
        return True
    else:
        logger.info(f"\n  ⚠️  Partial success - {len(behaviors)} behaviors extracted")
        return False


def test_polarity_detection():
    """
    Test 4: Polarity-Based Conflict Detection
    
    Scenario: User states positive and negative preferences for same target
    Expected: Both stored, conflict potentially flagged
    """
    user_id = f"demo_user_pol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("\n" + "="*80)
    logger.info("TEST 4: POLARITY-BASED CONFLICT DETECTION")
    logger.info("="*80)
    logger.info("\nPrompts:")
    logger.info("  1. 'I love Python'  (POSITIVE)")
    logger.info("  2. 'I cannot use Python at work'  (NEGATIVE)\n")
    
    result1 = run_behavior_extraction("I love Python")
    if result1.success:
        store_behavior(result1, user_id)
        logger.info("  ✓ Stored positive preference")
    
    result2 = run_behavior_extraction("I cannot use Python at work")
    if result2.success:
        store_behavior(result2, user_id)
        logger.info("  ✓ Processed negative constraint")
    
    behaviors = get_user_behaviors(user_id)
    
    logger.info(f"\n📊 RESULT:")
    logger.info(f"  Behaviors stored: {len(behaviors)}\n")
    
    polarities = {b['polarity'] for b in behaviors if b['polarity']}
    
    for idx, behavior in enumerate(behaviors, 1):
        logger.info(f"  {idx}. [{behavior['intent']}] {behavior['target']}")
        logger.info(f"     Polarity: {behavior['polarity']}")
        logger.info(f"     Text: {behavior['behavior_text'][:60]}...")
    
    if len(polarities) == 2 and "POSITIVE" in polarities and "NEGATIVE" in polarities:
        logger.info(f"\n  ✅ SUCCESS - Opposite polarities detected!")
        logger.info(f"     System can identify conflicts based on polarity")
        return True
    else:
        logger.info(f"\n  ⚠️  Partial success")
        return False


def run_all_demonstrations():
    """Run all demonstration tests"""
    logger.info("\n" + "="*80)
    logger.info("CANONICAL BEHAVIOR SYSTEM - DEMONSTRATION TESTS")
    logger.info("="*80)
    logger.info("Testing key improvements of the canonical behavior refactor\n")
    
    results = []
    
    try:
        results.append(("Duplicate Detection", test_duplicate_detection()))
    except Exception as e:
        logger.error(f"Test 1 failed with exception: {e}")
        results.append(("Duplicate Detection", False))
    
    try:
        results.append(("Context Reasoning", test_context_reasoning()))
    except Exception as e:
        logger.error(f"Test 2 failed with exception: {e}")
        results.append(("Context Reasoning", False))
    
    try:
        results.append(("Multi-Domain Extraction", test_multi_domain_extraction()))
    except Exception as e:
        logger.error(f"Test 3 failed with exception: {e}")
        results.append(("Multi-Domain Extraction", False))
    
    try:
        results.append(("Polarity Detection", test_polarity_detection()))
    except Exception as e:
        logger.error(f"Test 4 failed with exception: {e}")
        results.append(("Polarity Detection", False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80 + "\n")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} - {name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total = len(results)
    
    logger.info(f"\n  Overall: {passed_count}/{total} tests passed ({passed_count/total*100:.0f}%)")
    logger.info("\n" + "="*80 + "\n")
    
    return passed_count == total


if __name__ == "__main__":
    success = run_all_demonstrations()
    sys.exit(0 if success else 1)
