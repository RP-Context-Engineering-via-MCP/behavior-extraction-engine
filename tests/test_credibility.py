"""
Test suite for credibility calculation module.

Tests initial credibility calculation with various behavior scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.credibilityCalculator import (
    calculate_initial_credibility,
    should_store_behavior,
    get_decay_rate
)


def test_high_quality_behavior():
    """Test credibility for high quality behavior (high conf + clarity + specific)"""
    print("\n" + "="*80)
    print("TEST 1: High Quality Behavior")
    print("="*80)
    
    behavior = "prefers Python over JavaScript for backend development"
    confidence = 0.95
    clarity = 1.0
    linguistic_strength = 0.7
    
    credibility = calculate_initial_credibility(confidence, clarity, linguistic_strength, behavior)
    should_store = should_store_behavior(credibility)
    
    print(f"Behavior: '{behavior}'")
    print(f"Confidence: {confidence}")
    print(f"Clarity: {clarity}")
    print(f"Word count: {len(behavior.split())}")
    print(f"\n→ Initial Credibility: {credibility}")
    print(f"→ Should Store: {should_store}")
    print(f"→ Expected: High credibility (>0.8), should store")
    
    assert credibility > 0.8, f"Expected high credibility, got {credibility}"
    assert should_store, "High quality behavior should be stored"
    print("✓ PASSED")


def test_medium_quality_behavior():
    """Test credibility for medium quality behavior"""
    print("\n" + "="*80)
    print("TEST 2: Medium Quality Behavior")
    print("="*80)
    
    behavior = "likes code examples"
    confidence = 0.75
    clarity = 0.7
    linguistic_strength = 0.5
    
    credibility = calculate_initial_credibility(confidence, clarity, linguistic_strength, behavior)
    should_store = should_store_behavior(credibility)
    
    print(f"Behavior: '{behavior}'")
    print(f"Confidence: {confidence}")
    print(f"Clarity: {clarity}")
    print(f"Word count: {len(behavior.split())}")
    print(f"\n→ Initial Credibility: {credibility}")
    print(f"→ Should Store: {should_store}")
    print(f"→ Expected: Medium credibility (0.6-0.8), should store")
    
    assert 0.6 <= credibility <= 0.8, f"Expected medium credibility, got {credibility}"
    assert should_store, "Medium quality behavior should be stored"
    print("✓ PASSED")


def test_low_quality_behavior():
    """Test credibility for low quality behavior (should be filtered)"""
    print("\n" + "="*80)
    print("TEST 3: Low Quality Behavior (Below Threshold)")
    print("="*80)
    
    behavior = "uses tools"
    confidence = 0.4
    clarity = 0.3
    linguistic_strength = 0.3
    
    credibility = calculate_initial_credibility(confidence, clarity, linguistic_strength, behavior)
    should_store = should_store_behavior(credibility)
    
    print(f"Behavior: '{behavior}'")
    print(f"Confidence: {confidence}")
    print(f"Clarity: {clarity}")
    print(f"Word count: {len(behavior.split())}")
    print(f"\n→ Initial Credibility: {credibility}")
    print(f"→ Should Store: {should_store}")
    print(f"→ Expected: Low credibility (<0.5), should NOT store")
    
    assert credibility < 0.5, f"Expected low credibility, got {credibility}"
    assert not should_store, "Low quality behavior should be filtered out"
    print("✓ PASSED")


def test_very_specific_behavior():
    """Test that very specific behaviors get bonus"""
    print("\n" + "="*80)
    print("TEST 4: Very Specific Behavior (Specificity Bonus)")
    print("="*80)
    
    behavior = "prefers using React with TypeScript and functional components with hooks for building modern web applications"
    confidence = 0.85
    clarity = 0.9
    linguistic_strength = 0.7
    
    credibility = calculate_initial_credibility(confidence, clarity, linguistic_strength, behavior)
    
    # Compare with shorter version
    short_behavior = "uses React"
    credibility_short = calculate_initial_credibility(confidence, clarity, short_behavior)
    
    print(f"Long Behavior: '{behavior}'")
    print(f"Word count: {len(behavior.split())}")
    print(f"→ Credibility: {credibility}")
    
    print(f"\nShort Behavior: '{short_behavior}'")
    print(f"Word count: {len(short_behavior.split())}")
    print(f"→ Credibility: {credibility_short}")
    
    print(f"\n→ Difference: {credibility - credibility_short:.4f}")
    print(f"→ Expected: Long behavior should have higher credibility due to specificity bonus")
    
    assert credibility > credibility_short, "Specific behavior should have higher credibility"
    print("✓ PASSED")


def test_edge_cases():
    """Test edge cases and input validation"""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases")
    print("="*80)
    
    # Test with out-of-range confidence (should clamp)
    credibility1 = calculate_initial_credibility(1.5, 0.8, 0.2, "test behavior")
    print(f"Out-of-range confidence (1.5): {credibility1} (should clamp to 1.0)")
    assert credibility1 <= 1.0, "Should clamp high confidence"
    
    # Test with negative clarity (should clamp)
    credibility2 = calculate_initial_credibility(0.7, -0.2, 0.5, "test behavior")
    print(f"Negative clarity (-0.2): {credibility2} (should clamp to 0.0)")
    assert credibility2 >= 0.0, "Should clamp negative clarity"
    
    # Test with empty behavior text
    credibility3 = calculate_initial_credibility(0.9, 0.9, 0.5, "")
    print(f"Empty behavior text: {credibility3} (should return 0.0)")
    assert credibility3 == 0.0, "Empty text should return 0.0"
    
    # Test with single word behavior
    credibility4 = calculate_initial_credibility(0.9, 0.9, 0.1, "preference")
    print(f"Single word behavior: {credibility4}")
    assert 0.0 <= credibility4 <= 1.0, "Should return valid credibility"
    
    print("✓ PASSED")


def test_decay_rate():
    """Test decay rate retrieval"""
    print("\n" + "="*80)
    print("TEST 6: Decay Rate")
    print("="*80)
    
    behavior = "prefers Python"
    decay_rate = get_decay_rate(behavior)
    
    print(f"Behavior: '{behavior}'")
    print(f"→ Decay Rate: {decay_rate}")
    print(f"→ Expected: Default decay rate (0.015)")
    
    assert decay_rate == 0.015, f"Expected 0.015, got {decay_rate}"
    print("✓ PASSED")


def main():
    print("\n" + "#"*80)
    print("# CREDIBILITY CALCULATION TEST SUITE")
    print("#"*80)
    print("\nTesting credibility calculation with various behavior scenarios...\n")
    
    try:
        test_high_quality_behavior()
        test_medium_quality_behavior()
        test_low_quality_behavior()
        test_very_specific_behavior()
        test_edge_cases()
        test_decay_rate()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        raise


if __name__ == "__main__":
    main()
