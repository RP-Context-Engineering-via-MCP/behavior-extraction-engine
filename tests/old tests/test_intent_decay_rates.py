"""
Test intent-based decay rate implementation.
Verifies that different intents receive appropriate decay rates.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.credibilityCalculator import get_decay_rate
from config.configurations import INTENT_DECAY_RATES, DEFAULT_DECAY_RATE


def test_intent_based_decay_rates():
    """Test that each intent type returns the correct decay rate."""
    print("\n" + "="*70)
    print("Testing Intent-Based Decay Rates")
    print("="*70)
    
    test_cases = [
        ("HABIT", 0.04),
        ("PREFERENCE", 0.015),
        ("COMMUNICATION", 0.015),
        ("SKILL", 0.005),
        ("CONSTRAINT", 0.001),
    ]
    
    all_passed = True
    
    for intent, expected_rate in test_cases:
        actual_rate = get_decay_rate(intent=intent)
        status = "✓ PASS" if actual_rate == expected_rate else "✗ FAIL"
        print(f"{status} | Intent: {intent:15} | Expected: {expected_rate:6} | Got: {actual_rate:6}")
        
        if actual_rate != expected_rate:
            all_passed = False
    
    print("-"*70)
    return all_passed


def test_unknown_intent():
    """Test that unknown intent falls back to default decay rate."""
    print("\nTesting Unknown Intent Fallback:")
    print("-"*70)
    
    unknown_intent = "UNKNOWN_INTENT"
    actual_rate = get_decay_rate(intent=unknown_intent)
    expected_rate = DEFAULT_DECAY_RATE
    
    status = "✓ PASS" if actual_rate == expected_rate else "✗ FAIL"
    print(f"{status} | Intent: {unknown_intent:15} | Expected (default): {expected_rate:6} | Got: {actual_rate:6}")
    
    return actual_rate == expected_rate


def test_no_intent_provided():
    """Test that no intent falls back to default decay rate."""
    print("\nTesting No Intent Provided:")
    print("-"*70)
    
    actual_rate = get_decay_rate()
    expected_rate = DEFAULT_DECAY_RATE
    
    status = "✓ PASS" if actual_rate == expected_rate else "✗ FAIL"
    print(f"{status} | Intent: None            | Expected (default): {expected_rate:6} | Got: {actual_rate:6}")
    
    return actual_rate == expected_rate


def test_config_values():
    """Test that configuration values are set correctly."""
    print("\nTesting Configuration Values:")
    print("-"*70)
    
    expected_config = {
        "HABIT": 0.04,
        "PREFERENCE": 0.015,
        "COMMUNICATION": 0.015,
        "SKILL": 0.005,
        "CONSTRAINT": 0.001
    }
    
    all_passed = True
    
    for intent, expected_rate in expected_config.items():
        actual_rate = INTENT_DECAY_RATES.get(intent)
        status = "✓ PASS" if actual_rate == expected_rate else "✗ FAIL"
        print(f"{status} | Config[{intent:15}] = {actual_rate}")
        
        if actual_rate != expected_rate:
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all decay rate tests."""
    print("\n" + "="*70)
    print("INTENT-BASED DECAY RATE TEST SUITE")
    print("="*70)
    
    results = {
        "Intent-Based Decay Rates": test_intent_based_decay_rates(),
        "Unknown Intent Fallback": test_unknown_intent(),
        "No Intent Provided": test_no_intent_provided(),
        "Configuration Values": test_config_values()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status} | {test_name}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
