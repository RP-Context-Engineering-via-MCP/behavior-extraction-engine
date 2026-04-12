"""
Test lazy decay mechanism for credibility calculation.
Verifies exponential decay formula: C_current = C_stored × e^(-λ × days)
NOTE: Decay is applied per FULL DAY, not per second.
"""

import sys
import os
import time
import math
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.credibilityCalculator import apply_lazy_decay
from config.configurations import INTENT_DECAY_RATES, DECAY_GRACE_PERIOD_SECONDS


def test_no_decay_in_grace_period():
    """Test that decay is not applied during grace period."""
    print("\n" + "="*70)
    print("Test 1: No Decay During Grace Period")
    print("="*70)
    
    stored_credibility = 0.85
    decay_rate = 0.015
    current_time = int(time.time())
    last_decay_applied_at = current_time + 3600  # 1 hour in future (grace period)
    
    new_cred, decay_applied, time_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=last_decay_applied_at,
        current_time=current_time
    )
    
    no_change = new_cred == stored_credibility
    not_applied = decay_applied == False
    
    print(f"{'✓ PASS' if no_change else '✗ FAIL'} | Credibility unchanged: {stored_credibility} → {new_cred}")
    print(f"{'✓ PASS' if not_applied else '✗ FAIL'} | Decay not applied: {decay_applied}")
    print(f"  Still in grace period (ends in {last_decay_applied_at - current_time} seconds)")
    
    return no_change and not_applied


def test_decay_after_1_day():
    """Test decay calculation after exactly 1 full day."""
    print("\n" + "="*70)
    print("Test 2: Decay After Exactly 1 Full Day (PREFERENCE)")
    print("="*70)
    
    stored_credibility = 0.85
    decay_rate = INTENT_DECAY_RATES["PREFERENCE"]  # 0.015 per day
    current_time = int(time.time())
    one_day_ago = current_time - (24 * 60 * 60)  # Exactly 1 day = 86400 seconds
    
    new_cred, decay_applied, days_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=one_day_ago,
        current_time=current_time
    )
    
    # Calculate expected: C = 0.85 × e^(-0.015 × 1 day)
    expected = stored_credibility * math.exp(-decay_rate * 1)
    matches_expected = abs(new_cred - expected) < 0.0001
    correct_days = days_elapsed == 1
    
    print(f"{'✓ PASS' if decay_applied else '✗ FAIL'} | Decay applied: {decay_applied}")
    print(f"{'✓ PASS' if correct_days else '✗ FAIL'} | Days elapsed: {days_elapsed} (expected 1)")
    print(f"{'✓ PASS' if matches_expected else '✗ FAIL'} | Matches formula: {new_cred:.6f} ≈ {expected:.6f}")
    print(f"  Original:   {stored_credibility:.6f}")
    print(f"  After 1 day: {new_cred:.6f}")
    print(f"  Loss:       {stored_credibility - new_cred:.6f}")
    print(f"  Decay Rate: {decay_rate} per day")
    
    return decay_applied and matches_expected and correct_days


def test_no_decay_partial_day():
    """Test that decay is NOT applied for partial days (< 24 hours)."""
    print("\n" + "="*70)
    print("Test 3: No Decay for Partial Day (23 hours 59 minutes)")
    print("="*70)
    
    stored_credibility = 0.85
    decay_rate = INTENT_DECAY_RATES["PREFERENCE"]
    current_time = int(time.time())
    almost_one_day_ago = current_time - (86400 - 60)  # 23h 59m = 86340 seconds
    
    new_cred, decay_applied, days_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=almost_one_day_ago,
        current_time=current_time
    )
    
    no_change = new_cred == stored_credibility
    not_applied = decay_applied == False
    zero_days = days_elapsed == 0
    
    print(f"{'✓ PASS' if no_change else '✗ FAIL'} | Credibility unchanged: {stored_credibility} → {new_cred}")
    print(f"{'✓ PASS' if not_applied else '✗ FAIL'} | Decay not applied: {decay_applied}")
    print(f"{'✓ PASS' if zero_days else '✗ FAIL'} | Days elapsed: {days_elapsed} (expected 0)")
    print(f"  Time elapsed: {86400 - 60} seconds (< 1 full day)")
    
    return no_change and not_applied and zero_days


def test_decay_after_30_days():
    """Test decay calculation after 30 full days."""
    print("\n" + "="*70)
    print("Test 4: Decay After 30 Full Days (PREFERENCE)")
    print("="*70)
    
    stored_credibility = 0.90
    decay_rate = INTENT_DECAY_RATES["PREFERENCE"]  # 0.015 per day
    current_time = int(time.time())
    thirty_days_ago = current_time - (30 * 24 * 60 * 60)
    
    new_cred, decay_applied, days_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=thirty_days_ago,
        current_time=current_time
    )
    
    # Expected: C = 0.90 × e^(-0.015 × 30)
    expected = stored_credibility * math.exp(-decay_rate * 30)
    matches_expected = abs(new_cred - expected) < 0.0001
    correct_days = days_elapsed == 30
    
    print(f"{'✓ PASS' if decay_applied else '✗ FAIL'} | Decay applied: {decay_applied}")
    print(f"{'✓ PASS' if correct_days else '✗ FAIL'} | Days elapsed: {days_elapsed} (expected 30)")
    print(f"{'✓ PASS' if matches_expected else '✗ FAIL'} | Matches formula: {new_cred:.6f} ≈ {expected:.6f}")
    print(f"  Original:    {stored_credibility:.6f}")
    print(f"  After 30d:   {new_cred:.6f}")
    print(f"  Loss:        {stored_credibility - new_cred:.6f}")
    
    return decay_applied and matches_expected and correct_days


def test_different_intent_decay_rates():
    """Test that different intents decay at different rates over 7 full days."""
    print("\n" + "="*70)
    print("Test 5: Different Intent Decay Rates (7 Full Days)")
    print("="*70)
    
    stored_credibility = 0.80
    current_time = int(time.time())
    seven_days_ago = current_time - (7 * 24 * 60 * 60)
    
    results = {}
    for intent, decay_rate in INTENT_DECAY_RATES.items():
        new_cred, _, days = apply_lazy_decay(
            stored_credibility=stored_credibility,
            decay_rate=decay_rate,
            last_decay_applied_at=seven_days_ago,
            current_time=current_time
        )
        results[intent] = new_cred
        print(f"  {intent:15} (λ={decay_rate:6.3f}/day): {stored_credibility:.4f} → {new_cred:.4f} (loss: {stored_credibility - new_cred:.4f}, {days} days)")
    
    # Verify order: HABIT decays most, CONSTRAINT decays least
    habit_decayed_most = results["HABIT"] < results["PREFERENCE"] < results["SKILL"] < results["CONSTRAINT"]
    
    print(f"\n{'✓ PASS' if habit_decayed_most else '✗ FAIL'} | Decay order correct: HABIT < PREFERENCE < SKILL < CONSTRAINT")
    
    return habit_decayed_most


def test_no_decay_when_null():
    """Test that no decay is applied when last_decay_applied_at is None."""
    print("\n" + "="*70)
    print("Test 6: No Decay When last_decay_applied_at is None")
    print("="*70)
    
    stored_credibility = 0.85
    decay_rate = 0.015
    
    new_cred, decay_applied, days_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=None
    )
    
    no_change = new_cred == stored_credibility
    not_applied = decay_applied == False
    no_days = days_elapsed == 0
    
    print(f"{'✓ PASS' if no_change else '✗ FAIL'} | Credibility unchanged: {stored_credibility} → {new_cred}")
    print(f"{'✓ PASS' if not_applied else '✗ FAIL'} | Decay not applied: {decay_applied}")
    print(f"{'✓ PASS' if no_days else '✗ FAIL'} | Days elapsed is 0: {days_elapsed}")
    
    return no_change and not_applied and no_days


def test_credibility_clamping():
    """Test that credibility is clamped to [0.0, 1.0] range."""
    print("\n" + "="*70)
    print("Test 7: Credibility Clamping")
    print("="*70)
    
    stored_credibility = 0.10
    decay_rate = INTENT_DECAY_RATES["HABIT"]  # Fast decay (0.04 per day)
    current_time = int(time.time())
    # Very old behavior (365 days)
    one_year_ago = current_time - (365 * 24 * 60 * 60)
    
    new_cred, decay_applied, days_elapsed = apply_lazy_decay(
        stored_credibility=stored_credibility,
        decay_rate=decay_rate,
        last_decay_applied_at=one_year_ago,
        current_time=current_time
    )
    
    in_valid_range = 0.0 <= new_cred <= 1.0
    correct_days = days_elapsed == 365
    
    print(f"{'✓ PASS' if in_valid_range else '✗ FAIL'} | Credibility in valid range [0.0, 1.0]: {new_cred:.6f}")
    print(f"{'✓ PASS' if correct_days else '✗ FAIL'} | Days elapsed: {days_elapsed} (expected 365)")
    print(f"  Original (365 days ago): {stored_credibility:.6f}")
    print(f"  After decay:            {new_cred:.6f}")
    print(f"  Decay Rate (HABIT):     {decay_rate} per day")
    
    return in_valid_range and correct_days


def test_exponential_decay_formula():
    """Verify the exponential decay formula is correctly implemented (per day)."""
    print("\n" + "="*70)
    print("Test 8: Exponential Decay Formula Verification (Per Day)")
    print("="*70)
    
    test_cases = [
        (0.80, 0.015, 1),      # 1 day
        (0.90, 0.040, 2),      # 2 days with fast decay
        (0.75, 0.001, 7),      # 7 days with slow decay
        (0.85, 0.015, 30),     # 30 days
    ]
    
    all_match = True
    for stored_cred, decay_rate, days in test_cases:
        current_time = int(time.time())
        last_applied = current_time - (days * 24 * 60 * 60)
        
        new_cred, _, days_elapsed = apply_lazy_decay(
            stored_credibility=stored_cred,
            decay_rate=decay_rate,
            last_decay_applied_at=last_applied,
            current_time=current_time
        )
        
        # Manual calculation: C = C_stored × e^(-λ × days)
        expected = stored_cred * math.exp(-decay_rate * days)
        matches = abs(new_cred - expected) < 0.000001
        correct_days = days_elapsed == days
        
        print(f"  {'✓' if matches and correct_days else '✗'} C={stored_cred:.2f}, λ={decay_rate:.3f}/day, {days} days: {new_cred:.6f} ≈ {expected:.6f}")
        
        if not (matches and correct_days):
            all_match = False
    
    print(f"\n{'✓ PASS' if all_match else '✗ FAIL'} | All calculations match exponential formula")
    
    return all_match


def run_all_tests():
    """Run all lazy decay tests."""
    print("\n" + "="*70)
    print("LAZY DECAY MECHANISM TEST SUITE")
    print("Formula: C_current = C_stored × e^(-λ × days)")
    print("NOTE: Decay applied per FULL DAY only (86400 seconds)")
    print("="*70)
    
    results = {
        "No Decay in Grace Period": test_no_decay_in_grace_period(),
        "Decay After 1 Full Day": test_decay_after_1_day(),
        "No Decay for Partial Day": test_no_decay_partial_day(),
        "Decay After 30 Days": test_decay_after_30_days(),
        "Different Intent Rates": test_different_intent_decay_rates(),
        "No Decay When Null": test_no_decay_when_null(),
        "Credibility Clamping": test_credibility_clamping(),
        "Exponential Formula": test_exponential_decay_formula()
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
        print("✓ ALL TESTS PASSED - Lazy Decay Implementation Complete!")
        print("\nKey Formula: C_current = C_stored × e^(-λ × days)")
        print("  where λ = decay_rate (intent-based, per day)")
        print("        days = full days elapsed (floor division)")
        print("\n⚠️  IMPORTANT: Decay only applied for FULL DAYS (≥86400 seconds)")
        print("    Example: Last decay at 6am today, retrieval at 5:59am tomorrow = NO DECAY")
        print("             Last decay at 6am today, retrieval at 6:00am tomorrow = 1 DAY DECAY")
    else:
        print("✗ SOME TESTS FAILED - Please review implementation")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
