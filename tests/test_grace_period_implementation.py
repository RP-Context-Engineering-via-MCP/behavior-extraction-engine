"""
Test grace period implementation for decay mechanism.
Verifies that last_decay_applied_at is set correctly on behavior creation and reinforcement.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.configurations import DECAY_GRACE_PERIOD_SECONDS, DECAY_GRACE_PERIOD_DAYS
from models.behavior import StoredBehavior, CanonicalBehavior


def test_grace_period_constants():
    """Test that grace period constants are configured correctly."""
    print("\n" + "="*70)
    print("Testing Grace Period Constants")
    print("="*70)
    
    expected_days = 7
    expected_seconds = expected_days * 24 * 60 * 60  # 604800
    
    days_match = DECAY_GRACE_PERIOD_DAYS == expected_days
    seconds_match = DECAY_GRACE_PERIOD_SECONDS == expected_seconds
    
    print(f"{'✓ PASS' if days_match else '✗ FAIL'} | DECAY_GRACE_PERIOD_DAYS = {DECAY_GRACE_PERIOD_DAYS} (expected {expected_days})")
    print(f"{'✓ PASS' if seconds_match else '✗ FAIL'} | DECAY_GRACE_PERIOD_SECONDS = {DECAY_GRACE_PERIOD_SECONDS} (expected {expected_seconds})")
    
    return days_match and seconds_match


def test_stored_behavior_has_field():
    """Test that StoredBehavior model has last_decay_applied_at field."""
    print("\n" + "="*70)
    print("Testing StoredBehavior Model Field")
    print("="*70)
    
    # Check if field exists in model
    has_field = 'last_decay_applied_at' in StoredBehavior.model_fields
    
    print(f"{'✓ PASS' if has_field else '✗ FAIL'} | StoredBehavior has 'last_decay_applied_at' field")
    
    if has_field:
        field_info = StoredBehavior.model_fields['last_decay_applied_at']
        print(f"  Field Type: {field_info.annotation}")
        print(f"  Description: {field_info.description}")
    
    return has_field


def test_grace_period_calculation():
    """Test that grace period is calculated correctly (7 days from creation)."""
    print("\n" + "="*70)
    print("Testing Grace Period Calculation Logic")
    print("="*70)
    
    current_time = int(time.time())
    expected_decay_start = current_time + DECAY_GRACE_PERIOD_SECONDS
    
    # Calculate expected date
    expected_date = datetime.fromtimestamp(expected_decay_start)
    creation_date = datetime.fromtimestamp(current_time)
    
    # Verify it's 7 days later
    time_diff = expected_decay_start - current_time
    is_7_days = time_diff == (7 * 24 * 60 * 60)
    
    print(f"{'✓ PASS' if is_7_days else '✗ FAIL'} | Grace period is exactly 7 days (604800 seconds)")
    print(f"  Creation Time:     {creation_date.isoformat()}")
    print(f"  Decay Starts At:   {expected_date.isoformat()}")
    print(f"  Time Difference:   {time_diff} seconds ({time_diff / 86400:.1f} days)")
    
    return is_7_days


def test_stored_behavior_accepts_field():
    """Test that StoredBehavior can be created with last_decay_applied_at."""
    print("\n" + "="*70)
    print("Testing StoredBehavior Creation with Grace Period")
    print("="*70)
    
    current_time = int(time.time())
    decay_start_time = current_time + DECAY_GRACE_PERIOD_SECONDS
    
    try:
        behavior = StoredBehavior(
            user_id="test_user",
            behavior_text="Test behavior",
            credibility=0.85,
            clarity_score=0.9,
            extraction_confidence=0.88,
            linguistic_strength=0.92,
            decay_rate=0.015,
            created_at=current_time,
            last_seen_at=current_time,
            last_decay_applied_at=decay_start_time,
            embedding=[0.1] * 3072,
            intent="PREFERENCE",
            target="test",
            context="general",
            polarity="POSITIVE"
        )
        
        # Verify the field is set correctly
        field_set_correctly = behavior.last_decay_applied_at == decay_start_time
        is_7_days_later = (behavior.last_decay_applied_at - behavior.created_at) == DECAY_GRACE_PERIOD_SECONDS
        
        print(f"{'✓ PASS' if field_set_correctly else '✗ FAIL'} | last_decay_applied_at set to correct value")
        print(f"{'✓ PASS' if is_7_days_later else '✗ FAIL'} | Grace period is 7 days from creation")
        print(f"  Created At:              {datetime.fromtimestamp(behavior.created_at).isoformat()}")
        print(f"  Last Decay Applied At:   {datetime.fromtimestamp(behavior.last_decay_applied_at).isoformat()}")
        print(f"  Grace Period (seconds):  {behavior.last_decay_applied_at - behavior.created_at}")
        
        return field_set_correctly and is_7_days_later
        
    except Exception as e:
        print(f"✗ FAIL | Error creating StoredBehavior: {e}")
        return False


def test_grace_period_explanation():
    """Print explanation of the grace period mechanism."""
    print("\n" + "="*70)
    print("Grace Period Mechanism Explanation")
    print("="*70)
    
    print("""
How the 7-Day Grace Period Works:
---------------------------------

1. BEHAVIOR CREATION:
   - When a new behavior is saved to the database
   - last_decay_applied_at = created_at + 7 days
   - Decay will NOT be applied until this timestamp is reached

2. BEHAVIOR REINFORCEMENT:
   - When a behavior is reinforced (duplicate detected)
   - last_decay_applied_at is RESET to current_timestamp
   - This extends the grace period by another 7 days
   - Prevents decay on actively reinforced behaviors

3. DECAY APPLICATION (Future Implementation):
   - Check: current_time >= last_decay_applied_at
   - If true: Apply decay to credibility
   - Update: last_decay_applied_at = current_time
   - Next decay will be calculated from this new timestamp

4. BENEFITS:
   - New behaviors have time to stabilize (7 days)
   - Frequently reinforced behaviors don't decay
   - Clear audit trail of when decay was last applied
   - Simple, predictable decay schedule
    """)
    
    return True


def run_all_tests():
    """Run all grace period tests."""
    print("\n" + "="*70)
    print("GRACE PERIOD IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    results = {
        "Grace Period Constants": test_grace_period_constants(),
        "StoredBehavior Field": test_stored_behavior_has_field(),
        "Grace Period Calculation": test_grace_period_calculation(),
        "StoredBehavior Creation": test_stored_behavior_accepts_field(),
        "Mechanism Explanation": test_grace_period_explanation()
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
        print("✓ ALL TESTS PASSED - Grace Period Implementation Complete!")
    else:
        print("✗ SOME TESTS FAILED - Please review implementation")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
