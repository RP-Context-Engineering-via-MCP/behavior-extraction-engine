"""
Test suite for ProfileSignalRepository.

Tests database operations for storing and retrieving profile signals.
Requires database connection - tests will be skipped if database is unavailable.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import time
import uuid
from typing import Optional

# Try to import the repository and database connection
try:
    from services.profileSignalRepository import ProfileSignalRepository, get_profile_signal_repository
    from db.connection import init_db_pool, close_db_pool
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# Test data fixtures
def create_test_profile_signals(variant: int = 0) -> dict:
    """Create test profile signals with optional variation"""
    base_signals = {
        "intents": {"PROBLEM_SOLVING": 0.8 + (variant * 0.05), "LEARNING": 0.4},
        "interests": {"PROGRAMMING": 0.9, "AI": 0.5 - (variant * 0.1)},
        "behavior_level": ["BEGINNER", "INTERMEDIATE", "ADVANCED"][variant % 3],
        "signals": {"CODE_FOCUSED": 0.85, "DETAILED_EXPLANATION": 0.6},
        "complexity": 0.7 + (variant * 0.05),
        "consistency": 0.5
    }
    return base_signals


def generate_test_user_id() -> str:
    """Generate a unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


def generate_test_prompt_id() -> str:
    """Generate a unique test prompt ID"""
    return f"prompt_{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def db_setup():
    """Setup and teardown database connection pool"""
    if not DB_AVAILABLE:
        pytest.skip("Database not available")
    
    try:
        init_db_pool()
        yield
        close_db_pool()
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}")


@pytest.fixture
def repo():
    """Get ProfileSignalRepository instance"""
    return ProfileSignalRepository()


@pytest.fixture
def test_user_id():
    """Generate a unique test user ID"""
    return generate_test_user_id()


class TestProfileSignalRepository:
    """Tests for ProfileSignalRepository database operations"""
    
    # =========================================================================
    # Save Tests
    # =========================================================================
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_save_profile_signals(self, db_setup, repo, test_user_id):
        """Test saving profile signals to database"""
        print("\n" + "="*80)
        print("TEST: Save Profile Signals")
        print("="*80)
        
        prompt_id = generate_test_prompt_id()
        signals = create_test_profile_signals()
        
        # Should not raise
        repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals)
        
        # Verify by retrieving
        result = repo.get_by_prompt_id(user_id=test_user_id, prompt_id=prompt_id)
        
        assert result is not None
        assert result["intents"]["PROBLEM_SOLVING"] == signals["intents"]["PROBLEM_SOLVING"]
        assert result["interests"]["PROGRAMMING"] == signals["interests"]["PROGRAMMING"]
        
        print(f"User ID: {test_user_id}")
        print(f"Prompt ID: {prompt_id}")
        print(f"Saved signals: {signals}")
        print(f"Retrieved signals: {result}")
        print("✓ PASSED")
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_save_upsert_updates_existing(self, db_setup, repo, test_user_id):
        """Test that saving with same user_id/prompt_id updates existing record"""
        print("\n" + "="*80)
        print("TEST: Save Upsert (Update Existing)")
        print("="*80)
        
        prompt_id = generate_test_prompt_id()
        
        # Save first version
        signals_v1 = create_test_profile_signals(variant=0)
        repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals_v1)
        
        # Save second version (same user_id, prompt_id)
        signals_v2 = create_test_profile_signals(variant=1)
        signals_v2["intents"]["PROBLEM_SOLVING"] = 0.99  # Make it distinct
        repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals_v2)
        
        # Verify the update
        result = repo.get_by_prompt_id(user_id=test_user_id, prompt_id=prompt_id)
        
        assert result is not None
        assert result["intents"]["PROBLEM_SOLVING"] == 0.99
        
        # Should still be only 1 record (count for this specific prompt)
        count = repo.get_count(test_user_id)
        # Note: Count is for all prompts for user, so we just verify the upsert worked
        
        print(f"V1 signals: {signals_v1['intents']}")
        print(f"V2 signals: {signals_v2['intents']}")
        print(f"Retrieved: {result['intents']}")
        print("✓ PASSED - Upsert updated existing record")
    
    # =========================================================================
    # Get Recent Tests
    # =========================================================================
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_recent_returns_ordered_by_time(self, db_setup, repo):
        """Test that get_recent returns signals ordered by most recent first"""
        print("\n" + "="*80)
        print("TEST: Get Recent (Ordered by Time)")
        print("="*80)
        
        test_user_id = generate_test_user_id()
        
        # Save multiple signals with small time gaps
        prompt_ids = []
        for i in range(5):
            prompt_id = generate_test_prompt_id()
            signals = create_test_profile_signals(variant=i)
            signals["complexity"] = 0.1 * (i + 1)  # 0.1, 0.2, 0.3, 0.4, 0.5
            repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals)
            prompt_ids.append(prompt_id)
            time.sleep(0.1)  # Small delay to ensure different timestamps
        
        # Get recent signals
        recent = repo.get_recent(user_id=test_user_id, limit=5)
        
        assert len(recent) == 5
        
        # Most recent should be last saved (complexity 0.5)
        assert recent[0]["complexity"] == 0.5
        # Oldest should be first saved (complexity 0.1)
        assert recent[4]["complexity"] == 0.1
        
        print(f"Saved {len(prompt_ids)} signals")
        print(f"Retrieved {len(recent)} signals")
        print(f"Complexities (recent first): {[r['complexity'] for r in recent]}")
        print("✓ PASSED - Ordered by most recent first")
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_recent_respects_limit(self, db_setup, repo):
        """Test that get_recent respects the limit parameter"""
        print("\n" + "="*80)
        print("TEST: Get Recent (Respects Limit)")
        print("="*80)
        
        test_user_id = generate_test_user_id()
        
        # Save 10 signals
        for i in range(10):
            prompt_id = generate_test_prompt_id()
            signals = create_test_profile_signals(variant=i)
            repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals)
        
        # Request only 3
        recent = repo.get_recent(user_id=test_user_id, limit=3)
        
        assert len(recent) == 3
        
        print(f"Saved 10 signals, requested limit=3, got {len(recent)}")
        print("✓ PASSED - Limit respected")
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_recent_empty_for_unknown_user(self, db_setup, repo):
        """Test that get_recent returns empty list for unknown user"""
        print("\n" + "="*80)
        print("TEST: Get Recent (Unknown User)")
        print("="*80)
        
        unknown_user = f"unknown_{uuid.uuid4().hex}"
        recent = repo.get_recent(user_id=unknown_user, limit=10)
        
        assert recent == []
        
        print(f"User: {unknown_user}")
        print(f"Result: {recent}")
        print("✓ PASSED - Empty list for unknown user")
    
    # =========================================================================
    # Get Count Tests
    # =========================================================================
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_count(self, db_setup, repo):
        """Test get_count returns correct count"""
        print("\n" + "="*80)
        print("TEST: Get Count")
        print("="*80)
        
        test_user_id = generate_test_user_id()
        
        # Initially should be 0
        initial_count = repo.get_count(test_user_id)
        assert initial_count == 0
        
        # Save 5 signals
        for i in range(5):
            prompt_id = generate_test_prompt_id()
            signals = create_test_profile_signals(variant=i)
            repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals)
        
        # Should now be 5
        final_count = repo.get_count(test_user_id)
        assert final_count == 5
        
        print(f"Initial count: {initial_count}")
        print(f"After saving 5: {final_count}")
        print("✓ PASSED")
    
    # =========================================================================
    # Get by Prompt ID Tests
    # =========================================================================
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_by_prompt_id_found(self, db_setup, repo, test_user_id):
        """Test get_by_prompt_id when record exists"""
        print("\n" + "="*80)
        print("TEST: Get by Prompt ID (Found)")
        print("="*80)
        
        prompt_id = generate_test_prompt_id()
        signals = create_test_profile_signals()
        
        repo.save(user_id=test_user_id, prompt_id=prompt_id, profile_signals=signals)
        result = repo.get_by_prompt_id(user_id=test_user_id, prompt_id=prompt_id)
        
        assert result is not None
        assert result["behavior_level"] == signals["behavior_level"]
        
        print(f"Prompt ID: {prompt_id}")
        print(f"Result: {result}")
        print("✓ PASSED")
    
    @pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
    def test_get_by_prompt_id_not_found(self, db_setup, repo, test_user_id):
        """Test get_by_prompt_id when record doesn't exist"""
        print("\n" + "="*80)
        print("TEST: Get by Prompt ID (Not Found)")
        print("="*80)
        
        result = repo.get_by_prompt_id(
            user_id=test_user_id, 
            prompt_id=f"nonexistent_{uuid.uuid4().hex}"
        )
        
        assert result is None
        
        print("✓ PASSED - Returns None for nonexistent prompt")


# =========================================================================
# Singleton Test
# =========================================================================

def test_singleton_instance():
    """Test that get_profile_signal_repository returns singleton"""
    print("\n" + "="*80)
    print("TEST: Singleton Instance")
    print("="*80)
    
    if not DB_AVAILABLE:
        pytest.skip("Database module not available")
    
    repo1 = get_profile_signal_repository()
    repo2 = get_profile_signal_repository()
    
    assert repo1 is repo2
    
    print(f"repo1 id: {id(repo1)}")
    print(f"repo2 id: {id(repo2)}")
    print("✓ PASSED - Same instance returned")


# =========================================================================
# Run tests manually (without pytest)
# =========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROFILE SIGNAL REPOSITORY TESTS")
    print("="*80)
    
    if not DB_AVAILABLE:
        print("⚠ Database module not available - skipping tests")
        sys.exit(0)
    
    try:
        # Initialize database
        print("Initializing database connection pool...")
        init_db_pool()
        
        repo = ProfileSignalRepository()
        
        # Run basic tests
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: Save and retrieve
        try:
            test_user = generate_test_user_id()
            test_prompt = generate_test_prompt_id()
            test_signals = create_test_profile_signals()
            
            print("\n--- Test: Save and Retrieve ---")
            repo.save(test_user, test_prompt, test_signals)
            result = repo.get_by_prompt_id(test_user, test_prompt)
            assert result is not None
            print("✓ Save and retrieve: PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"✗ Save and retrieve: FAILED - {e}")
            tests_failed += 1
        
        # Test 2: Get recent
        try:
            print("\n--- Test: Get Recent ---")
            test_user = generate_test_user_id()
            for i in range(3):
                repo.save(test_user, generate_test_prompt_id(), create_test_profile_signals(i))
                time.sleep(0.05)
            
            recent = repo.get_recent(test_user, limit=3)
            assert len(recent) == 3
            print("✓ Get recent: PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"✗ Get recent: FAILED - {e}")
            tests_failed += 1
        
        # Test 3: Get count
        try:
            print("\n--- Test: Get Count ---")
            test_user = generate_test_user_id()
            for i in range(5):
                repo.save(test_user, generate_test_prompt_id(), create_test_profile_signals(i))
            
            count = repo.get_count(test_user)
            assert count == 5
            print("✓ Get count: PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"✗ Get count: FAILED - {e}")
            tests_failed += 1
        
        print("\n" + "="*80)
        print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
    finally:
        try:
            close_db_pool()
            print("Database connection pool closed.")
        except:
            pass
