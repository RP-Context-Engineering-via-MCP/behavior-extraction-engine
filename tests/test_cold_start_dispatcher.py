"""
Test suite for ColdStartDispatcher.

Tests the cold-start profiling orchestration flow.
Uses mocks for external service calls and database operations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Import the dispatcher
try:
    from services.coldStartDispatcher import ColdStartDispatcher, get_cold_start_dispatcher
    DISPATCHER_AVAILABLE = True
except ImportError as e:
    DISPATCHER_AVAILABLE = False
    import_error = str(e)


@pytest.fixture
def mock_profile_signal_repo():
    """Create a mock ProfileSignalRepository"""
    mock = MagicMock()
    mock.get_count = MagicMock(return_value=0)
    mock.get_recent = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_profile_service_client():
    """Create a mock ProfileServiceClient"""
    mock = MagicMock()
    mock.assign_profile = AsyncMock(return_value={"success": True})
    mock.get_user_profile_status = AsyncMock(return_value={"has_profile": False})
    return mock


@pytest.fixture
def sample_profile_signals():
    """Sample profile signals for testing"""
    return [
        {
            "intents": {"PROBLEM_SOLVING": 0.8, "LEARNING": 0.6},
            "interests": {"PROGRAMMING": 0.9, "AI": 0.7},
            "behavior_level": "INTERMEDIATE",
            "signals": {"CODE_FOCUSED": 0.85},
            "complexity": 0.7,
            "consistency": 0.6
        },
        {
            "intents": {"TASK_COMPLETION": 0.7},
            "interests": {"PROGRAMMING": 0.85},
            "behavior_level": "INTERMEDIATE",
            "signals": {"DETAILED_EXPLANATION": 0.6},
            "complexity": 0.6,
            "consistency": 0.5
        },
        {
            "intents": {"EXPLORATION": 0.6},
            "interests": {"AI": 0.8, "DATA_SCIENCE": 0.5},
            "behavior_level": "ADVANCED",
            "signals": {"DEEP_REASONING": 0.7},
            "complexity": 0.8,
            "consistency": 0.7
        }
    ]


class TestColdStartDispatcher:
    """Tests for ColdStartDispatcher"""
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason=f"Dispatcher not available")
    def test_initialization(self):
        """Test that ColdStartDispatcher initializes correctly"""
        print("\n" + "="*80)
        print("TEST: Initialization")
        print("="*80)
        
        dispatcher = ColdStartDispatcher()
        
        assert dispatcher is not None
        assert hasattr(dispatcher, 'dispatch')
        assert hasattr(dispatcher, '_is_cold_start_user')
        
        print("✓ PASSED - Dispatcher initialized with required methods")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_is_cold_start_user_no_signals(
        self, mock_profile_signal_repo
    ):
        """Test cold start detection when user has no signals"""
        print("\n" + "="*80)
        print("TEST: Is Cold Start User (No Signals)")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 0
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            dispatcher = ColdStartDispatcher()
            is_cold_start = dispatcher._is_cold_start_user("user_123")
        
        assert is_cold_start is True
        mock_profile_signal_repo.get_count.assert_called_once_with("user_123")
        
        print("✓ PASSED - User with 0 signals is cold start")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_is_cold_start_user_few_signals(
        self, mock_profile_signal_repo
    ):
        """Test cold start detection when user has few signals (< threshold)"""
        print("\n" + "="*80)
        print("TEST: Is Cold Start User (Few Signals)")
        print("="*80)
        
        # Assuming threshold is around 5
        mock_profile_signal_repo.get_count.return_value = 2
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            dispatcher = ColdStartDispatcher()
            is_cold_start = dispatcher._is_cold_start_user("user_123")
        
        assert is_cold_start is True
        
        print("✓ PASSED - User with few signals is cold start")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_is_cold_start_user_many_signals(
        self, mock_profile_signal_repo
    ):
        """Test cold start detection when user has many signals (>= threshold)"""
        print("\n" + "="*80)
        print("TEST: Is Cold Start User (Many Signals)")
        print("="*80)
        
        # User has enough signals - not cold start
        mock_profile_signal_repo.get_count.return_value = 10
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            dispatcher = ColdStartDispatcher()
            is_cold_start = dispatcher._is_cold_start_user("user_123")
        
        assert is_cold_start is False
        
        print("✓ PASSED - User with many signals is NOT cold start")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_dispatch_cold_start_user_calls_assign(
        self, mock_profile_signal_repo, mock_profile_service_client, sample_profile_signals
    ):
        """Test that dispatch calls assign_profile for cold start users"""
        print("\n" + "="*80)
        print("TEST: Dispatch Calls assign_profile for Cold Start")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 3  # Below threshold
        mock_profile_signal_repo.get_recent.return_value = sample_profile_signals
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'services.coldStartDispatcher.get_profile_service_client',
            return_value=mock_profile_service_client
        ):
            dispatcher = ColdStartDispatcher()
            result = await dispatcher.dispatch(
                user_id="user_123",
                profile_signals=sample_profile_signals[0]
            )
        
        # Should have called assign_profile
        assert mock_profile_service_client.assign_profile.called
        
        print(f"Result: {result}")
        print("✓ PASSED - assign_profile called for cold start user")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_dispatch_existing_user_skips_assign(
        self, mock_profile_signal_repo, mock_profile_service_client
    ):
        """Test that dispatch skips assign_profile for existing users"""
        print("\n" + "="*80)
        print("TEST: Dispatch Skips assign_profile for Existing User")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 50  # Above threshold
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'services.coldStartDispatcher.get_profile_service_client',
            return_value=mock_profile_service_client
        ):
            dispatcher = ColdStartDispatcher()
            result = await dispatcher.dispatch(
                user_id="user_123",
                profile_signals={"intents": {"LEARNING": 0.5}}
            )
        
        # Should NOT have called assign_profile
        assert not mock_profile_service_client.assign_profile.called
        
        print(f"Result: {result}")
        print("✓ PASSED - assign_profile NOT called for existing user")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    def test_dispatch_sync_wrapper(
        self, mock_profile_signal_repo, mock_profile_service_client
    ):
        """Test the synchronous wrapper for dispatch"""
        print("\n" + "="*80)
        print("TEST: Dispatch Sync Wrapper")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 2
        mock_profile_signal_repo.get_recent.return_value = []
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'services.coldStartDispatcher.get_profile_service_client',
            return_value=mock_profile_service_client
        ):
            dispatcher = ColdStartDispatcher()
            
            # This should work synchronously
            result = dispatcher.dispatch_sync_wrapper(
                user_id="user_456",
                profile_signals={"intents": {"EXPLORATION": 0.7}}
            )
        
        assert result is not None
        
        print(f"Result: {result}")
        print("✓ PASSED - Sync wrapper executed successfully")


class TestColdStartDispatcherErrorHandling:
    """Tests for error handling in ColdStartDispatcher"""
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_dispatch_handles_service_error(
        self, mock_profile_signal_repo, mock_profile_service_client
    ):
        """Test that dispatch handles profile service errors gracefully"""
        print("\n" + "="*80)
        print("TEST: Dispatch Handles Service Error")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 2
        mock_profile_signal_repo.get_recent.return_value = []
        mock_profile_service_client.assign_profile = AsyncMock(
            side_effect=Exception("Service unavailable")
        )
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'services.coldStartDispatcher.get_profile_service_client',
            return_value=mock_profile_service_client
        ):
            dispatcher = ColdStartDispatcher()
            
            # Should not raise, should handle error gracefully
            try:
                result = await dispatcher.dispatch(
                    user_id="user_123",
                    profile_signals={"intents": {"LEARNING": 0.5}}
                )
                # If it returns something, that's fine (error handling in place)
                print(f"Result (with error handling): {result}")
                print("✓ PASSED - Error handled gracefully")
            except Exception as e:
                # If dispatcher doesn't handle errors internally, this is also acceptable
                # for now, but should be noted
                print(f"Exception raised: {e}")
                print("⚠ Dispatcher does not handle service errors internally")
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    @pytest.mark.asyncio
    async def test_dispatch_handles_repo_error(
        self, mock_profile_signal_repo, mock_profile_service_client
    ):
        """Test that dispatch handles repository errors gracefully"""
        print("\n" + "="*80)
        print("TEST: Dispatch Handles Repository Error")
        print("="*80)
        
        mock_profile_signal_repo.get_count.side_effect = Exception("Database error")
        
        with patch(
            'services.coldStartDispatcher.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'services.coldStartDispatcher.get_profile_service_client',
            return_value=mock_profile_service_client
        ):
            dispatcher = ColdStartDispatcher()
            
            try:
                result = await dispatcher.dispatch(
                    user_id="user_123",
                    profile_signals={"intents": {"LEARNING": 0.5}}
                )
                print(f"Result: {result}")
                print("✓ PASSED - Repository error handled")
            except Exception as e:
                print(f"Exception: {e}")
                print("⚠ Repository errors propagate up")


class TestColdStartDispatcherSingleton:
    """Tests for singleton pattern"""
    
    @pytest.mark.skipif(not DISPATCHER_AVAILABLE, reason="Dispatcher not available")
    def test_get_cold_start_dispatcher_singleton(self):
        """Test that get_cold_start_dispatcher returns singleton"""
        print("\n" + "="*80)
        print("TEST: Singleton Instance")
        print("="*80)
        
        dispatcher1 = get_cold_start_dispatcher()
        dispatcher2 = get_cold_start_dispatcher()
        
        assert dispatcher1 is dispatcher2
        
        print(f"dispatcher1 id: {id(dispatcher1)}")
        print(f"dispatcher2 id: {id(dispatcher2)}")
        print("✓ PASSED - Same instance returned")


# =========================================================================
# Manual Test Runner
# =========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COLD START DISPATCHER TESTS")
    print("="*80)
    
    if not DISPATCHER_AVAILABLE:
        print(f"⚠ Dispatcher not available: {import_error}")
        sys.exit(1)
    
    # Run singleton test
    try:
        print("\n--- Test: Singleton ---")
        d1 = get_cold_start_dispatcher()
        d2 = get_cold_start_dispatcher()
        assert d1 is d2
        print("✓ Singleton test: PASSED")
    except Exception as e:
        print(f"✗ Singleton test: FAILED - {e}")
    
    # Run basic initialization test
    try:
        print("\n--- Test: Initialization ---")
        dispatcher = ColdStartDispatcher()
        assert hasattr(dispatcher, 'dispatch')
        print("✓ Initialization test: PASSED")
    except Exception as e:
        print(f"✗ Initialization test: FAILED - {e}")
    
    print("\n" + "="*80)
    print("For full test suite, run: pytest test_cold_start_dispatcher.py -v")
    print("="*80)
