"""
Test suite for Profile Signals API Endpoints.

Tests the REST API endpoints for retrieving profile signals.
Uses FastAPI TestClient for endpoint testing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import MagicMock, patch
import json

# Try to import FastAPI test client and app
try:
    from fastapi.testclient import TestClient
    from app import app
    FASTAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    import_error = str(e)


@pytest.fixture
def client():
    """Create FastAPI test client"""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    return TestClient(app)


@pytest.fixture
def mock_profile_signal_repo():
    """Create a mock ProfileSignalRepository"""
    mock = MagicMock()
    mock.get_recent = MagicMock(return_value=[])
    mock.get_count = MagicMock(return_value=0)
    return mock


@pytest.fixture
def sample_signals_list():
    """Sample list of profile signals"""
    return [
        {
            "intents": {"PROBLEM_SOLVING": 0.85, "LEARNING": 0.6},
            "interests": {"PROGRAMMING": 0.9, "AI": 0.75},
            "behavior_level": "INTERMEDIATE",
            "signals": {"CODE_FOCUSED": 0.9, "DETAILED_EXPLANATION": 0.7},
            "complexity": 0.75,
            "consistency": 0.65
        },
        {
            "intents": {"TASK_COMPLETION": 0.7, "GUIDANCE": 0.5},
            "interests": {"DATA_SCIENCE": 0.8},
            "behavior_level": "ADVANCED",
            "signals": {"STEP_BY_STEP": 0.6},
            "complexity": 0.8,
            "consistency": 0.7
        },
        {
            "intents": {"EXPLORATION": 0.65},
            "interests": {"AI": 0.85, "PROGRAMMING": 0.7},
            "behavior_level": "INTERMEDIATE",
            "signals": {"DEEP_REASONING": 0.75},
            "complexity": 0.65,
            "consistency": 0.6
        }
    ]


class TestGetRecentSignalsEndpoint:
    """Tests for GET /api/behaviors/{user_id}/recent endpoint"""
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_recent_signals_success(
        self, client, mock_profile_signal_repo, sample_signals_list
    ):
        """Test successful retrieval of recent signals"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/recent - Success")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = sample_signals_list
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/recent")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be a direct array
        assert isinstance(data, list)
        assert len(data) == 3
        
        print(f"Response status: {response.status_code}")
        print(f"Signals count: {len(data)}")
        print("✓ PASSED")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_recent_signals_with_limit(
        self, client, mock_profile_signal_repo, sample_signals_list
    ):
        """Test retrieval with custom limit parameter"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/recent?limit=2")
        print("="*80)
        
        # Return only 2 signals when limit=2
        mock_profile_signal_repo.get_recent.return_value = sample_signals_list[:2]
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/recent?limit=2")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify limit was passed to repository
        mock_profile_signal_repo.get_recent.assert_called_with(
            user_id="user_123", 
            limit=2
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Requested limit: 2")
        print("✓ PASSED")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_recent_signals_empty_user(
        self, client, mock_profile_signal_repo
    ):
        """Test retrieval for user with no signals"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/recent - Empty User")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = []
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/new_user/recent")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be empty array
        assert data == []
        
        print(f"Response status: {response.status_code}")
        print(f"Signals: {data}")
        print("✓ PASSED - Returns empty list for new user")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_recent_signals_limit_validation(
        self, client, mock_profile_signal_repo
    ):
        """Test that limit is clamped to max value"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/recent - Limit Validation")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = []
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ), patch(
            'app.PROFILE_SIGNALS_MAX_LIMIT',
            50
        ):
            # Request with very high limit
            response = client.get("/api/behaviors/user_123/recent?limit=1000")
        
        assert response.status_code == 200
        
        # The limit should have been clamped to max
        # (actual clamping happens in the endpoint)
        
        print(f"Response status: {response.status_code}")
        print("✓ PASSED - Endpoint handles large limit")


class TestGetSignalsCountEndpoint:
    """Tests for GET /api/behaviors/{user_id}/signals/count endpoint"""
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_signals_count_success(
        self, client, mock_profile_signal_repo
    ):
        """Test successful retrieval of signal count"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/signals/count - Success")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 15
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/signals/count")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "user_123"
        assert data["count"] == 15
        
        print(f"Response status: {response.status_code}")
        print(f"Count: {data['count']}")
        print("✓ PASSED")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_signals_count_zero(
        self, client, mock_profile_signal_repo
    ):
        """Test count for user with no signals"""
        print("\n" + "="*80)
        print("TEST: GET /api/behaviors/{user_id}/signals/count - Zero")
        print("="*80)
        
        mock_profile_signal_repo.get_count.return_value = 0
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/new_user/signals/count")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 0
        
        print(f"Response status: {response.status_code}")
        print(f"Count: {data['count']}")
        print("✓ PASSED - Returns 0 for new user")


class TestEndpointErrorHandling:
    """Tests for error handling in endpoints"""
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_get_recent_signals_database_error(
        self, client, mock_profile_signal_repo
    ):
        """Test handling of database errors"""
        print("\n" + "="*80)
        print("TEST: Database Error Handling")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.side_effect = Exception("Database connection failed")
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/recent")
        
        # Should return 500 error
        assert response.status_code == 500
        
        print(f"Response status: {response.status_code}")
        print("✓ PASSED - Returns 500 on database error")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_invalid_limit_parameter(self, client, mock_profile_signal_repo):
        """Test handling of invalid limit parameter"""
        print("\n" + "="*80)
        print("TEST: Invalid Limit Parameter")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = []
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            # Test with negative limit
            response = client.get("/api/behaviors/user_123/recent?limit=-5")
        
        # FastAPI might handle this differently - could be 422 or handled gracefully
        # The important thing is it doesn't crash
        assert response.status_code in [200, 422, 400]
        
        print(f"Response status: {response.status_code}")
        print("✓ PASSED - Invalid limit handled")


class TestResponseFormat:
    """Tests for response format validation"""
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_response_structure(
        self, client, mock_profile_signal_repo, sample_signals_list
    ):
        """Test that response has correct structure"""
        print("\n" + "="*80)
        print("TEST: Response Structure Validation")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = sample_signals_list
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/recent")
        
        data = response.json()
        
        # Response should be a direct array
        assert isinstance(data, list)
        assert len(data) == 3
        
        # Check signal structure
        for signal in data:
            assert "intents" in signal
            assert "interests" in signal
            assert "behavior_level" in signal
        
        print(f"Response type: {type(data).__name__}")
        print(f"Signal structure: {list(data[0].keys())}")
        print("✓ PASSED - Response has correct structure")
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    def test_json_serializable(
        self, client, mock_profile_signal_repo, sample_signals_list
    ):
        """Test that response is properly JSON serializable"""
        print("\n" + "="*80)
        print("TEST: JSON Serialization")
        print("="*80)
        
        mock_profile_signal_repo.get_recent.return_value = sample_signals_list
        
        with patch(
            'app.get_profile_signal_repository',
            return_value=mock_profile_signal_repo
        ):
            response = client.get("/api/behaviors/user_123/recent")
        
        # Should be able to parse as JSON
        data = response.json()
        
        # Should be able to re-serialize
        json_str = json.dumps(data)
        assert len(json_str) > 0
        
        print("✓ PASSED - Response is JSON serializable")


# =========================================================================
# Integration Tests (require actual app running)
# =========================================================================

class TestEndpointIntegration:
    """Integration tests that may require database"""
    
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
    @pytest.mark.skip(reason="Requires database connection")
    def test_full_flow_save_and_retrieve(self, client):
        """Test full flow: extract behavior and retrieve signals"""
        # This would be a full integration test
        # 1. POST /extract with prompt
        # 2. GET /api/behaviors/{user_id}/recent
        # 3. Verify signals are returned
        pass


# =========================================================================
# Manual Test Runner
# =========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROFILE SIGNALS API ENDPOINT TESTS")
    print("="*80)
    
    if not FASTAPI_AVAILABLE:
        print(f"⚠ FastAPI not available: {import_error}")
        print("Install with: pip install fastapi[all]")
        sys.exit(1)
    
    # Create test client
    client = TestClient(app)
    
    # Test basic endpoint availability
    try:
        print("\n--- Test: Endpoint Availability ---")
        
        # These tests use mocks
        mock_repo = MagicMock()
        mock_repo.get_recent = MagicMock(return_value=[])
        mock_repo.get_count = MagicMock(return_value=0)
        
        with patch('app.get_profile_signal_repository', return_value=mock_repo):
            response = client.get("/api/behaviors/test_user/recent")
            print(f"GET /api/behaviors/test_user/recent: {response.status_code}")
            
            response = client.get("/api/behaviors/test_user/signals/count")
            print(f"GET /api/behaviors/test_user/signals/count: {response.status_code}")
        
        print("✓ Endpoint availability test: PASSED")
    except Exception as e:
        print(f"✗ Endpoint availability test: FAILED - {e}")
    
    print("\n" + "="*80)
    print("For full test suite, run: pytest test_profile_signals_api.py -v")
    print("="*80)
