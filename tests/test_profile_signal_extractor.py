"""
Test suite for ProfileSignalExtractor.

Tests validation of GPT-4 profile_signals output against controlled vocabularies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from services.profileSignalExtractor import ProfileSignalExtractor


class TestProfileSignalExtractor:
    """Tests for ProfileSignalExtractor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = ProfileSignalExtractor()
    
    # =========================================================================
    # Valid Input Tests
    # =========================================================================
    
    def test_valid_profile_signals_minimal(self):
        """Test validation with minimal valid input"""
        print("\n" + "="*80)
        print("TEST: Valid Profile Signals (Minimal)")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "INTERMEDIATE",
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert "intents" in result
        assert "interests" in result
        assert result["intents"]["LEARNING"] == 0.8
        assert result["interests"]["PROGRAMMING"] == 0.7
        assert result["behavior_level"] == "INTERMEDIATE"
        assert result["complexity"] == 0.5
        assert result["consistency"] == 0.5
        
        print(f"Input: {raw}")
        print(f"Output: {result}")
        print("✓ PASSED")
    
    def test_valid_profile_signals_complete(self):
        """Test validation with complete valid input"""
        print("\n" + "="*80)
        print("TEST: Valid Profile Signals (Complete)")
        print("="*80)
        
        raw = {
            "intents": {
                "PROBLEM_SOLVING": 0.9,
                "LEARNING": 0.6,
                "TASK_COMPLETION": 0.3
            },
            "interests": {
                "PROGRAMMING": 0.85,
                "AI": 0.7,
                "DATA_SCIENCE": 0.4
            },
            "behavior_level": "ADVANCED",
            "signals": {
                "CODE_FOCUSED": 0.9,
                "DETAILED_EXPLANATION": 0.7,
                "STEP_BY_STEP": 0.5
            },
            "complexity": 0.82,
            "consistency": 0.75
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert len(result["intents"]) == 3
        assert len(result["interests"]) == 3
        assert len(result["signals"]) == 3
        assert result["behavior_level"] == "ADVANCED"
        assert result["complexity"] == 0.82
        
        print(f"Input intents: {raw['intents']}")
        print(f"Output intents: {result['intents']}")
        print("✓ PASSED")
    
    # =========================================================================
    # Invalid Vocabulary Filtering Tests
    # =========================================================================
    
    def test_filters_invalid_intents(self):
        """Test that invalid intents are filtered out"""
        print("\n" + "="*80)
        print("TEST: Filter Invalid Intents")
        print("="*80)
        
        raw = {
            "intents": {
                "LEARNING": 0.8,
                "INVALID_INTENT": 0.9,  # Should be filtered
                "UNKNOWN": 0.5  # Should be filtered
            },
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "BEGINNER",
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert "LEARNING" in result["intents"]
        assert "INVALID_INTENT" not in result["intents"]
        assert "UNKNOWN" not in result["intents"]
        assert len(result["intents"]) == 1
        
        print(f"Input intents: {list(raw['intents'].keys())}")
        print(f"Output intents: {list(result['intents'].keys())}")
        print("✓ PASSED - Invalid intents filtered")
    
    def test_filters_invalid_interests(self):
        """Test that invalid interests are filtered out"""
        print("\n" + "="*80)
        print("TEST: Filter Invalid Interests")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {
                "PROGRAMMING": 0.7,
                "COOKING": 0.9,  # Should be filtered (not in vocabulary)
                "SPORTS": 0.5  # Should be filtered
            },
            "behavior_level": "BEGINNER",
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert "PROGRAMMING" in result["interests"]
        assert "COOKING" not in result["interests"]
        assert "SPORTS" not in result["interests"]
        assert len(result["interests"]) == 1
        
        print(f"Input interests: {list(raw['interests'].keys())}")
        print(f"Output interests: {list(result['interests'].keys())}")
        print("✓ PASSED - Invalid interests filtered")
    
    def test_filters_invalid_signals(self):
        """Test that invalid signals are filtered out"""
        print("\n" + "="*80)
        print("TEST: Filter Invalid Signals")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "BEGINNER",
            "signals": {
                "CODE_FOCUSED": 0.9,
                "FAST_RESPONSE": 0.8,  # Should be filtered
                "INVALID_SIGNAL": 0.5  # Should be filtered
            },
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert "CODE_FOCUSED" in result["signals"]
        assert "FAST_RESPONSE" not in result["signals"]
        assert "INVALID_SIGNAL" not in result["signals"]
        assert len(result["signals"]) == 1
        
        print("✓ PASSED - Invalid signals filtered")
    
    # =========================================================================
    # Score Clamping Tests
    # =========================================================================
    
    def test_clamps_scores_above_one(self):
        """Test that scores above 1.0 are clamped"""
        print("\n" + "="*80)
        print("TEST: Clamp Scores Above 1.0")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 1.5},  # Should be clamped to 1.0
            "interests": {"PROGRAMMING": 2.0},  # Should be clamped to 1.0
            "behavior_level": "BEGINNER",
            "signals": {"CODE_FOCUSED": 99.0},  # Should be clamped to 1.0
            "complexity": 1.5,  # Should be clamped to 1.0
            "consistency": 100.0  # Should be clamped to 1.0
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert result["intents"]["LEARNING"] == 1.0
        assert result["interests"]["PROGRAMMING"] == 1.0
        assert result["signals"]["CODE_FOCUSED"] == 1.0
        assert result["complexity"] == 1.0
        assert result["consistency"] == 1.0
        
        print("✓ PASSED - All scores clamped to 1.0")
    
    def test_clamps_scores_below_zero(self):
        """Test that scores below 0.0 are clamped"""
        print("\n" + "="*80)
        print("TEST: Clamp Scores Below 0.0")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": -0.5},  # Should be clamped to 0.0
            "interests": {"PROGRAMMING": -1.0},  # Should be clamped to 0.0
            "behavior_level": "BEGINNER",
            "signals": {},
            "complexity": -0.1,  # Should be clamped to 0.0
            "consistency": -99.0  # Should be clamped to 0.0
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert result["intents"]["LEARNING"] == 0.0
        assert result["interests"]["PROGRAMMING"] == 0.0
        assert result["complexity"] == 0.0
        assert result["consistency"] == 0.0
        
        print("✓ PASSED - All scores clamped to 0.0")
    
    # =========================================================================
    # Behavior Level Tests
    # =========================================================================
    
    def test_valid_behavior_levels(self):
        """Test all valid behavior levels"""
        print("\n" + "="*80)
        print("TEST: Valid Behavior Levels")
        print("="*80)
        
        for level in ["BEGINNER", "INTERMEDIATE", "ADVANCED"]:
            raw = {
                "intents": {"LEARNING": 0.8},
                "interests": {"PROGRAMMING": 0.7},
                "behavior_level": level,
                "signals": {},
                "complexity": 0.5,
                "consistency": 0.5
            }
            result = self.extractor.parse_and_validate(raw)
            assert result["behavior_level"] == level
            print(f"  ✓ {level}")
        
        print("✓ PASSED - All valid levels accepted")
    
    def test_invalid_behavior_level_defaults_to_beginner(self):
        """Test that invalid behavior level defaults to BEGINNER"""
        print("\n" + "="*80)
        print("TEST: Invalid Behavior Level Defaults")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "EXPERT",  # Invalid - should default to BEGINNER
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert result["behavior_level"] == "BEGINNER"
        print(f"Input level: 'EXPERT' → Output level: '{result['behavior_level']}'")
        print("✓ PASSED - Invalid level defaulted to BEGINNER")
    
    # =========================================================================
    # Error Cases
    # =========================================================================
    
    def test_raises_error_for_empty_input(self):
        """Test that empty input raises ValueError"""
        print("\n" + "="*80)
        print("TEST: Empty Input Error")
        print("="*80)
        
        with pytest.raises(ValueError) as excinfo:
            self.extractor.parse_and_validate({})
        
        assert "at least one valid intent" in str(excinfo.value).lower() or "empty" in str(excinfo.value).lower()
        print(f"Error message: {excinfo.value}")
        print("✓ PASSED - Raised ValueError for empty input")
    
    def test_raises_error_for_none_input(self):
        """Test that None input raises ValueError"""
        print("\n" + "="*80)
        print("TEST: None Input Error")
        print("="*80)
        
        with pytest.raises(ValueError) as excinfo:
            self.extractor.parse_and_validate(None)
        
        print(f"Error message: {excinfo.value}")
        print("✓ PASSED - Raised ValueError for None input")
    
    def test_raises_error_for_no_valid_intents(self):
        """Test that input with no valid intents raises ValueError"""
        print("\n" + "="*80)
        print("TEST: No Valid Intents Error")
        print("="*80)
        
        raw = {
            "intents": {"INVALID": 0.8},  # No valid intents
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "BEGINNER",
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        with pytest.raises(ValueError) as excinfo:
            self.extractor.parse_and_validate(raw)
        
        assert "at least one valid intent" in str(excinfo.value).lower()
        print(f"Error message: {excinfo.value}")
        print("✓ PASSED - Raised ValueError for no valid intents")
    
    def test_raises_error_for_no_valid_interests(self):
        """Test that input with no valid interests raises ValueError"""
        print("\n" + "="*80)
        print("TEST: No Valid Interests Error")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {"COOKING": 0.7},  # No valid interests
            "behavior_level": "BEGINNER",
            "signals": {},
            "complexity": 0.5,
            "consistency": 0.5
        }
        
        with pytest.raises(ValueError) as excinfo:
            self.extractor.parse_and_validate(raw)
        
        assert "at least one valid interest" in str(excinfo.value).lower()
        print(f"Error message: {excinfo.value}")
        print("✓ PASSED - Raised ValueError for no valid interests")
    
    # =========================================================================
    # Default Value Tests
    # =========================================================================
    
    def test_default_complexity_and_consistency(self):
        """Test that missing complexity/consistency default to 0.5"""
        print("\n" + "="*80)
        print("TEST: Default Complexity and Consistency")
        print("="*80)
        
        raw = {
            "intents": {"LEARNING": 0.8},
            "interests": {"PROGRAMMING": 0.7},
            "behavior_level": "BEGINNER",
            "signals": {}
            # complexity and consistency missing
        }
        
        result = self.extractor.parse_and_validate(raw)
        
        assert result["complexity"] == 0.5
        assert result["consistency"] == 0.5
        
        print("✓ PASSED - Defaults applied for missing values")
    
    # =========================================================================
    # Vocabulary Reference Test
    # =========================================================================
    
    def test_get_valid_vocabularies(self):
        """Test that get_valid_vocabularies returns all vocabularies"""
        print("\n" + "="*80)
        print("TEST: Get Valid Vocabularies")
        print("="*80)
        
        vocabs = self.extractor.get_valid_vocabularies()
        
        assert "intents" in vocabs
        assert "interests" in vocabs
        assert "signals" in vocabs
        assert "behavior_levels" in vocabs
        
        assert "LEARNING" in vocabs["intents"]
        assert "PROGRAMMING" in vocabs["interests"]
        assert "CODE_FOCUSED" in vocabs["signals"]
        assert "BEGINNER" in vocabs["behavior_levels"]
        
        print(f"Intents: {vocabs['intents']}")
        print(f"Interests: {vocabs['interests']}")
        print(f"Signals: {vocabs['signals']}")
        print(f"Behavior Levels: {vocabs['behavior_levels']}")
        print("✓ PASSED")


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROFILE SIGNAL EXTRACTOR TESTS")
    print("="*80)
    
    test_instance = TestProfileSignalExtractor()
    test_instance.setup_method()
    
    # Run all tests
    tests = [
        test_instance.test_valid_profile_signals_minimal,
        test_instance.test_valid_profile_signals_complete,
        test_instance.test_filters_invalid_intents,
        test_instance.test_filters_invalid_interests,
        test_instance.test_filters_invalid_signals,
        test_instance.test_clamps_scores_above_one,
        test_instance.test_clamps_scores_below_zero,
        test_instance.test_valid_behavior_levels,
        test_instance.test_invalid_behavior_level_defaults_to_beginner,
        test_instance.test_raises_error_for_empty_input,
        test_instance.test_raises_error_for_none_input,
        test_instance.test_raises_error_for_no_valid_intents,
        test_instance.test_raises_error_for_no_valid_interests,
        test_instance.test_default_complexity_and_consistency,
        test_instance.test_get_valid_vocabularies,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
