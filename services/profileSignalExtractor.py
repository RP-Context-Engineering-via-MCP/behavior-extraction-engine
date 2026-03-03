"""
Profile Signal Extractor

Parses and validates the profile_signals block from GPT-4 response.
Does NOT call GPT-4 itself — just validates and cleans the output against
controlled vocabularies required by the Profile Service.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ProfileSignalExtractor:
    """
    Validates GPT-4 profile_signals output against controlled vocabularies.
    
    The Profile Service expects specific vocabulary terms that differ from
    canonical behavior intents. This class ensures only valid terms are
    passed through, with appropriate defaults for missing/invalid values.
    """
    
    # Profile-level intents (different from canonical behavior intents)
    VALID_INTENTS = {
        "LEARNING",           # User wants to understand a concept
        "TASK_COMPLETION",    # User wants something done, result-focused
        "PROBLEM_SOLVING",    # User is debugging or solving technical issues
        "EXPLORATION",        # User is brainstorming or generating ideas
        "GUIDANCE",           # User is seeking advice or personal direction
        "ENGAGEMENT"          # Casual, fun, low-commitment interaction
    }
    
    # Interest areas for user profiling
    VALID_INTERESTS = {
        "AI",                 # Artificial intelligence, machine learning
        "DATA_SCIENCE",       # Data analysis, statistics
        "WRITING",            # Drafting, editing, summarizing
        "PROGRAMMING",        # Coding, debugging, algorithms
        "CREATIVE",           # Stories, scripts, ideation
        "HEALTH",             # Well-being, diet, exercise
        "PERSONAL_GROWTH",    # Career, life guidance
        "ENTERTAINMENT"       # Games, quizzes, leisure
    }
    
    # Behavioral signals indicating user preferences for response style
    VALID_SIGNALS = {
        "DEEP_REASONING",         # Open-ended curiosity-driven queries
        "DETAILED_EXPLANATION",   # User wants thorough explanation
        "CODE_FOCUSED",           # Response expected to contain code
        "STEP_BY_STEP",           # User wants progressive breakdown
        "QUICK_ANSWER",           # User wants concise response
        "CREATIVE_OUTPUT",        # User wants generated creative content
        "EMPATHETIC_RESPONSE",    # User needs emotional tone
        "ITERATIVE_REFINEMENT"    # User expects multiple turn refinement
    }
    
    # User expertise/behavior levels
    VALID_LEVELS = {"BEGINNER", "INTERMEDIATE", "ADVANCED"}
    
    def parse_and_validate(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate raw profile_signals from GPT-4 output.
        
        Filters out invalid vocabulary terms, clamps numeric values to [0.0, 1.0],
        and provides defaults for missing fields.
        
        Args:
            raw: Raw profile_signals dict from GPT-4 response
            
        Returns:
            Validated profile_signals dict with structure:
            {
                "intents": {"INTENT_NAME": confidence, ...},
                "interests": {"INTEREST_NAME": confidence, ...},
                "behavior_level": "BEGINNER" | "INTERMEDIATE" | "ADVANCED",
                "signals": {"SIGNAL_NAME": confidence, ...},
                "complexity": float (0.0-1.0),
                "consistency": float (0.0-1.0)
            }
            
        Raises:
            ValueError: If no valid intents or interests are found
        """
        if not raw:
            raise ValueError("profile_signals cannot be empty or None")
        
        # Extract and validate intents
        intents = self._extract_scored_dict(
            raw.get("intents", {}), 
            self.VALID_INTENTS,
            "intents"
        )
        
        # Extract and validate interests
        interests = self._extract_scored_dict(
            raw.get("interests", {}), 
            self.VALID_INTERESTS,
            "interests"
        )
        
        # Validate behavior level
        behavior_level = raw.get("behavior_level", "BEGINNER")
        if behavior_level not in self.VALID_LEVELS:
            logger.warning(
                f"Invalid behavior_level '{behavior_level}', defaulting to BEGINNER"
            )
            behavior_level = "BEGINNER"
        
        # Extract and validate signals
        signals = self._extract_scored_dict(
            raw.get("signals", {}), 
            self.VALID_SIGNALS,
            "signals"
        )
        
        # Clamp complexity and consistency to [0.0, 1.0]
        complexity = self._clamp_float(raw.get("complexity", 0.5))
        consistency = self._clamp_float(raw.get("consistency", 0.5))
        
        # Validation: must have at least one valid intent and interest
        if not intents:
            raise ValueError(
                "profile_signals must contain at least one valid intent. "
                f"Valid intents: {self.VALID_INTENTS}"
            )
        if not interests:
            raise ValueError(
                "profile_signals must contain at least one valid interest. "
                f"Valid interests: {self.VALID_INTERESTS}"
            )
        
        validated = {
            "intents": intents,
            "interests": interests,
            "behavior_level": behavior_level,
            "signals": signals,
            "complexity": complexity,
            "consistency": consistency
        }
        
        logger.debug(f"Validated profile_signals: {validated}")
        return validated
    
    def _extract_scored_dict(
        self, 
        raw_dict: Dict[str, Any], 
        valid_keys: set,
        field_name: str
    ) -> Dict[str, float]:
        """
        Extract key-score pairs, filtering to valid keys and clamping scores.
        
        Args:
            raw_dict: Raw dictionary from GPT-4 output
            valid_keys: Set of allowed keys
            field_name: Name of field for logging
            
        Returns:
            Dictionary with valid keys and clamped float scores
        """
        result = {}
        invalid_keys = []
        
        for key, value in raw_dict.items():
            if key in valid_keys:
                try:
                    score = self._clamp_float(float(value))
                    result[key] = score
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid score for {field_name}.{key}: {value}, skipping"
                    )
            else:
                invalid_keys.append(key)
        
        if invalid_keys:
            logger.warning(
                f"Filtered invalid {field_name}: {invalid_keys}. "
                f"Valid options: {valid_keys}"
            )
        
        return result
    
    def _clamp_float(self, value: float) -> float:
        """Clamp a float value to the range [0.0, 1.0]."""
        try:
            return min(1.0, max(0.0, float(value)))
        except (ValueError, TypeError):
            return 0.5  # Default for invalid values
    
    def get_valid_vocabularies(self) -> Dict[str, set]:
        """
        Return all valid vocabularies for reference or documentation.
        
        Returns:
            Dictionary containing all valid vocabulary sets
        """
        return {
            "intents": self.VALID_INTENTS,
            "interests": self.VALID_INTERESTS,
            "signals": self.VALID_SIGNALS,
            "behavior_levels": self.VALID_LEVELS
        }
