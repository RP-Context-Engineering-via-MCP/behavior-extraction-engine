import logging
import math
from typing import Dict, Any
from config.configurations import (
    CREDIBILITY_WEIGHTS,
    DEFAULT_DECAY_RATE,
    CREDIBILITY_PRUNE_THRESHOLD,
    BASE_REINFORCEMENT_BOOST
)

logger = logging.getLogger(__name__)

# Reinforcement configuration (kept for backward compatibility)
BASE_BOOST = BASE_REINFORCEMENT_BOOST  # From config
MIN_BOOST = 0.001  # Minimum meaningful boost


def calculate_initial_credibility(
    confidence: float,
    clarity: float,
    behavior_text: str,
    linguistic_strength: float
) -> float:
    """
    Calculate initial credibility score for a newly extracted behavior.
    
    The credibility is calculated using a weighted combination of:
    1. Confidence score from GPT (how certain the model is)
    2. Clarity score from GPT (how unambiguous the behavior is)
    3. Linguistic strength from GPT (how strongly user expressed the behavior)
    
    Args:
        confidence: GPT confidence score (0.0-1.0)
        clarity: GPT clarity score (0.0-1.0)
        linguistic_strength: GPT linguistic strength score (0.0-1.0)
        behavior_text: The extracted behavior description
        
    Returns:
        float: Initial credibility score (0.0-1.0)
        
    Example:
        >>> calculate_initial_credibility(0.95, 1.0, 0.7, "prefers Python over JavaScript for backend")
    """
    
    # Validate inputs
    if not 0.0 <= confidence <= 1.0:
        logger.warning(f"Confidence {confidence} out of range [0,1], clamping")
        confidence = max(0.0, min(1.0, confidence))
    
    if not 0.0 <= clarity <= 1.0:
        logger.warning(f"Clarity {clarity} out of range [0,1], clamping")
        clarity = max(0.0, min(1.0, clarity))

    if not 0.0 <= linguistic_strength <= 1.0:
        logger.warning(f"Linguistic strength {linguistic_strength} out of range [0,1], clamping")
        linguistic_strength = max(0.0, min(1.0, linguistic_strength))
    
    if not behavior_text or not behavior_text.strip():
        logger.error("Empty behavior text provided")
        return 0.0
    
    # Get weights from config
    w_confidence = CREDIBILITY_WEIGHTS.get("confidence", 0.5)
    w_clarity = CREDIBILITY_WEIGHTS.get("clarity", 0.5)
    w_linguistic = CREDIBILITY_WEIGHTS.get("linguistic_strength", 0.0)
    
    # Calculate credibility from all factors
    initial_credibility = (w_confidence * confidence) + (w_clarity * clarity) + (w_linguistic * linguistic_strength)
    
    # Ensure result is in valid range [0.0, 1.0]
    initial_credibility = max(0.0, min(1.0, initial_credibility))
    
    logger.debug(
        f"Calculated initial credibility: {initial_credibility:.3f} "
        f"(conf={confidence:.2f}, clarity={clarity:.2f}, "
        f"ling_str={linguistic_strength:.2f})"
    )
    
    return round(initial_credibility, 4)


def should_store_behavior(credibility: float) -> bool:
    """
    Determine if a behavior should be stored based on its credibility.
    
    Behaviors below the prune threshold are considered too low quality
    to store in the database.
    
    Args:
        credibility: The calculated credibility score
        
    Returns:
        bool: True if behavior should be stored, False otherwise
    """
    should_store = credibility > CREDIBILITY_PRUNE_THRESHOLD
    
    if not should_store:
        logger.info(
            f"Behavior filtered out: credibility {credibility:.3f} "
            f"<= threshold {CREDIBILITY_PRUNE_THRESHOLD}"
        )
    
    return should_store


def get_decay_rate(behavior_text: str) -> float:
    """
    Get the decay rate for a behavior.
    
    Currently returns the default decay rate for all behaviors.
    Future enhancement: Implement category-based or keyword-based
    dynamic decay rates (e.g., medical constraints decay slower).
    
    Args:
        behavior_text: The behavior description
        
    Returns:
        float: Decay rate per time unit
    """
    # For now, return default decay rate
    # TODO: Implement smart decay based on behavior patterns
    return DEFAULT_DECAY_RATE


def calculate_reinforcement_boost(
    current_credibility: float,
    current_reinforcement_count: int
) -> float:
    """
    Calculate credibility boost using diminishing returns formula.
    
    The boost decreases as reinforcement_count increases, following the formula:
    boost = BASE_BOOST × (1 / sqrt(reinforcement_count))
    
    This ensures:
    - First reinforcement gives maximum boost (~0.05 or 5%)
    - Subsequent reinforcements give progressively smaller boosts
    - Behavior can't inflate to unreasonable credibility levels
    - New behaviors can still compete with heavily reinforced ones
    
    Args:
        current_credibility: Current credibility score (0.0-1.0)
        current_reinforcement_count: Number of times already reinforced (≥1)
        
    Returns:
        float: Boost amount to add to credibility (always positive, may be capped)
        
    Example:
        >>> calculate_reinforcement_boost(0.75, 1)  # First reinforcement
        0.05
        >>> calculate_reinforcement_boost(0.75, 4)  # Fourth reinforcement
        0.025
        >>> calculate_reinforcement_boost(0.98, 10)  # High credibility, many reinforcements
        0.0158  (but capped to reach 1.0)
    """
    # Validate inputs
    if not 0.0 <= current_credibility <= 1.0:
        logger.warning(f"Credibility {current_credibility} out of range [0,1], clamping")
        current_credibility = max(0.0, min(1.0, current_credibility))
    
    if current_reinforcement_count < 1:
        logger.warning(f"Invalid reinforcement_count {current_reinforcement_count}, using 1")
        current_reinforcement_count = 1
    
    # Calculate diminishing boost using square root
    # As count increases, sqrt increases slower, so 1/sqrt decreases
    raw_boost = BASE_BOOST * (1.0 / math.sqrt(current_reinforcement_count))
    
    # Ensure boost is meaningful but not too small
    if raw_boost < MIN_BOOST:
        raw_boost = MIN_BOOST
    
    # Cap boost to not exceed maximum credibility (1.0)
    max_possible_boost = 1.0 - current_credibility
    actual_boost = min(raw_boost, max_possible_boost)
    
    # Ensure non-negative
    actual_boost = max(0.0, actual_boost)
    
    logger.debug(
        f"Reinforcement boost calculated: {actual_boost:.4f} "
        f"(current_cred={current_credibility:.3f}, count={current_reinforcement_count}, "
        f"raw_boost={raw_boost:.4f})"
    )
    
    return round(actual_boost, 4)


def apply_reinforcement_boost(
    current_credibility: float,
    current_reinforcement_count: int
) -> float:
    """
    Apply reinforcement boost to credibility and return new value.
    
    Convenience function that combines calculate_reinforcement_boost()
    with the addition operation.
    
    Args:
        current_credibility: Current credibility score (0.0-1.0)
        current_reinforcement_count: Number of times already reinforced (≥1)
        
    Returns:
        float: New credibility after applying boost (0.0-1.0)
        
    Example:
        >>> apply_reinforcement_boost(0.75, 1)
        0.80
        >>> apply_reinforcement_boost(0.95, 10)
        0.9658
    """
    boost = calculate_reinforcement_boost(current_credibility, current_reinforcement_count)
    new_credibility = min(1.0, current_credibility + boost)
    
    logger.info(
        f"Credibility reinforced: {current_credibility:.3f} → {new_credibility:.3f} "
        f"(+{boost:.4f}, count={current_reinforcement_count})"
    )
    
    return round(new_credibility, 4)


# Future functions to implement:

def apply_decay(
    current_credibility: float,
    decay_rate: float,
    time_elapsed: int
) -> float:
    """
    Apply time-based decay to credibility.
    
    Args:
        current_credibility: Current credibility score
        decay_rate: Decay rate per time unit
        time_elapsed: Time elapsed since last update (in seconds or days)
        
    Returns:
        float: New credibility after decay
        
    Note:
        To be implemented when decay mechanism is activated.
    """
    # TODO: Implement exponential or linear decay
    # Example: credibility = current_credibility * exp(-decay_rate * time_elapsed)
    raise NotImplementedError("Decay mechanism not yet implemented")


def reinforce_credibility(
    current_credibility: float,
    reinforcement_strength: float = 0.1
) -> float:
    """
    Increase credibility when a behavior is re-observed.
    
    Args:
        current_credibility: Current credibility score
        reinforcement_strength: How much to boost credibility (0.0-1.0)
        
    Returns:
        float: New credibility after reinforcement
        
    Note:
        To be implemented when reinforcement mechanism is activated.
    """
    # TODO: Implement reinforcement logic
    # Example: credibility = min(current_credibility + reinforcement_strength, 1.0)
    raise NotImplementedError("Reinforcement mechanism not yet implemented")
