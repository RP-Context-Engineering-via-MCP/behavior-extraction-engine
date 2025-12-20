import logging
from typing import Dict, Any
from config.configurations import (
    CREDIBILITY_WEIGHTS,
    DEFAULT_DECAY_RATE,
    CREDIBILITY_PRUNE_THRESHOLD
)

logger = logging.getLogger(__name__)


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
