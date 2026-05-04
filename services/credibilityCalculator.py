import logging
import math
import time
from typing import Dict, Any, Optional, Tuple
from config.configurations import (
    CREDIBILITY_WEIGHTS,
    DEFAULT_DECAY_RATE,
    CREDIBILITY_PRUNE_THRESHOLD,
    BASE_REINFORCEMENT_BOOST,
    INTENT_DECAY_RATES
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

    Two-factor weighted combination:
    1. extraction_quality — average of GPT confidence and clarity. These
       two scores measure the same axis ("is this a clean, real, extractable
       behavior?") and are intentionally collapsed so the formula does not
       double-count them.
    2. linguistic_strength — GPT's score for how strongly the user expressed
       the behavior. This is the dominant signal because it is the only one
       that distinguishes hedged statements ("I might try Rust") from strong
       ones ("I always use Python").

    Args:
        confidence: GPT confidence score (0.0-1.0)
        clarity: GPT clarity score (0.0-1.0)
        linguistic_strength: GPT linguistic strength score (0.0-1.0)
        behavior_text: The extracted behavior description

    Returns:
        float: Initial credibility score (0.0-1.0)

    Example:
        >>> calculate_initial_credibility(0.95, 1.0, "prefers Python over JavaScript for backend", 0.7)
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

    # Get weights from config (2-factor formula)
    w_extraction = CREDIBILITY_WEIGHTS.get("extraction_quality", 0.25)
    w_linguistic = CREDIBILITY_WEIGHTS.get("linguistic_strength", 0.75)

    # Combine confidence and clarity into a single extraction_quality signal —
    # they describe the same "cleanly extractable behavior?" axis.
    extraction_quality = (confidence + clarity) / 2.0

    initial_credibility = (w_extraction * extraction_quality) + (w_linguistic * linguistic_strength)

    # Ensure result is in valid range [0.0, 1.0]
    initial_credibility = max(0.0, min(1.0, initial_credibility))

    logger.debug(
        f"Calculated initial credibility: {initial_credibility:.3f} "
        f"(extraction_quality={extraction_quality:.2f}, "
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


def get_decay_rate(behavior_text: str = None, intent: Optional[str] = None) -> float:
    """
    Get the decay rate for a behavior based on its intent.
    
    Different behavioral intents have different decay rates:
    - HABIT (0.04): Habits can change quickly with new routines
    - PREFERENCE (0.015): Preferences are moderately stable
    - COMMUNICATION (0.015): Communication styles are moderately stable
    - SKILL (0.005): Skills persist longer once learned
    - CONSTRAINT (0.001): Constraints are most persistent (medical, etc.)
    
    Args:
        behavior_text: The behavior description (kept for backward compatibility)
        intent: The behavioral intent type (HABIT, PREFERENCE, SKILL, etc.)
        
    Returns:
        float: Decay rate per time unit
        
    Example:
        >>> get_decay_rate(intent="SKILL")
        0.005
        >>> get_decay_rate(intent="HABIT")
        0.04
    """
    # If intent is provided, use intent-based decay rate
    if intent and intent in INTENT_DECAY_RATES:
        decay_rate = INTENT_DECAY_RATES[intent]
        logger.debug(f"Using intent-based decay rate: {decay_rate} for intent '{intent}'")
        return decay_rate
    
    # Fallback to default decay rate if intent not provided or not recognized
    if intent:
        logger.warning(f"Unknown intent '{intent}', using default decay rate {DEFAULT_DECAY_RATE}")
    
    return DEFAULT_DECAY_RATE


def apply_lazy_decay(
    stored_credibility: float,
    decay_rate: float,
    last_decay_applied_at: Optional[int],
    current_time: Optional[int] = None
) -> Tuple[float, bool, int]:
    """
    Apply exponential decay to credibility based on full days elapsed.
    
    This implements a "lazy" decay mechanism where decay is calculated on-the-fly
    when behaviors are retrieved, rather than through batch processing.
    
    IMPORTANT: Decay is only applied for FULL DAYS elapsed (86400 seconds = 1 day).
    This prevents constant micro-adjustments and makes decay predictable.
    
    Formula: C_current = C_stored × e^(-λ × days_elapsed)
    Where:
    - C_stored: Credibility value stored in database
    - λ (lambda): Decay rate (intent-based, per day)
    - days_elapsed: Number of FULL days since last decay application
    - e: Euler's number (~2.71828)
    
    Args:
        stored_credibility: The credibility value currently in the database
        decay_rate: Intent-based decay rate (λ) per day
        last_decay_applied_at: Timestamp when decay was last applied (or grace period end)
        current_time: Current timestamp (defaults to now if not provided)
        
    Returns:
        Tuple of (new_credibility, decay_applied, days_elapsed):
        - new_credibility: Calculated credibility after decay
        - decay_applied: Whether decay was actually applied (False if < 1 full day)
        - days_elapsed: Number of full days elapsed since last decay application
        
    Example:
        >>> # If decay last applied today at 6am, retrieval at 5:59am tomorrow = no decay
        >>> # If decay last applied today at 6am, retrieval at 6:00am tomorrow = 1 day decay
        >>> stored_cred = 0.85
        >>> decay_rate = 0.015  # PREFERENCE intent (per day)
        >>> last_applied = int(time.time()) - (30 * 24 * 60 * 60)  # 30 days ago
        >>> new_cred, applied, days = apply_lazy_decay(stored_cred, decay_rate, last_applied)
        >>> # new_cred will reflect 30 full days of decay
        
    Notes:
        - If last_decay_applied_at is None, no decay is applied
        - If current_time < last_decay_applied_at, no decay (still in grace period)
        - Decay only applied if at least 1 FULL day (86400 seconds) has elapsed
        - Credibility is clamped to [0.0, 1.0] range
        - Uses exponential decay for smooth, continuous degradation
    """
    # Constants
    SECONDS_PER_DAY = 86400  # 24 * 60 * 60
    
    # Default to current time if not provided
    if current_time is None:
        current_time = int(time.time())
    
    # If last_decay_applied_at is not set, no decay can be applied
    if last_decay_applied_at is None:
        logger.debug("No last_decay_applied_at timestamp, skipping decay")
        return (stored_credibility, False, 0)
    
    # If still in grace period, no decay
    if current_time < last_decay_applied_at:
        time_until_decay = last_decay_applied_at - current_time
        logger.debug(
            f"Still in grace period, {time_until_decay} seconds until decay starts"
        )
        return (stored_credibility, False, 0)
    
    # Calculate time elapsed since last decay application (in seconds)
    time_elapsed_seconds = current_time - last_decay_applied_at
    
    # Calculate FULL DAYS elapsed (integer division)
    days_elapsed = time_elapsed_seconds // SECONDS_PER_DAY
    
    # If less than 1 full day has elapsed, don't apply decay
    if days_elapsed < 1:
        logger.debug(
            f"Less than 1 full day elapsed ({time_elapsed_seconds} seconds), "
            f"skipping decay"
        )
        return (stored_credibility, False, 0)
    
    # Apply exponential decay formula: C_current = C_stored × e^(-λ × days)
    decay_factor = math.exp(-decay_rate * days_elapsed)
    new_credibility = stored_credibility * decay_factor
    
    # Clamp to valid range [0.0, 1.0]
    new_credibility = max(0.0, min(1.0, new_credibility))
    
    # Calculate credibility loss for logging
    credibility_loss = stored_credibility - new_credibility
    
    logger.info(
        f"Lazy decay applied: {stored_credibility:.4f} → {new_credibility:.4f} "
        f"(loss: {credibility_loss:.4f}, {days_elapsed} full days elapsed, "
        f"decay_rate: {decay_rate}/day, factor: {decay_factor:.6f})"
    )
    
    return (new_credibility, True, days_elapsed)


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
