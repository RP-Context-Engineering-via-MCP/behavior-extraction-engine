"""
Utility functions for calculating similarity/distance between embeddings and behaviors.
"""
from typing import List, Dict
import math
import logging

logger = logging.getLogger(__name__)


def cosine_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine distance between two embedding vectors.
    
    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where:
    - 0 = identical vectors (0 distance)
    - 1 = orthogonal vectors (no similarity)
    - 2 = opposite vectors (maximum distance)
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine distance as float
        
    Raises:
        ValueError: If embeddings are empty or have different dimensions
    """
    if not embedding1 or not embedding2:
        raise ValueError("Embeddings cannot be empty")
    
    if len(embedding1) != len(embedding2):
        raise ValueError(f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}")
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Zero magnitude vector encountered")
    
    # Cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to [-1, 1] to handle floating point errors
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    
    # Cosine distance
    distance = 1 - cosine_similarity
    
    return distance


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Range: [-1, 1] where:
    - 1 = identical direction (perfect similarity)
    - 0 = orthogonal (no similarity)
    - -1 = opposite direction
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity as float
        
    Raises:
        ValueError: If embeddings are empty or have different dimensions
    """
    distance = cosine_distance(embedding1, embedding2)
    return 1 - distance


def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate Euclidean distance between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Euclidean distance as float
        
    Raises:
        ValueError: If embeddings are empty or have different dimensions
    """
    if not embedding1 or not embedding2:
        raise ValueError("Embeddings cannot be empty")
    
    if len(embedding1) != len(embedding2):
        raise ValueError(f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}")
    
    squared_diff_sum = sum((a - b) ** 2 for a, b in zip(embedding1, embedding2))
    return math.sqrt(squared_diff_sum)


def calculate_behavior_distance(
    behavior1_text: str, 
    behavior2_text: str, 
    embedding1: List[float], 
    embedding2: List[float],
    metric: str = "cosine"
) -> Dict[str, any]:
    """
    Calculate distance between two behaviors using their embeddings.
    
    Args:
        behavior1_text: First behavior description
        behavior2_text: Second behavior description
        embedding1: Embedding vector for behavior1
        embedding2: Embedding vector for behavior2
        metric: Distance metric to use ("cosine" or "euclidean")
        
    Returns:
        Dictionary containing:
        - distance: The calculated distance
        - similarity: Cosine similarity (only for cosine metric)
        - metric: The metric used
        - behavior1: First behavior text
        - behavior2: Second behavior text
        - embedding_dimensions: Dimension of embeddings
        
    Raises:
        ValueError: If inputs are invalid or metric is unknown
    """
    if metric not in ["cosine", "euclidean"]:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")
    
    result = {
        "behavior1": behavior1_text,
        "behavior2": behavior2_text,
        "metric": metric,
        "embedding_dimensions": len(embedding1)
    }
    
    if metric == "cosine":
        distance = cosine_distance(embedding1, embedding2)
        similarity = cosine_similarity(embedding1, embedding2)
        result["distance"] = distance
        result["similarity"] = similarity
        result["interpretation"] = _interpret_cosine_distance(distance)
    else:  # euclidean
        distance = euclidean_distance(embedding1, embedding2)
        result["distance"] = distance
        result["interpretation"] = "Euclidean distance (lower is more similar)"
    
    logger.info(f"Calculated {metric} distance between behaviors: {distance:.4f}")
    
    return result


def _interpret_cosine_distance(distance: float) -> str:
    """
    Provide human-readable interpretation of cosine distance.
    
    Args:
        distance: Cosine distance value
        
    Returns:
        Interpretation string
    """
    if distance < 0.1:
        return "Nearly identical"
    elif distance < 0.3:
        return "Very similar"
    elif distance < 0.5:
        return "Similar"
    elif distance < 0.7:
        return "Somewhat similar"
    elif distance < 0.9:
        return "Somewhat different"
    elif distance < 1.2:
        return "Different"
    else:
        return "Very different"
