"""
Utility functions for generating text embeddings.
"""
from typing import List
from services.openAiClient import embed_text
import logging

logger = logging.getLogger(__name__)


def get_text_embedding(text: str) -> List[float]:
    """
    Get embedding vector for a natural language text.
    
    Args:
        text: Natural language text to embed
        
    Returns:
        List of floats representing the embedding vector (3072 dimensions for text-embedding-3-large)
        
    Raises:
        ValueError: If text is empty or invalid
        Exception: If embedding generation fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        embedding = embed_text(text.strip())
        logger.info(f"Generated embedding for text (length: {len(text)}, dimensions: {len(embedding)})")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise


def get_behavior_embedding(behavior_text: str) -> List[float]:
    """
    Get embedding vector for a behavior description.
    This is a specialized wrapper around get_text_embedding for behaviors.
    
    Args:
        behavior_text: Behavior description text
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        ValueError: If behavior_text is empty or invalid
        Exception: If embedding generation fails
    """
    return get_text_embedding(behavior_text)
