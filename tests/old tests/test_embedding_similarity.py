import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.openAiClient import embed_text
import numpy as np


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def test_embedding_similarity():
    """Test embedding similarity between segment and behaviors."""
    
    # Test data
    segment = "as a javascript developer i like react over angular"
    behavior1 = "I like React over Angular."
    behavior2 = "I don't like Angular compared to React."
    
    print("=" * 80)
    print("EMBEDDING SIMILARITY TEST")
    print("=" * 80)
    print(f"\nSegment: '{segment}'")
    print(f"Behavior 1: '{behavior1}'")
    print(f"Behavior 2: '{behavior2}'")
    print("\n" + "-" * 80)
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    segment_embedding = embed_text(segment)
    behavior1_embedding = embed_text(behavior1)
    behavior2_embedding = embed_text(behavior2)
    
    print(f"✓ Segment embedding dimension: {len(segment_embedding)}")
    print(f"✓ Behavior 1 embedding dimension: {len(behavior1_embedding)}")
    print(f"✓ Behavior 2 embedding dimension: {len(behavior2_embedding)}")
    
    # Calculate similarities
    print("\n" + "-" * 80)
    print("SIMILARITY SCORES (Cosine Similarity)")
    print("-" * 80)
    
    similarity_seg_b1 = cosine_similarity(segment_embedding, behavior1_embedding)
    similarity_seg_b2 = cosine_similarity(segment_embedding, behavior2_embedding)
    similarity_b1_b2 = cosine_similarity(behavior1_embedding, behavior2_embedding)
    
    print(f"\nSegment ↔ Behavior 1: {similarity_seg_b1:.6f}")
    print(f"Segment ↔ Behavior 2: {similarity_seg_b2:.6f}")
    print(f"Behavior 1 ↔ Behavior 2: {similarity_b1_b2:.6f}")
    
    # Analysis
    print("\n" + "-" * 80)
    print("ANALYSIS")
    print("-" * 80)
    
    if similarity_seg_b1 > similarity_seg_b2:
        diff = similarity_seg_b1 - similarity_seg_b2
        print(f"\n✓ Behavior 1 is MORE similar to segment (by {diff:.6f})")
    elif similarity_seg_b2 > similarity_seg_b1:
        diff = similarity_seg_b2 - similarity_seg_b1
        print(f"\n✓ Behavior 2 is MORE similar to segment (by {diff:.6f})")
    else:
        print(f"\n= Both behaviors are EQUALLY similar to segment")
    
    print(f"\nBehavior similarity to each other: {similarity_b1_b2:.6f}")
    
    # Interpretation guide
    print("\n" + "-" * 80)
    print("SIMILARITY INTERPRETATION")
    print("-" * 80)
    print("0.9 - 1.0  : Nearly identical")
    print("0.8 - 0.9  : Very similar")
    print("0.7 - 0.8  : Similar")
    print("0.6 - 0.7  : Somewhat similar")
    print("< 0.6      : Different")
    print("=" * 80)
    
    # Assertions for automated testing
    assert len(segment_embedding) > 0, "Segment embedding should not be empty"
    assert len(behavior1_embedding) > 0, "Behavior 1 embedding should not be empty"
    assert len(behavior2_embedding) > 0, "Behavior 2 embedding should not be empty"
    assert 0 <= similarity_seg_b1 <= 1, "Similarity should be between 0 and 1"
    assert 0 <= similarity_seg_b2 <= 1, "Similarity should be between 0 and 1"
    assert 0 <= similarity_b1_b2 <= 1, "Similarity should be between 0 and 1"
    
    print("\n✅ All assertions passed!")


if __name__ == "__main__":
    test_embedding_similarity()
