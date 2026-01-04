import os 
from dotenv import load_dotenv


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

GPT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

DEFAULT_DECAY_RATE = 0.015

# Credibility calculation weights
# confidence: GPT's confidence in the extraction (0.0-1.0)
# clarity: How unambiguous the behavior is (0.0-1.0)
# linguistic_strength: How strongly the user expressed the behavior (0.0-1.0)
CREDIBILITY_WEIGHTS = {
    "confidence": 0.40,
    "clarity": 0.35,
    "linguistic_strength": 0.25
}

# Minimum credibility threshold for storing behaviors in database
# Behaviors below this threshold are filtered out as low quality
CREDIBILITY_PRUNE_THRESHOLD = 0.5

DATABASE_URL = os.getenv("DATABASE_URL")

SAMPLE_USERID = os.getenv("SAMPLE_USERID", "user_12345")

# ============================================================================
# Phase 1 & Phase 2: Similarity and Conflict Detection Thresholds
# ============================================================================
# 
# IMPORTANT (Canonical Behavior Refactor):
# These thresholds are now RETRIEVAL HINTS ONLY, not decision boundaries.
# 
# With the new canonical behavior system, final decisions are made by:
# - Intent + Target matching (structured fields)
# - Context reasoning (general vs specific)  
# - Polarity comparison (POSITIVE vs NEGATIVE)
#
# Embeddings are used ONLY for candidate retrieval, not classification.
# Distance values help narrow the search space but do not determine 
# whether behaviors are duplicates, related, or conflicting.
# ============================================================================

# Similarity distance thresholds for RETRIEVAL (not classification)
# Based on cosine distance between embeddings (0.0 = identical, 2.0 = opposite)
# Can be relaxed since structured matching handles precision
DUPLICATE_THRESHOLD = 0.25       # Retrieval hint for likely duplicates (was 0.12)
SIMILAR_THRESHOLD = 0.35         # Retrieval hint for related behaviors (was 0.20)
CONFLICT_THRESHOLD_MIN = 0.35    # Retrieval hint for potential conflicts (was 0.20)
CONFLICT_THRESHOLD_MAX = 0.70    # Retrieval cutoff for unrelated behaviors (was 0.55)

# Phase 2: Conflict resolution threshold
# When credibility difference exceeds this, auto-resolve (higher credibility wins)
# When below this, flag for user decision (Phase 3)
CREDIBILITY_DIFFERENCE_THRESHOLD = 0.3

# Phase 1: Reinforcement boost calculation
# Formula: BASE_REINFORCEMENT_BOOST / sqrt(reinforcement_count)
# This creates diminishing returns for repeated behaviors
BASE_REINFORCEMENT_BOOST = 0.05

# Phase 3: User confirmation expiration (in seconds)
CONFIRMATION_EXPIRATION_DAYS = 7
CONFIRMATION_EXPIRATION_SECONDS = CONFIRMATION_EXPIRATION_DAYS * 24 * 60 * 60  # 604800 seconds

SEMANTIC_RELEVANCE_THRESHOLD = 0.55