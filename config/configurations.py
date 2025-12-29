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

# Similarity distance thresholds for behavior classification
# Based on cosine distance between embeddings (0.0 = identical, 2.0 = opposite)
DUPLICATE_THRESHOLD = 0.12       # 0.00-0.05: Exact match, reinforce existing
SIMILAR_THRESHOLD = 0.20         # 0.05-0.15: Related variations, insert both
CONFLICT_THRESHOLD_MIN = 0.20    # 0.15-0.40: Potential conflict, needs LLM analysis
CONFLICT_THRESHOLD_MAX = 0.55    # 0.40+: Unrelated behaviors

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