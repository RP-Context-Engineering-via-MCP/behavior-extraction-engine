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

# Decay grace period
# New behaviors do not decay for the first 7 days to allow stabilization
DECAY_GRACE_PERIOD_DAYS = 7
DECAY_GRACE_PERIOD_SECONDS = DECAY_GRACE_PERIOD_DAYS * 24 * 60 * 60  # 604800 seconds

# Intent-based decay rates
# Different behavioral intents decay at different rates based on their nature:
# - HABIT: Decays faster (0.04) - habits can change quickly
# - PREFERENCE: Medium decay (0.015) - preferences are moderately stable
# - COMMUNICATION: Medium decay (0.015) - communication styles are moderately stable
# - SKILL: Decays slowly (0.005) - skills persist longer
# - CONSTRAINT: Decays very slowly (0.001) - constraints are most persistent
INTENT_DECAY_RATES = {
    "HABIT": 0.04,
    "PREFERENCE": 0.015,
    "COMMUNICATION": 0.015,
    "SKILL": 0.005,
    "CONSTRAINT": 0.001
}

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
CREDIBILITY_PRUNE_THRESHOLD = 0.4

DATABASE_URL = os.getenv("DATABASE_URL")

SAMPLE_USERID = os.getenv("SAMPLE_USERID", "user_12345")

# Similarity distance thresholds for RETRIEVAL (not classification)
# Based on cosine distance between embeddings (0.0 = identical, 2.0 = opposite)
# Can be relaxed since structured matching handles precision
# DUPLICATE_THRESHOLD = 0.25       # Retrieval hint for likely duplicates (was 0.12)
# SIMILAR_THRESHOLD = 0.35         # Retrieval hint for related behaviors (was 0.20)
# CONFLICT_THRESHOLD_MIN = 0.35    # Retrieval hint for potential conflicts (was 0.20)
# CONFLICT_THRESHOLD_MAX = 0.70    # Retrieval cutoff for unrelated behaviors (was 0.55)

# Phase 2: Conflict resolution threshold
# When credibility difference exceeds this, auto-resolve (higher credibility wins)
# When below this, flag for user decision (Phase 3)
# CREDIBILITY_DIFFERENCE_THRESHOLD = 0.3

# Phase 1: Reinforcement boost calculation
# Formula: BASE_REINFORCEMENT_BOOST / sqrt(reinforcement_count)
# This creates diminishing returns for repeated behaviors
BASE_REINFORCEMENT_BOOST = 0.05

# Phase 3: User confirmation expiration (in seconds)
CONFIRMATION_EXPIRATION_DAYS = 7
CONFIRMATION_EXPIRATION_SECONDS = CONFIRMATION_EXPIRATION_DAYS * 24 * 60 * 60  # 604800 seconds

SEMANTIC_RELEVANCE_THRESHOLD = 0.55
RELATED_BEHAVIORS_DISTANCE_THRESHOLD = 0.65