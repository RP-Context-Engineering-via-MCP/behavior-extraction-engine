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
RELATED_BEHAVIORS_DISTANCE_THRESHOLD = 0.72

# ============================================================================
# TGHR (Tuple-Guided Hybrid Retrieval) Configuration
# Weights for combining dense (semantic) and sparse (BM25) search scores
# hybrid_score = DENSE_W * semantic + SPARSE_W * bm25 + INTENT_BOOST * intent_match
# ============================================================================

# Dense score weight: Semantic similarity via cosine distance
# Higher weight = more emphasis on meaning/synonym matching
HYBRID_DENSE_WEIGHT = 0.55

# Sparse score weight: BM25 lexical/keyword matching via tsvector
# Higher weight = more emphasis on exact keyword overlap
# Now uses OR-based tsquery so BM25 actually produces non-zero scores
HYBRID_SPARSE_WEIGHT = 0.30

# Intent boost weight: Soft boost when behavior intent matches LLM-predicted intents
# This is NOT a hard filter — behaviors with non-matching intents still appear
# if they score well on dense + sparse. Intent match just ranks them higher.
HYBRID_INTENT_BOOST_WEIGHT = 0.15

# Default limit for hybrid search results before threshold filtering
HYBRID_SEARCH_LIMIT = 30

# Minimum hybrid score threshold (below this, behaviors are considered irrelevant)
# Score range: 0.0 (no match) to ~1.0 (perfect match on all 3 signals)
HYBRID_SCORE_THRESHOLD = 0.10

# Relevance gap cutoff: stop returning results when score drops more than
# this fraction below the top result. E.g., 0.55 means if top score is 0.50,
# any result below 0.50 * (1 - 0.55) = 0.225 is dropped.
# Set higher than 0.40 because BM25 keyword matches create artificial spikes
# in the top result — a 40% gap would kill equally relevant behaviors that
# simply lack exact keyword overlap.
RELEVANCE_GAP_DROP_RATIO = 0.55

# Soft cap: maximum number of results returned after gap cutoff filtering.
# Prevents over-retrieval for broad/vague queries where many behaviors
# cluster in a similar score range and gap cutoff alone can't prune them.
# Test 4 showed Phase 3 averaging 14.2 behaviors/query with peaks of 27.
MAX_RETRIEVAL_RESULTS = 20

# All known intent types for behavioral classification
ALL_INTENT_TYPES = ["HABIT", "PREFERENCE", "CONSTRAINT", "SKILL", "COMMUNICATION"]

# Intent affinity matrix: graduated boost for related (but non-matching) intents.
# Key: frozenset of two intent types (symmetric relationship)
# Value: affinity score 0.0-1.0 (how related the intents are)
# Exact match always gets 1.0 (handled separately). These cover cross-intent relevance.
# Example: A HABIT behavior about vitamins is relevant when searching for CONSTRAINTs
#          about health, so HABIT<->CONSTRAINT affinity = 0.50.
INTENT_AFFINITY = {
    frozenset({"HABIT", "CONSTRAINT"}): 0.50,        # routines <-> restrictions (health, exercise)
    frozenset({"HABIT", "PREFERENCE"}): 0.40,        # habits often reflect preferences
    frozenset({"PREFERENCE", "CONSTRAINT"}): 0.35,   # preferences <-> restrictions overlap
    frozenset({"COMMUNICATION", "PREFERENCE"}): 0.30,
    frozenset({"COMMUNICATION", "HABIT"}): 0.20,
    frozenset({"SKILL", "HABIT"}): 0.20,
    frozenset({"SKILL", "PREFERENCE"}): 0.15,
    frozenset({"SKILL", "CONSTRAINT"}): 0.10,
    frozenset({"COMMUNICATION", "CONSTRAINT"}): 0.10,
    frozenset({"COMMUNICATION", "SKILL"}): 0.10,
}

# ============================================================================
# Redis Configuration (for Drift Detection Service integration)
# Events are published to Redis Streams for real-time drift detection
# ============================================================================

# Redis connection URL (default: local Redis on port 6379)
# Format: redis://[username:password@]host:port/db
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis Stream name for behavior events
REDIS_STREAM_NAME = os.getenv("REDIS_STREAM_NAME", "behavior.events")

# Whether event publishing is enabled (disable for testing without Redis)
REDIS_EVENTS_ENABLED = os.getenv("REDIS_EVENTS_ENABLED", "true").lower() == "true"

# ============================================================================
# Profile Service Integration Configuration
# Used for cold-start profiling and drift fallback communication
# ============================================================================

# Profile Service base URL for API calls
PROFILE_SERVICE_BASE_URL = os.getenv("PROFILE_SERVICE_BASE_URL", "http://localhost:8001")

# Default number of recent profile signals to return
PROFILE_SIGNALS_DEFAULT_LIMIT = int(os.getenv("PROFILE_SIGNALS_DEFAULT_LIMIT", "10"))

# Maximum number of recent profile signals allowed per request
PROFILE_SIGNALS_MAX_LIMIT = int(os.getenv("PROFILE_SIGNALS_MAX_LIMIT", "50"))