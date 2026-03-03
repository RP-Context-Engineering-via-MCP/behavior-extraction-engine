"""
Algorithm constants for the Behavior Detection and Management system.

These are pure numeric/structural constants — not environment-backed.
They control scoring, thresholds, weights, and retrieval behaviour.
Changing a value here affects the system's intelligence; they are NOT
deployment-environment secrets.

Environment-backed values (API keys, URLs) live in configurations.py.
"""

# ---------------------------------------------------------------------------
# Decay configuration
# ---------------------------------------------------------------------------

DEFAULT_DECAY_RATE: float = 0.015

# New behaviors do not decay for the first 7 days to allow stabilisation.
DECAY_GRACE_PERIOD_DAYS: int = 7
DECAY_GRACE_PERIOD_SECONDS: int = DECAY_GRACE_PERIOD_DAYS * 24 * 60 * 60  # 604 800 s

# Intent-based decay rates (per full day):
#   HABIT       → 0.04  – habits can change quickly
#   PREFERENCE  → 0.015 – moderately stable
#   COMMUNICATION → 0.015 – moderately stable
#   SKILL       → 0.005 – skills persist longer
#   CONSTRAINT  → 0.001 – most persistent (medical, hard rules)
INTENT_DECAY_RATES: dict[str, float] = {
    "HABIT": 0.04,
    "PREFERENCE": 0.015,
    "COMMUNICATION": 0.015,
    "SKILL": 0.005,
    "CONSTRAINT": 0.001,
}

# ---------------------------------------------------------------------------
# Credibility scoring
# ---------------------------------------------------------------------------

# Weights used when calculating the initial credibility of an extracted behavior.
#   confidence        – GPT's certainty in the extraction          (0.0-1.0)
#   clarity           – how unambiguous the behavior statement is  (0.0-1.0)
#   linguistic_strength – how strongly the user expressed it       (0.0-1.0)
CREDIBILITY_WEIGHTS: dict[str, float] = {
    "confidence": 0.40,
    "clarity": 0.35,
    "linguistic_strength": 0.25,
}

# Behaviors whose calculated credibility falls at or below this value are
# filtered out before storage (too low quality to be useful).
CREDIBILITY_PRUNE_THRESHOLD: float = 0.4

# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------

# Formula: boost = BASE_REINFORCEMENT_BOOST / sqrt(reinforcement_count)
# Creates diminishing returns for repeated reinforcement signals.
BASE_REINFORCEMENT_BOOST: float = 0.05

# ---------------------------------------------------------------------------
# Conflict resolution — user confirmation window
# ---------------------------------------------------------------------------

CONFIRMATION_EXPIRATION_DAYS: int = 7
CONFIRMATION_EXPIRATION_SECONDS: int = (
    CONFIRMATION_EXPIRATION_DAYS * 24 * 60 * 60  # 604 800 s
)

# ---------------------------------------------------------------------------
# Similarity / retrieval thresholds
# ---------------------------------------------------------------------------

# Maximum cosine distance for a candidate to be considered semantically
# relevant during the store_behavior pipeline (duplicate / conflict gate).
SEMANTIC_RELEVANCE_THRESHOLD: float = 0.55

# Maximum cosine *distance* value used by the /v2/extract endpoint to
# decide which retrieved behaviors are returned to the caller as "related".
# (distance = 1 - hybrid_score, so lower is closer)
# 0.73 provides a small safety margin for behaviors whose densely-computed
# distance lands just above 0.72 after intent boosting.
RELATED_BEHAVIORS_DISTANCE_THRESHOLD: float = 0.73

# ---------------------------------------------------------------------------
# TGHR – Tuple-Guided Hybrid Retrieval (3D search) configuration
# ---------------------------------------------------------------------------
# hybrid_score = DENSE_W * semantic + SPARSE_W * bm25 + INTENT_BOOST_W * intent_match
#
# Weight rationale (must sum to 1.0):
#   Dense  0.50 — primary relevance signal (semantic similarity)
#   Sparse 0.25 — keyword overlap bonus (BM25 via tsvector)
#   Intent 0.25 — LLM-predicted intent is a high-quality signal; giving it
#                 more weight anchors retrieval to the semantic "type" of the
#                 query, preventing fringe behaviors from crowding out
#                 intent-matched ones that sit slightly further in dense space.

# Dense (semantic / cosine) signal weight.
HYBRID_DENSE_WEIGHT: float = 0.50

# Sparse (BM25 / tsvector) signal weight.
HYBRID_SPARSE_WEIGHT: float = 0.25

# Intent-match soft boost weight.
# NOT a hard filter — non-matching intents still appear if dense+sparse score well.
HYBRID_INTENT_BOOST_WEIGHT: float = 0.25

# Maximum results fetched from DB before threshold / gap filtering.
# 40 provides enough headroom for users with ~50-100 stored behaviors without
# over-querying in production (where the partitioned table keeps per-user counts
# well below a few hundred in typical use).
HYBRID_SEARCH_LIMIT: int = 40

# Results whose hybrid_score falls below this value are treated as irrelevant.
HYBRID_SCORE_THRESHOLD: float = 0.10

# Relevance-gap cutoff ratio.
# If a result's score drops more than this fraction below the top result,
# all further results are discarded.
# Set higher than 0.40 because BM25 spikes in the top result would otherwise
# kill equally-relevant behaviors that lack exact keyword overlap.
RELEVANCE_GAP_DROP_RATIO: float = 0.55

# Hard cap on the number of results returned after gap filtering.
# Prevents over-retrieval on broad / vague queries.
MAX_RETRIEVAL_RESULTS: int = 20

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

ALL_INTENT_TYPES: list[str] = [
    "HABIT",
    "PREFERENCE",
    "CONSTRAINT",
    "SKILL",
    "COMMUNICATION",
]

# Intent affinity matrix — graduated boost for related (non-identical) intents.
# Key: frozenset of two intent types (symmetric).
# Value: affinity 0.0-1.0 (exact match handled separately as 1.0).
#
# Example: a HABIT behavior about vitamins is relevant when searching for
# CONSTRAINT behaviors about health → HABIT↔CONSTRAINT affinity = 0.65.
INTENT_AFFINITY: dict[frozenset, float] = {
    # HABIT behaviors are frequently the stored equivalent of PREFERENCE /
    # CONSTRAINT statements ("always uses dark mode" ≡ "prefers dark mode").
    # Raising these affinities ensures HABIT behaviors surface when a user
    # queries for their preferences or constraints, and vice-versa.
    frozenset({"HABIT", "CONSTRAINT"}): 0.65,   # was 0.50
    frozenset({"HABIT", "PREFERENCE"}): 0.60,   # was 0.40
    frozenset({"PREFERENCE", "CONSTRAINT"}): 0.35,
    frozenset({"COMMUNICATION", "PREFERENCE"}): 0.30,
    frozenset({"COMMUNICATION", "HABIT"}): 0.20,
    frozenset({"SKILL", "HABIT"}): 0.20,
    frozenset({"SKILL", "PREFERENCE"}): 0.15,
    frozenset({"SKILL", "CONSTRAINT"}): 0.10,
    frozenset({"COMMUNICATION", "CONSTRAINT"}): 0.10,
    frozenset({"COMMUNICATION", "SKILL"}): 0.10,
}
