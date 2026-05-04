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
#   COMMUNICATION → 0.001 – least stable
#   SKILL       → 0.004 – skills persist longer
#   CONSTRAINT  → 0.00015 – most persistent (medical, hard rules)
INTENT_DECAY_RATES: dict[str, float] = {
    "HABIT": 0.04,
    "PREFERENCE": 0.015,
    "COMMUNICATION": 0.01,
    "SKILL": 0.004,
    "CONSTRAINT": 0.0015,
}

# ---------------------------------------------------------------------------
# Credibility scoring
# ---------------------------------------------------------------------------

# Weights used when calculating the initial credibility of an extracted behavior.
# Reduced to 2 factors (was 3) — confidence and clarity were near-duplicates of
# the same "is this a clean, real behavior?" signal and double-counted, leaving
# linguistic_strength under-weighted at 0.25 even though it is the only signal
# that distinguishes hedged ("I might try Rust") from strong ("I always use
# Python") statements.  They are now averaged into a single extraction_quality
# term, freeing linguistic_strength to carry the dominant weight.
#   extraction_quality (= avg of LLM confidence + clarity) – weight 0.25
#   linguistic_strength (LLM's intensity score)            – weight 0.75
CREDIBILITY_WEIGHTS: dict[str, float] = {
    "extraction_quality": 0.25,
    "linguistic_strength": 0.75,
}

# Behaviors whose calculated credibility falls at or below this value are
# filtered out before storage (too low quality to be useful).
# Lowered from 0.40 → 0.30 to accompany the formula change above: weak-but-real
# behaviors ("I sometimes use Pomodoro") now legitimately settle around 0.36
# and would otherwise be filtered out.
CREDIBILITY_PRUNE_THRESHOLD: float = 0.30

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
# Reinforcement-Divergence Auto-Resolution
# ---------------------------------------------------------------------------
# After each reinforcement, the system checks whether the reinforced behavior
# is involved in any PENDING conflict.  If the two conflicting behaviors have
# diverged enough in both reinforcement count AND credibility, the conflict is
# automatically resolved in favour of the stronger behaviour (OLD_WINS / NEW_WINS).
#
# Thresholds (either condition triggers resolution):
#   reinforcement_gap ≥ AUTO_RESOLVE_MIN_REINFORCEMENT_GAP  OR
#   credibility_gap   ≥ AUTO_RESOLVE_MIN_CREDIBILITY_GAP
# ⇒ stronger behaviour wins, weaker is SUPERSEDED.
#
# If neither threshold is met but the conflict is older than
# CONFLICT_EXPIRY_DAYS, the system expires it as BOTH_CORRECT (the user
# clearly doesn't mind both coexisting).

AUTO_RESOLVE_MIN_REINFORCEMENT_GAP: int = 3
AUTO_RESOLVE_MIN_CREDIBILITY_GAP: float = 0.15

CONFLICT_EXPIRY_DAYS: int = 30
CONFLICT_EXPIRY_SECONDS: int = CONFLICT_EXPIRY_DAYS * 24 * 60 * 60  # 2 592 000 s

# ---------------------------------------------------------------------------
# Similarity / retrieval thresholds
# ---------------------------------------------------------------------------

# Maximum cosine distance for a candidate to be considered semantically
# relevant during the store_behavior pipeline (duplicate / conflict gate).
SEMANTIC_RELEVANCE_THRESHOLD: float = 0.6

# Maximum cosine distance at which two behaviors with the same intent +
# same polarity + same context are upgraded to DUPLICATE even if their
# target strings differ.  Was hardcoded as 0.15 in extractor.py; raised to
# 0.28 to catch paraphrases like "dark mode" / "dark theme" or short vs
# verbose forms without admitting false duplicates (precision was 1.00 with
# headroom to spare).
PARAPHRASE_DUPLICATE_DISTANCE: float = 0.28

# Secondary gate applied on top of PARAPHRASE_DUPLICATE_DISTANCE: the
# *targets themselves* (not the full canonical sentence) must also be
# semantically close before two behaviors can be merged as DUPLICATE.
# Without this, the canonical-sentence distance can be tight purely because
# intent + context + polarity match — letting through cases like
# "fastapi" vs "asynchronous support" (same web-framework context,
# disjoint concepts).  Tuned for short target phrases; loosen if true
# paraphrases like "dark mode" / "dark theme" start being missed.
PARAPHRASE_TARGET_DISTANCE: float = 0.30

# Maximum cosine *distance* value used by the /v2/extract endpoint to
# decide which retrieved behaviors are returned to the caller as "related".
# (distance = 1 - hybrid_score, so lower is closer)
# 0.73 provides a small safety margin for behaviors whose densely-computed
# distance lands just above 0.72 after intent boosting.
RELATED_BEHAVIORS_DISTANCE_THRESHOLD: float = 0.73

# ---------------------------------------------------------------------------
# Hybrid Multi-Signal Behavior Retrieval (HMBR) — replaces the old LRA
# ---------------------------------------------------------------------------
# Pipeline:
#   Pillar 1: Multi-signal candidate generation
#             - prose-embedding cosine over `embedding`
#             - canonical-embedding cosine over `canonical_embedding`
#             - lexical BM25 over `search_vector`
#             union → ~80-150 candidate pool, each scored on every signal
#   Pillar 2: Personalised PageRank graph expansion
#             - seeds: candidates from Pillar 1, mass ∝ S_seed
#             - edges: CO_PROMPT (high), SEMANTIC_SIMILAR (medium),
#                      CO_SESSION (low)
#             - 4 power-iterations with α=0.15 restart probability
#   Pillar 3: Adaptive fusion + elbow selection
#             - S_final = Σ w_i · S_i  (weights chosen by query_type)
#             - cut by knee detection on sorted scores; soft cap fallback
# ---------------------------------------------------------------------------

# Per-probe per-lane top-K fetched from DB before fusion.
# 20 gives up to 20×3lanes×3probes=180 candidate slots (many overlap) — ample
# for fusion while saving ~33% DB scan time vs the original 30.
HMBR_PER_LANE_TOP_K: int = 20

# Lexical-lane minimum BM25-style match score (ts_rank); below this we
# treat the row as "not a lexical hit" and assign S_lexical = 0.
HMBR_LEXICAL_MIN_RANK: float = 0.01

# Recency time constant (days).  S_recency = exp(-Δt / τ).
HMBR_RECENCY_TAU_DAYS: float = 14.0

# Same-session boost: behaviors in the *current* session get this much
# added to S_final after fusion, before elbow detection.
HMBR_SESSION_BOOST: float = 0.15

# Hard cap on returned results after elbow detection.
HMBR_MAX_RESULTS: int = 12

# Soft floor: a candidate must have S_final ≥ this to be returned even
# if elbow detection includes it.  Prevents a thin tail from leaking in
# when the score distribution is flat.
HMBR_MIN_FINAL_SCORE: float = 0.20

# ----- Personalised PageRank --------------------------------------------
# Restart probability (a.k.a. teleport rate) — fraction of mass returned
# to the seed distribution at each iteration.  α=0.15 is the standard
# choice; lower values let mass diffuse further into the graph.
HMBR_PPR_ALPHA: float = 0.15
HMBR_PPR_ITERATIONS: int = 4

# Edge-type weights for the transition matrix.  CO_PROMPT propagates
# the most because direct co-occurrence is the strongest signal;
# SEMANTIC_SIMILAR is medium; CO_SESSION is loose.
HMBR_EDGE_WEIGHT_CO_PROMPT: float = 1.0
HMBR_EDGE_WEIGHT_SEMANTIC_SIMILAR: float = 0.7
HMBR_EDGE_WEIGHT_CO_SESSION: float = 0.4

# Maximum graph nodes loaded for PPR.  Above this we sample down by
# proximity to seeds — keeps the PPR matrix size bounded for users with
# very large behavior libraries.
HMBR_GRAPH_MAX_NODES: int = 400

# ----- SEMANTIC_SIMILAR edge generation ---------------------------------
# When a behavior is inserted, find its top-N closest existing behaviors
# by canonical_embedding and create SEMANTIC_SIMILAR edges to them when
# their cosine distance is below the threshold.
HMBR_SEMANTIC_SIMILAR_TOP_N: int = 5
HMBR_SEMANTIC_SIMILAR_MAX_DISTANCE: float = 0.25  # i.e. similarity ≥ 0.75

# ----- Fusion weights per query_type ------------------------------------
# Each row sums to 1.0.  Lanes:  sem | canon | lex | rec | cred | useful | ppr
HMBR_FUSION_WEIGHTS: dict[str, dict[str, float]] = {
    "NARROW": {
        "sem":   0.35, "canon": 0.20, "lex":   0.25,
        "rec":   0.05, "cred":  0.05, "useful": 0.05, "ppr": 0.05,
    },
    "BROAD": {
        "sem":   0.22, "canon": 0.20, "lex":   0.10,
        "rec":   0.08, "cred":  0.10, "useful": 0.05, "ppr": 0.25,
    },
    "EXPLORATORY": {
        "sem":   0.10, "canon": 0.10, "lex":   0.05,
        "rec":   0.25, "cred":  0.15, "useful": 0.10, "ppr": 0.25,
    },
    "TASK": {
        "sem":   0.20, "canon": 0.25, "lex":   0.10,
        "rec":   0.05, "cred":  0.05, "useful": 0.05, "ppr": 0.30,
    },
    "RECALL": {
        "sem":   0.15, "canon": 0.10, "lex":   0.30,
        "rec":   0.25, "cred":  0.05, "useful": 0.05, "ppr": 0.10,
    },
}

# ----- Multi-probe HyDE -------------------------------------------------
# The LLM emits 1..MAX_STANDALONE_QUERIES retrieval probes per prompt.
MAX_STANDALONE_QUERIES: int = 3

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
