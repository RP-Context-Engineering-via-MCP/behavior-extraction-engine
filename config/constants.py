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
# Layered Retrieval Architecture (LRA) — replaces the old additive TGHR
# ---------------------------------------------------------------------------
# Pipeline:
#   Stage 1: Pure dense (cosine) retrieval from DB → top K candidates
#   Stage 2: In-memory multiplicative intent re-ranking
#            S_base = 1 - D_cosine
#            M = 1 + (INTENT_RERANK_ALPHA * affinity)
#            S_final = S_base * M
#   Stage 3: Absolute semantic floor + relevance-gap dynamic cutoff
# ---------------------------------------------------------------------------

# Maximum candidates fetched from DB before in-memory re-ranking.
# 40 provides enough headroom for users with ~50-100 stored behaviors.
HYBRID_SEARCH_LIMIT: int = 40

# Intent re-ranking alpha — multiplicative weight for intent affinity.
# S_final = S_base * (1 + INTENT_RERANK_ALPHA * affinity)
# At alpha=0.35, a perfect intent match lifts the score by 35%.
# A zero-affinity intent leaves the score unchanged.
# Raised from 0.25 → 0.35 to better recover on-intent near-misses
# under all-MiniLM-L6's diffuse cosine geometry.
INTENT_RERANK_ALPHA: float = 0.35

# Absolute semantic floor — any behavior with S_final below this is
# mathematically discarded.  Prevents injecting weakly-related behaviors
# into the LLM context window, reducing hallucination risk.
# Lowered from 0.48 → 0.40 to reflect MiniLM-L6's diffuse geometry on
# abstract↔concrete asymmetric query patterns (e.g., "QA practices" ↔
# "always writes unit tests"), then 0.40 → 0.35 to recover near-misses
# that the relevance-gap stage was already correctly ranking but the
# floor was discarding before the gap stage could see them.
SEMANTIC_FLOOR_THRESHOLD: float = 0.35

# Soft-fallback floor — only activated when the primary floor (above)
# drops ALL candidates to zero results.  Recovers near-miss behaviors
# that the strict floor filters out.  Because this only fires when the
# primary search returns nothing, it CANNOT affect already-passing queries.
# Must be < SEMANTIC_FLOOR_THRESHOLD to actually rescue anything (the
# previous 0.48 == 0.48 made this branch a no-op).
SEMANTIC_FLOOR_FALLBACK: float = 0.32

# Maximum results returned in fallback mode.  Capped low to prevent
# noise from diluting the LLM context when the match quality is marginal.
MAX_FALLBACK_RESULTS: int = 3

# Relevance-gap cutoff ratio (Top-Score Relative Drop-off).
# T_dynamic = S_max * (1 - RELEVANCE_GAP_DROP_RATIO)
# Any behaviour below T_dynamic is cut.
# Loosened from 0.15 → 0.30 → 0.45 progressively as we observed that
# even 0.30 was over-pruning: with S_max ≈ 0.7 (typical good match)
# T_dynamic was 0.49, which on diffuse 384-dim cosine left only the
# top-1 candidate. 0.45 keeps the second/third tier visible while the
# absolute SEMANTIC_FLOOR_THRESHOLD still guards against tail noise.
RELEVANCE_GAP_DROP_RATIO: float = 0.45

# Hard cap on the number of results returned after gap filtering.
# Prevents over-retrieval on broad / vague queries.
MAX_RETRIEVAL_RESULTS: int = 10

# ---------------------------------------------------------------------------
# Multi-probe HyDE retrieval
# ---------------------------------------------------------------------------
# The LLM emits 1..MAX_STANDALONE_QUERIES short canonical probes per user
# prompt.  Each probe is embedded and used to query pgvector independently.
# Per-candidate scoring uses the BEST (minimum) cosine distance across
# probes, with a 10%-per-extra-probe agreement multiplier when a candidate
# appears in multiple probes' top-K.
#
# Multiple probes attack the abstract↔concrete vocabulary asymmetry
# inherent to conversational-prompt → canonical-stored-behavior retrieval
# (e.g., "modern web frontend" ↔ "uses React for frontend").  Keeping the
# scoring on the existing s_base scale means τ_min/ρ thresholds carry over
# without recalibration.
MAX_STANDALONE_QUERIES: int = 3

# ---------------------------------------------------------------------------
# Graph expansion relevance gate
# ---------------------------------------------------------------------------
# After 1-hop co-occurrence walk in get_graph_expanded_behaviors, every
# neighbor is checked against the query embedding(s) and dropped if its
# best cosine distance to any probe exceeds the edge-type-specific
# threshold below.
#
# Without any gate, graph expansion replays write-time co-occurrence
# (e.g., "Python" and "cooking" mentioned in the same prompt) into
# read-time noise — even when dense retrieval correctly excluded them.
#
# CO_PROMPT edges are the riskiest source of cross-domain noise: any
# two behaviors mentioned in one prompt are linked, regardless of how
# unrelated they are.  Apply a tight gate here.
#
# CO_SESSION edges already carry an implicit pragmatic association —
# the user kept these behaviors together inside one session boundary,
# so even topically distant neighbors are usually relevant context.
# Use a much looser gate so that, e.g., "I'm working on a Django
# project, and I prefer dark mode" still surfaces the dark-mode
# preference when the query is about Django, without needing semantic
# proximity between the two concepts.
GRAPH_EXPANSION_DISTANCE_THRESHOLD: float = 0.55
GRAPH_EXPANSION_SESSION_DISTANCE_THRESHOLD: float = 0.85

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
