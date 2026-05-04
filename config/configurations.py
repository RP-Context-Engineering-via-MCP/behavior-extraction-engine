"""
Runtime configuration for the Behavior Detection and Management system.

Environment-backed secrets and service URLs are validated here via
pydantic-settings.  A missing required variable raises a clear error
at process startup rather than silently returning None mid-request.

Pure algorithm constants (thresholds, weights, decay rates, etc.) live
in config/constants.py and are re-exported here so all existing
`from config.configurations import X` statements keep working without
change.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import (  # re-export for backward compatibility
    ALL_INTENT_TYPES,
    AUTO_RESOLVE_MIN_CREDIBILITY_GAP,
    AUTO_RESOLVE_MIN_REINFORCEMENT_GAP,
    BASE_REINFORCEMENT_BOOST,
    CONFIRMATION_EXPIRATION_DAYS,
    CONFIRMATION_EXPIRATION_SECONDS,
    CONFLICT_EXPIRY_DAYS,
    CONFLICT_EXPIRY_SECONDS,
    CREDIBILITY_PRUNE_THRESHOLD,
    CREDIBILITY_WEIGHTS,
    DECAY_GRACE_PERIOD_DAYS,
    DECAY_GRACE_PERIOD_SECONDS,
    DEFAULT_DECAY_RATE,
    INTENT_AFFINITY,
    INTENT_DECAY_RATES,
    MAX_STANDALONE_QUERIES,
    PARAPHRASE_DUPLICATE_DISTANCE,
    PARAPHRASE_TARGET_DISTANCE,
    RELATED_BEHAVIORS_DISTANCE_THRESHOLD,
    SEMANTIC_RELEVANCE_THRESHOLD,
    # --- HMBR ---
    HMBR_PER_LANE_TOP_K,
    HMBR_LEXICAL_MIN_RANK,
    HMBR_RECENCY_TAU_DAYS,
    HMBR_SESSION_BOOST,
    HMBR_MAX_RESULTS,
    HMBR_MIN_FINAL_SCORE,
    HMBR_PPR_ALPHA,
    HMBR_PPR_ITERATIONS,
    HMBR_EDGE_WEIGHT_CO_PROMPT,
    HMBR_EDGE_WEIGHT_SEMANTIC_SIMILAR,
    HMBR_EDGE_WEIGHT_CO_SESSION,
    HMBR_GRAPH_MAX_NODES,
    HMBR_SEMANTIC_SIMILAR_TOP_N,
    HMBR_SEMANTIC_SIMILAR_MAX_DISTANCE,
    HMBR_FUSION_WEIGHTS,
)

__all__ = [
    # --- environment-backed names (loaded below) --------------------------
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_VERSION",
    "GPT_MODEL",
    "EMBED_MODEL",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "DATABASE_URL",
    "SAMPLE_USERID",
    "REDIS_URL",
    "REDIS_STREAM_NAME",
    "REDIS_EVENTS_ENABLED",
    "USER_MANAGEMENT_SERVICE_BASE_URL",
    "PROFILE_SIGNALS_DEFAULT_LIMIT",
    "PROFILE_SIGNALS_MAX_LIMIT",
    # --- algorithm constants (re-exported from constants.py) --------------
    "ALL_INTENT_TYPES",
    "AUTO_RESOLVE_MIN_CREDIBILITY_GAP",
    "AUTO_RESOLVE_MIN_REINFORCEMENT_GAP",
    "BASE_REINFORCEMENT_BOOST",
    "CONFIRMATION_EXPIRATION_DAYS",
    "CONFIRMATION_EXPIRATION_SECONDS",
    "CONFLICT_EXPIRY_DAYS",
    "CONFLICT_EXPIRY_SECONDS",
    "CREDIBILITY_PRUNE_THRESHOLD",
    "CREDIBILITY_WEIGHTS",
    "DECAY_GRACE_PERIOD_DAYS",
    "DECAY_GRACE_PERIOD_SECONDS",
    "DEFAULT_DECAY_RATE",
    "INTENT_AFFINITY",
    "INTENT_DECAY_RATES",
    "MAX_STANDALONE_QUERIES",
    "PARAPHRASE_DUPLICATE_DISTANCE",
    "PARAPHRASE_TARGET_DISTANCE",
    "RELATED_BEHAVIORS_DISTANCE_THRESHOLD",
    "SEMANTIC_RELEVANCE_THRESHOLD",
    # HMBR
    "HMBR_PER_LANE_TOP_K",
    "HMBR_LEXICAL_MIN_RANK",
    "HMBR_RECENCY_TAU_DAYS",
    "HMBR_SESSION_BOOST",
    "HMBR_MAX_RESULTS",
    "HMBR_MIN_FINAL_SCORE",
    "HMBR_PPR_ALPHA",
    "HMBR_PPR_ITERATIONS",
    "HMBR_EDGE_WEIGHT_CO_PROMPT",
    "HMBR_EDGE_WEIGHT_SEMANTIC_SIMILAR",
    "HMBR_EDGE_WEIGHT_CO_SESSION",
    "HMBR_GRAPH_MAX_NODES",
    "HMBR_SEMANTIC_SIMILAR_TOP_N",
    "HMBR_SEMANTIC_SIMILAR_MAX_DISTANCE",
    "HMBR_FUSION_WEIGHTS",
]


# ---------------------------------------------------------------------------
# Settings — validated at startup via pydantic-settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Environment-backed configuration.

    All fields without a default are *required* — the process will refuse to
    start if they are absent from the environment / .env file, which prevents
    subtle None-related failures deep inside request handling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # Ignore unknown env vars — keeps .env flexible
    )

    # Azure OpenAI ----------------------------------------------------------------
    azure_openai_endpoint: str
    azure_openai_key: str
    # These are model/API version constants that could theoretically be overridden
    # per deployment, so we keep them here with sensible defaults.
    azure_openai_api_version: str = "2024-12-01-preview"
    gpt_model: str = "gpt-4.1-mini"
    embed_model: str = "all-MiniLM-L6-v2"

    # Database -------------------------------------------------------------------
    database_url: str

    # Supabase (optional — only required if the Supabase client is used) ----------
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    # Redis (for Drift Detection Service integration) ----------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_name: str = "behavior.events"
    redis_events_enabled: bool = True

    # User Management Service Integration ----------------------------------------
    user_management_service_base_url: str = "http://user-management-service:8080"
    profile_signals_default_limit: int = 10
    profile_signals_max_limit: int = 50

    # Misc -----------------------------------------------------------------------
    sample_userid: str = "user_12345"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated Settings singleton (created once, cached forever)."""
    return Settings()


# ---------------------------------------------------------------------------
# Module-level flat names — kept for backward compatibility so that all
# existing `from config.configurations import X` calls keep working.
# ---------------------------------------------------------------------------

_s = get_settings()

AZURE_OPENAI_ENDPOINT: str = _s.azure_openai_endpoint
AZURE_OPENAI_KEY: str = _s.azure_openai_key
AZURE_OPENAI_API_VERSION: str = _s.azure_openai_api_version
GPT_MODEL: str = _s.gpt_model
EMBED_MODEL: str = _s.embed_model

DATABASE_URL: str = _s.database_url

SUPABASE_URL: Optional[str] = _s.supabase_url
SUPABASE_KEY: Optional[str] = _s.supabase_key

REDIS_URL: str = _s.redis_url
REDIS_STREAM_NAME: str = _s.redis_stream_name
REDIS_EVENTS_ENABLED: bool = _s.redis_events_enabled

USER_MANAGEMENT_SERVICE_BASE_URL: str = _s.user_management_service_base_url
PROFILE_SIGNALS_DEFAULT_LIMIT: int = _s.profile_signals_default_limit
PROFILE_SIGNALS_MAX_LIMIT: int = _s.profile_signals_max_limit

SAMPLE_USERID: str = _s.sample_userid
