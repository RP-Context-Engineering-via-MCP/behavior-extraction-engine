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
    BASE_REINFORCEMENT_BOOST,
    CONFIRMATION_EXPIRATION_DAYS,
    CONFIRMATION_EXPIRATION_SECONDS,
    CREDIBILITY_PRUNE_THRESHOLD,
    CREDIBILITY_WEIGHTS,
    DECAY_GRACE_PERIOD_DAYS,
    DECAY_GRACE_PERIOD_SECONDS,
    DEFAULT_DECAY_RATE,
    HYBRID_DENSE_WEIGHT,
    HYBRID_INTENT_BOOST_WEIGHT,
    HYBRID_SCORE_THRESHOLD,
    HYBRID_SEARCH_LIMIT,
    HYBRID_SPARSE_WEIGHT,
    INTENT_AFFINITY,
    INTENT_DECAY_RATES,
    MAX_RETRIEVAL_RESULTS,
    RELATED_BEHAVIORS_DISTANCE_THRESHOLD,
    RELEVANCE_GAP_DROP_RATIO,
    SEMANTIC_RELEVANCE_THRESHOLD,
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
    # --- algorithm constants (re-exported from constants.py) --------------
    "ALL_INTENT_TYPES",
    "BASE_REINFORCEMENT_BOOST",
    "CONFIRMATION_EXPIRATION_DAYS",
    "CONFIRMATION_EXPIRATION_SECONDS",
    "CREDIBILITY_PRUNE_THRESHOLD",
    "CREDIBILITY_WEIGHTS",
    "DECAY_GRACE_PERIOD_DAYS",
    "DECAY_GRACE_PERIOD_SECONDS",
    "DEFAULT_DECAY_RATE",
    "HYBRID_DENSE_WEIGHT",
    "HYBRID_INTENT_BOOST_WEIGHT",
    "HYBRID_SCORE_THRESHOLD",
    "HYBRID_SEARCH_LIMIT",
    "HYBRID_SPARSE_WEIGHT",
    "INTENT_AFFINITY",
    "INTENT_DECAY_RATES",
    "MAX_RETRIEVAL_RESULTS",
    "RELATED_BEHAVIORS_DISTANCE_THRESHOLD",
    "RELEVANCE_GAP_DROP_RATIO",
    "SEMANTIC_RELEVANCE_THRESHOLD",
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
    embed_model: str = "text-embedding-3-large"

    # Database -------------------------------------------------------------------
    database_url: str

    # Supabase (optional — only required if the Supabase client is used) ----------
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

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

SAMPLE_USERID: str = _s.sample_userid
