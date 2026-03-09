"""
Logging configuration.

Usage:
    from config.logging_config import setup_logging
    setup_logging()          # call once at process start (app.py / run.py)
"""

import logging
import os
import sys


def setup_logging(log_level: str | None = None) -> None:
    """
    Configure basic logging for the application.

    All loggers created with logging.getLogger(__name__) will inherit this
    format automatically.

    Args:
        log_level: Override the log level.  Falls back to the LOG_LEVEL env
                   variable, then to INFO.
    """
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
