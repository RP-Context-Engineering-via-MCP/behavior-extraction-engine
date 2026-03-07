"""
Structured JSON logging configuration.

Why JSON?
  Promtail ships log lines to Loki as-is.  If each line is valid JSON Loki
  can index individual fields, enabling LogQL queries like:
    {service="behavior-detection"} | json | level="ERROR"
    {service="behavior-detection"} | json | user_id="abc123"

Usage:
    from config.logging_config import setup_logging
    setup_logging()          # call once at process start (app.py / run.py)
"""

import logging
import os
import sys

from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str | None = None) -> None:
    """
    Replace the root logger's handler with a JSON-emitting StreamHandler.

    All loggers created with logging.getLogger(__name__) will inherit this
    format automatically.

    Args:
        log_level: Override the log level.  Falls back to the LOG_LEVEL env
                   variable, then to INFO.
    """
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    # Build a JSON formatter.
    # The 'fmt' string lists which LogRecord attributes to include as top-level
    # JSON keys in every log line.
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        },
        datefmt="%Y-%m-%dT%H:%M:%S",
        static_fields={
            "service": os.getenv("SERVICE_NAME", "behavior-detection"),
            "env": os.getenv("APP_ENV", "development"),
        },
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
