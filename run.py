#!/usr/bin/env python3
"""
FastAPI service runner for Behavior Detection and Management system.
Run this script to start the service: python run.py
"""

import uvicorn

from config.logging_config import setup_logging

# Configure logging before uvicorn starts
setup_logging()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info",  # Changed from "info" to "debug"
        access_log=True
    )
