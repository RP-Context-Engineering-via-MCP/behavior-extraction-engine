"""
Application factory for the Behavior Extraction API.

This module is intentionally thin:
  - Creates the FastAPI instance with its lifespan (startup / shutdown hooks)
  - Registers CORS middleware and static-file mount
  - Includes the APIRouter that owns all route definitions

Business logic lives in services/.
HTTP schemas live in api/schemas.py.
Route handlers live in api/router.py.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.router import router
from db.connection import close_db_pool, init_db_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database connection pool")
    init_db_pool()
    logger.info("Database connection pool initialized")

    yield

    logger.info("Shutting down database connection pool")
    close_db_pool()
    logger.info("Database connection pool shut down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Behavior Extraction API",
    description="Extract and manage user behaviors from natural language prompts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: specify exact origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend assets (optional - only mount if directory exists)
frontend_dir = Path("frontend")
if frontend_dir.exists() and frontend_dir.is_dir():
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
    logger.info("Mounted frontend static files at /frontend")
else:
    logger.info("Frontend directory not found, skipping static files mount")

# Register all API routes
app.include_router(router)

