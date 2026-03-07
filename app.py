"""
Application factory for the Behavior Extraction API.

This module is intentionally thin:
  - Creates the FastAPI instance with its lifespan (startup / shutdown hooks)
  - Registers CORS middleware
  - Includes the APIRouter that owns all route definitions

Business logic lives in services/.
HTTP schemas live in api/schemas.py.
Route handlers live in api/router.py.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router
from config.logging_config import setup_logging
from db.connection import close_db_pool, init_db_pool

# Initialise structured JSON logging before anything else logs a message
setup_logging()
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

# Register all API routes
app.include_router(router)

