import os
import psycopg
from pgvector.psycopg import register_vector
from config.configurations import DATABASE_URL
import logging

logger = logging.getLogger(__name__)


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment variables.")

def get_db_connection():
    """
    return a psycopg3 connection with pgvactor registered
    """
    try:
        connection = psycopg.connect(DATABASE_URL, autocommit=False)
        register_vector(connection)
        logger.info("Database connection established and pgvector registered.")
        return connection
    except psycopg.OperationalError as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise RuntimeError(f"Database connection error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database connection: {e}")
        raise