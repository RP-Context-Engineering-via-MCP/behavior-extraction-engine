import os
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from config.configurations import DATABASE_URL
import threading
import logging

logger = logging.getLogger(__name__)


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment variables.")

def get_db_connection():
    """
        Return a psycopg3 connection with pgvector registered.
        Automatically ensures pgvector extension is available.
    """
    try:
        connection = psycopg.connect(DATABASE_URL, autocommit=False)
        register_vector(connection)

        logger.info("Database connection established and pgvector extension ensured.")
        return connection
    except psycopg.OperationalError as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise RuntimeError(f"Database connection error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database connection: {e}")
        raise

_pool = None
_pool_lock = threading.Lock()

def init_db_pool():
    global _pool
    _pool = ConnectionPool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        configure=lambda conn: register_vector(conn)
    )

def get_db_pool_connection():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                init_db_pool()
    return _pool.connection()

def close_db_pool():
    """close the connection pool greacefully"""
    global _pool
    if _pool is not None:
        _pool.close()
        logger.info("Database connection pool closed.")