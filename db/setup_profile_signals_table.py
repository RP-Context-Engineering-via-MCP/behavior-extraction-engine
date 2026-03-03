"""
Setup Profile Signals Table - Python Script
============================================================================
This script creates the user_profile_signals table in your PostgreSQL database

Prerequisites:
- psycopg installed (already in requirements.txt)
- Database connection details in .env file

Usage:
    python setup_profile_signals_table.py
============================================================================
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import psycopg
except ImportError:
    print("Error: psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(1)

# SQL to create the table
CREATE_TABLE_SQL = """
-- Profile Signals Table (for Profile Service Integration)
CREATE TABLE IF NOT EXISTS user_profile_signals (
    id              UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         TEXT    NOT NULL,
    prompt_id       TEXT    NOT NULL,
    profile_signals JSONB   NOT NULL,
    extracted_at    BIGINT  NOT NULL,
    UNIQUE (user_id, prompt_id)
);
"""

CREATE_INDEX_SQL = """
-- Index for efficient retrieval of recent signals per user
CREATE INDEX IF NOT EXISTS idx_user_profile_signals_user_extracted
    ON user_profile_signals (user_id, extracted_at DESC);
"""

VERIFY_SQL = """
SELECT 
    COUNT(*) as row_count,
    'user_profile_signals' as table_name
FROM user_profile_signals;
"""

def main():
    print("=" * 60)
    print("Profile Signals Table Setup")
    print("=" * 60)
    print()
    
    # Load environment variables
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not found in .env file!")
        sys.exit(1)
    
    print(f"Database URL: {database_url}")
    print()
    
    try:
        print("Connecting to database...")
        conn = psycopg.connect(database_url)
        
        print("Creating user_profile_signals table...")
        with conn.cursor() as cur:
            # Create table
            cur.execute(CREATE_TABLE_SQL)
            print("  ✓ Table created")
            
            # Create index
            cur.execute(CREATE_INDEX_SQL)
            print("  ✓ Index created")
            
            # Commit changes
            conn.commit()
            
            # Verify
            cur.execute(VERIFY_SQL)
            result = cur.fetchone()
            print()
            print(f"Verification:")
            print(f"  Table: {result[1]}")
            print(f"  Rows: {result[0]}")
        
        print()
        print("=" * 60)
        print("✓ Setup Complete!")
        print("=" * 60)
        print()
        print("The user_profile_signals table has been created successfully.")
        print("You can now restart your API server.")
        
    except psycopg.Error as e:
        print()
        print("✗ Database Error:")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print()
        print("✗ Unexpected Error:")
        print(f"  {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
