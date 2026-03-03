-- ============================================================================
-- Profile Signals Table Creation Script
-- ============================================================================
-- This script creates the user_profile_signals table needed for Profile 
-- Service integration (cold-start profiling and drift fallback).
--
-- Run this script if you get the error:
-- "relation 'user_profile_signals' does not exist"
--
-- Usage:
--   psql -U username -d database_name -f create_profile_signals_table.sql
-- ============================================================================

-- Create user_profile_signals table
CREATE TABLE IF NOT EXISTS user_profile_signals (
    id              UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         TEXT    NOT NULL,
    prompt_id       TEXT    NOT NULL,
    profile_signals JSONB   NOT NULL,
    extracted_at    BIGINT  NOT NULL,
    UNIQUE (user_id, prompt_id)
);

-- Create index for efficient retrieval of recent signals per user
-- This index is used by the drift fallback endpoint to quickly fetch
-- the most recent N signals for a user, ordered by extracted_at DESC
CREATE INDEX IF NOT EXISTS idx_user_profile_signals_user_extracted
    ON user_profile_signals (user_id, extracted_at DESC);

-- Verify table creation
SELECT 
    'user_profile_signals table created successfully' AS status,
    COUNT(*) AS row_count
FROM user_profile_signals;

-- Display table structure
\d user_profile_signals
