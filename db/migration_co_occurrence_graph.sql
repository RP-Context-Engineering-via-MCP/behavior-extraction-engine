-- ==========================================================================
-- Migration: Behavior Co-Occurrence Graph
-- ==========================================================================
-- Creates the behavior_co_occurrences table and its indexes for the
-- graph-based retrieval expansion layer.
--
-- This table stores directional edges between behaviors that were
-- expressed together (same prompt or same session).  The graph is
-- queried at retrieval time to surface associated behaviors that are
-- embedding-distant from the query but pragmatically related.
--
-- Run this ONCE after deploying the code changes.
-- ==========================================================================

-- 1. Create the co-occurrence table
CREATE TABLE IF NOT EXISTS behavior_co_occurrences (
    behavior_id_1  TEXT    NOT NULL,
    behavior_id_2  TEXT    NOT NULL,
    user_id        TEXT    NOT NULL,
    edge_type      TEXT    NOT NULL
        CHECK (edge_type IN ('CO_PROMPT', 'CO_SESSION')),
    weight         FLOAT  NOT NULL DEFAULT 1.0,
    created_at     BIGINT NOT NULL,

    -- Composite PK prevents duplicate edges of the same type
    PRIMARY KEY (behavior_id_1, behavior_id_2, edge_type),

    -- Referential integrity to the behaviors table
    CONSTRAINT fk_co_occ_behavior_1
        FOREIGN KEY (behavior_id_1, user_id)
        REFERENCES behaviors(behavior_id, user_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_co_occ_behavior_2
        FOREIGN KEY (behavior_id_2, user_id)
        REFERENCES behaviors(behavior_id, user_id)
        ON DELETE CASCADE
);

-- 2. Create indexes for fast 1-hop graph expansion
--    Both directions are needed because edges are stored once (a→b)
--    but queried from either side.

-- Forward lookup: given behavior_id_1, find all neighbors
CREATE INDEX IF NOT EXISTS idx_co_occ_forward
ON behavior_co_occurrences(user_id, behavior_id_1);

-- Reverse lookup: given behavior_id_2, find all neighbors
CREATE INDEX IF NOT EXISTS idx_co_occ_reverse
ON behavior_co_occurrences(user_id, behavior_id_2);

-- 3. Composite index for UPSERT conflict detection (ON CONFLICT path)
--    The PK already covers (behavior_id_1, behavior_id_2, edge_type),
--    so this is handled automatically.

-- ==========================================================================
-- Verification: run after migration to confirm
-- ==========================================================================
-- SELECT tablename FROM pg_tables WHERE tablename = 'behavior_co_occurrences';
-- SELECT indexname FROM pg_indexes WHERE tablename = 'behavior_co_occurrences';
