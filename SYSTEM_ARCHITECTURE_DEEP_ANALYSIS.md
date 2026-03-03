# Behavior Extraction Engine - Deep System Architecture Analysis

> **Document Purpose**: This document provides a 100% comprehensive analysis of the Behavior Extraction Engine system architecture, components, data flows, algorithms, and implementation details. It is designed to enable any LLM or developer to fully understand the entire system without ambiguity.

**Last Updated**: March 3, 2026  
**System Version**: 1.0.0  
**Technology Stack**: Python 3.11, FastAPI, PostgreSQL with pgvector, Azure OpenAI GPT-4.1-mini, Redis Streams

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Architecture](#2-core-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Data Models](#4-data-models)
5. [Database Schema](#5-database-schema)
6. [Core Services](#6-core-services)
7. [API Endpoints](#7-api-endpoints)
8. [Algorithms and Business Logic](#8-algorithms-and-business-logic)
9. [Configuration System](#9-configuration-system)
10. [External Integrations](#10-external-integrations)
11. [Testing Strategy](#11-testing-strategy)
12. [Deployment](#12-deployment)
13. [Data Flow Diagrams](#13-data-flow-diagrams)
14. [Security and Performance](#14-security-and-performance)
15. [Future Extensibility](#15-future-extensibility)

---

## 1. System Overview

### 1.1 Purpose

The **Behavior Extraction Engine** is a sophisticated AI-powered system that:
- **Extracts** user behaviors, preferences, constraints, habits, skills, and communication patterns from natural language text
- **Canonicalizes** behaviors into structured, machine-readable forms (intent, target, context, polarity)
- **Stores** behaviors with semantic embeddings in a vector database
- **Manages** behavior lifecycle including duplicate detection, conflict resolution, reinforcement, and decay
- **Retrieves** relevant behaviors using hybrid search (semantic + lexical + metadata)
- **Integrates** with Profile Service for cold-start user profiling and drift detection

### 1.2 Key Features

1. **AI-Powered Extraction**: Uses Azure OpenAI GPT-4.1-mini to extract behaviors with 3072-dimensional embeddings (text-embedding-3-large)
2. **Canonical Structure**: All behaviors are normalized into 4-tuple: (intent, target, context, polarity)
3. **Credibility Scoring**: Multi-factor scoring based on confidence, clarity, and linguistic strength
4. **Intelligent Deduplication**: Detects duplicates using semantic + structured matching
5. **Conflict Detection**: Identifies and resolves contradictory behaviors (polarity conflicts, cross-intent conflicts)
6. **Behavior State Management**: Tracks lifecycle (NEW → ACTIVE → SUPERSEDED/FLAGGED/ARCHIVED)
7. **Lazy Decay**: Time-based credibility decay with grace periods and intent-specific rates
8. **Hybrid Retrieval (TGHR)**: Tuple-Guided Hybrid Retrieval combining dense (semantic), sparse (BM25), and metadata (intent affinity)
9. **Profile Signal Extraction**: Extracts holistic user profile characteristics for profile matching
10. **Event Publishing**: Publishes behavior events to Redis Streams for drift detection service

### 1.3 System Context

The Behavior Extraction Engine operates as part of a larger ecosystem:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATIONS                       │
│                   (Chat UI, Mobile App, etc.)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               BEHAVIOR EXTRACTION ENGINE (This System)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Extraction  │  │   Storage    │  │  Retrieval   │          │
│  │   Service    │  │   Service    │  │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌──────────────┐
│   Azure     │  │  PostgreSQL    │  │    Redis     │
│   OpenAI    │  │  + pgvector    │  │   Streams    │
│  (GPT-4.1)  │  │   (Database)   │  │   (Events)   │
└─────────────┘  └────────────────┘  └──────┬───────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Drift Detection  │
                                   │     Service      │
                                   └──────────────────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Profile Service  │
                                   │ (Cold-Start &    │
                                   │ Profile Match)   │
                                   └──────────────────┘
```

---

## 2. Core Architecture

### 2.1 Layered Architecture

The system follows a **layered architecture** pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│              (FastAPI REST API Endpoints)                │
│                        app.py                            │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                      SERVICE LAYER                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  extractor  │  │  behavior   │  │ credibility │     │
│  │   .py       │  │ Repository  │  │ Calculator  │     │
│  │             │  │    .py      │  │    .py      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  openAI     │  │   profile   │  │    event    │     │
│  │  Client     │  │   Signal    │  │  Publisher  │     │
│  │   .py       │  │ Extractor   │  │    .py      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │coldStart    │  │  profile    │                      │
│  │Dispatcher   │  │   Service   │                      │
│  │   .py       │  │  Client.py  │                      │
│  └─────────────┘  └─────────────┘                      │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     DATA ACCESS LAYER                    │
│              (Database Connection Pool)                  │
│                  db/connection.py                        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     DATABASE LAYER                       │
│         PostgreSQL 15+ with pgvector extension           │
│   Tables: behaviors, behavior_conflicts,                 │
│   user_profile_signals, prompt_segments                  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### 2.2.1 Application Entry Points

- **`app.py`**: Main FastAPI application with all REST API endpoints, middleware, CORS, and lifecycle management
- **`run.py`**: Uvicorn server launcher for development with hot-reload enabled

#### 2.2.2 Configuration Layer

- **`config/configurations.py`**: Centralized configuration management
  - Environment variable loading via `python-dotenv`
  - Azure OpenAI credentials and model names
  - Database connection strings (Supabase/PostgreSQL)
  - Redis configuration for event publishing
  - All thresholds, weights, and algorithmic parameters
  - Profile Service integration URLs

#### 2.2.3 Data Models

- **`models/behavior.py`**: Pydantic models for type-safe data validation
  - `ExtractedBehavior`: Behavior extracted from LLM with scores
  - `StoredBehavior`: Behavior persisted in database with all fields
  - `ExtractionResult`: Response from extraction service
  - `BehaviorSegment`: Segment of prompt with associated behaviors
  - `SimilarityResult`: Result from similarity search
  - `ConflictAnalysisResult`: LLM conflict analysis output
  - Enums: `BehaviorState`, `ConflictType`, `RelationType`, etc.

#### 2.2.4 Service Layer

**Extraction & Processing:**
- **`services/extractor.py`**: Main orchestrator for behavior extraction workflow
- **`services/openAiClient.py`**: Azure OpenAI API communication (extraction, embedding, conflict analysis)
- **`services/credibilityCalculator.py`**: Multi-factor credibility scoring and decay calculations

**Storage & Retrieval:**
- **`services/behaviorRepository.py`**: Database operations (CRUD, search, reinforcement, conflicts)
- **`services/profileSignalRepository.py`**: Profile signal persistence and retrieval

**Profile Integration:**
- **`services/profileSignalExtractor.py`**: Validates profile signals against controlled vocabularies
- **`services/coldStartDispatcher.py`**: Orchestrates cold-start profile assignment flow
- **`services/profileServiceClient.py`**: HTTP client for Profile Service communication

**Event Publishing:**
- **`services/eventPublisher.py`**: Publishes behavior events to Redis Streams for drift detection

#### 2.2.5 Utility Layer

- **`utils/embedding_utils.py`**: Embedding generation helpers
- **`utils/similarity_utils.py`**: Distance metrics (cosine, euclidean, manhattan)

#### 2.2.6 Database Layer

- **`db/connection.py`**: PostgreSQL connection pool management with pgvector registration
- **`db/create_profile_signals_table.sql`**: Profile signals table schema
- **`db/existing_db_scripts.txt`**: Main database schemas for behaviors, conflicts, etc.

---

## 3. Technology Stack

### 3.1 Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Primary programming language |
| **Web Framework** | FastAPI | 0.115.6 | REST API with async support |
| **Server** | Uvicorn | 0.32.1 | ASGI server with hot-reload |
| **Database** | PostgreSQL | 15+ | Primary data store |
| **Vector Search** | pgvector | 0.3.6 | Vector similarity search extension |
| **DB Driver** | psycopg[binary] | 3.2.3 | PostgreSQL adapter with binary protocol |
| **Connection Pool** | psycopg-pool | 3.2.3 | Database connection pooling |
| **LLM Provider** | Azure OpenAI | GPT-4.1-mini | Behavior extraction and analysis |
| **Embedding Model** | text-embedding-3-large | 3072-dim | Semantic embedding generation |
| **Event Bus** | Redis | 5.0.1 | Event streaming for drift detection |
| **Validation** | Pydantic | 2.10.4 | Data validation and serialization |
| **HTTP Client** | httpx | 0.28.1 | Async HTTP for service communication |
| **Testing** | pytest | 8.3.4 | Unit and integration testing |

### 3.2 Python Dependencies

```
# Web Framework
fastapi==0.115.6
uvicorn[standard]==0.32.1

# OpenAI / Azure OpenAI
openai==1.57.4

# Database & Vector Support
psycopg[binary]==3.2.3
psycopg-pool==3.2.3
pgvector==0.3.6
supabase==2.11.0

# Data Validation
pydantic==2.10.4

# Environment Variables
python-dotenv==1.0.1

# HTTP Client
httpx==0.28.1

# Redis (for event publishing)
redis==5.0.1

# Testing
pytest==8.3.4
pytest-asyncio==0.24.0
```

### 3.3 Database Extensions

```sql
-- Required PostgreSQL extension
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for similarity search
```

---

## 4. Data Models

### 4.1 Core Behavior Models

#### 4.1.1 ExtractedBehavior (Extraction Output)

```python
class ExtractedBehavior(BaseModel):
    description: str                    # Human-readable behavior summary
    confidence: float                   # GPT confidence (0.0-1.0)
    clarity: float                      # Statement clarity (0.0-1.0)
    linguistic_strength: float          # Expression intensity (0.0-1.0)
    extracted_at: str                   # ISO timestamp
    
    # Canonical structure (4-tuple)
    intent: Optional[Literal[           # Behavioral intent category
        "PREFERENCE",                   # Soft desires/likes
        "CONSTRAINT",                   # Hard rules/restrictions
        "HABIT",                        # Frequency patterns
        "SKILL",                        # Capabilities/expertise
        "COMMUNICATION"                 # Interaction style
    ]]
    target: Optional[str]               # Primary object (concise noun)
    context: Optional[str]              # Scope (IDE, morning, etc.)
    polarity: Optional[Literal[         # Behavioral direction
        "POSITIVE",                     # Likes, prefers, uses
        "NEGATIVE"                      # Dislikes, avoids, restricts
    ]]
```

#### 4.1.2 StoredBehavior (Database Representation)

```python
class StoredBehavior(BaseModel):
    # Identity
    behavior_id: str                    # Unique ID (beh_xxxxxxxx)
    user_id: str                        # User identifier
    session_id: str                     # Session isolation key
    
    # Content
    behavior_text: str                  # Extracted behavior description
    embedding: List[float]              # 3072-dim vector (text-embedding-3-large)
    
    # Canonical structure
    intent: Literal["PREFERENCE"|"CONSTRAINT"|"HABIT"|"SKILL"|"COMMUNICATION"]
    target: str                         # Canonicalized target noun
    context: str                        # Context scope
    polarity: Literal["POSITIVE"|"NEGATIVE"]
    
    # Quality metrics
    credibility: float                  # Weighted score (0.0-1.0)
    extraction_confidence: float        # GPT confidence
    clarity_score: float                # Statement clarity
    linguistic_strength: float          # Expression intensity
    
    # Lifecycle management
    behavior_state: str                 # NEW|ACTIVE|SUPERSEDED|FLAGGED|ARCHIVED
    reinforcement_count: int            # Times reinforced (starts at 1)
    decay_rate: float                   # Decay speed (intent-specific)
    
    # Timestamps
    created_at: int                     # Unix timestamp (creation)
    last_seen_at: int                   # Unix timestamp (last reinforcement)
    last_decay_applied_at: int          # Unix timestamp (last decay computation)
    last_accessed_at: int               # Unix timestamp (last retrieval)
    
    # History
    prompt_history_ids: List[str]       # Prompts that triggered this behavior
    superseded_by_id: Optional[str]     # ID of behavior that replaced this
    related_behaviors: List[dict]       # Similar behaviors metadata
```

#### 4.1.3 Canonical Behavior Structure

Every behavior is normalized into a **4-tuple canonical form**:

```python
# Canonical 4-tuple
(intent, target, context, polarity)

# Examples:
("PREFERENCE", "Python", "backend", "POSITIVE")        # Prefers Python for backend
("CONSTRAINT", "gluten", "general", "NEGATIVE")        # Cannot eat gluten
("HABIT", "exercise", "morning", "POSITIVE")           # Exercises in the morning
("SKILL", "TypeScript", "frontend", "POSITIVE")        # Skilled in TypeScript
```

**Intent Types (ordered by precedence for conflict resolution):**

1. **CONSTRAINT**: Hard rules (cannot, must not, never, allergic to, forbidden)
   - Highest priority in conflicts
   - High linguistic strength (0.8+)
   - Very slow decay rate (0.001)
   
2. **PREFERENCE**: Soft desires (likes, prefers, enjoys, favors)
   - Medium priority
   - Medium decay rate (0.015)
   
3. **HABIT**: Frequency patterns (usually, always, regularly, tends to)
   - Medium priority
   - Faster decay rate (0.04)
   
4. **SKILL**: Capabilities (experienced with, proficient in, knows)
   - Lower priority in conflicts
   - Slow decay rate (0.005)
   
5. **COMMUNICATION**: Interaction style (prefers brief answers, wants examples)
   - Lowest priority
   - Medium decay rate (0.015)

**Target Canonicalization Rules:**

- Always use **full, standard, widely recognized names**
- **NEVER** use abbreviations or acronyms
- Examples:
  * `JS` → `JavaScript`
  * `TS` → `TypeScript`
  * `PY` → `Python`
  * `K8s` → `Kubernetes`
  * `DB` → `database`
  * `C#` → `C Sharp`
  * `CPP` → `C Plus Plus`

### 4.2 Extraction Models

#### 4.2.1 ExtractionResult

```python
class ExtractionResult(BaseModel):
    segments: List[BehaviorSegment]     # Prompt segments with behaviors
    success: bool                       # Extraction success flag
    error: Optional[str]                # Error message if failed
    extraction_time: float              # Time taken (milliseconds)
    standalone_query: Optional[str]     # Enriched query (with context resolved)
    required_intents: Optional[List[str]]  # LLM-predicted relevant intents
    profile_signals: Optional[dict]     # Profile characteristics for matching
```

#### 4.2.2 BehaviorSegment

```python
class BehaviorSegment(BaseModel):
    text: str                           # Original segment text
    behaviors: List[ExtractedBehavior]  # Extracted behaviors from segment
```

### 4.3 Search and Similarity Models

#### 4.3.1 SimilarityResult

```python
class SimilarityResult(BaseModel):
    behavior_id: str
    behavior_text: str
    distance: float                     # Cosine or hybrid distance
    classification: SimilarityClassification  # DUPLICATE|SIMILAR|POTENTIAL_CONFLICT
    credibility: float                  # Current credibility (after decay)
    reinforcement_count: int
    
    # Canonical fields
    intent: str
    target: str
    context: str
    polarity: str
    
    # Additional metadata
    last_seen_at: int
    created_at: int
```

#### 4.3.2 SimilarityClassification (Enum)

```python
class SimilarityClassification(str, Enum):
    DUPLICATE = "DUPLICATE"                   # Near-identical (distance < 0.12)
    SIMILAR = "SIMILAR"                       # Related (distance 0.12-0.20)
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT" # May conflict (distance 0.20-0.55)
    DISTANT = "DISTANT"                       # Unrelated (distance > 0.55)
```

### 4.4 Conflict Models

#### 4.4.1 ConflictType (Enum)

```python
class ConflictType(str, Enum):
    RESOLVABLE = "RESOLVABLE"                 # Auto-resolve via credibility
    USER_DECISION_NEEDED = "USER_DECISION_NEEDED"  # Requires user input
```

#### 4.4.2 ConflictAnalysisResult

```python
class ConflictAnalysisResult(BaseModel):
    analysis_type: ConflictAnalysisType       # POLARITY_CONFLICT | CROSS_INTENT | etc.
    conflict_detected: bool
    explanation: str                          # LLM explanation
    resolution_confidence: float              # GPT confidence in analysis
```

### 4.5 Profile Signal Models

Profile signals use **different vocabularies** than canonical behaviors for Profile Service integration:

```python
profile_signals = {
    "intents": {                        # User's PURPOSE (not behavior intent)
        "LEARNING": 0.85,               # Wants to understand deeply
        "TASK_COMPLETION": 0.40,        # Result-focused
        "PROBLEM_SOLVING": 0.70,        # Debugging/technical issues
        "EXPLORATION": 0.20,            # Brainstorming
        "GUIDANCE": 0.10,               # Seeking advice
        "ENGAGEMENT": 0.05              # Casual interaction
    },
    "interests": {                      # Topic areas
        "PROGRAMMING": 0.90,
        "AI": 0.60,
        "DATA_SCIENCE": 0.30
    },
    "behavior_level": "ADVANCED",       # BEGINNER|INTERMEDIATE|ADVANCED
    "signals": {                        # Response style preferences
        "CODE_FOCUSED": 0.85,
        "DETAILED_EXPLANATION": 0.70,
        "STEP_BY_STEP": 0.40
    },
    "complexity": 0.78,                 # Prompt complexity (0.0-1.0)
    "consistency": 0.5                  # Default (calculated across sessions)
}
```

---

## 5. Database Schema

### 5.1 Main Tables

#### 5.1.1 `behaviors` Table (Partitioned by user_id)

```sql
CREATE TABLE public.behaviors (
    -- Identity
    behavior_id            TEXT,
    user_id                TEXT NOT NULL,
    session_id             TEXT NOT NULL,
    
    -- Content
    behavior_text          TEXT NOT NULL,
    embedding              vector(3072),          -- pgvector type
    search_vector          tsvector,              -- For BM25 sparse search
    
    -- Canonical structure
    intent                 TEXT,                  -- PREFERENCE|CONSTRAINT|HABIT|SKILL|COMMUNICATION
    target                 TEXT,                  -- Canonicalized noun
    context                TEXT,                  -- Scope (general|IDE|morning|etc.)
    polarity               TEXT,                  -- POSITIVE|NEGATIVE
    
    -- Quality metrics
    credibility            DOUBLE PRECISION,
    extraction_confidence  DOUBLE PRECISION,
    clarity_score          DOUBLE PRECISION,
    linguistic_strength    DOUBLE PRECISION,
    
    -- Lifecycle
    behavior_state         TEXT DEFAULT 'ACTIVE' CHECK (behavior_state IN ('NEW', 'ACTIVE', 'SUPERSEDED', 'FLAGGED', 'ARCHIVED')),
    reinforcement_count    INTEGER,
    decay_rate             DOUBLE PRECISION,
    superseded_by_id       TEXT,
    related_behaviors      JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at             BIGINT,
    last_seen_at           BIGINT,
    last_decay_applied_at  BIGINT,
    last_accessed_at       BIGINT,
    
    -- History
    prompt_history_ids     TEXT[],
    context_notes          TEXT,
    
    PRIMARY KEY (behavior_id, user_id)
)
PARTITION BY LIST (user_id);

-- Default partition for all users
CREATE TABLE IF NOT EXISTS public.behaviors_default
    PARTITION OF public.behaviors
    DEFAULT;
```

**Key Indexes:**

```sql
-- HNSW index for fast vector similarity search
CREATE INDEX behaviors_embedding_hnsw
ON behaviors USING hnsw (embedding vector_cosine_ops);

-- B-tree indexes for filtering and sorting
CREATE INDEX idx_behaviors_state 
ON behaviors(user_id, behavior_state);

CREATE INDEX idx_behaviors_last_seen 
ON behaviors(user_id, last_seen_at);

CREATE INDEX idx_behaviors_credibility 
ON behaviors(user_id, credibility DESC) 
WHERE behavior_state = 'ACTIVE';

-- GIN index for full-text search (BM25)
CREATE INDEX idx_behaviors_search_vector
ON behaviors USING gin(search_vector);
```

#### 5.1.2 `behavior_conflicts` Table

```sql
CREATE TABLE behavior_conflicts (
    conflict_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               TEXT NOT NULL,
    behavior_id_1         TEXT NOT NULL,
    behavior_id_2         TEXT NOT NULL,
    conflict_type         TEXT NOT NULL CHECK (conflict_type IN ('RESOLVABLE', 'USER_DECISION_NEEDED')),
    similarity_distance   FLOAT NOT NULL,
    llm_analysis          TEXT,
    resolution_status     TEXT NOT NULL DEFAULT 'PENDING' 
                          CHECK (resolution_status IN ('PENDING', 'AUTO_RESOLVED', 'USER_RESOLVED', 'EXPIRED')),
    resolved_at           BIGINT,
    resolution_choice     TEXT,
    created_at            BIGINT NOT NULL,
    
    -- Foreign keys
    CONSTRAINT fk_conflict_behavior_1 
        FOREIGN KEY (behavior_id_1, user_id) 
        REFERENCES behaviors(behavior_id, user_id) 
        ON DELETE CASCADE,
    
    CONSTRAINT fk_conflict_behavior_2 
        FOREIGN KEY (behavior_id_2, user_id) 
        REFERENCES behaviors(behavior_id, user_id) 
        ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_conflicts_user 
ON behavior_conflicts(user_id, resolution_status);

CREATE INDEX idx_conflicts_pending 
ON behavior_conflicts(user_id, created_at) 
WHERE resolution_status = 'PENDING';
```

#### 5.1.3 `user_profile_signals` Table

```sql
CREATE TABLE user_profile_signals (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT NOT NULL,
    prompt_id        TEXT NOT NULL,
    profile_signals  JSONB NOT NULL,     -- Validated profile signals dict
    extracted_at     BIGINT NOT NULL,
    
    UNIQUE (user_id, prompt_id)
);

-- Index for efficient recent signal retrieval
CREATE INDEX idx_user_profile_signals_user_extracted
ON user_profile_signals (user_id, extracted_at DESC);
```

#### 5.1.4 `prompt_segments` Table

```sql
CREATE TABLE prompt_segments (
    segment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       TEXT NOT NULL,
    segment_text  TEXT NOT NULL,
    created_at    BIGINT NOT NULL
);

CREATE INDEX idx_user_segments
ON prompt_segments (user_id);
```

#### 5.1.5 `user_confirmation_queue` Table

```sql
CREATE TABLE user_confirmation_queue (
    request_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    conflict_id     UUID,
    question        TEXT NOT NULL,
    options         JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING' 
                    CHECK (status IN ('PENDING', 'RESOLVED', 'EXPIRED')),
    user_response   TEXT,
    responded_at    BIGINT,
    created_at      BIGINT NOT NULL,
    expires_at      BIGINT NOT NULL,
    
    CONSTRAINT fk_confirmation_conflict 
        FOREIGN KEY (conflict_id) 
        REFERENCES behavior_conflicts(conflict_id) 
        ON DELETE CASCADE
);

CREATE INDEX idx_confirmation_user_pending 
ON user_confirmation_queue(user_id, created_at) 
WHERE status = 'PENDING';
```

### 5.2 Database Connection Management

**Connection Pool Configuration** (in `db/connection.py`):

```python
_pool = ConnectionPool(
    DATABASE_URL,
    min_size=2,              # Minimum connections
    max_size=10,             # Maximum connections
    open=True,
    timeout=30,              # Connection timeout (seconds)
    max_idle=600,            # 10 minutes idle timeout
    max_lifetime=3600,       # 1 hour max lifetime
    check=ConnectionPool.check_connection,  # Health checks
    configure=lambda conn: register_vector(conn)  # pgvector registration
)
```

**Usage Pattern:**

```python
with get_db_pool_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM behaviors WHERE user_id = %s", [user_id])
        results = cur.fetchall()
    conn.commit()  # Explicit commit (autocommit=False)
```

---

## 6. Core Services

### 6.1 Extraction Service (`services/extractor.py`)

**Primary Responsibilities:**
1. Orchestrate behavior extraction workflow
2. Validate GPT-4 extraction responses
3. Convert raw JSON to Pydantic models
4. Coordinate storage with conflict detection
5. Manage detailed flow tracking for UI

**Key Functions:**

#### 6.1.1 `run_behavior_extraction(prompt: str) -> ExtractionResult`

Main extraction function:

```python
def run_behavior_extraction(prompt: str) -> ExtractionResult:
    """
    Extract behaviors from a user prompt.
    
    Workflow:
    1. Call GPT-4 via openAiClient.extract_behavior()
    2. Validate response structure
    3. Convert to Pydantic models
    4. Validate profile_signals
    5. Return ExtractionResult
    """
    raw_response = extract_behavior(prompt)
    # Validation and Pydantic conversion...
    return ExtractionResult(...)
```

#### 6.1.2 `run_behavior_extraction_with_history(prompt, recent_history) -> ExtractionResult`

Extraction with conversation context:

```python
def run_behavior_extraction_with_history(
    prompt: str, 
    recent_history: List[dict]
) -> ExtractionResult:
    """
    Extract behaviors with conversation history for context resolution.
    
    Returns:
    - segments: Extracted behaviors
    - standalone_query: Enriched query (references resolved)
    - required_intents: LLM-predicted relevant intents for TGHR
    """
    raw_response = extract_behavior_with_history(prompt, recent_history)
    # Returns standalone_query like:
    # "which one is better for backend?" → 
    # "which is better for backend: Python or JavaScript?"
```

#### 6.1.3 `store_behavior(extraction_result, user_id, session_id) -> List[StoredBehavior]`

Store extracted behaviors with duplicate/conflict detection:

```python
def store_behavior(
    extraction_result: ExtractionResult,
    user_id: str,
    session_id: str
) -> List[StoredBehavior]:
    """
    Store behaviors with intelligent deduplication and conflict handling.
    
    For each extracted behavior:
    1. Calculate initial credibility
    2. Filter low-quality behaviors (< threshold)
    3. Generate embedding
    4. Search for similar existing behaviors
    5. Classify relationship (duplicate/conflict/new)
    6. Reinforce, resolve conflict, or insert as new
    7. Publish events to Redis
    """
```

#### 6.1.4 `store_behavior_with_tracking(...) -> DetailedExtractionResult`

Enhanced storage with detailed flow tracking for UI:

```python
def store_behavior_with_tracking(...) -> DetailedExtractionResult:
    """
    Store behaviors with detailed tracking of each behavior's processing path.
    
    Returns:
    - total_extracted, total_stored, total_reinforced, total_conflicts, total_pruned
    - flow_info: List[BehaviorFlowInfo] with action taken for each behavior
      * NEW_STORED, REINFORCED, CONFLICT_DETECTED, PRUNED, etc.
    """
```

**Behavior Relationship Detection:**

```python
class RelationType(str, Enum):
    DUPLICATE = "DUPLICATE"                      # Same 4-tuple → reinforce
    POLARITY_CONFLICT = "POLARITY_CONFLICT"      # Same intent+target+context, opposite polarity
    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"    # Same intent+context, different target
    CROSS_INTENT_CONFLICT = "CROSS_INTENT_CONFLICT"  # CONSTRAINT vs other intent
    RELATED = "RELATED"                          # Same intent+target, different context
    COMPATIBLE = "COMPATIBLE"                    # No conflict, coexist
```

**Intent Conflict Matrix:**

```python
INTENT_CONFLICT_MATRIX = {
    "PREFERENCE": {"PREFERENCE", "CONSTRAINT"},
    "SKILL": {"SKILL", "CONSTRAINT"},
    "HABIT": {"HABIT", "CONSTRAINT"},
    "CONSTRAINT": {"PREFERENCE", "SKILL", "HABIT", "CONSTRAINT", "COMMUNICATION"},
    "COMMUNICATION": {"COMMUNICATION", "CONSTRAINT"},
}
```

### 6.2 OpenAI Client (`services/openAiClient.py`)

**Primary Responsibilities:**
1. Communicate with Azure OpenAI API
2. Generate behavior extractions with canonical structure
3. Generate semantic embeddings (3072-dim)
4. Perform conflict analysis via LLM

**Azure OpenAI Configuration:**

```python
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview",
    timeout=30
)

GPT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"
```

#### 6.2.1 `extract_behavior(prompt: str) -> dict`

Main extraction function with comprehensive system prompt:

```python
def extract_behavior(prompt: str) -> dict:
    """
    Extract behaviors from prompt using GPT-4.1-mini.
    
    System prompt enforces:
    - Canonical 4-tuple extraction (intent, target, context, polarity)
    - Target canonicalization (JS → JavaScript, etc.)
    - Confidence, clarity, linguistic_strength scoring
    - Profile signals extraction (different vocabulary)
    - Strict JSON output format
    
    Returns:
    {
        "segments": [...],
        "profile_signals": {...},
        "success": True/False,
        "error": None,
        "metadata": {"extraction_time_ms": ..., "tokens_used": ...}
    }
    """
```

**System Prompt Highlights:**

1. **Canonical Structure Enforcement:**
   ```
   FOR EACH BEHAVIOR, PRODUCE:
   - intent: CONSTRAINT | PREFERENCE | HABIT | SKILL | COMMUNICATION
   - target: Concise canonical noun (1-3 words)
   - context: Scope (IDE, morning, etc.) or "general"
   - polarity: POSITIVE | NEGATIVE
   - confidence, clarity, linguistic_strength: 0.0-1.0
   ```

2. **Target Canonicalization:**
   ```
   ALWAYS use FULL, STANDARD names:
   - JS → JavaScript
   - TS → TypeScript
   - PY → Python
   - K8s → Kubernetes
   - DB → database
   ```

3. **Comparative Statement Handling:**
   ```
   "I prefer X over Y" → Extract ONLY X with POSITIVE polarity
   Do NOT extract Y as separate negative behavior
   ```

4. **Profile Signals:**
   ```
   Extract holistic profile characteristics:
   - intents: LEARNING, TASK_COMPLETION, PROBLEM_SOLVING, etc.
   - interests: AI, PROGRAMMING, CREATIVE, etc.
   - behavior_level: BEGINNER | INTERMEDIATE | ADVANCED
   - signals: CODE_FOCUSED, DETAILED_EXPLANATION, etc.
   ```

#### 6.2.2 `extract_behavior_with_history(prompt, recent_history) -> dict`

Extraction with conversation context resolution:

```python
def extract_behavior_with_history(
    prompt: str, 
    recent_history: List[dict]
) -> dict:
    """
    Extract behaviors and enrich prompt to standalone query.
    
    System prompt includes:
    - Behavior extraction (same as extract_behavior)
    - Standalone query generation: resolve references like "it", "that", "above"
    - Required intents prediction: predict which intents are relevant for TGHR
    
    Additional output:
    - standalone_query: "What foods before morning run?" (resolved)
    - required_intents: ["PREFERENCE", "CONSTRAINT"]
    """
```

#### 6.2.3 `embed_text(text: str) -> List[float]`

Generate semantic embedding:

```python
def embed_text(text: str) -> List[float]:
    """
    Generate 3072-dimensional embedding using text-embedding-3-large.
    
    Returns:
    - List[float] with 3072 dimensions
    - Normalized for cosine similarity
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding  # 3072-dim vector
```

#### 6.2.4 `analyze_conflict(behavior1, behavior2, analysis_type) -> ConflictAnalysisResult`

LLM-based conflict analysis:

```python
def analyze_conflict(
    behavior1: dict,
    behavior2: dict,
    analysis_type: ConflictAnalysisType
) -> ConflictAnalysisResult:
    """
    Analyze potential conflict between two behaviors using GPT-4.
    
    Analysis types:
    - POLARITY_CONFLICT: Same intent+target+context, opposite polarity
    - CROSS_INTENT_CONFLICT: CONSTRAINT vs other intent
    - GENERAL: Other potential conflicts
    
    Returns:
    - conflict_detected: bool
    - explanation: str (LLM reasoning)
    - resolution_confidence: float (0.0-1.0)
    """
```

### 6.3 Behavior Repository (`services/behaviorRepository.py`)

**Primary Responsibilities:**
1. Database CRUD operations
2. Hybrid search (TGHR: Tuple-Guided Hybrid Retrieval)
3. Reinforcement and decay management
4. Conflict persistence
5. Event publishing to Redis

#### 6.3.1 `insert_behavior(payload: dict)`

Insert new behavior into database:

```python
def insert_behavior(payload: dict):
    """
    Insert new behavior with all fields.
    
    - Defaults behavior_state to 'ACTIVE'
    - Builds enriched search_text for tsvector (BM25)
    - Publishes behavior.created event to Redis
    """
    payload['search_text'] = f"{behavior_text} {target} {context}".strip()
    
    cur.execute(
        """
        INSERT INTO behaviors (
            behavior_id, user_id, behavior_text, embedding,
            credibility, intent, target, context, polarity,
            decay_rate, reinforcement_count, created_at,
            last_seen_at, last_decay_applied_at, session_id,
            prompt_history_ids, behavior_state,
            search_vector
        ) VALUES (
            ...,
            to_tsvector('english', %(search_text)s)
        )
        """, payload
    )
    
    # Publish event
    publisher.publish_behavior_created(...)
```

#### 6.3.2 `search_similar_behavior_3D(...) -> HybridSearchResponse`

**TGHR (Tuple-Guided Hybrid Retrieval)**: Combines 3 signals for intelligent retrieval:

```python
def search_similar_behavior_3D(
    user_id: str,
    query_embedding: List[float],
    query_text: str,
    session_id: str,
    required_intents: List[str] = None,
    limit: int = 30
) -> HybridSearchResponse:
    """
    3D Hybrid Search: Dense (semantic) + Sparse (BM25) + Metadata (intent).
    
    Scoring formula:
    hybrid_score = 
        (DENSE_W × semantic_score) +       # 0.55 × (1 - cosine_distance)
        (SPARSE_W × bm25_score) +          # 0.30 × ts_rank_cd(...)
        (INTENT_W × intent_affinity)       # 0.15 × graduated_boost
    
    Intent affinity is GRADUATED (not binary):
    - Exact match: 1.0
    - Related intents (e.g., HABIT↔CONSTRAINT): 0.50
    - Unrelated: 0.0
    
    Returns:
    - results: List[SimilarityResult] (filtered by threshold and gap)
    - decay_updates: Pending lazy decay updates
    - accessed_behavior_ids: IDs for last_accessed_at updates
    """
```

**TGHR Weights (configurable):**

```python
HYBRID_DENSE_WEIGHT = 0.55        # Semantic similarity
HYBRID_SPARSE_WEIGHT = 0.30       # BM25 keyword matching
HYBRID_INTENT_BOOST_WEIGHT = 0.15 # Intent affinity boost
```

**Intent Affinity Matrix** (graduated boost):

```python
INTENT_AFFINITY = {
    frozenset({"HABIT", "CONSTRAINT"}): 0.50,
    frozenset({"HABIT", "PREFERENCE"}): 0.40,
    frozenset({"PREFERENCE", "CONSTRAINT"}): 0.35,
    frozenset({"COMMUNICATION", "PREFERENCE"}): 0.30,
    # ... more combinations
}
```

**OR-based BM25 Query:**

```python
def _build_or_tsquery(text: str) -> str:
    """
    Convert text to OR-based tsquery for partial keyword matching.
    
    'What foods should I eat before my morning run'
    → 'food | eat | morn | run'
    
    This prevents zero scores when short behaviors only partially
    overlap with long queries (AND-based tsquery issue).
    """
```

**Result Filtering:**

1. **Threshold filter**: Drop results below `HYBRID_SCORE_THRESHOLD` (0.10)
2. **Relevance gap filter**: Drop results when score drops more than `RELEVANCE_GAP_DROP_RATIO` (0.55) below top result
3. **Soft cap**: Return max `MAX_RETRIEVAL_RESULTS` (20) even if more pass filters

**Lazy Decay Application:**

```python
# Apply decay in-memory during retrieval
new_credibility, decay_applied, days_elapsed = apply_lazy_decay(
    stored_credibility=float(stored_credibility),
    decay_rate=float(decay_rate),
    last_decay_applied_at=last_decay_applied_at,
    current_time=current_time
)

if decay_applied:
    # Collect for async batch update
    decay_updates.append((new_credibility, current_time, behavior_id, user_id))
```

#### 6.3.3 `reinforce_behavior(...) -> ReinforcementResult`

Reinforce existing behavior (duplicate detected):

```python
def reinforce_behavior(
    behavior_id: str,
    user_id: str,
    segment_id: str = None
) -> ReinforcementResult:
    """
    Reinforce behavior: increment count, boost credibility, update timestamp.
    
    Reinforcement boost formula (diminishing returns):
    boost = BASE_REINFORCEMENT_BOOST / sqrt(reinforcement_count)
    
    new_credibility = min(1.0, current_credibility + boost)
    new_count = current_count + 1
    last_seen_at = current_timestamp
    
    Publishes behavior.reinforced event to Redis.
    """
```

#### 6.3.4 `persist_retrieval_updates_batch(decay_updates, accessed_ids, user_id)`

Async batch persistence of retrieval updates:

```python
def persist_retrieval_updates_batch(
    decay_updates: List[tuple],
    accessed_behavior_ids: List[str],
    user_id: str
):
    """
    Batch update behaviors after retrieval (background task).
    
    1. Update credibility for behaviors with lazy decay applied
    2. Update last_accessed_at for all retrieved behaviors
    
    Uses unnest() for efficient bulk updates.
    """
```

#### 6.3.5 `insert_conflict(conflict_data: dict)`

Persist conflict to database:

```python
def insert_conflict(conflict_data: dict):
    """
    Insert conflict record.
    
    Fields:
    - conflict_type: RESOLVABLE | USER_DECISION_NEEDED
    - resolution_status: PENDING | AUTO_RESOLVED | USER_RESOLVED | EXPIRED
    - llm_analysis: GPT explanation
    """
```

#### 6.3.6 `resolve_conflict(conflict_id, user_id, resolution_choice) -> dict`

User-driven conflict resolution:

```python
def resolve_conflict(
    conflict_id: str,
    user_id: str,
    resolution_choice: Literal["OLD_WINS", "NEW_WINS", "BOTH_CORRECT"]
) -> dict:
    """
    Resolve conflict based on user's choice.
    
    OLD_WINS: Keep existing, discard new
    NEW_WINS: Supersede old with new
    BOTH_CORRECT: Keep both, mark as compatible
    
    Publishes behavior.conflict.resolved event.
    """
```

### 6.4 Credibility Calculator (`services/credibilityCalculator.py`)

**Primary Responsibilities:**
1. Calculate initial credibility from extraction scores
2. Compute reinforcement boost (diminishing returns)
3. Apply lazy decay (time-based credibility reduction)
4. Manage grace periods and intent-specific decay rates

#### 6.4.1 `calculate_initial_credibility(...) -> float`

Multi-factor credibility scoring:

```python
def calculate_initial_credibility(
    confidence: float,
    clarity: float,
    linguistic_strength: float,
    behavior_text: str
) -> float:
    """
    Calculate initial credibility using weighted combination.
    
    Formula:
    credibility = 
        (0.40 × confidence) +
        (0.35 × clarity) +
        (0.25 × linguistic_strength)
    
    Range: 0.0-1.0
    
    Weights from CREDIBILITY_WEIGHTS config.
    """
```

#### 6.4.2 `should_store_behavior(credibility: float) -> bool`

Quality gate for storage:

```python
def should_store_behavior(credibility: float) -> bool:
    """
    Determine if behavior meets quality threshold.
    
    Threshold: CREDIBILITY_PRUNE_THRESHOLD = 0.4
    
    Behaviors below 0.4 credibility are filtered out.
    """
    return credibility > 0.4
```

#### 6.4.3 `get_decay_rate(intent: str) -> float`

Intent-specific decay rates:

```python
def get_decay_rate(intent: str) -> float:
    """
    Return decay rate based on behavioral intent.
    
    Intent-based decay rates:
    - HABIT: 0.04        (fast decay - habits change quickly)
    - PREFERENCE: 0.015  (medium decay)
    - COMMUNICATION: 0.015
    - SKILL: 0.005       (slow decay - skills persist)
    - CONSTRAINT: 0.001  (very slow - constraints are persistent)
    
    Default: 0.015
    """
    return INTENT_DECAY_RATES.get(intent, DEFAULT_DECAY_RATE)
```

#### 6.4.4 `apply_lazy_decay(...) -> Tuple[float, bool, int]`

Lazy decay with grace period:

```python
def apply_lazy_decay(
    stored_credibility: float,
    decay_rate: float,
    last_decay_applied_at: Optional[int],
    current_time: int
) -> Tuple[float, bool, int]:
    """
    Apply time-based decay to credibility (lazy evaluation).
    
    Formula:
    new_credibility = stored_credibility × e^(-decay_rate × days_elapsed)
    
    Grace period: 7 days (DECAY_GRACE_PERIOD_DAYS)
    - No decay for first 7 days after creation
    
    Returns:
    - new_credibility: Decayed credibility
    - decay_applied: True if decay was applied
    - days_elapsed: Days since last decay
    """
```

**Decay Formula Explanation:**

```
Exponential decay: credibility(t) = credibility₀ × e^(-λt)

Where:
- λ (lambda) = decay_rate (intent-specific)
- t = time elapsed in days
- credibility₀ = initial or last credibility

Example (PREFERENCE with decay_rate = 0.015):
- Day 0: 0.80
- Day 10: 0.80 × e^(-0.015×10) = 0.80 × 0.860 = 0.688
- Day 30: 0.80 × e^(-0.015×30) = 0.80 × 0.638 = 0.510
- Day 60: 0.80 × e^(-0.015×60) = 0.80 × 0.407 = 0.326
```

#### 6.4.5 `calculate_reinforcement_boost(reinforcement_count: int) -> float`

Diminishing returns for reinforcement:

```python
def calculate_reinforcement_boost(reinforcement_count: int) -> float:
    """
    Calculate credibility boost for reinforcement.
    
    Formula: boost = BASE_REINFORCEMENT_BOOST / sqrt(reinforcement_count)
    
    BASE_REINFORCEMENT_BOOST = 0.05
    
    Diminishing returns:
    - 1st reinforcement: 0.05 / sqrt(1) = 0.050
    - 2nd reinforcement: 0.05 / sqrt(2) = 0.035
    - 3rd reinforcement: 0.05 / sqrt(3) = 0.029
    - 10th reinforcement: 0.05 / sqrt(10) = 0.016
    
    Minimum boost: 0.001
    """
    boost = BASE_REINFORCEMENT_BOOST / math.sqrt(reinforcement_count)
    return max(boost, MIN_BOOST)
```

### 6.5 Profile Service Integration

#### 6.5.1 Profile Signal Extractor (`services/profileSignalExtractor.py`)

Validates profile signals against controlled vocabularies:

```python
class ProfileSignalExtractor:
    """
    Validates GPT-4 profile_signals output.
    
    Controlled vocabularies:
    - VALID_INTENTS: LEARNING, TASK_COMPLETION, PROBLEM_SOLVING, etc.
    - VALID_INTERESTS: AI, PROGRAMMING, CREATIVE, etc.
    - VALID_SIGNALS: CODE_FOCUSED, DETAILED_EXPLANATION, etc.
    - VALID_LEVELS: BEGINNER, INTERMEDIATE, ADVANCED
    """
    
    def parse_and_validate(self, raw: dict) -> dict:
        """
        Validate and clean profile_signals.
        
        - Filters out invalid vocabulary terms
        - Clamps numeric values to [0.0, 1.0]
        - Provides defaults for missing fields
        
        Raises ValueError if no valid intents or interests.
        """
```

#### 6.5.2 Profile Signal Repository (`services/profileSignalRepository.py`)

Persists profile signals for drift fallback:

```python
class ProfileSignalRepository:
    def save(self, user_id: str, prompt_id: str, profile_signals: dict):
        """
        Save/upsert profile signals.
        
        Uses ON CONFLICT to update if (user_id, prompt_id) exists.
        """
    
    def get_recent(self, user_id: str, limit: int = 10) -> List[dict]:
        """
        Retrieve most recent profile signals for drift fallback.
        
        Used by Profile Service when drift detection triggers.
        """
```

#### 6.5.3 Cold Start Dispatcher (`services/coldStartDispatcher.py`)

Orchestrates cold-start profile assignment:

```python
class ColdStartDispatcher:
    async def dispatch(
        self,
        user_id: str,
        prompt_id: str,
        profile_signals: dict
    ) -> Optional[dict]:
        """
        Process profile signals after extraction.
        
        Workflow:
        1. Always save locally (for drift fallback)
        2. Check if user is in COLD_START mode
        3. If COLD_START, call Profile Service for assignment
        
        Profile Service response:
        - PENDING: More prompts needed
        - ASSIGNED: Profile assigned, user exits COLD_START
        """
```

#### 6.5.4 Profile Service Client (`services/profileServiceClient.py`)

HTTP client for Profile Service API:

```python
class ProfileServiceClient:
    async def assign_profile(
        self,
        user_id: str,
        profile_signals: dict
    ) -> Optional[dict]:
        """
        POST /api/predefined-profiles/assign-profile
        
        Response:
        {
            "status": "PENDING" | "ASSIGNED",
            "prompts_collected": 3,
            "assigned_profile_id": "tech_enthusiast_v1",
            "confidence": 0.87
        }
        """
    
    async def get_user_status(self, user_id: str) -> Optional[dict]:
        """
        GET /api/predefined-profiles/user/{user_id}
        
        Response:
        {
            "mode": "COLD_START" | "ASSIGNED",
            "profile_id": "tech_enthusiast_v1"
        }
        """
```

### 6.6 Event Publisher (`services/eventPublisher.py`)

Publishes behavior events to Redis Streams for Drift Detection Service:

```python
class BehaviorEventPublisher:
    """
    Publishes events to Redis Stream.
    
    Stream name: "behavior.events" (configurable)
    """
    
    def publish_behavior_created(self, user_id, behavior_id, ...):
        """Event: behavior.created"""
    
    def publish_behavior_reinforced(self, user_id, behavior_id, ...):
        """Event: behavior.reinforced"""
    
    def publish_behavior_superseded(self, user_id, behavior_id, ...):
        """Event: behavior.superseded"""
    
    def publish_conflict_resolved(self, user_id, conflict_id, ...):
        """Event: behavior.conflict.resolved"""
```

**Event Format:**

```json
{
    "event_type": "behavior.created",
    "event_id": "evt_abc123...",
    "published_at": 1709481600,
    "payload": {
        "user_id": "user_12345",
        "behavior_id": "beh_abc123",
        "target": "Python",
        "intent": "PREFERENCE",
        "context": "backend",
        "polarity": "POSITIVE",
        "credibility": 0.85,
        "reinforcement_count": 1,
        "state": "ACTIVE",
        "created_at": 1709481600,
        "last_seen_at": 1709481600
    }
}
```

---

## 7. API Endpoints

### 7.1 Core Extraction Endpoints

#### 7.1.1 POST `/extract`

Basic behavior extraction and storage.

**Request:**
```json
{
    "prompt": "I prefer Python over JavaScript for backend development",
    "user_id": "user_12345",
    "session_id": "default"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "extraction": {
            "segments": [...],
            "extraction_time_ms": 523.45,
            "total_segments": 1,
            "total_behaviors_extracted": 1
        },
        "storage": {
            "stored_behaviors": [
                {
                    "behavior_id": "beh_abc123",
                    "user_id": "user_12345",
                    "behavior_text": "prefers Python for backend",
                    "credibility": 0.85,
                    "canonical": {
                        "intent": "PREFERENCE",
                        "target": "Python",
                        "context": "backend",
                        "polarity": "POSITIVE"
                    },
                    "reinforcement_count": 1,
                    "created_at": 1709481600
                }
            ],
            "total_behaviors_stored": 1,
            "behaviors_filtered": 0
        }
    },
    "error": null
}
```

#### 7.1.2 POST `/extract-detailed`

Extraction with detailed flow tracking (for UI visualization).

**Additional response fields:**
```json
{
    "data": {
        "processing": {
            "total_extracted": 3,
            "total_stored": 1,
            "total_reinforced": 1,
            "total_conflicts": 1,
            "total_pruned": 0
        },
        "flow_info": [
            {
                "behavior_description": "prefers Python for backend",
                "action": "NEW_STORED",
                "credibility": 0.85,
                "canonical": {...},
                "stored_behavior_id": "beh_abc123"
            },
            {
                "behavior_description": "experienced with TypeScript",
                "action": "REINFORCED",
                "matched_behavior_id": "beh_xyz789",
                "distance": 0.08
            },
            {
                "behavior_description": "dislikes Python for frontend",
                "action": "CONFLICT_DETECTED",
                "conflict_info": {
                    "conflict_type": "POLARITY_CONFLICT",
                    "existing_behavior": "likes Python for frontend"
                }
            }
        ]
    }
}
```

#### 7.1.3 POST `/v2/extract`

**Optimized endpoint** with conversation history and fast retrieval.

**Request:**
```json
{
    "prompt": "which one is better for backend?",
    "recent_history": [
        {"role": "user", "text": "I'm choosing between Python and JavaScript"},
        {"role": "assistant", "text": "Both are popular choices"}
    ],
    "user_id": "user_12345",
    "session_id": "default"
}
```

**Response (fast return with related behaviors):**
```json
{
    "success": true,
    "data": {
        "standalone_query": "which is better for backend: Python or JavaScript?",
        "required_intents": ["PREFERENCE", "SKILL"],
        "original_prompt": "which one is better for backend?",
        "related_behaviors": [
            {
                "behavior_id": "beh_abc123",
                "behavior_text": "prefers Python for backend",
                "distance": 0.15,
                "intent": "PREFERENCE",
                "target": "Python",
                "context": "backend",
                "polarity": "POSITIVE",
                "credibility": 0.87
            }
        ],
        "extraction_time_ms": 485.2
    }
}
```

**Background tasks** (async, non-blocking):
- Behavior storage with conflict detection
- Retrieval updates (decay persistence, access timestamps)

### 7.2 Retrieval Endpoints

#### 7.2.1 GET `/behaviors/{user_id}`

Get all behaviors for a user.

**Query params:**
- `session_id` (optional): Filter by session

**Response:**
```json
{
    "success": true,
    "data": {
        "user_id": "user_12345",
        "session_id": "default",
        "total_behaviors": 15,
        "behaviors": [...]
    }
}
```

#### 7.2.2 POST `/api/behaviors/by-ids`

Get specific behaviors by IDs.

**Request:**
```json
{
    "user_id": "user_12345",
    "behavior_ids": ["beh_abc123", "beh_xyz789"]
}
```

**Response:**
```json
[
    {
        "behavior_id": "beh_abc123",
        "behavior_text": "prefers Python for backend",
        "canonical": {...},
        "credibility": 0.85
    }
]
```

### 7.3 Conflict Management Endpoints

#### 7.3.1 GET `/conflicts/{user_id}`

Get all conflicts for a user.

**Response:**
```json
{
    "success": true,
    "data": {
        "user_id": "user_12345",
        "total_conflicts": 2,
        "conflicts": [
            {
                "conflict_id": "uuid-...",
                "behavior_1": {...},
                "behavior_2": {...},
                "conflict_type": "POLARITY_CONFLICT",
                "resolution_status": "PENDING",
                "created_at": 1709481600
            }
        ]
    }
}
```

#### 7.3.2 POST `/conflicts/resolve`

Resolve a conflict.

**Request:**
```json
{
    "conflict_id": "uuid-...",
    "user_id": "user_12345",
    "resolution_choice": "NEW_WINS"
}
```

**Choices:**
- `OLD_WINS`: Keep existing, discard new
- `NEW_WINS`: Supersede old with new
- `BOTH_CORRECT`: Keep both

### 7.4 Profile Signal Endpoints

#### 7.4.1 GET `/api/behaviors/{user_id}/signals/recent`

Get recent profile signals for drift fallback.

**Query params:**
- `limit` (default: 10, max: 50)

**Response:**
```json
{
    "user_id": "user_12345",
    "total_signals": 10,
    "signals": [
        {
            "prompt_id": "prompt_abc123",
            "profile_signals": {
                "intents": {"PROBLEM_SOLVING": 0.85},
                "interests": {"PROGRAMMING": 0.90}
            },
            "extracted_at": 1709481600
        }
    ]
}
```

#### 7.4.2 GET `/api/behaviors/{user_id}/signals/count`

Get total count of stored profile signals.

**Response:**
```json
{
    "user_id": "user_12345",
    "count": 42
}
```

### 7.5 Utility Endpoints

#### 7.5.1 GET `/health`

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "service": "behavior_extraction"
}
```

#### 7.5.2 POST `/behaviors/similarity`

Compare two behaviors (POC/testing).

**Request:**
```json
{
    "behavior1": "I prefer dark mode",
    "behavior2": "I like dark themes",
    "metric": "cosine"
}
```

**Response:**
```json
{
    "similarity": 0.94,
    "distance": 0.06,
    "metric": "cosine"
}
```

---

## 8. Algorithms and Business Logic

### 8.1 Credibility Scoring Algorithm

**Initial Credibility Calculation:**

```
credibility = (w_conf × confidence) + (w_clar × clarity) + (w_ling × linguistic_strength)

Where:
- w_conf = 0.40  (confidence weight)
- w_clar = 0.35  (clarity weight)
- w_ling = 0.25  (linguistic strength weight)

Range: 0.0-1.0
Threshold: 0.4 (behaviors below this are pruned)
```

**Example:**
```
confidence = 0.92
clarity = 0.88
linguistic_strength = 0.75

credibility = (0.40 × 0.92) + (0.35 × 0.88) + (0.25 × 0.75)
            = 0.368 + 0.308 + 0.1875
            = 0.8635
            
Result: 0.864 (high quality behavior, will be stored)
```

### 8.2 Time-Based Decay Algorithm

**Exponential Decay Formula:**

```
credibility(t) = credibility₀ × e^(-λt)

Where:
- credibility₀ = initial or last credibility
- λ = decay_rate (intent-specific)
- t = time elapsed in days
- e = Euler's number (2.71828...)

Grace period: 7 days (no decay for first week)
```

**Intent-Specific Decay Rates:**

| Intent | Decay Rate (λ) | Half-Life | Rationale |
|--------|---------------|-----------|-----------|
| HABIT | 0.040 | ~17 days | Habits change frequently |
| PREFERENCE | 0.015 | ~46 days | Preferences are moderately stable |
| COMMUNICATION | 0.015 | ~46 days | Communication style evolves slowly |
| SKILL | 0.005 | ~139 days | Skills persist longer |
| CONSTRAINT | 0.001 | ~693 days | Constraints are very persistent |

**Example (PREFERENCE with λ=0.015):**

```
Initial credibility: 0.80
Grace period: 7 days (no decay)

Day 7:  0.80 × e^(-0.015×0)  = 0.800
Day 10: 0.80 × e^(-0.015×3)  = 0.765
Day 20: 0.80 × e^(-0.015×13) = 0.659
Day 30: 0.80 × e^(-0.015×23) = 0.566
Day 60: 0.80 × e^(-0.015×53) = 0.363
```

### 8.3 Reinforcement Algorithm

**Diminishing Returns Formula:**

```
boost = BASE_BOOST / √reinforcement_count

BASE_BOOST = 0.05
MIN_BOOST = 0.001

new_credibility = min(1.0, current_credibility + boost)
new_count = current_count + 1
last_seen_at = current_timestamp
```

**Boost Progression:**

| Reinforcement | Boost | Total Gain (if starting at 0.7) |
|---------------|-------|----------------------------------|
| 1st | 0.0500 | 0.7500 |
| 2nd | 0.0354 | 0.7854 |
| 3rd | 0.0289 | 0.8143 |
| 5th | 0.0224 | 0.8590 |
| 10th | 0.0158 | 0.9222 |
| 20th | 0.0112 | 0.9650 |
| 50th | 0.0071 | 0.9915 |

**Key insight:** Early reinforcements have stronger impact; later reinforcements provide minimal boost (prevents unrealistic 1.0 credibility).

### 8.4 Duplicate Detection Algorithm

**Multi-Stage Classification:**

```
1. Compute semantic similarity (cosine distance on embeddings)
2. Compare canonical structure (intent, target, context, polarity)
3. Classify relationship:

DUPLICATE (reinforce):
- Intent matches
- Target matches (case-insensitive)
- Context is same or generalizable
- Polarity matches
- Distance < 0.12 OR (structure matches AND distance < 0.25)

SIMILAR (related but not duplicate):
- Intent matches
- Target matches
- Context differs significantly
- Distance 0.12-0.20

POLARITY_CONFLICT:
- Intent matches
- Target matches
- Context matches
- Polarity differs (POSITIVE vs NEGATIVE)

CROSS_INTENT_CONFLICT:
- CONSTRAINT conflicts with any other intent
- Same target and context

POTENTIAL_CONFLICT:
- Same intent and context
- Different target
- Distance 0.20-0.55

COMPATIBLE (no action):
- No structural overlap
- Distance > 0.55
```

### 8.5 Hybrid Retrieval (TGHR) Algorithm

**TGHR: Tuple-Guided Hybrid Retrieval**

Combines 3 signals:

```
hybrid_score = (w_dense × dense_score) + 
               (w_sparse × sparse_score) + 
               (w_intent × intent_affinity)

Where:
- dense_score = 1 - cosine_distance     (semantic similarity)
- sparse_score = ts_rank_cd(...)        (BM25 keyword matching)
- intent_affinity = graduated_boost     (intent relationship)

Weights:
- w_dense = 0.55    (semantic)
- w_sparse = 0.30   (lexical)
- w_intent = 0.15   (metadata)
```

**Intent Affinity (Graduated Boost):**

Instead of binary (match=1.0, no-match=0.0), uses graduated affinity:

```
Exact match:         1.00
HABIT ↔ CONSTRAINT:  0.50
HABIT ↔ PREFERENCE:  0.40
PREFERENCE ↔ CONSTRAINT: 0.35
...
Unrelated:           0.00
```

**Example:**

Query: "What should I eat before morning exercise?"
- Required intents: [PREFERENCE, CONSTRAINT]

Candidate: "Always takes vitamins in the morning" (HABIT)
- Dense score: 0.75 (semantic overlap: "morning")
- Sparse score: 0.40 (keyword match: "morning")
- Intent affinity: 0.50 (HABIT ↔ CONSTRAINT)

```
hybrid_score = (0.55 × 0.75) + (0.30 × 0.40) + (0.15 × 0.50)
             = 0.4125 + 0.12 + 0.075
             = 0.6075
```

**Filtering:**

1. **Threshold**: Drop if hybrid_score < 0.10
2. **Gap filter**: Drop if score < top_score × (1 - 0.55)
3. **Soft cap**: Return max 20 results

### 8.6 Conflict Resolution Algorithm

**Phase 1: Detection**

When new behavior is extracted:
1. Search for similar behaviors (semantic + structured)
2. Classify relationship (DUPLICATE, POLARITY_CONFLICT, etc.)
3. If conflict detected, proceed to Phase 2

**Phase 2: Auto-Resolution (RESOLVABLE conflicts)**

```
if conflict_type == POLARITY_CONFLICT:
    credibility_diff = abs(existing.credibility - new.credibility)
    
    if credibility_diff > 0.3:
        # Significant credibility difference → auto-resolve
        if new.credibility > existing.credibility:
            supersede(existing, new)  # NEW_WINS
            resolution_status = AUTO_RESOLVED
        else:
            discard(new)              # OLD_WINS
            resolution_status = AUTO_RESOLVED
    else:
        # Close credibility → user decision needed
        conflict_type = USER_DECISION_NEEDED
        resolution_status = PENDING
```

**Phase 3: User Decision (USER_DECISION_NEEDED conflicts)**

Present conflict to user with options:
- **OLD_WINS**: Keep existing, discard new
- **NEW_WINS**: Supersede old with new (mark old as SUPERSEDED)
- **BOTH_CORRECT**: Keep both, allow coexistence

---

## 9. Configuration System

### 9.1 Environment Variables

**Required:**
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key

# Redis (for events)
REDIS_URL=redis://localhost:6379/0
REDIS_STREAM_NAME=behavior.events
REDIS_EVENTS_ENABLED=true

# Profile Service
PROFILE_SERVICE_BASE_URL=http://localhost:8001
```

**Optional:**
```bash
# Sample test user
SAMPLE_USERID=user_12345

# Profile signal limits
PROFILE_SIGNALS_DEFAULT_LIMIT=10
PROFILE_SIGNALS_MAX_LIMIT=50
```

### 9.2 Algorithmic Configuration

**Credibility:**
```python
CREDIBILITY_WEIGHTS = {
    "confidence": 0.40,
    "clarity": 0.35,
    "linguistic_strength": 0.25
}
CREDIBILITY_PRUNE_THRESHOLD = 0.4
```

**Decay:**
```python
DEFAULT_DECAY_RATE = 0.015
DECAY_GRACE_PERIOD_DAYS = 7

INTENT_DECAY_RATES = {
    "HABIT": 0.04,
    "PREFERENCE": 0.015,
    "COMMUNICATION": 0.015,
    "SKILL": 0.005,
    "CONSTRAINT": 0.001
}
```

**Reinforcement:**
```python
BASE_REINFORCEMENT_BOOST = 0.05
```

**Hybrid Retrieval (TGHR):**
```python
HYBRID_DENSE_WEIGHT = 0.55          # Semantic
HYBRID_SPARSE_WEIGHT = 0.30         # BM25
HYBRID_INTENT_BOOST_WEIGHT = 0.15   # Intent affinity
HYBRID_SEARCH_LIMIT = 30
HYBRID_SCORE_THRESHOLD = 0.10
RELEVANCE_GAP_DROP_RATIO = 0.55
MAX_RETRIEVAL_RESULTS = 20
```

**Similarity Thresholds:**
```python
RELATED_BEHAVIORS_DISTANCE_THRESHOLD = 0.72
SEMANTIC_RELEVANCE_THRESHOLD = 0.55
```

---

## 10. External Integrations

### 10.1 Azure OpenAI Integration

**Models Used:**
- **GPT-4.1-mini**: Behavior extraction, conflict analysis, query enrichment
- **text-embedding-3-large**: Semantic embeddings (3072 dimensions)

**API Configuration:**
```python
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview",
    timeout=30
)
```

**Rate Limiting:**
- Implement retry logic with exponential backoff
- Track token usage via metadata
- Batch operations when possible

### 10.2 Redis Streams Integration

**Purpose:** Event publishing for Drift Detection Service

**Configuration:**
```python
REDIS_URL = "redis://localhost:6379/0"
REDIS_STREAM_NAME = "behavior.events"
REDIS_EVENTS_ENABLED = True
```

**Event Types:**
1. `behavior.created`
2. `behavior.reinforced`
3. `behavior.superseded`
4. `behavior.conflict.resolved`

**Event Format:**
```
XADD behavior.events * 
    event_type "behavior.created"
    event_id "evt_abc123"
    published_at 1709481600
    payload "{...json...}"
```

**Consumer:** Drift Detection Service reads from stream

### 10.3 Profile Service Integration

**Endpoints Used:**
1. **POST** `/api/predefined-profiles/assign-profile`: Cold-start profile assignment
2. **GET** `/api/predefined-profiles/user/{user_id}`: Check user mode (COLD_START vs ASSIGNED)

**Cold-Start Flow:**
```
1. User sends first prompt
2. Extract behaviors + profile_signals
3. Check if user is in COLD_START mode
4. If yes, send profile_signals to Profile Service
5. Profile Service accumulates signals
6. After N prompts (e.g., 5), assigns profile
7. User exits COLD_START, enters ASSIGNED mode
```

**Drift Fallback Flow:**
```
1. Profile Service detects drift
2. Calls GET /api/behaviors/{user_id}/signals/recent
3. Retrieves last N profile signals
4. Re-runs profile matching with historical data
```

---

## 11. Testing Strategy

### 11.1 Test Structure

```
tests/
├── test_behavior_extraction.py          # Extraction logic
├── test_behavior_extraction_simple.py   # Performance tests
├── test_credibility.py                  # Credibility scoring
├── test_conflict_detection.py           # Conflict identification
├── test_conflict_resolution.py          # Conflict resolution
├── test_duplicate_detection.py          # Deduplication
├── test_embedding_similarity.py         # Vector similarity
├── test_event_publishing.py             # Redis events
├── test_lazy_decay.py                   # Decay calculations
├── test_intent_decay_rates.py           # Intent-specific decay
├── test_grace_period_implementation.py  # Grace period logic
├── test_profile_signal_extractor.py     # Profile signal validation
├── test_profile_signal_repository.py    # Profile signal persistence
├── test_profile_signals_api.py          # Profile signal endpoints
├── test_cold_start_dispatcher.py        # Cold-start orchestration
├── test_3d_retrieval_e2e.py            # TGHR end-to-end
├── test_3d_comprehensive_e2e.py        # Comprehensive TGHR
└── full flow tests/
    ├── test_e2e_complete_system.py      # Full system integration
    ├── test_flow_1_new_unique_behavior.py
    ├── test_flow_2_duplicate_detection.py
    ├── test_flow_3_conflict_new_wins.py
    └── test_flow_conflict_auto_resolution.py
```

### 11.2 Test Categories

**Unit Tests:**
- Individual function testing
- Mocked dependencies
- Fast execution

**Integration Tests:**
- Database operations
- OpenAI API calls (or mocked)
- Service interactions

**End-to-End Tests:**
- Complete workflows
- Real database
- Full API request/response cycles

### 11.3 Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_credibility.py

# With verbose output
pytest -v tests/

# With coverage
pytest --cov=services --cov=models tests/

# Async tests
pytest --asyncio-mode=auto tests/
```

---

## 12. Deployment

### 12.1 Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc postgresql-client

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import requests; requests.get('http://localhost:8001/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

**docker-compose.yml:**
```yaml
services:
  app:
    build: .
    container_name: behavior-extraction-engine
    ports:
      - "8001:8001"
    environment:
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: redis://shared-redis:6379/0
      AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT}
      AZURE_OPENAI_KEY: ${AZURE_OPENAI_KEY}
    volumes:
      - ./:/app
    networks:
      - shared-network
    restart: unless-stopped

networks:
  shared-network:
    external: true
```

### 12.2 Build and Run

```bash
# Build image
docker build -t behavior-extraction-engine .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 12.3 Database Setup

```bash
# Connect to PostgreSQL
psql -U username -d database_name

# Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

# Run table creation scripts
\i db/existing_db_scripts.txt
\i db/create_profile_signals_table.sql
```

### 12.4 Production Considerations

**Performance:**
- Connection pooling (min=2, max=10)
- HNSW index for vector search
- GIN index for full-text search
- Query optimization for hybrid retrieval

**Monitoring:**
- Health check endpoint (`/health`)
- Structured logging
- Token usage tracking
- Error rate monitoring

**Security:**
- CORS configuration for production origins
- API key management via environment variables
- Database connection string encryption
- Rate limiting (implement at API gateway level)

**Scalability:**
- Horizontal scaling via load balancer
- Database read replicas for retrieval
- Redis cluster for events
- Asynchronous background tasks

---

## 13. Data Flow Diagrams

### 13.1 Complete Extraction and Storage Flow

```
┌────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│  POST /extract { prompt, user_id, session_id }                 │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│              STEP 1: Behavior Extraction                        │
│                  (extractor.py)                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1.1 Call Azure OpenAI GPT-4.1-mini                      │  │
│  │      - System prompt enforces canonical structure        │  │
│  │      - Extract (intent, target, context, polarity)       │  │
│  │      - Score (confidence, clarity, linguistic_strength)  │  │
│  │      - Extract profile_signals                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1.2 Validate and Convert to Pydantic Models            │  │
│  │      - ExtractedBehavior validation                      │  │
│  │      - Profile signal validation                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│           STEP 2: Quality Filtering & Embedding                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.1 Calculate Initial Credibility                       │  │
│  │      credibility = 0.4×conf + 0.35×clarity + 0.25×ling  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.2 Filter Low Quality (< 0.4 credibility)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.3 Generate Embedding (3072-dim)                       │  │
│  │      - Azure OpenAI text-embedding-3-large               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.4 Get Intent-Specific Decay Rate                     │  │
│  │      HABIT: 0.04, PREFERENCE: 0.015, SKILL: 0.005, etc. │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│      STEP 3: Similarity Search & Classification                 │
│                (behaviorRepository.py)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3.1 Search Similar Behaviors (same user + session)     │  │
│  │      - Cosine similarity on embeddings                   │  │
│  │      - Limit to top 10 results                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3.2 Classify Relationship for Each Candidate           │  │
│  │      ┌───────────────────────────────────────────────┐  │  │
│  │      │ DUPLICATE?                                     │  │  │
│  │      │ - Same intent, target, context, polarity       │  │  │
│  │      │ - Distance < 0.12 OR structure match           │  │  │
│  │      └───────────────────────────────────────────────┘  │  │
│  │      ┌───────────────────────────────────────────────┐  │  │
│  │      │ POLARITY_CONFLICT?                            │  │  │
│  │      │ - Same intent, target, context                │  │  │
│  │      │ - Opposite polarity                           │  │  │
│  │      └───────────────────────────────────────────────┘  │  │
│  │      ┌───────────────────────────────────────────────┐  │  │
│  │      │ CROSS_INTENT_CONFLICT?                        │  │  │
│  │      │ - CONSTRAINT vs other intent                  │  │  │
│  │      │ - Same target + context                       │  │  │
│  │      └───────────────────────────────────────────────┘  │  │
│  │      ┌───────────────────────────────────────────────┐  │  │
│  │      │ COMPATIBLE (no action)                        │  │  │
│  │      └───────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│          STEP 4: Action Based on Classification                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  If DUPLICATE:                                           │  │
│  │    → Reinforce existing behavior                         │  │
│  │       - Increment reinforcement_count                    │  │
│  │       - Boost credibility (diminishing returns)          │  │
│  │       - Update last_seen_at                             │  │
│  │       - Publish behavior.reinforced event               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  If POLARITY_CONFLICT:                                   │  │
│  │    → Check credibility difference                        │  │
│  │       - If diff > 0.3: Auto-resolve (higher wins)       │  │
│  │       - If diff ≤ 0.3: Flag for user decision           │  │
│  │       - Insert conflict record                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  If CROSS_INTENT_CONFLICT:                              │  │
│  │    → CONSTRAINT always wins                             │  │
│  │       - Supersede non-constraint behavior               │  │
│  │       - Insert conflict record                          │  │
│  │       - Publish behavior.superseded event               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  If COMPATIBLE:                                          │  │
│  │    → Insert as new behavior                             │  │
│  │       - Store in database                               │  │
│  │       - Publish behavior.created event                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│          STEP 5: Profile Signal Dispatch (Optional)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  5.1 Save profile_signals locally (always)              │  │
│  │      - Insert into user_profile_signals table           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  5.2 Check if user in COLD_START mode                   │  │
│  │      - Query Profile Service                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  5.3 If COLD_START: Send to Profile Service            │  │
│  │      - POST /api/predefined-profiles/assign-profile     │  │
│  │      - Profile Service accumulates signals              │  │
│  │      - Assigns profile after N prompts                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                    RESPONSE TO CLIENT                           │
│  {                                                              │
│    "extraction": {...},                                         │
│    "storage": {...stored behaviors...},                        │
│    "conflicts": [...if any...]                                 │
│  }                                                              │
└────────────────────────────────────────────────────────────────┘
```

### 13.2 Hybrid Retrieval (TGHR) Flow

```
┌────────────────────────────────────────────────────────────────┐
│           POST /v2/extract with conversation history            │
│  { prompt, recent_history, user_id, session_id }               │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│        STEP 1: Context-Aware Extraction                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GPT-4 extracts:                                         │  │
│  │  - Behaviors (same as basic extraction)                   │  │
│  │  - standalone_query: Enriched query with context         │  │
│  │    "which one is better?" →                             │  │
│  │    "which is better for backend: Python or JavaScript?" │  │
│  │  - required_intents: Predicted relevant intents         │  │
│  │    ["PREFERENCE", "CONSTRAINT"]                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│        STEP 2: 3D Hybrid Search (TGHR)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.1 Generate embedding for standalone_query            │  │
│  │      embedding = embed_text(standalone_query)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.2 Build OR-based tsquery for BM25                    │  │
│  │      "Python backend better" → "python | backend | bet" │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.3 Calculate graduated intent affinity                │  │
│  │      required_intents = ["PREFERENCE", "CONSTRAINT"]    │  │
│  │      Behavior with intent="HABIT":                      │  │
│  │        - HABIT ↔ CONSTRAINT: 0.50                       │  │
│  │        - HABIT ↔ PREFERENCE: 0.40                       │  │
│  │        → max_affinity = 0.50                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.4 Execute hybrid SQL query                           │  │
│  │      SELECT                                              │  │
│  │        behavior_id, behavior_text,                      │  │
│  │        embedding <=> %s AS cosine_distance,             │  │
│  │        ts_rank_cd(search_vector, tsquery) AS bm25,      │  │
│  │        (                                                 │  │
│  │          0.55 * (1 - cosine_distance) +                 │  │
│  │          0.30 * bm25_score +                            │  │
│  │          0.15 * intent_affinity                         │  │
│  │        ) AS hybrid_score                                │  │
│  │      FROM behaviors                                      │  │
│  │      WHERE user_id = %s AND session_id = %s            │  │
│  │      ORDER BY hybrid_score DESC                         │  │
│  │      LIMIT 30;                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.5 Apply lazy decay in-memory                         │  │
│  │      For each result:                                    │  │
│  │        - Calculate decay: cred × e^(-λt)                │  │
│  │        - If decay applied: add to decay_updates         │  │
│  │        - Track for last_accessed_at update              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2.6 Filter results                                      │  │
│  │      - Drop if hybrid_score < 0.10                      │  │
│  │      - Drop if score < top × (1 - 0.55)                 │  │
│  │      - Cap at 20 results                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│      STEP 3: Return Results IMMEDIATELY                         │
│  {                                                              │
│    "standalone_query": "...",                                   │
│    "required_intents": ["PREFERENCE", "CONSTRAINT"],           │
│    "related_behaviors": [                                       │
│      {                                                          │
│        "behavior_id": "beh_abc123",                            │
│        "behavior_text": "prefers Python for backend",          │
│        "distance": 0.15,                                        │
│        "intent": "PREFERENCE",                                 │
│        "target": "Python",                                     │
│        "credibility": 0.87                                     │
│      }                                                          │
│    ]                                                            │
│  }                                                              │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│     STEP 4: Background Tasks (Async, Non-Blocking)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Task 1: Store extracted behaviors                       │  │
│  │    - Insert new behaviors                                │  │
│  │    - Detect duplicates/conflicts                        │  │
│  │    - Reinforce/resolve as needed                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Task 2: Persist retrieval updates (batch)              │  │
│  │    - Update credibility for decayed behaviors           │  │
│  │    - Update last_accessed_at for retrieved behaviors    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## 14. Security and Performance

### 14.1 Security Measures

**API Security:**
- CORS configured for specific origins (production)
- Input validation via Pydantic models
- SQL injection prevention (parameterized queries)
- Environment variable protection (no hardcoded secrets)

**Database Security:**
- Connection pooling with health checks
- Foreign key constraints for data integrity
- User-specific data partitioning
- Soft deletes (state management instead of hard deletes)

**LLM Security:**
- Prompt length validation (min: 3, max: 8000 chars)
- Temperature=0.0 for deterministic output
- Response format validation (JSON schema)
- Token usage monitoring

### 14.2 Performance Optimizations

**Database:**
- HNSW index for fast vector search (O(log n) instead of O(n))
- GIN index for full-text search
- Connection pooling (reuse connections)
- Batch updates for retrieval (single query vs N queries)
- Partitioning by user_id for large tables

**Caching:**
- Embedding caching (if same text)
- Connection pool caching
- Redis for event buffering

**Asynchronous Processing:**
- Background tasks for non-critical operations
- Lazy decay computation (on-read instead of on-write)
- Batch persistence of updates

**Query Optimization:**
- Compound indexes for multi-column filters
- LIMIT clauses to reduce result set
- WHERE filters before ORDER BY
- Avoid SELECT * (only needed columns)

### 14.3 Monitoring and Logging

**Structured Logging:**
```python
logger.info(f"Extracted {len(behaviors)} behaviors for user: {user_id}")
logger.debug(f"[3D] Hybrid search: {len(results)} results, threshold: {threshold}")
logger.error(f"Failed to store behavior: {error}")
```

**Metrics to Track:**
- Extraction success rate
- Average extraction time
- Token usage per request
- Database query latency
- Cache hit rate
- Error rate by endpoint

**Health Check:**
```
GET /health
→ {"status": "healthy", "service": "behavior_extraction"}
```

---

## 15. Future Extensibility

### 15.1 Planned Enhancements

**1. Advanced Conflict Resolution:**
- Multi-behavior conflict analysis
- Temporal conflict detection (behaviors valid at different times)
- Context-sensitive conflict resolution

**2. Behavior Evolution Tracking:**
- Track how behaviors change over time
- Visualize behavior drift patterns
- Detect seasonal/cyclical behaviors

**3. Enhanced Profile Integration:**
- Real-time profile updates
- Dynamic profile switching based on context
- Multi-profile support per user

**4. Improved Retrieval:**
- Learned weights for TGHR (ML-based tuning)
- Personalized retrieval based on user interaction history
- Cross-session retrieval (when appropriate)

**5. API Enhancements:**
- Webhook support for async notifications
- GraphQL API for flexible querying
- Batch operations API

### 15.2 Extensibility Points

**New Intent Types:**
Add to `INTENT_CONFLICT_MATRIX` and `INTENT_DECAY_RATES`:
```python
INTENT_DECAY_RATES["NEW_INTENT"] = 0.02
INTENT_CONFLICT_MATRIX["NEW_INTENT"] = {...}
```

**Custom Credibility Weights:**
Adjust in `config/configurations.py`:
```python
CREDIBILITY_WEIGHTS = {
    "confidence": 0.50,  # Increase confidence importance
    "clarity": 0.30,
    "linguistic_strength": 0.20
}
```

**Additional Event Types:**
Extend `BehaviorEventPublisher`:
```python
def publish_custom_event(self, ...):
    self._publish_event("behavior.custom", ...)
```

**Custom Similarity Metrics:**
Add to `utils/similarity_utils.py`:
```python
def custom_distance(embedding1, embedding2) -> float:
    # Custom implementation
    pass
```

---

## Conclusion

This document provides a comprehensive, LLM-readable analysis of the Behavior Extraction Engine system. It covers:

✅ **Architecture**: Layered design with clear separation of concerns  
✅ **Data Models**: Pydantic models with canonical 4-tuple structure  
✅ **Database**: PostgreSQL with pgvector, optimized indexes  
✅ **Algorithms**: Credibility scoring, decay, reinforcement, TGHR  
✅ **Services**: Extraction, storage, retrieval, profile integration  
✅ **API**: REST endpoints with detailed request/response formats  
✅ **Integrations**: Azure OpenAI, Redis, Profile Service  
✅ **Testing**: Comprehensive test suite  
✅ **Deployment**: Docker, docker-compose, production considerations  
✅ **Extensibility**: Clear extension points for future enhancements  

**Key Innovations:**

1. **Canonical Behavior Structure**: 4-tuple (intent, target, context, polarity) enables precise matching
2. **TGHR (Tuple-Guided Hybrid Retrieval)**: 3-signal retrieval with graduated intent affinity
3. **Lazy Decay**: Efficient time-based credibility decay computed on-read
4. **Intent-Specific Decay Rates**: Different behavioral types decay at different rates
5. **Profile Signal Extraction**: Dual-purpose extraction (behaviors + profile characteristics)
6. **Event-Driven Architecture**: Redis Streams for real-time drift detection

This system is production-ready, extensively tested, and designed for scalability and maintainability.
