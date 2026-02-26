# Behavior Detection and Management System

A sophisticated behavior extraction and management system that uses AI to detect, validate, store, and manage user behaviors with semantic filtering, credibility scoring, duplicate detection, and conflict resolution.

## Features

- **AI-Powered Behavior Extraction**: Uses Azure OpenAI (GPT-4.1-mini) to extract user behaviors from text
- **Canonical Behavior Structure**: Extracts structured behaviors with intent, target, context, and polarity
- **Semantic Filtering**: Validates extracted behaviors against source text using embedding similarity (text-embedding-3-large)
- **Credibility Scoring**: Calculates confidence scores based on clarity, confidence, and linguistic strength
- **Duplicate Detection**: Identifies and reinforces existing behaviors with intelligent matching
- **Conflict Detection**: Detects and manages conflicting behaviors (polarity conflicts, cross-intent conflicts)
- **Behavior State Management**: Tracks behavior lifecycle (NEW, ACTIVE, SUPERSEDED, FLAGGED, ARCHIVED)
- **Vector Database Storage**: Stores behaviors with 3072-dimensional embeddings in PostgreSQL with pgvector
- **Connection Pooling**: Efficient database connections with psycopg connection pooling
- **FastAPI Backend**: RESTful API with comprehensive endpoints for extraction, retrieval, and analysis
- **Comprehensive Testing**: Extensive test suite covering extraction, credibility, conflicts, and end-to-end flows

## Project Structure

```
├── app.py                          # FastAPI application with API endpoints
├── run.py                          # Alternative entry point
├── config/
│   ├── __init__.py
│   └── configurations.py           # Environment variables and configuration
├── db/
│   ├── __init__.py
│   ├── connection.py               # Database connection management with pooling
│   ├── existing_db_scripts.txt     # Database schema scripts
│   └── db scripts.txt              # Additional database scripts
├── models/
│   ├── __init__.py
│   └── behavior.py                 # Pydantic models for data validation
├── services/
│   ├── __init__.py
│   ├── behaviorRepository.py       # Database operations and queries
│   ├── credibilityCalculator.py    # Credibility and reinforcement calculation
│   ├── extractor.py                # Main behavior extraction orchestration
│   ├── openAiClient.py             # Azure OpenAI integration
│   └── supabaseClient.py           # Supabase client (optional)
├── utils/
│   ├── __init__.py
│   ├── embedding_utils.py          # Embedding generation utilities
│   └── similarity_utils.py         # Distance and similarity calculations
├── tests/
│   ├── test_behavior_extraction.py
│   ├── test_credibility.py
│   ├── test_conflict_detection.py
│   ├── test_duplicate_detection.py
│   ├── test_comprehensive_flows.py
│   └── full flow tests/            # End-to-end integration tests
└── docs/                           # Architecture and API documentation
```

## Prerequisites

- Python 3.10 or higher
- PostgreSQL database with pgvector extension
- Azure OpenAI account with API access (GPT-4.1-mini and text-embedding-3-large models)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "Behavior detaction and management"
```

### 2. Create Virtual Environment

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_KEY=your_azure_openai_key

# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database_name

# Application Configuration (Optional)
SAMPLE_USERID=user_12345

# Supabase Configuration (Optional - if using Supabase)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 5. Database Setup

Ensure your PostgreSQL database has the pgvector extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Run the database schema scripts found in `db/existing_db_scripts.txt` to create the required tables:
- `prompt_segments` - Stores user prompt segments
- `behaviors` - Main table for storing extracted behaviors (partitioned by user_id)
- `conflicts` - Stores detected behavior conflicts

### 6. Run the Application

#### Development Server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Access the API:
- API: http://localhost:8000
- Interactive API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test file:
```bash
python tests/test_behavior_extraction.py
```

Run end-to-end flow tests:
```bash
python -m pytest "tests/full flow tests/"
```

## API Endpoints

### 1. Extract Behaviors (Basic)
Extracts behaviors from a prompt and stores them in the database.

```http
POST /extract
Content-Type: application/json

{
  "prompt": "I prefer Python over JavaScript for backend development.",
  "user_id": "user_12345",
  "session_id": "session_001"  // Optional, defaults to "default"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "extraction": {
      "segments": [...],
      "extraction_time_ms": 1234.56,
      "total_segments": 1,
      "total_behaviors_extracted": 1
    },
    "storage": {
      "stored_behaviors": [...],
      "total_behaviors_stored": 1,
      "behaviors_filtered": 0
    },
    "user_id": "user_12345"
  }
}
```

### 2. Extract Behaviors with Detailed Flow Tracking
Provides detailed information about what happened to each behavior (duplicate, conflict, new, etc.).

```http
POST /extract-detailed
Content-Type: application/json

{
  "prompt": "I like working on frontend projects in React",
  "user_id": "user_12345",
  "session_id": "session_001"  // Optional, defaults to "default"
}
```

**Response includes:**
- Flow tracking for each behavior (NEW, REINFORCED, CONFLICT, PRUNED)
- Matched behaviors and distances
- Conflict information
- Processing statistics

### 3. Get User Behaviors
Retrieves all stored behaviors for a specific user.

```http
GET /behaviors/{user_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user_id": "user_12345",
    "total_behaviors": 10,
    "behaviors": [...]
  }
}
```

### 4. Get User Conflicts
Retrieves all detected conflicts for a specific user.

```http
GET /conflicts/{user_id}
```

**Response includes:**
- Conflict type (POLARITY_CONFLICT, CROSS_INTENT_CONFLICT, etc.)
- Both conflicting behaviors with full details
- Conflict detection timestamp
- Resolution status

### 5. Calculate Behavior Similarity (POC)
Analyzes similarity between two behavior descriptions using embeddings.

```http
POST /similarity
Content-Type: application/json

{
  "behavior1": "I prefer Python for backend",
  "behavior2": "I like using Python for server-side code",
  "metric": "cosine"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "distance": 0.1234,
    "similarity_score": 0.8766,
    "metric": "cosine",
    "interpretation": "Very similar behaviors"
  }
}
```

### 6. Health Check
```http
GET /health
```

Returns service health status.

## Configuration

All configuration settings are in [config/configurations.py](config/configurations.py).

### AI Models
- **GPT Model**: `gpt-4.1-mini` - For behavior extraction and conflict analysis
- **Embedding Model**: `text-embedding-3-large` - Generates 3072-dimensional embeddings
- **API Version**: `2024-12-01-preview`

### Credibility Calculation Weights
Credibility is calculated using weighted factors:
- `confidence`: 0.40 - GPT's confidence in extraction accuracy
- `clarity`: 0.35 - How unambiguous the behavior statement is
- `linguistic_strength`: 0.25 - How strongly the user expressed the behavior

### Thresholds
- **Credibility Prune Threshold**: `0.4` - Minimum credibility to store a behavior
- **Semantic Relevance Threshold**: `0.55` - Threshold for semantic filtering
- **Base Reinforcement Boost**: `0.05` - Boost applied when reinforcing behaviors (with diminishing returns)
- **Default Decay Rate**: `0.015` - Rate at which behavior credibility decays over time

### Behavior States
Behaviors can be in one of these states:
- `NEW` - Freshly extracted, pending validation
- `ACTIVE` - Validated and active in the system
- `SUPERSEDED` - Replaced by a newer, conflicting behavior
- `FLAGGED` - Marked for review due to conflicts
- `ARCHIVED` - No longer active

### Intent Types (Canonical Structure)
- `PREFERENCE` - User likes/dislikes
- `CONSTRAINT` - Requirements or limitations
- `HABIT` - Regular patterns
- `SKILL` - Abilities or expertise
- `COMMUNICATION` - Communication preferences

### Conflict Detection Rules
- **PREFERENCE** can conflict with PREFERENCE and CONSTRAINT
- **SKILL** can conflict with SKILL and CONSTRAINT
- **HABIT** can conflict with HABIT and CONSTRAINT
- **CONSTRAINT** can conflict with ANY intent type
- **COMMUNICATION** can conflict with COMMUNICATION and CONSTRAINT

## Key Components

### Behavior Extraction Flow
1. **Input**: User text/prompt via API
2. **Extraction**: Azure OpenAI extracts behaviors with canonical structure (intent, target, context, polarity)
3. **Semantic Filtering**: Validates behaviors against source text using embedding similarity
4. **Credibility Scoring**: Calculates confidence scores from extraction metrics
5. **Duplicate Detection**: Searches for similar existing behaviors using vector similarity
6. **Conflict Detection**: Identifies conflicts based on intent rules and semantic analysis
7. **Storage**: Stores valid behaviors with embeddings and metadata in PostgreSQL
8. **State Management**: Tracks behavior lifecycle and relationships

### Canonical Behavior Structure
Each behavior is structured with four key components:
- **Intent**: What type of behavior (PREFERENCE, SKILL, HABIT, CONSTRAINT, COMMUNICATION)
- **Target**: The primary object or subject of the behavior
- **Context**: The scope or situation where behavior applies
- **Polarity**: Direction of the behavior (POSITIVE or NEGATIVE)

This structure enables intelligent duplicate detection and conflict resolution based on semantic meaning rather than just text similarity.

### Core Models

#### Data Models ([models/behavior.py](models/behavior.py))
- `BehaviorSegment`: Represents a text segment with extracted behaviors
- `ExtractedBehavior`: AI-extracted behavior with credibility metrics and canonical fields
- `StoredBehavior`: Complete behavior record with database fields and embeddings
- `ExtractionResult`: Complete extraction results with segments and metadata
- `DetailedExtractionResult`: Enhanced result with flow tracking
- `BehaviorFlowInfo`: Tracks what happened to each behavior during processing
- `ConflictType`: Enumeration of conflict types (POLARITY, CROSS_INTENT, etc.)
- `BehaviorState`: Enumeration of behavior lifecycle states

#### Service Layer

**[services/extractor.py](services/extractor.py)** - Main Orchestration
- `run_behavior_extraction()`: Coordinates the full extraction pipeline
- `store_behavior()`: Stores behaviors with duplicate/conflict detection
- `store_behavior_with_tracking()`: Enhanced storage with detailed flow tracking
- Intent conflict matrix and relationship detection
- Behavior comparison and classification logic

**[services/openAiClient.py](services/openAiClient.py)** - AI Integration
- `extract_behavior()`: Calls GPT for behavior extraction
- `embed_text()`: Generates embeddings using text-embedding-3-large
- `analyze_conflict()`: Uses GPT to analyze detected conflicts

**[services/credibilityCalculator.py](services/credibilityCalculator.py)** - Scoring
- `calculate_initial_credibility()`: Computes credibility from extraction metrics
- `calculate_reinforcement_boost()`: Calculates boost for duplicate behaviors
- `should_store_behavior()`: Determines if behavior meets quality threshold

**[services/behaviorRepository.py](services/behaviorRepository.py)** - Database Operations
- `insert_behavior()`: Inserts new behavior records
- `search_similar_behaviors()`: Vector similarity search for duplicates/conflicts
- `reinforce_behavior()`: Updates behavior on duplicate detection
- `insert_conflict()`: Records detected conflicts
- `supersede_behavior()`: Marks behaviors as superseded
- `get_behaviors_by_user()`: Retrieves all user behaviors
- `get_user_conflicts()`: Retrieves user conflicts

**[utils/embedding_utils.py](utils/embedding_utils.py)** - Embedding Operations
- Wrapper functions for embedding generation

**[utils/similarity_utils.py](utils/similarity_utils.py)** - Distance Calculations
- Cosine, Euclidean, and Manhattan distance calculations
- Similarity interpretation helpers

## Database Schema

### Tables

#### behaviors
Partitioned table storing extracted behaviors (partitioned by user_id for scalability).

**Key Fields:**
- `behavior_id` (TEXT, PK): Unique behavior identifier
- `user_id` (TEXT, PK): User identifier (partition key)
- `session_id` (TEXT): Session/conversation identifier
- `behavior_text` (TEXT): The behavior description
- `embedding` (VECTOR(3072)): Behavior embedding for similarity search
- `credibility` (DOUBLE PRECISION): Current credibility score
- `extraction_confidence`, `clarity_score`, `linguistic_strength`: Extraction metrics
- `reinforcement_count` (INTEGER): Number of times reinforced
- `decay_rate` (DOUBLE PRECISION): Credibility decay rate
- `created_at`, `last_seen_at` (BIGINT): Timestamps
- `prompt_history_ids` (TEXT[]): References to prompt segments
- `behavior_state` (TEXT): Current state (ACTIVE, SUPERSEDED, etc.)
- `intent`, `target`, `context`, `polarity`: Canonical structure fields
- `superseded_by_id` (TEXT): ID of behavior that superseded this one

#### prompt_segments
Stores original user prompt segments.

**Key Fields:**
- `segment_id` (UUID, PK): Unique segment identifier
- `user_id` (TEXT): User identifier
- `segment_text` (TEXT): The prompt text
- `created_at` (BIGINT): Timestamp

#### conflicts
Records detected conflicts between behaviors.

**Key Fields:**
- `conflict_id` (TEXT, PK): Unique conflict identifier
- `user_id` (TEXT): User identifier
- `behavior_id_1`, `behavior_id_2` (TEXT): The conflicting behaviors
- `conflict_type` (TEXT): Type of conflict (POLARITY_CONFLICT, etc.)
- `analysis_result` (JSONB): Detailed analysis from GPT
- `resolution_status` (TEXT): PENDING, RESOLVED, etc.
- `detected_at` (BIGINT): Detection timestamp

## Development

### Adding New Features
1. Update models in [models/behavior.py](models/behavior.py)
2. Implement logic in appropriate service module:
   - Extraction logic → [services/extractor.py](services/extractor.py)
   - Database operations → [services/behaviorRepository.py](services/behaviorRepository.py)
   - AI operations → [services/openAiClient.py](services/openAiClient.py)
3. Add comprehensive tests in `tests/` directory
4. Update API endpoints in [app.py](app.py)
5. Update configuration in [config/configurations.py](config/configurations.py) if needed

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Add docstrings for all functions and classes
- Log important operations and errors
- Use Pydantic models for data validation

### Testing Strategy
- Unit tests for individual components
- Integration tests for service interactions
- End-to-end tests for complete flows in `tests/full flow tests/`
- Test files follow naming convention: `test_*.py`

## Troubleshooting

### Database Connection Issues
- Verify `DATABASE_URL` is correct in `.env` file
- Ensure PostgreSQL server is running and accessible
- Check that pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`
- Verify database credentials have necessary permissions
- Check connection pool settings in [db/connection.py](db/connection.py)

### OpenAI API Issues
- Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY` in `.env`
- Check Azure OpenAI resource has required models deployed:
  - `gpt-4.1-mini`
  - `text-embedding-3-large`
- Monitor API quota and rate limits in Azure portal
- Verify API version compatibility (currently using `2024-12-01-preview`)
- Check network connectivity to Azure endpoints

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.10+)
- Check for conflicting package versions

### Extraction Issues
- If behaviors aren't being extracted, check the credibility threshold
- Review extraction confidence in API responses
- Check logs for semantic filtering rejections
- Verify prompt quality - clearer prompts yield better extraction

### Duplicate Detection Not Working
- Check embedding generation is successful (embeddings should be 3072 dimensions)
- Review distance thresholds for your use case
- Examine canonical field extraction (intent, target, context, polarity)
- Check database indexes on user_id for partition performance

### High Memory Usage
- Review connection pool settings (min_size, max_size) in [db/connection.py](db/connection.py)
- Check for unclosed database connections
- Monitor concurrent requests to the API
- Consider adjusting max_idle and max_lifetime for connections

## Architecture

### Key Design Decisions

1. **Partitioned Table by User**: The `behaviors` table is partitioned by `user_id` to enable horizontal scaling as user base grows.

2. **Canonical Structure**: Behaviors are structured with intent, target, context, and polarity to enable intelligent matching beyond simple text similarity.

3. **Connection Pooling**: Uses psycopg connection pooling with health checks to efficiently manage database connections.

4. **Vector Similarity Search**: Uses pgvector for efficient similarity search with 3072-dimensional embeddings.

5. **State Management**: Behaviors have explicit states (ACTIVE, SUPERSEDED, etc.) to track their lifecycle and enable historical tracking.

6. **Conflict Detection**: Multi-layered approach using both semantic similarity and structured field comparison.

7. **Reinforcement Learning**: Behaviors gain credibility through repeated observations with diminishing returns.

## Documentation

Additional documentation is available in the `docs/` directory:
- [API_SPECIFICATION.md](docs/API_SPECIFICATION.md) - Detailed API documentation
- [ARCHITECTURE_BEHAVIOR_DRIFT_SHIFT.md](docs/ARCHITECTURE_BEHAVIOR_DRIFT_SHIFT.md) - System architecture
- [COMPLETE_SYSTEM_FLOWS.md](docs/COMPLETE_SYSTEM_FLOWS.md) - Flow diagrams and descriptions
- [CANONICAL_IMPLEMENTATION_SUMMARY.md](docs/CANONICAL_IMPLEMENTATION_SUMMARY.md) - Canonical structure details

## Performance Considerations

- **Embedding Generation**: Batch embeddings when processing multiple behaviors
- **Database Queries**: Leverage partitioning and indexes for user-based queries
- **Vector Search**: Limit search radius and result count to optimize performance
- **Connection Pooling**: Tune pool size based on concurrent load
- **Caching**: Consider caching embeddings for frequently accessed behaviors

## Security Considerations

- Store Azure OpenAI credentials securely (use environment variables, never commit to repo)
- Use database connection strings with strong passwords
- In production, configure CORS to allow only specific origins (currently set to "*")
- Implement rate limiting on API endpoints
- Sanitize user inputs before storage
- Use prepared statements (already implemented) to prevent SQL injection

## Known Limitations

- Embedding generation requires Azure OpenAI API calls (cost and latency)
- Vector similarity search performance degrades with very large datasets (consider approximate nearest neighbor algorithms)
- Conflict resolution currently requires manual review for certain conflict types
- No built-in user authentication (implement at API gateway or application level)

## Future Enhancements

- Implement automatic conflict resolution strategies
- Add behavior expiration based on age and credibility
- Implement behavior analytics and insights
- Add support for behavior hierarchies and relationships
- Implement real-time notifications for conflicts
- Add batch processing for large prompt sets
- Implement behavior evolution tracking over time

## License

[Add your license information here]

## Contributors

[Add contributor information here]

## Contact

[Add contact information here]
