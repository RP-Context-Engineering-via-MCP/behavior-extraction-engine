# Behavior Detection and Management System

A sophisticated behavior extraction and management system that uses AI to detect, validate, and store user behaviors with semantic filtering and credibility scoring.

## Features

- **AI-Powered Behavior Extraction**: Uses Azure OpenAI to extract user behaviors from text
- **Semantic Filtering**: Validates extracted behaviors against source text using embedding similarity
- **Credibility Scoring**: Calculates confidence scores based on clarity, confidence, and linguistic strength
- **Vector Database Storage**: Stores behaviors with embeddings in PostgreSQL with pgvector
- **FastAPI Backend**: RESTful API for behavior extraction and retrieval
- **Comprehensive Testing**: Includes extensive test suite for all components

## Project Structure

```
behavior_engine/
├── app.py                      # FastAPI application
├── config/
│   └── configurations.py       # Configuration and environment variables
├── db/
│   ├── connection.py          # Database connection management
│   └── init_extensions.py     # Database initialization scripts
├── models/
│   └── behavior.py            # Pydantic models for data validation
├── services/
│   ├── behaviorRepository.py  # Database operations
│   ├── credibilityCalculator.py  # Credibility scoring logic
│   ├── extractor.py           # Main behavior extraction logic
│   ├── openAiClient.py        # Azure OpenAI integration
│   └── supabaseClient.py      # Supabase client (optional)
├── tests/
│   ├── test_behavior_extraction.py
│   ├── test_credibility.py
│   ├── test_embedding_similarity.py
│   └── test_prompt_to_persistence_flow.py
└── utils/
```

## Prerequisites

- Python 3.10 or higher
- PostgreSQL database with pgvector extension
- Azure OpenAI account with API access
- (Optional) Supabase account

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

# Supabase Configuration (Optional)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Application Configuration
SAMPLE_USERID=user_12345
```

### 5. Database Setup

Ensure your PostgreSQL database has the pgvector extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Run the database initialization:

```bash
python -m behavior_engine.db.init_extensions
```

### 6. Run the Application

#### Development Server:
```bash
cd behavior_engine
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Access the API:
- API: http://localhost:8000
- Interactive API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## Running Tests

Run all tests:
```bash
python -m pytest behavior_engine/tests/
```

Run specific test file:
```bash
python behavior_engine/tests/test_behavior_extraction.py
```

## API Endpoints

### Extract Behaviors
```http
POST /extract
Content-Type: application/json

{
  "prompt": "I prefer Python over JavaScript for backend development."
}
```

### Search Similar Behaviors
```http
POST /search
Content-Type: application/json

{
  "user_id": "user_12345",
  "query_text": "programming preferences",
  "top_k": 5
}
```

## Configuration

### Credibility Calculation Weights
Adjust in `config/configurations.py`:
- `confidence`: 0.40 - GPT's confidence in extraction
- `clarity`: 0.35 - Behavior unambiguity
- `linguistic_strength`: 0.25 - Expression strength

### Credibility Threshold
Minimum threshold for storing behaviors: `0.5`

### Semantic Similarity Threshold
Default threshold for semantic filtering: `0.65`

## Key Components

### Behavior Extraction Flow
1. **Input**: User text/prompt
2. **Extraction**: Azure OpenAI extracts potential behaviors
3. **Semantic Filtering**: Validates behaviors against source text
4. **Credibility Scoring**: Calculates confidence scores
5. **Storage**: Stores valid behaviors with embeddings
6. **Retrieval**: Semantic search for similar behaviors

### Models
- `BehaviorSegment`: Individual text segment for analysis
- `ExtractedBehavior`: Behavior extracted by AI
- `StoredBehavior`: Validated and stored behavior with metadata
- `ExtractionResult`: Complete extraction results

## Development

### Adding New Features
1. Update models in `models/behavior.py`
2. Implement logic in appropriate service module
3. Add tests in `tests/` directory
4. Update API endpoints in `app.py`

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Add docstrings for functions and classes

## Troubleshooting

### Database Connection Issues
- Verify DATABASE_URL is correct in `.env`
- Ensure PostgreSQL is running
- Check pgvector extension is installed

### OpenAI API Issues
- Verify API credentials in `.env`
- Check API quota and rate limits
- Ensure correct model names are configured

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version`

## License

[Add your license information here]

## Contributors

[Add contributor information here]

## Contact

[Add contact information here]
