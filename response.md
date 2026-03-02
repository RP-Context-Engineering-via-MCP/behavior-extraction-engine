The file system isn't available right now so I can't write the file directly. Here is the complete document — you can copy and save it as a `.md` file:

---

# Behavior Resolution Mechanism — Profile Support Design

## Table of Contents
1. Overview
2. What Already Exists vs What Needs to Be Built
3. Architecture Diagram
4. New Component 1 — Profile Signal Extractor
5. New Component 2 — Profile Signal Repository
6. New Component 3 — Profile Service Client
7. New Component 4 — Cold Start Dispatcher
8. New Component 5 — Recent Behaviors Endpoint
9. Database Schema — New Table
10. Extended GPT-4 Extraction Prompt
11. Complete Processing Flow
12. Configuration & Environment Variables
13. Component Input/Output Reference

---

## 1. Overview

### 1.1 The Problem

Behavior Resolution currently outputs canonical behaviors:
```json
{
    "intent": "PREFERENCE",
    "target": "Python",
    "context": "backend",
    "polarity": "POSITIVE",
    "confidence": 0.95
}
```

The Profile Service needs a completely different shape:
```json
{
    "intents": {"LEARNING": 0.8, "PROBLEM_SOLVING": 0.6},
    "interests": {"PROGRAMMING": 0.7, "AI": 0.4},
    "behavior_level": "INTERMEDIATE",
    "signals": {"DETAILED_EXPLANATION": 0.5},
    "complexity": 0.65,
    "consistency": 0.72
}
```

The vocabularies don't map through any simple rule. The gap can only be bridged accurately by GPT-4 while it still has the full prompt in context.

### 1.2 The Solution

Extend the existing GPT-4 extraction call to output **two payloads simultaneously** from the same prompt — no second API call:

- `canonical_behaviors` — existing output, unchanged, feeds Drift Detection
- `profile_signals` — new output, feeds the Profile Service

### 1.3 Two Triggers That Use This Data

| Trigger | When | How |
|---|---|---|
| Cold start | Every prompt for a COLD_START user | HTTP POST to Profile Service after extraction |
| Drift fallback | Profile Service receives drift event | Profile Service calls `GET /api/behaviors/{user_id}/recent` |

---

## 2. What Already Exists vs What Needs to Be Built

### Already Exists (no changes)

| Component | Purpose |
|---|---|
| GPT-4 extraction pipeline | Extracts canonical behaviors from raw prompts |
| Canonical behavior storage | Stores behaviors in `behaviors` table |
| Relationship detection | DUPLICATE / POLARITY_CONFLICT / COMPATIBLE etc. |
| Conflict resolution | Auto-resolve or flag conflicting behaviors |
| `behavior.events` Redis Stream publisher | Publishes lifecycle events to Drift Detection |

### Needs to Be Built (new)

| Component | Purpose |
|---|---|
| `ProfileSignalExtractor` | Validates GPT-4 `profile_signals` output |
| `user_profile_signals` table | Stores `profile_signals` per prompt per user |
| `ProfileSignalRepository` | Saves and queries `user_profile_signals` |
| `ProfileServiceClient` | HTTP client that calls Profile Service |
| `ColdStartDispatcher` | Decides when and whether to call Profile Service |
| `GET /api/behaviors/{user_id}/recent` | Returns last N `profile_signals` for drift fallback |

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIOR RESOLUTION MECHANISM                         │
│                                                                          │
│  Raw Prompt                                                              │
│      │                                                                   │
│      ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              GPT-4 Extraction (EXTENDED)                          │  │
│  │  OUTPUT: {                                                        │  │
│  │    canonical_behaviors: [...],   <- existing, unchanged           │  │
│  │    profile_signals: {...}        <- NEW                           │  │
│  │  }                                                                │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                           │
│              ┌───────────────┴───────────────┐                          │
│              │                               │                          │
│              ▼                               ▼                          │
│  ┌───────────────────────┐     ┌──────────────────────────────────────┐ │
│  │  Existing Pipeline    │     │  ProfileSignalRepository.save()      │ │
│  │  - Relationship detect│     │  INSERT into user_profile_signals    │ │
│  │  - Conflict resolve   │     └──────────────────┬───────────────────┘ │
│  │  - behavior.events ──►│Redis                   │                     │
│  └───────────────────────┘                        ▼                     │
│                                    ┌──────────────────────────────────┐ │
│                                    │      ColdStartDispatcher         │ │
│                                    │  - Check user mode               │ │
│                                    │  - If COLD_START:                │ │
│                                    │    call ProfileServiceClient     │ │
│                                    └──────────────┬───────────────────┘ │
│                                                   │ HTTP POST           │
└───────────────────────────────────────────────────┼─────────────────────┘
                                                    ▼
                                       Predefined Profile Service

┌─────────────────────────────────────────────────────────────────────────┐
│  GET /api/behaviors/{user_id}/recent?limit=N                            │
│  <- Called by Profile Service during DRIFT_FALLBACK                     │
│  -> Returns last N profile_signals from user_profile_signals table      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. New Component 1 — Profile Signal Extractor

**File:** `app/extraction/profile_signal_extractor.py`

Parses and validates the `profile_signals` block from the GPT-4 response. Does NOT call GPT-4 itself — just validates and cleans the output.

### Controlled Vocabularies

**Intents (profile-level — different from canonical intents):**

| Intent | When to use |
|---|---|
| `LEARNING` | User wants to understand a concept |
| `TASK_COMPLETION` | User wants something done, result-focused |
| `PROBLEM_SOLVING` | User is debugging or solving technical issues |
| `EXPLORATION` | User is brainstorming or generating ideas |
| `GUIDANCE` | User is seeking advice or personal direction |
| `ENGAGEMENT` | Casual, fun, low-commitment interaction |

**Interest Areas:**

| Interest | Description |
|---|---|
| `AI` | Artificial intelligence, machine learning |
| `DATA_SCIENCE` | Data analysis, statistics |
| `WRITING` | Drafting, editing, summarizing |
| `PROGRAMMING` | Coding, debugging, algorithms |
| `CREATIVE` | Stories, scripts, ideation |
| `HEALTH` | Well-being, diet, exercise |
| `PERSONAL_GROWTH` | Career, life guidance |
| `ENTERTAINMENT` | Games, quizzes, leisure |

**Behavior Signals:**

| Signal | Description |
|---|---|
| `DEEP_REASONING` | Open-ended curiosity-driven queries |
| `DETAILED_EXPLANATION` | User wants thorough explanation |
| `CODE_FOCUSED` | Response expected to contain code |
| `STEP_BY_STEP` | User wants progressive breakdown |
| `QUICK_ANSWER` | User wants concise response |
| `CREATIVE_OUTPUT` | User wants generated creative content |
| `EMPATHETIC_RESPONSE` | User needs emotional tone |
| `ITERATIVE_REFINEMENT` | User expects multiple turn refinement |

### Implementation

```python
# app/extraction/profile_signal_extractor.py

class ProfileSignalExtractor:

    VALID_INTENTS = {
        "LEARNING", "TASK_COMPLETION", "PROBLEM_SOLVING",
        "EXPLORATION", "GUIDANCE", "ENGAGEMENT"
    }
    VALID_INTERESTS = {
        "AI", "DATA_SCIENCE", "WRITING", "PROGRAMMING",
        "CREATIVE", "HEALTH", "PERSONAL_GROWTH", "ENTERTAINMENT"
    }
    VALID_SIGNALS = {
        "DEEP_REASONING", "DETAILED_EXPLANATION", "CODE_FOCUSED",
        "STEP_BY_STEP", "QUICK_ANSWER", "CREATIVE_OUTPUT",
        "EMPATHETIC_RESPONSE", "ITERATIVE_REFINEMENT"
    }
    VALID_LEVELS = {"BEGINNER", "INTERMEDIATE", "ADVANCED"}

    def parse_and_validate(self, raw: dict) -> dict:
        intents = {
            k: float(v) for k, v in raw.get("intents", {}).items()
            if k in self.VALID_INTENTS
        }
        interests = {
            k: float(v) for k, v in raw.get("interests", {}).items()
            if k in self.VALID_INTERESTS
        }
        behavior_level = raw.get("behavior_level", "BEGINNER")
        if behavior_level not in self.VALID_LEVELS:
            behavior_level = "BEGINNER"

        signals = {
            k: float(v) for k, v in raw.get("signals", {}).items()
            if k in self.VALID_SIGNALS
        }
        complexity  = min(1.0, max(0.0, float(raw.get("complexity", 0.5))))
        consistency = min(1.0, max(0.0, float(raw.get("consistency", 0.5))))

        if not intents:
            raise ValueError("profile_signals must contain at least one valid intent")
        if not interests:
            raise ValueError("profile_signals must contain at least one valid interest")

        return {
            "intents": intents,
            "interests": interests,
            "behavior_level": behavior_level,
            "signals": signals,
            "complexity": complexity,
            "consistency": consistency
        }
```

---

## 5. New Component 2 — Profile Signal Repository

**File:** `app/repositories/profile_signal_repository.py`

Persists `profile_signals` per prompt per user. Enables the `/recent` endpoint to serve historical signals to the Profile Service during drift fallback.

```python
# app/repositories/profile_signal_repository.py

import json, time
from typing import List
from app.db import get_db_connection

class ProfileSignalRepository:

    async def save(self, user_id: str, prompt_id: str, profile_signals: dict) -> None:
        async with get_db_connection() as conn:
            await conn.execute(
                """
                INSERT INTO user_profile_signals
                    (user_id, prompt_id, profile_signals, extracted_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, prompt_id) DO UPDATE
                    SET profile_signals = EXCLUDED.profile_signals,
                        extracted_at    = EXCLUDED.extracted_at
                """,
                user_id, prompt_id, json.dumps(profile_signals), int(time.time())
            )

    async def get_recent(self, user_id: str, limit: int = 10) -> List[dict]:
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT profile_signals FROM user_profile_signals
                WHERE user_id = $1
                ORDER BY extracted_at DESC
                LIMIT $2
                """,
                user_id, limit
            )
            return [json.loads(row["profile_signals"]) for row in rows]

    async def get_count(self, user_id: str) -> int:
        async with get_db_connection() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM user_profile_signals WHERE user_id = $1",
                user_id
            ) or 0
```

---

## 6. New Component 3 — Profile Service Client

**File:** `app/clients/profile_service_client.py`

HTTP client that calls `POST /api/predefined-profiles/assign-profile` after each prompt for a COLD_START user.

```python
# app/clients/profile_service_client.py

import httpx
from app.config.settings import settings

class ProfileServiceClient:

    async def assign_profile(self, user_id: str, profile_signals: dict) -> dict | None:
        url = f"{settings.profile_service_base_url}/api/predefined-profiles/assign-profile"
        payload = {
            "user_id": user_id,
            "mode": "COLD_START",
            "extracted_behavior": profile_signals
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[ProfileServiceClient] HTTP error for {user_id}: {e}")
            return None
        except httpx.TimeoutException:
            print(f"[ProfileServiceClient] Timeout for {user_id}")
            return None
        except Exception as e:
            print(f"[ProfileServiceClient] Error for {user_id}: {e}")
            return None
```

---

## 7. New Component 4 — Cold Start Dispatcher

**File:** `app/core/cold_start_dispatcher.py`

Runs after every prompt extraction. Saves `profile_signals` always (needed for drift fallback), then conditionally calls the Profile Service if the user is still in COLD_START mode.

**User mode check strategy:** Calls `GET /api/predefined-profiles/user/{user_id}` on the Profile Service — the single source of truth for profile state. Behavior Resolution does not shadow this state. Stops calling once `assigned_profile_id` is not null.

```python
# app/core/cold_start_dispatcher.py

import httpx
from app.clients.profile_service_client import ProfileServiceClient
from app.repositories.profile_signal_repository import ProfileSignalRepository
from app.config.settings import settings

class ColdStartDispatcher:

    def __init__(self):
        self._profile_client = ProfileServiceClient()
        self._signal_repo    = ProfileSignalRepository()

    async def dispatch(self, user_id: str, prompt_id: str, profile_signals: dict) -> None:

        # 1. Always save — needed for drift fallback regardless of mode
        await self._signal_repo.save(user_id, prompt_id, profile_signals)

        # 2. Check if user still needs cold-start profiling
        if not await self._is_cold_start_user(user_id):
            return

        # 3. Call Profile Service
        result = await self._profile_client.assign_profile(user_id, profile_signals)
        if result:
            print(
                f"[ColdStartDispatcher] user={user_id} "
                f"status={result.get('status')} "
                f"profile={result.get('assigned_profile_id')}"
            )

    async def _is_cold_start_user(self, user_id: str) -> bool:
        url = f"{settings.profile_service_base_url}/api/predefined-profiles/user/{user_id}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    return True   # New user, cold start needed
                response.raise_for_status()
                data = response.json()
                if data.get("assigned_profile_id"):
                    return False  # Already assigned, cold start over
                return data.get("user_mode") == "COLD_START"
        except Exception as e:
            print(f"[ColdStartDispatcher] Could not check mode for {user_id}: {e}")
            return False  # Fail safe — do not call if mode is uncertain
```

---

## 8. New Component 5 — Recent Behaviors Endpoint

**File:** Addition to `app/api/behavior_routes.py`

Serves the last N `profile_signals` for a user. Called exclusively by the Profile Service during drift fallback.

**Endpoint:** `GET /api/behaviors/{user_id}/recent?limit=10`

**Response:**
```json
{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "count": 2,
    "behaviors": [
        {
            "intents": {"PROBLEM_SOLVING": 0.9},
            "interests": {"PROGRAMMING": 0.8},
            "behavior_level": "ADVANCED",
            "signals": {"CODE_FOCUSED": 0.8},
            "complexity": 0.82,
            "consistency": 0.75
        }
    ]
}
```

```python
# Addition to app/api/behavior_routes.py

from fastapi import APIRouter, Query
from app.repositories.profile_signal_repository import ProfileSignalRepository

router = APIRouter()
signal_repo = ProfileSignalRepository()

@router.get("/behaviors/{user_id}/recent")
async def get_recent_behaviors(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50)
):
    behaviors = await signal_repo.get_recent(user_id=user_id, limit=limit)
    return {"user_id": user_id, "count": len(behaviors), "behaviors": behaviors}
```

---

## 9. Database Schema — New Table

```sql
CREATE TABLE user_profile_signals (
    id              UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         TEXT    NOT NULL,
    prompt_id       TEXT    NOT NULL,
    profile_signals JSONB   NOT NULL,
    extracted_at    BIGINT  NOT NULL,
    UNIQUE (user_id, prompt_id)
);

CREATE INDEX idx_user_profile_signals_user_extracted
    ON user_profile_signals (user_id, extracted_at DESC);
```

| Column | Type | Description |
|---|---|---|
| `user_id` | TEXT | User identifier |
| `prompt_id` | TEXT | Links to the specific prompt |
| `profile_signals` | JSONB | Full profile_signals dict |
| `extracted_at` | BIGINT | Unix timestamp for ORDER BY in recent query |

---

## 10. Extended GPT-4 Extraction Prompt

Append this block to the **existing** extraction system prompt. No second API call:

```
Additionally, extract a "profile_signals" block from the same prompt
using these controlled vocabularies only:

intents (all that apply, confidence 0.0-1.0):
  LEARNING, TASK_COMPLETION, PROBLEM_SOLVING,
  EXPLORATION, GUIDANCE, ENGAGEMENT

interests (all that apply, confidence 0.0-1.0):
  AI, DATA_SCIENCE, WRITING, PROGRAMMING,
  CREATIVE, HEALTH, PERSONAL_GROWTH, ENTERTAINMENT

behavior_level (one of):
  BEGINNER | INTERMEDIATE | ADVANCED

signals (all that apply, confidence 0.0-1.0):
  DEEP_REASONING, DETAILED_EXPLANATION, CODE_FOCUSED,
  STEP_BY_STEP, QUICK_ANSWER, CREATIVE_OUTPUT,
  EMPATHETIC_RESPONSE, ITERATIVE_REFINEMENT

complexity (float 0.0-1.0):
  Prompt complexity based on length, constraints, multi-step
  nature, and technical depth.

consistency (float 0.0-1.0):
  Consistency with observed session history. Default 0.5 if
  no history available.
```

**Extended GPT-4 output structure:**
```json
{
    "segments": [
        {
            "text": "...",
            "behaviors": [
                {
                    "intent": "PREFERENCE",
                    "target": "Python",
                    "context": "backend",
                    "polarity": "POSITIVE",
                    "confidence": 0.95
                }
            ]
        }
    ],
    "profile_signals": {
        "intents": {"PROBLEM_SOLVING": 0.85, "LEARNING": 0.40},
        "interests": {"PROGRAMMING": 0.80, "AI": 0.45},
        "behavior_level": "ADVANCED",
        "signals": {"CODE_FOCUSED": 0.75, "ITERATIVE_REFINEMENT": 0.60},
        "complexity": 0.78,
        "consistency": 0.72
    }
}
```

---

## 11. Complete Processing Flow

### Cold Start (Per Prompt)
```
User sends prompt
      │
      ▼
GPT-4 Extraction (EXTENDED)
      │
      ├──► canonical_behaviors ──► Relationship Detection
      │                         ──► behavior.events (Redis) ──► Drift Detection
      │
      └──► profile_signals ──► ProfileSignalExtractor.parse_and_validate()
                                      │
                               ProfileSignalRepository.save()
                                      │
                               ColdStartDispatcher._is_cold_start_user()
                                      │
                           ┌──────────┴──────────┐
                      COLD_START            Already assigned
                           │                     │
                           ▼                  (stop)
                  ProfileServiceClient.assign_profile()
                  POST /api/predefined-profiles/assign-profile
                           │
                           ▼
                  Profile Service ──► PENDING or ASSIGNED
```

### Drift Fallback
```
Drift Detection ──► drift.events (Redis)
                           │
                  Profile Service consumes
                  (MODERATE or STRONG severity only)
                           │
                  GET /api/behaviors/{user_id}/recent?limit=10
                           │
                  Behavior Resolution returns List[profile_signals]
                           │
                  Profile Service runs DRIFT_FALLBACK matching
                           │
                  profile.assigned published to Redis
```

---

## 12. Configuration & Environment Variables

```python
# Addition to app/config/settings.py
profile_service_base_url: str          # http://predefined-profile-service:8000
profile_signals_default_limit: int = 10
profile_signals_max_limit: int = 50
```

```env
# .env.example addition
PROFILE_SERVICE_BASE_URL=http://predefined-profile-service:8000
PROFILE_SIGNALS_DEFAULT_LIMIT=10
PROFILE_SIGNALS_MAX_LIMIT=50
```

---

## 13. Component Input/Output Reference

| Component | Input | Output |
|---|---|---|
| `ProfileSignalExtractor.parse_and_validate()` | Raw `profile_signals` from GPT-4 | Validated `profile_signals` dict |
| `ProfileSignalRepository.save()` | `user_id`, `prompt_id`, `profile_signals` | Row in `user_profile_signals` |
| `ProfileSignalRepository.get_recent()` | `user_id`, `limit` | `List[profile_signals dict]` |
| `ProfileSignalRepository.get_count()` | `user_id` | `int` |
| `ProfileServiceClient.assign_profile()` | `user_id`, `profile_signals` | Assignment result dict or `None` |
| `ColdStartDispatcher.dispatch()` | `user_id`, `prompt_id`, `profile_signals` | None — side effects only |
| `ColdStartDispatcher._is_cold_start_user()` | `user_id` | `bool` |
| `GET /api/behaviors/{user_id}/recent` | `user_id`, `limit` | `{user_id, count, behaviors}` |
| Extended GPT-4 prompt | Raw prompt text | `{segments, profile_signals}` |

---

## Summary

### New Files to Create

| File | Purpose |
|---|---|
| `app/extraction/profile_signal_extractor.py` | Validates GPT-4 profile_signals output |
| `app/repositories/profile_signal_repository.py` | Saves and queries `user_profile_signals` |
| `app/clients/profile_service_client.py` | HTTP client to call Profile Service |
| `app/core/cold_start_dispatcher.py` | Orchestrates cold-start trigger logic |

### Existing Files to Modify

| File | Change |
|---|---|
| GPT-4 extraction system prompt | Append `profile_signals` extraction instructions |
| Extraction pipeline orchestrator | Call `ColdStartDispatcher.dispatch()` after extraction |
| `app/api/behavior_routes.py` | Add `GET /api/behaviors/{user_id}/recent` |
| `app/config/settings.py` | Add profile service URL and limit settings |
| Database migrations | Add `user_profile_signals` table and index |