"""
Comprehensive E2E Benchmark: 3D Hybrid Retrieval + Co-Occurrence Graph
========================================================================

Covers 10 behavior categories:
  Food & Diet, Exercise & Fitness, Sleep & Routine, Work & Productivity,
  Entertainment, Travel & Transport, Music, Social, Shopping & Health,
  Reading & Pets & Technology

Phase 1 — SEED:      15 diverse prompts via /extract (v1, synchronous storage)
                      CO_PROMPT graph edges are created per prompt.
Phase 2 — RETRIEVE:  25 prompt-only tests via /v2/extract
Phase 3 — HISTORY:   25 prompt + history tests via /v2/extract

Both Phase 2 and 3 validate:
  - related_behaviors  (from embedding / LRA search)
  - associated_behaviors (from 1-hop graph expansion)

Each run uses a unique user_id to avoid interference with previous runs.
All actions within a run share the same session_id.

Results saved to: docs/behavior retrieval testing/YYYY_MM_DD_HHMM_testN.txt

Run:
    python -m tests.test_3d_comprehensive_e2e

Prerequisites:
    - Server running on http://localhost:8000
    - Database migrated with search_vector column + GIN index
    - Database migrated with behavior_co_occurrences table
"""

import requests
import time
import json
import sys
import os
import re
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"

# Unique per-run IDs — prevents contamination from previous test runs
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
USER_ID = f"test_graph_{RUN_TS}"
SESSION_ID = f"sess_{RUN_TS}"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "behavior retrieval testing"
)

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ─── Output Buffer ──────────────────────────────────────────────────────────
_log_lines = []


def log(msg=""):
    """Print and buffer a line (plain)."""
    _log_lines.append(msg)
    print(msg)


def log_c(msg, color):
    """Print with color, buffer without."""
    _log_lines.append(msg)
    print(f"{color}{msg}{RESET}")


def section(title):
    log()
    log("=" * 90)
    log(f"  {title}")
    log("=" * 90)
    log()


def ok(msg):
    log_c(f"  ✓ {msg}", GREEN)


def fail(msg):
    log_c(f"  ✗ {msg}", RED)


def warn(msg):
    log_c(f"  ⚠ {msg}", YELLOW)


# ─── API Helpers ─────────────────────────────────────────────────────────────
def call_v1_extract(prompt: str, session_id: str = None, max_retries: int = 3) -> dict:
    """POST /extract — synchronous extraction + storage, with retry on transient errors."""
    sid = session_id or SESSION_ID
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/extract",
                json={"prompt": prompt, "user_id": USER_ID, "session_id": sid},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            last_error = e
            status_code = e.response.status_code if e.response is not None else 0
            # Retry on server-side errors (5xx) and 422 (often transient LLM failures)
            if status_code in (422, 500, 502, 503, 504):
                warn(f"Attempt {attempt}/{max_retries} failed ({status_code}), retrying in 3s...")
                time.sleep(3)
                continue
            raise  # Non-retryable HTTP error (400, 404, etc.)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            warn(f"Attempt {attempt}/{max_retries} failed ({type(e).__name__}), retrying in 3s...")
            time.sleep(3)
            continue
    # All retries exhausted
    raise last_error


def call_v2_extract(prompt: str, recent_history: list = None, session_id: str = None) -> dict:
    """POST /v2/extract — extraction + 3D hybrid retrieval + graph expansion."""
    sid = session_id or SESSION_ID
    body = {"prompt": prompt, "user_id": USER_ID, "session_id": sid}
    if recent_history:
        body["recent_history"] = recent_history
    resp = requests.post(f"{BASE_URL}/v2/extract", json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED DATA — 15 diverse prompts covering 10+ behavior categories
# ═══════════════════════════════════════════════════════════════════════════════
SEED_PROMPTS = [
    # 1: Food allergies & preferences
    "I am allergic to peanuts and I love eating dark chocolate after dinner. "
    "I prefer organic vegetables and avoid processed foods.",

    # 2: Exercise habits
    "I go for a 5km morning jog every day before breakfast. "
    "I avoid heavy workouts in the evening because it affects my sleep.",

    # 3: Sleep & routine
    "I usually sleep by 10pm and wake up at 5:30am. "
    "I don't drink coffee after 3pm because it keeps me awake.",

    # 4: Diet specifics
    "I follow a low-carb, high-protein diet. "
    "I eat eggs and avocado for breakfast every morning. "
    "I try to avoid sugar and white bread.",

    # 5: Cuisine preferences
    "I love sushi and Japanese food in general. "
    "I dislike spicy food and never eat anything with chili peppers.",

    # 6: Work habits
    "I work from home on Mondays and Fridays. "
    "I prefer having meetings in the morning and doing deep focus work in the afternoon. "
    "I take a short walk during my lunch break.",

    # 7: Entertainment
    "I watch Netflix every evening for about an hour. "
    "I prefer documentaries and thriller series. "
    "I don't enjoy reality TV shows at all.",

    # 8: Travel preferences
    "I always choose window seats on flights. "
    "I prefer taking trains over buses for short trips. "
    "I never check in luggage if I can carry on instead.",

    # 9: Music preferences
    "I listen to lo-fi beats while working to help me concentrate. "
    "I enjoy jazz music on weekend mornings. I really cannot stand heavy metal.",

    # 10: Social habits
    "I prefer texting over phone calls for most communication. "
    "I usually meet friends on Saturday evenings for dinner. "
    "I tend to avoid large crowded events and parties.",

    # 11: Shopping habits
    "I always compare prices online before buying any electronics. "
    "I prefer shopping online rather than going to physical stores. "
    "I try to buy eco-friendly and sustainable products whenever possible.",

    # 12: Health & wellness
    "I take vitamin D and omega-3 supplements every morning with breakfast. "
    "I drink at least 2 liters of water throughout the day. "
    "I avoid taking painkillers unless it's absolutely necessary.",

    # 13: Reading habits
    "I read for about 30 minutes before going to bed every night. "
    "I mainly enjoy non-fiction books about psychology and science. "
    "I prefer using my Kindle over physical paper books.",

    # 14: Pet care
    "I walk my golden retriever twice a day, once in the morning and once in the evening. "
    "I only feed him natural organic dog food. "
    "I take him to the vet for checkups every six months.",

    # 15: Technology preferences
    "I always use dark mode on all my devices and apps. "
    "I back up my important files to cloud storage every Sunday. "
    "I make it a rule not to check social media before noon.",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — 25 PROMPT-ONLY RETRIEVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
PROMPT_ONLY_TESTS = [
    # ── Food & Diet (6) ──────────────────────────────────────────────────────
    {
        "prompt": "What foods should I eat before my morning run?",
        "category": "Food & Diet",
        "description": "exercise habits, food preferences, breakfast foods, peanut allergy",
        "expected_keywords": ["jog", "egg", "avocado", "breakfast", "peanut", "morning"],
    },
    {
        "prompt": "Is dark chocolate a healthy evening snack for me?",
        "category": "Food & Diet",
        "description": "dark chocolate preference, dietary constraints, sugar avoidance",
        "expected_keywords": ["chocolate", "dinner", "sugar", "processed"],
    },
    {
        "prompt": "Can you recommend a high protein lunch?",
        "category": "Food & Diet",
        "description": "high-protein diet, low-carb, food avoidances",
        "expected_keywords": ["protein", "bread", "carb", "processed", "sugar"],
    },
    {
        "prompt": "What should I order at a Japanese restaurant?",
        "category": "Food & Diet",
        "description": "sushi/Japanese food love, spicy food avoidance, peanut allergy",
        "expected_keywords": ["sushi", "japanese", "spicy", "chili", "peanut"],
    },
    {
        "prompt": "What ingredients should I completely avoid when cooking?",
        "category": "Food & Diet",
        "description": "peanut allergy, processed food avoidance, sugar, spicy, bread",
        "expected_keywords": ["peanut", "processed", "sugar", "bread", "spicy", "chili"],
    },
    {
        "prompt": "What are all my dietary restrictions and food preferences?",
        "category": "Food & Diet",
        "description": "comprehensive: allergy, diet, avoidances, cuisine preferences",
        "expected_keywords": ["peanut", "sugar", "bread", "processed", "protein", "organic"],
    },

    # ── Exercise & Sleep (5) ─────────────────────────────────────────────────
    {
        "prompt": "What time should I stop drinking coffee?",
        "category": "Exercise & Sleep",
        "description": "coffee constraint, sleep schedule",
        "expected_keywords": ["coffee", "afternoon", "sleep"],
    },
    {
        "prompt": "What exercises can I do first thing in the morning?",
        "category": "Exercise & Sleep",
        "description": "morning jog habit, exercise preferences",
        "expected_keywords": ["jog", "morning", "workout", "5km"],
    },
    {
        "prompt": "How can I improve my sleep quality?",
        "category": "Exercise & Sleep",
        "description": "sleep habits, coffee avoidance, evening workout avoidance",
        "expected_keywords": ["sleep", "coffee", "wake", "evening", "workout"],
    },
    {
        "prompt": "What does my typical morning routine look like?",
        "category": "Exercise & Sleep",
        "description": "wake time, jog, breakfast foods, supplements",
        "expected_keywords": ["jog", "wake", "morning", "egg", "avocado", "vitamin"],
    },
    {
        "prompt": "Should I exercise in the evening after work?",
        "category": "Exercise & Sleep",
        "description": "evening workout avoidance, sleep impact",
        "expected_keywords": ["workout", "evening", "heavy", "sleep"],
    },

    # ── Work & Productivity (4) ──────────────────────────────────────────────
    {
        "prompt": "How should I structure my workday for maximum productivity?",
        "category": "Work & Productivity",
        "description": "work from home, morning meetings, afternoon focus",
        "expected_keywords": ["meeting", "morning", "focus", "afternoon", "work", "home"],
    },
    {
        "prompt": "When is the best time for me to schedule important meetings?",
        "category": "Work & Productivity",
        "description": "morning meeting preference",
        "expected_keywords": ["meeting", "morning"],
    },
    {
        "prompt": "What do I usually do during my lunch break?",
        "category": "Work & Productivity",
        "description": "lunch break walk habit",
        "expected_keywords": ["walk", "lunch", "break"],
    },
    {
        "prompt": "Which days do I work remotely from home?",
        "category": "Work & Productivity",
        "description": "work from home schedule",
        "expected_keywords": ["home", "monday", "friday", "work"],
    },

    # ── Entertainment & Music (4) ────────────────────────────────────────────
    {
        "prompt": "What should I watch on TV tonight after dinner?",
        "category": "Entertainment",
        "description": "Netflix habit, documentary/thriller preference, reality avoidance",
        "expected_keywords": ["netflix", "document", "thriller", "reality"],
    },
    {
        "prompt": "What kind of music should I play while I work?",
        "category": "Music",
        "description": "lo-fi music while working",
        "expected_keywords": ["lo-fi", "lofi", "music", "work", "concentrate"],
    },
    {
        "prompt": "What types of TV shows and genres do I enjoy watching?",
        "category": "Entertainment",
        "description": "documentary and thriller preference, reality TV avoidance",
        "expected_keywords": ["document", "thriller", "reality", "netflix"],
    },
    {
        "prompt": "What music do I enjoy on weekends?",
        "category": "Music",
        "description": "jazz on weekends, heavy metal avoidance",
        "expected_keywords": ["jazz", "weekend", "metal"],
    },

    # ── Travel, Social, Shopping, Health, Reading, Pets, Tech (6) ────────────
    {
        "prompt": "I'm booking a flight for vacation, any seating preferences?",
        "category": "Travel",
        "description": "window seat preference, luggage avoidance",
        "expected_keywords": ["window", "seat", "flight", "luggage"],
    },
    {
        "prompt": "When do my friends and I usually get together?",
        "category": "Social",
        "description": "Saturday evening dinner with friends",
        "expected_keywords": ["saturday", "evening", "friend", "dinner"],
    },
    {
        "prompt": "What supplements and vitamins do I take daily?",
        "category": "Health",
        "description": "vitamin D, omega-3, morning supplements",
        "expected_keywords": ["vitamin", "omega", "supplement", "morning"],
    },
    {
        "prompt": "What do I usually do before going to bed at night?",
        "category": "Reading",
        "description": "reading before bed, Kindle, sleep habit",
        "expected_keywords": ["read", "kindle", "bed", "sleep", "book"],
    },
    {
        "prompt": "What device and app preferences do I have?",
        "category": "Technology",
        "description": "dark mode, cloud backup, social media avoidance",
        "expected_keywords": ["dark", "cloud", "backup", "social media"],
    },
    {
        "prompt": "How should I travel for a short trip between two nearby cities?",
        "category": "Travel",
        "description": "trains over buses preference",
        "expected_keywords": ["train", "bus"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — 25 PROMPT-WITH-HISTORY RETRIEVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
HISTORY_TESTS = [
    # ── Food & Diet (6) ──────────────────────────────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I want to plan my meals for tomorrow"},
            {"role": "assistant", "text": "Sure! Let me help you plan. What time do you usually have breakfast?"},
        ],
        "prompt": "I usually eat around 6am before my morning walk",
        "category": "Food & Diet",
        "description": "meal planning → breakfast foods + exercise habits",
        "expected_keywords": ["jog", "egg", "avocado", "breakfast", "morning", "walk"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I'm going to a sushi restaurant tonight"},
            {"role": "assistant", "text": "That sounds great! Do you have any dietary restrictions I should know about?"},
        ],
        "prompt": "yes I have some food allergies you should know about",
        "category": "Food & Diet",
        "description": "sushi restaurant → peanut allergy + spicy avoidance + sushi love",
        "expected_keywords": ["peanut", "spicy", "sushi", "chili", "allerg"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I'm cooking dinner for friends tonight"},
            {"role": "assistant", "text": "Nice! What kind of cuisine are you thinking?"},
        ],
        "prompt": "What ingredients must I absolutely keep out of the food?",
        "category": "Food & Diet",
        "description": "cooking → allergy + food avoidances",
        "expected_keywords": ["peanut", "chili", "spicy", "processed", "sugar"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I need some snack ideas for after my evening workout"},
            {"role": "assistant", "text": "A post-workout snack should be a good mix of protein and carbs."},
        ],
        "prompt": "I follow a specific diet, suggest something accordingly",
        "category": "Food & Diet",
        "description": "post-workout snack → high-protein diet + food constraints",
        "expected_keywords": ["protein", "carb", "sugar", "bread", "diet"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I want a healthy afternoon snack"},
            {"role": "assistant", "text": "I can suggest some options. Do you have preferences?"},
        ],
        "prompt": "something that fits my usual food preferences",
        "category": "Food & Diet",
        "description": "snack → organic preference + dietary constraints",
        "expected_keywords": ["organic", "processed", "sugar", "chocolate"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I'm thinking about changing my breakfast"},
            {"role": "assistant", "text": "What do you currently eat for breakfast?"},
        ],
        "prompt": "the usual stuff, but I want more variety",
        "category": "Food & Diet",
        "description": "breakfast change → egg + avocado + dietary constraints",
        "expected_keywords": ["egg", "avocado", "breakfast", "morning", "protein"],
    },

    # ── Exercise & Sleep (4) ─────────────────────────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I've been having trouble sleeping lately"},
            {"role": "assistant", "text": "I'm sorry to hear that. Let's look at factors affecting your sleep."},
        ],
        "prompt": "could it be related to what I drink in the afternoon?",
        "category": "Exercise & Sleep",
        "description": "sleep trouble → coffee constraint + sleep schedule",
        "expected_keywords": ["coffee", "afternoon", "sleep"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I'm training for a 10km race"},
            {"role": "assistant", "text": "Great goal! What's your current exercise routine?"},
        ],
        "prompt": "I already have a regular running habit",
        "category": "Exercise & Sleep",
        "description": "race training → morning jog habit + workout preferences",
        "expected_keywords": ["jog", "morning", "5km", "workout"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I feel really tired in the evenings"},
            {"role": "assistant", "text": "That could be related to your daily routine. What does your day look like?"},
        ],
        "prompt": "what habits might be affecting my energy levels?",
        "category": "Exercise & Sleep",
        "description": "tiredness → sleep, exercise, coffee, routine",
        "expected_keywords": ["sleep", "coffee", "workout", "evening", "wake", "jog"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I need to plan my schedule for tomorrow"},
            {"role": "assistant", "text": "Sure! What time do you usually start your day?"},
        ],
        "prompt": "pretty early, I'm a morning person",
        "category": "Exercise & Sleep",
        "description": "schedule planning → wake time + morning routine + jog",
        "expected_keywords": ["wake", "morning", "jog", "sleep", "egg"],
    },

    # ── Work & Productivity (4) ──────────────────────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I just started a new remote job"},
            {"role": "assistant", "text": "Congratulations! How are you setting up your work routine?"},
        ],
        "prompt": "I already work from home on some days",
        "category": "Work & Productivity",
        "description": "remote work → WFH schedule + meetings + focus time",
        "expected_keywords": ["home", "work", "meeting", "focus", "monday", "friday"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I'm struggling to focus during the day"},
            {"role": "assistant", "text": "Let's figure out when you're most productive."},
        ],
        "prompt": "I think I have better focus at certain times",
        "category": "Work & Productivity",
        "description": "focus → afternoon deep work, morning meetings",
        "expected_keywords": ["focus", "afternoon", "meeting", "morning"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I need to improve my work-life balance"},
            {"role": "assistant", "text": "What does your typical day look like after work?"},
        ],
        "prompt": "I have certain routines to unwind in the evening",
        "category": "Work & Productivity",
        "description": "work-life → evening Netflix, reading, sleep routine",
        "expected_keywords": ["netflix", "evening", "read", "bed", "sleep"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I have back-to-back meetings today"},
            {"role": "assistant", "text": "That's intense. When do you usually schedule your meetings?"},
        ],
        "prompt": "I prefer them at a certain time of day",
        "category": "Work & Productivity",
        "description": "meeting scheduling → morning meeting preference",
        "expected_keywords": ["meeting", "morning"],
    },

    # ── Entertainment & Music (3) ────────────────────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I'm bored at home tonight"},
            {"role": "assistant", "text": "Would you like some entertainment suggestions?"},
        ],
        "prompt": "yes, recommend something I'd actually enjoy watching",
        "category": "Entertainment",
        "description": "bored → Netflix, documentaries, thrillers",
        "expected_keywords": ["netflix", "document", "thriller", "reality"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I need some background music for studying"},
            {"role": "assistant", "text": "What kind of music helps you concentrate?"},
        ],
        "prompt": "I have a go-to genre for focus and concentration",
        "category": "Music",
        "description": "study music → lo-fi for working",
        "expected_keywords": ["lo-fi", "lofi", "music", "concentrate", "work"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "My friend wants to play heavy metal at the party"},
            {"role": "assistant", "text": "Not everyone enjoys the same music. What are your music preferences?"},
        ],
        "prompt": "I have strong opinions about certain music genres",
        "category": "Music",
        "description": "music preference → jazz, lo-fi, heavy metal avoidance",
        "expected_keywords": ["jazz", "metal", "lo-fi", "lofi"],
    },

    # ── Travel & Social (3) ──────────────────────────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I'm planning a short weekend trip"},
            {"role": "assistant", "text": "Sounds fun! How are you planning to get there?"},
        ],
        "prompt": "I have preferences for how I travel short distances",
        "category": "Travel",
        "description": "short trip → train over bus preference",
        "expected_keywords": ["train", "bus"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "There's a big company party this weekend"},
            {"role": "assistant", "text": "Are you planning to go?"},
        ],
        "prompt": "I'm not sure, I have mixed feelings about big social events",
        "category": "Social",
        "description": "party → avoids crowds, prefers small groups, Saturday evenings",
        "expected_keywords": ["crowd", "large", "avoid", "party", "friend"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "My friend keeps calling me but I prefer other ways"},
            {"role": "assistant", "text": "Different people have different communication preferences."},
        ],
        "prompt": "I definitely prefer a different way to stay in touch",
        "category": "Social",
        "description": "communication → texting over calls",
        "expected_keywords": ["text", "phone", "call"],
    },

    # ── Shopping, Health, Reading, Pets, Tech (5) ────────────────────────────
    {
        "recent_history": [
            {"role": "user", "text": "I have a doctor's appointment tomorrow"},
            {"role": "assistant", "text": "Is there anything health-related I can help you prepare for?"},
        ],
        "prompt": "What does my daily health and wellness routine look like?",
        "category": "Health",
        "description": "health checkup → vitamins, water, painkillers avoidance",
        "expected_keywords": ["vitamin", "water", "supplement", "painkiller"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I can't fall asleep tonight"},
            {"role": "assistant", "text": "Let's think about your usual bedtime routine."},
        ],
        "prompt": "I usually do something relaxing before bed",
        "category": "Reading",
        "description": "bedtime → reading, Kindle, sleep schedule",
        "expected_keywords": ["read", "kindle", "bed", "book", "sleep"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "My dog seems really energetic today"},
            {"role": "assistant", "text": "Maybe he needs some extra exercise. What's your usual routine with him?"},
        ],
        "prompt": "I walk him regularly but maybe I need to adjust the schedule",
        "category": "Pets",
        "description": "dog care → walk twice daily, organic food, vet checkups",
        "expected_keywords": ["walk", "dog", "morning", "evening", "vet"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I want to do some online shopping this weekend"},
            {"role": "assistant", "text": "What are you looking to buy?"},
        ],
        "prompt": "electronics mostly, I have a specific way I shop",
        "category": "Shopping",
        "description": "shopping → compare prices, online preference, sustainable",
        "expected_keywords": ["compare", "price", "online", "sustainable", "eco"],
    },
    {
        "recent_history": [
            {"role": "user", "text": "I just got a new laptop"},
            {"role": "assistant", "text": "Nice! Are you setting it up now?"},
        ],
        "prompt": "yes, I have specific preferences for how I set up my devices",
        "category": "Technology",
        "description": "device setup → dark mode, cloud backup, social media rules",
        "expected_keywords": ["dark", "cloud", "backup", "social media"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_next_test_number():
    """Scan results directory for existing test files and return next number."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = os.listdir(RESULTS_DIR)
    numbers = []
    for f in existing:
        match = re.search(r'test(\d+)', f)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def check_keywords(behaviors: list, expected_keywords: list) -> list:
    """Return which expected keywords appear in the behavior texts.
    Works for both related_behaviors and associated_behaviors."""
    all_text = " ".join(b.get("behavior_text", "").lower() for b in behaviors)
    return [kw for kw in expected_keywords if kw.lower() in all_text]


def print_behaviors(behaviors: list, label: str = "related", indent: str = "    "):
    """Log each returned behavior. Handles both related (distance) and associated (edge_weight)."""
    for b in behaviors:
        source = b.get("source", "embedding")
        if source == "graph":
            log(
                f"{indent}• [{b.get('intent','?')}] {b['behavior_text'][:80]}  "
                f"(edge_weight={b.get('edge_weight', 0):.2f}, "
                f"edge_type={b.get('edge_type','?')}, "
                f"credibility={b.get('credibility', 0):.4f}) [{label}/graph]"
            )
        else:
            dist_val = b.get('distance', 0)
            log(
                f"{indent}• [{b.get('intent','?')}] {b['behavior_text'][:80]}  "
                f"(distance={dist_val:.4f}, "
                f"credibility={b.get('credibility', 0):.4f}) [{label}]"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: SEED
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1_seed():
    section("PHASE 1 — SEED BEHAVIORS (v1 /extract, synchronous storage)")

    total_stored = 0
    for i, prompt in enumerate(SEED_PROMPTS, 1):
        log(f"\n▶ Seed prompt {i}/{len(SEED_PROMPTS)}")
        log(f"  Prompt: \"{prompt[:95]}...\"")

        try:
            result = call_v1_extract(prompt)
            if result.get("success"):
                storage = result.get("data", {}).get("storage", {})
                stored = storage.get("stored_behaviors", [])
                count = storage.get("total_behaviors_stored", len(stored))
                total_stored += count
                log(f"  Stored {count} behavior(s):")
                for b in stored:
                    c = b.get("canonical", {})
                    log(
                        f"    • [{c.get('intent','?')}] {b['behavior_text'][:75]}  "
                        f"(credibility={b['credibility']:.4f}, target={c.get('target','?')})"
                    )
                ok(f"{count} behavior(s) stored")
            else:
                fail(f"Extraction failed: {result.get('error')}")
        except Exception as e:
            fail(f"Exception: {e}")

        time.sleep(1)

    log(f"\n  Total behaviors seeded: {total_stored}")
    log(f"  Waiting 3 seconds for tsvector indexing to settle...")
    time.sleep(3)
    return total_stored


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC RETRIEVAL TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_test(test: dict, test_num: int, total: int, use_history: bool = False) -> dict:
    """
    Run a single retrieval test and return a result dict.

    Validates both embedding-based related_behaviors AND graph-based
    associated_behaviors.  Keywords are matched against the UNION of
    both lists so that graph-surfaced behaviors can contribute to the
    pass criteria.
    """
    prompt = test["prompt"]
    category = test.get("category", "General")
    expected_kw = test.get("expected_keywords", [])
    history = test.get("recent_history") if use_history else None

    log(f"\n▶ Test {test_num}/{total} [{category}]")
    if use_history and history:
        log(f"  History    : {json.dumps(history, indent=None)[:120]}...")
    log(f"  Prompt     : \"{prompt}\"")
    log(f"  Expecting  : {test.get('description', 'N/A')}")

    try:
        result = call_v2_extract(prompt, history)
        if not result.get("success"):
            fail(f"API returned error: {result.get('error')}")
            return {
                "test_num": test_num, "prompt": prompt, "category": category,
                "retrieved": 0, "associated": 0,
                "matched_kw": 0, "matched_kw_embedding": 0,
                "matched_kw_graph": 0, "total_kw": len(expected_kw),
                "passed": False, "error": result.get("error"),
                "standalone_query": "", "required_intents": [],
                "matched_keywords": [], "graph_keywords": [],
            }

        data = result.get("data", {})
        related = data.get("related_behaviors", [])
        associated = data.get("associated_behaviors", [])
        standalone = data.get("standalone_query", "")
        intents = data.get("required_intents", [])

        log(f"  Standalone query      : {standalone}")
        log(f"  Required intents      : {intents}")
        log(f"  Related behaviors     : {len(related)}  (embedding/LRA)")
        print_behaviors(related, label="embedding")

        if associated:
            log(f"  Associated behaviors  : {len(associated)}  (graph expansion)")
            print_behaviors(associated, label="graph")
        else:
            log(f"  Associated behaviors  : 0  (no graph neighbors)")

        # Keyword matching — embedding results
        matched_emb = check_keywords(related, expected_kw)
        # Keyword matching — graph results (new keywords only)
        matched_graph = check_keywords(associated, expected_kw)
        # UNION: unique keywords matched by either source
        matched_all = sorted(set(matched_emb + matched_graph))

        # Pass: at least 1 behavior (from either source) + at least 1 keyword
        total_behaviors = len(related) + len(associated)
        passed = total_behaviors >= 1 and len(matched_all) >= 1

        # Report
        graph_only_kw = sorted(set(matched_graph) - set(matched_emb))
        if passed:
            msg = (
                f"PASS — {len(related)} related + {len(associated)} associated, "
                f"keywords: {len(matched_all)}/{len(expected_kw)} {matched_all}"
            )
            if graph_only_kw:
                msg += f"  [GRAPH added: {graph_only_kw}]"
            ok(msg)
        else:
            reason = "(no behaviors returned)" if total_behaviors == 0 else "(no keyword match)"
            fail(
                f"FAIL — {len(related)} related + {len(associated)} associated, "
                f"keywords: {len(matched_all)}/{len(expected_kw)} {reason}"
            )

        return {
            "test_num": test_num, "prompt": prompt, "category": category,
            "retrieved": len(related), "associated": len(associated),
            "matched_kw": len(matched_all), "matched_kw_embedding": len(matched_emb),
            "matched_kw_graph": len(matched_graph), "total_kw": len(expected_kw),
            "passed": passed, "error": None,
            "standalone_query": standalone, "required_intents": intents,
            "matched_keywords": matched_all, "graph_keywords": graph_only_kw,
        }

    except Exception as e:
        fail(f"Exception: {e}")
        return {
            "test_num": test_num, "prompt": prompt, "category": category,
            "retrieved": 0, "associated": 0,
            "matched_kw": 0, "matched_kw_embedding": 0,
            "matched_kw_graph": 0, "total_kw": len(expected_kw),
            "passed": False, "error": str(e),
            "standalone_query": "", "required_intents": [],
            "matched_keywords": [], "graph_keywords": [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 & 3 RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_prompt_only():
    section("PHASE 2 — PROMPT-ONLY RETRIEVAL (25 tests)")
    results = []
    for i, test in enumerate(PROMPT_ONLY_TESTS, 1):
        r = run_single_test(test, test_num=i, total=len(PROMPT_ONLY_TESTS), use_history=False)
        results.append(r)
        time.sleep(1)
    return results


def phase_3_with_history():
    section("PHASE 3 — RETRIEVAL WITH HISTORY (25 tests)")
    results = []
    for i, test in enumerate(HISTORY_TESTS, 1):
        r = run_single_test(test, test_num=i, total=len(HISTORY_TESTS), use_history=True)
        results.append(r)
        time.sleep(1)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY & SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(total_seeded, p2_results, p3_results):
    section("BENCHMARK RESULTS")

    all_results = p2_results + p3_results

    # ── Phase 2 stats ────────────────────────────────────────────────────
    p2_pass = sum(1 for r in p2_results if r["passed"])
    p2_total = len(p2_results)
    p2_avg_ret = sum(r["retrieved"] for r in p2_results) / max(p2_total, 1)
    p2_avg_assoc = sum(r.get("associated", 0) for r in p2_results) / max(p2_total, 1)
    p2_avg_kw = sum(r["matched_kw"] / max(r["total_kw"], 1) for r in p2_results) / max(p2_total, 1) * 100

    # ── Phase 3 stats ────────────────────────────────────────────────────
    p3_pass = sum(1 for r in p3_results if r["passed"])
    p3_total = len(p3_results)
    p3_avg_ret = sum(r["retrieved"] for r in p3_results) / max(p3_total, 1)
    p3_avg_assoc = sum(r.get("associated", 0) for r in p3_results) / max(p3_total, 1)
    p3_avg_kw = sum(r["matched_kw"] / max(r["total_kw"], 1) for r in p3_results) / max(p3_total, 1) * 100

    # ── Overall ──────────────────────────────────────────────────────────
    total_pass = p2_pass + p3_pass
    total_tests = p2_total + p3_total
    overall_pct = total_pass / max(total_tests, 1) * 100

    log(f"  User ID          : {USER_ID}")
    log(f"  Session ID       : {SESSION_ID}")
    log(f"  Behaviors seeded : {total_seeded}")
    log()

    # ── Graph Impact ─────────────────────────────────────────────────────
    total_assoc = sum(r.get("associated", 0) for r in all_results)
    tests_with_assoc = sum(1 for r in all_results if r.get("associated", 0) > 0)
    graph_only_kw_count = sum(len(r.get("graph_keywords", [])) for r in all_results)
    # Tests that ONLY passed because graph-surfaced behaviors matched keywords
    graph_rescued = sum(
        1 for r in all_results
        if r["passed"] and r.get("matched_kw_embedding", 0) == 0 and r.get("matched_kw_graph", 0) > 0
    )

    log(f"  ── Graph Expansion Stats ──")
    log(f"    Tests with graph neighbors  : {tests_with_assoc}/{total_tests}")
    log(f"    Total associated behaviors  : {total_assoc}")
    log(f"    Unique graph-only keywords  : {graph_only_kw_count}  (matched by graph but not embedding)")
    log(f"    Tests rescued by graph      : {graph_rescued}  (would have failed without graph)")
    log()

    # ── Phase 2 Detail ───────────────────────────────────────────────────
    log(f"  Phase 2 — Prompt Only ({p2_total} tests):")
    log(f"    Passed:                  {p2_pass}/{p2_total} ({p2_pass/max(p2_total,1)*100:.1f}%)")
    log(f"    Failed:                  {p2_total - p2_pass}/{p2_total}")
    log(f"    Avg related/query:       {p2_avg_ret:.1f}")
    log(f"    Avg associated/query:    {p2_avg_assoc:.1f}")
    log(f"    Avg keyword coverage:    {p2_avg_kw:.1f}%")
    log()

    for r in p2_results:
        icon = "✓" if r["passed"] else "✗"
        g = f" +{r.get('associated',0)}g" if r.get("associated", 0) > 0 else ""
        log(f"    {icon} [{r['retrieved']:>2} ret{g}] (kw: {r['matched_kw']}/{r['total_kw']}) {r['prompt'][:55]}")
    log(f"    → {sum(r['retrieved'] for r in p2_results)} related + {sum(r.get('associated',0) for r in p2_results)} associated")
    log()

    # ── Phase 3 Detail ───────────────────────────────────────────────────
    log(f"  Phase 3 — With History ({p3_total} tests):")
    log(f"    Passed:                  {p3_pass}/{p3_total} ({p3_pass/max(p3_total,1)*100:.1f}%)")
    log(f"    Failed:                  {p3_total - p3_pass}/{p3_total}")
    log(f"    Avg related/query:       {p3_avg_ret:.1f}")
    log(f"    Avg associated/query:    {p3_avg_assoc:.1f}")
    log(f"    Avg keyword coverage:    {p3_avg_kw:.1f}%")
    log()

    for r in p3_results:
        icon = "✓" if r["passed"] else "✗"
        g = f" +{r.get('associated',0)}g" if r.get("associated", 0) > 0 else ""
        log(f"    {icon} [{r['retrieved']:>2} ret{g}] (kw: {r['matched_kw']}/{r['total_kw']}) {r['prompt'][:55]}")
    log(f"    → {sum(r['retrieved'] for r in p3_results)} related + {sum(r.get('associated',0) for r in p3_results)} associated")
    log()

    # ── Category Breakdown ───────────────────────────────────────────────
    log(f"  Category Breakdown:")
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "total": 0, "assoc": 0}
        categories[cat]["total"] += 1
        categories[cat]["assoc"] += r.get("associated", 0)
        if r["passed"]:
            categories[cat]["pass"] += 1

    for cat, stats in sorted(categories.items()):
        pct = stats["pass"] / max(stats["total"], 1) * 100
        log(f"    {cat:<25} {stats['pass']}/{stats['total']} ({pct:.0f}%)  graph_assoc={stats['assoc']}")
    log()

    # ── Failed Tests Detail ──────────────────────────────────────────────
    failed = [r for r in all_results if not r["passed"]]
    if failed:
        log(f"  Failed Tests:")
        for r in failed:
            phase = "P2" if r in p2_results else "P3"
            reason = r.get("error") or (
                "no behaviors" if (r["retrieved"] + r.get("associated", 0)) == 0
                else "no keyword match"
            )
            log(f"    ✗ [{phase}] Test {r['test_num']}: \"{r['prompt'][:50]}...\" ({reason})")
        log()

    # ── Overall ──────────────────────────────────────────────────────────
    log(f"  {'='*60}")
    if overall_pct >= 90:
        log_c(f"  OVERALL: {total_pass}/{total_tests} PASSED ({overall_pct:.1f}%)", GREEN)
    elif overall_pct >= 70:
        log_c(f"  OVERALL: {total_pass}/{total_tests} PASSED ({overall_pct:.1f}%)", YELLOW)
    else:
        log_c(f"  OVERALL: {total_pass}/{total_tests} PASSED ({overall_pct:.1f}%)", RED)
    log(f"  {'='*60}")
    log()

    return {
        "total_pass": total_pass,
        "total_tests": total_tests,
        "overall_pct": overall_pct,
        "p2_pass": p2_pass,
        "p3_pass": p3_pass,
        "graph_rescued": graph_rescued,
        "total_associated": total_assoc,
    }


def save_results():
    """Save buffered output to results file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_num = get_next_test_number()
    now = datetime.now()
    filename = f"{now.strftime('%Y_%m_%d_%H_%M')}_test{test_num}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))

    print(f"\n{GREEN}Results saved to: {filepath}{RESET}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    section("3D HYBRID RETRIEVAL + CO-OCCURRENCE GRAPH — E2E BENCHMARK")
    log(f"  User ID    : {USER_ID}  (unique per run)")
    log(f"  Session ID : {SESSION_ID}")
    log(f"  Server     : {BASE_URL}")
    log(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Test Count : {len(PROMPT_ONLY_TESTS) + len(HISTORY_TESTS)} "
        f"({len(PROMPT_ONLY_TESTS)} prompt-only + {len(HISTORY_TESTS)} with-history)")
    log(f"  Graph      : CO_PROMPT edges created per seed prompt")

    # Verify server
    try:
        requests.get(f"{BASE_URL}/docs", timeout=5)
        ok("Server is reachable")
    except requests.ConnectionError:
        fail(f"Cannot reach server at {BASE_URL}. Is it running?")
        sys.exit(1)

    # Phase 1: Seed
    total_seeded = phase_1_seed()
    if total_seeded == 0:
        fail("No behaviors seeded. Aborting.")
        sys.exit(1)

    # Phase 2: Prompt-only
    p2_results = phase_2_prompt_only()

    # Phase 3: With history
    p3_results = phase_3_with_history()

    # Summary
    elapsed = time.time() - start_time
    log(f"\n  Total execution time: {elapsed:.1f}s")
    summary = print_summary(total_seeded, p2_results, p3_results)

    # Save
    save_results()

    # Exit code
    sys.exit(0 if summary["overall_pct"] >= 70 else 1)


if __name__ == "__main__":
    main()
