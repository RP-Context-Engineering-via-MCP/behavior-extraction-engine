"""
E2E Test: 3D Hybrid Retrieval (TGHR) — Behavior Seeding → Retrieval Validation
================================================================================

This test uses a SINGLE user_id and session_id throughout all steps:

  Phase 1 — SEED:   Send prompts via /extract (v1, synchronous storage) to populate
                     the database with diverse behaviors (food, exercise, sleep, allergy).

  Phase 2 — RETRIEVE: Send related prompts via /v2/extract to verify the 3D hybrid
                       search returns the previously stored behaviors.

  Phase 3 — HISTORY: Send prompts WITH recent_history via /v2/extract, testing that
                      context-dependent prompts still retrieve correct behaviors.

Run with:
    python -m tests.test_3d_retrieval_e2e

Prerequisites:
    - Server running on http://localhost:8000  (python run.py)
    - Database migrated with search_vector column + GIN index
"""

import requests
import time
import json
import sys

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
USER_ID = "test_3d_retrieval_user_002"
SESSION_ID = "test_3d_session_002"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def section(title: str):
    print(f"\n{'='*90}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{'='*90}\n")


def step(label: str):
    print(f"\n{CYAN}▶ {label}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def call_v1_extract(prompt: str) -> dict:
    """POST /extract — synchronous extraction + storage."""
    resp = requests.post(
        f"{BASE_URL}/extract",
        json={
            "prompt": prompt,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_v2_extract(prompt: str, recent_history: list = None) -> dict:
    """POST /v2/extract — extraction + 3D hybrid retrieval."""
    body = {
        "prompt": prompt,
        "user_id": USER_ID,
        "session_id": SESSION_ID,
    }
    if recent_history:
        body["recent_history"] = recent_history
    resp = requests.post(f"{BASE_URL}/v2/extract", json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def print_stored_behaviors(result: dict):
    """Pretty-print behaviors stored by v1 /extract."""
    storage = result.get("data", {}).get("storage", {})
    stored = storage.get("stored_behaviors", [])
    print(f"  Stored {len(stored)} behavior(s):")
    for b in stored:
        canonical = b.get("canonical", {})
        print(
            f"    • [{canonical.get('intent','?')}] {b['behavior_text'][:80]}  "
            f"(credibility={b['credibility']:.4f}, target={canonical.get('target','?')})"
        )


def print_related_behaviors(result: dict):
    """Pretty-print related behaviors returned by v2 /v2/extract."""
    data = result.get("data", {})
    related = data.get("related_behaviors", [])
    standalone = data.get("standalone_query", "")
    intents = data.get("required_intents", [])
    print(f"  Standalone query : {standalone}")
    print(f"  Required intents : {intents}")
    print(f"  Related behaviors: {len(related)}")
    for b in related:
        print(
            f"    • [{b.get('intent','?')}] {b['behavior_text'][:80]}  "
            f"(distance={b['distance']:.4f}, credibility={b['credibility']:.4f})"
        )
    return related


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Seed behaviors via v1 /extract (synchronous)
# ═══════════════════════════════════════════════════════════════════════════════
def phase_1_seed():
    section("PHASE 1 — SEED BEHAVIORS (v1 /extract, synchronous storage)")
    
    seed_prompts = [
        # Prompt 1: Food preferences and constraints
        "I am allergic to peanuts and I love eating dark chocolate after dinner. "
        "I prefer organic vegetables and avoid processed foods.",

        # Prompt 2: Exercise habits
        "I go for a 5km morning jog every day before breakfast. "
        "I avoid heavy workouts in the evening because it affects my sleep.",

        # Prompt 3: Sleep and routine
        "I usually sleep by 10pm and wake up at 5:30am. "
        "I don't drink coffee after 3pm because it keeps me awake.",

        # Prompt 4: Dietary specifics
        "I follow a low-carb, high-protein diet. "
        "I eat eggs and avocado for breakfast every morning. "
        "I try to avoid sugar and white bread.",

        # Prompt 5: More food preferences
        "I love sushi and Japanese food in general. "
        "I dislike spicy food and never eat anything with chili peppers.",
    ]

    total_stored = 0
    for i, prompt in enumerate(seed_prompts, 1):
        step(f"Seed prompt {i}/{len(seed_prompts)}")
        print(f"  Prompt: \"{prompt[:90]}...\"")
        
        result = call_v1_extract(prompt)
        
        if result.get("success"):
            stored_count = result["data"]["storage"]["total_behaviors_stored"]
            total_stored += stored_count
            print_stored_behaviors(result)
            ok(f"{stored_count} behavior(s) stored")
        else:
            fail(f"Extraction failed: {result.get('error')}")

        # Small delay to avoid rate limiting
        time.sleep(1)

    print(f"\n  {BOLD}Total behaviors seeded: {total_stored}{RESET}")
    
    # Give background indexing a moment 
    print(f"\n  Waiting 3 seconds for tsvector indexing to settle...")
    time.sleep(3)
    
    return total_stored


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Retrieval test via v2 /v2/extract (no history)
# ═══════════════════════════════════════════════════════════════════════════════
def phase_2_retrieve():
    section("PHASE 2 — RETRIEVE BEHAVIORS (v2 /v2/extract, no history)")

    retrieval_prompts = [
        {
            "prompt": "What foods should I eat before my morning run?",
            "description": "Should retrieve: exercise habits, food preferences, breakfast foods, peanut allergy",
            "expected_keywords": ["jog", "breakfast", "egg", "avocado", "peanut", "allergic"],
        },
        {
            "prompt": "Is dark chocolate a healthy evening snack?",
            "description": "Should retrieve: dark chocolate preference, dietary constraints",
            "expected_keywords": ["chocolate", "dinner", "sugar", "organic"],
        },
        {
            "prompt": "Can you recommend a high protein lunch?",
            "description": "Should retrieve: low-carb/high-protein diet, food avoidances",
            "expected_keywords": ["protein", "low-carb", "processed", "sugar", "bread"],
        },
        {
            "prompt": "What should I eat at a Japanese restaurant?",
            "description": "Should retrieve: sushi/Japanese food love, spicy food avoidance, peanut allergy",
            "expected_keywords": ["sushi", "japanese", "spicy", "chili", "peanut"],
        },
        {
            "prompt": "What time should I stop drinking coffee?",
            "description": "Should retrieve: coffee/caffeine constraint, sleep schedule",
            "expected_keywords": ["coffee", "3pm", "sleep", "10pm", "awake"],
        },
    ]

    results_summary = []
    for i, test in enumerate(retrieval_prompts, 1):
        step(f"Retrieval prompt {i}/{len(retrieval_prompts)}")
        print(f"  Prompt     : \"{test['prompt']}\"")
        print(f"  Expecting  : {test['description']}")

        result = call_v2_extract(test["prompt"])

        if result.get("success"):
            related = print_related_behaviors(result)
            
            # Check how many expected keywords appear in returned behavior texts
            all_text = " ".join(b["behavior_text"].lower() for b in related)
            matched_kw = [kw for kw in test["expected_keywords"] if kw.lower() in all_text]
            total_kw = len(test["expected_keywords"])
            
            if len(related) > 0:
                ok(f"Retrieved {len(related)} behavior(s), keyword match: {len(matched_kw)}/{total_kw} {matched_kw}")
            else:
                fail(f"Retrieved 0 behaviors! Expected matches for: {test['expected_keywords']}")
            
            results_summary.append({
                "prompt": test["prompt"],
                "retrieved": len(related),
                "keyword_match": f"{len(matched_kw)}/{total_kw}",
            })
        else:
            fail(f"v2/extract failed: {result.get('error')}")
            results_summary.append({
                "prompt": test["prompt"],
                "retrieved": 0,
                "keyword_match": "N/A",
            })

        time.sleep(1)

    return results_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Retrieval with conversation history
# ═══════════════════════════════════════════════════════════════════════════════
def phase_3_history():
    section("PHASE 3 — RETRIEVE WITH HISTORY (v2 /v2/extract + recent_history)")

    history_tests = [
        {
            "recent_history": [
                {"role": "user", "text": "I want to plan my meals for tomorrow"},
                {"role": "assistant", "text": "Sure! Let me help you plan your meals. What time do you usually have breakfast?"},
            ],
            "prompt": "I usually eat around 6am before my walk",
            "description": "History: meal planning. Should retrieve breakfast foods + exercise habits",
        },
        {
            "recent_history": [
                {"role": "user", "text": "I'm going to a sushi restaurant tonight"},
                {"role": "assistant", "text": "That sounds great! Do you have any dietary restrictions I should know about?"},
            ],
            "prompt": "yes I have some food allergies",
            "description": "History: sushi restaurant. Should retrieve peanut allergy + spicy avoidance + sushi preference",
        },
        {
            "recent_history": [
                {"role": "user", "text": "I've been having trouble sleeping lately"},
                {"role": "assistant", "text": "I'm sorry to hear that. Let's look at some factors that might be affecting your sleep."},
            ],
            "prompt": "could it be related to what I drink in the afternoon?",
            "description": "History: sleep trouble. Should retrieve coffee constraint + sleep schedule",
        },
        {
            "recent_history": [
                {"role": "user", "text": "I need some snack ideas for after my evening workout"},
                {"role": "assistant", "text": "A post-workout snack should be a good mix of protein and carbs. What kind of diet do you follow?"},
            ],
            "prompt": "I follow a specific diet, suggest something accordingly",
            "description": "History: post-workout snack. Should retrieve high-protein diet + evening workout avoidance + food preferences",
        },
    ]

    results_summary = []
    for i, test in enumerate(history_tests, 1):
        step(f"History retrieval {i}/{len(history_tests)}")
        print(f"  History    : {json.dumps(test['recent_history'], indent=None)[:120]}...")
        print(f"  Prompt     : \"{test['prompt']}\"")
        print(f"  Expecting  : {test['description']}")

        result = call_v2_extract(test["prompt"], test["recent_history"])

        if result.get("success"):
            related = print_related_behaviors(result)
            
            if len(related) > 0:
                ok(f"Retrieved {len(related)} behavior(s) with history context")
            else:
                fail(f"Retrieved 0 behaviors! History context may not have helped")
            
            results_summary.append({
                "prompt": test["prompt"],
                "history_context": test["recent_history"][0]["text"][:50],
                "retrieved": len(related),
            })
        else:
            fail(f"v2/extract failed: {result.get('error')}")
            results_summary.append({
                "prompt": test["prompt"],
                "history_context": test["recent_history"][0]["text"][:50],
                "retrieved": 0,
            })

        time.sleep(1)

    return results_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    section("3D HYBRID RETRIEVAL (TGHR) — END-TO-END TEST")
    print(f"  User ID   : {USER_ID}")
    print(f"  Session ID: {SESSION_ID}")
    print(f"  Server    : {BASE_URL}")

    # Verify server is up
    try:
        requests.get(f"{BASE_URL}/docs", timeout=5)
        ok("Server is reachable")
    except requests.ConnectionError:
        fail(f"Cannot reach server at {BASE_URL}. Is it running? (python run.py)")
        sys.exit(1)

    # ── Phase 1: Seed ──
    total_seeded = phase_1_seed()
    if total_seeded == 0:
        fail("No behaviors were seeded. Cannot proceed with retrieval tests.")
        sys.exit(1)

    # ── Phase 2: Retrieve (no history) ──
    phase2_results = phase_2_retrieve()

    # ── Phase 3: Retrieve with history ──
    phase3_results = phase_3_history()

    # ── Summary ──
    section("TEST SUMMARY")
    
    print(f"  {BOLD}Phase 1 — Seeding:{RESET}")
    print(f"    Total behaviors seeded: {total_seeded}")

    print(f"\n  {BOLD}Phase 2 — Retrieval (no history):{RESET}")
    p2_total = sum(r["retrieved"] for r in phase2_results)
    p2_nonzero = sum(1 for r in phase2_results if r["retrieved"] > 0)
    for r in phase2_results:
        status_icon = GREEN + "✓" + RESET if r["retrieved"] > 0 else RED + "✗" + RESET
        print(f"    {status_icon} [{r['retrieved']} matched] (kw: {r['keyword_match']}) {r['prompt'][:60]}")
    print(f"    → {p2_nonzero}/{len(phase2_results)} prompts returned behaviors, {p2_total} total matches")

    print(f"\n  {BOLD}Phase 3 — Retrieval (with history):{RESET}")
    p3_total = sum(r["retrieved"] for r in phase3_results)
    p3_nonzero = sum(1 for r in phase3_results if r["retrieved"] > 0)
    for r in phase3_results:
        status_icon = GREEN + "✓" + RESET if r["retrieved"] > 0 else RED + "✗" + RESET
        print(f"    {status_icon} [{r['retrieved']} matched] {r['prompt'][:60]}")
    print(f"    → {p3_nonzero}/{len(phase3_results)} prompts returned behaviors, {p3_total} total matches")

    # Overall pass/fail
    total_tests = len(phase2_results) + len(phase3_results)
    total_pass = p2_nonzero + p3_nonzero
    print(f"\n  {'='*60}")
    if total_pass == total_tests:
        print(f"  {GREEN}{BOLD}ALL {total_tests} RETRIEVAL TESTS PASSED{RESET}")
    elif total_pass > 0:
        print(f"  {YELLOW}{BOLD}{total_pass}/{total_tests} RETRIEVAL TESTS PASSED{RESET}")
    else:
        print(f"  {RED}{BOLD}ALL {total_tests} RETRIEVAL TESTS FAILED{RESET}")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    main()
