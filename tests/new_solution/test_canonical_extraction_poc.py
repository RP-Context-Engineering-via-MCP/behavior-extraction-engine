"""
Proof of Concept Test: Canonical Behavior Extraction with GPT-4

Purpose:
    Validate that GPT-4 can reliably extract structured canonical behavior fields:
    - intent (PREFERENCE, CONSTRAINT, HABIT, SKILL, COMMUNICATION)
    - target (primary object of behavior)
    - context (optional scope: IDE, frontend, night, etc.)
    - polarity (POSITIVE, NEGATIVE)
    - strength (0.0-1.0, from linguistic_strength)

Test Strategy:
    1. Define 15 diverse test cases covering all intent types
    2. Extract behaviors using NEW canonical-aware prompt
    3. Validate field accuracy per test case
    4. Calculate overall extraction accuracy (target: 90%+)
    5. Identify edge cases or prompt improvements needed

Success Criteria:
    - Intent accuracy: 95%+
    - Target accuracy: 90%+
    - Context accuracy: 85%+ (when applicable)
    - Polarity accuracy: 95%+
    - Overall: 90%+ across all fields
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AzureOpenAI
from config.configurations import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT_MODEL
)
from models.behavior import CanonicalBehavior
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    timeout=30
)

# New canonical-aware system prompt (from implementation plan)
CANONICAL_EXTRACTION_PROMPT = """You are a behavior canonicalization engine.

Your task is to extract ONLY long-term, reusable user behaviors and represent them in a normalized, machine-reasonable form.

A behavior MUST be stable across time. Do NOT extract temporary states or one-time requests.

---

FOR EACH BEHAVIOR, YOU MUST PRODUCE A CANONICAL FORM WITH THESE FIELDS:

1. intent:
   - PREFERENCE → likes, prefers, enjoys, favors
   - CONSTRAINT → cannot, avoids, allergic to, restricted
   - HABIT → usually, always, regularly
   - SKILL → experienced with, proficient in
   - COMMUNICATION → prefers brief answers, wants examples

2. target:
   - The primary object of the behavior (e.g., dark mode, Python, late-night work)
   - Must be concise and noun-like

3. context:
   - Optional scope where the behavior applies
   - Examples: IDE, frontend, backend, night, work, general
   - If not specified, use "general"

4. polarity:
   - POSITIVE → likes / prefers / wants
   - NEGATIVE → dislikes / avoids / cannot

5. strength (0.0–1.0):
   - Use linguistic intensity
   - Strong words → 0.8–1.0
   - Normal preference → 0.6–0.8
   - Mild → 0.4–0.6
   - Weak/uncertain → <0.4

---

OUTPUT FORMAT (STRICT JSON ONLY):

{
  "segments": [
    {
      "text": "original segment text",
      "behaviors": [
        {
          "description": "short human-readable summary",
          "intent": "PREFERENCE",
          "target": "dark mode",
          "context": "IDE",
          "polarity": "POSITIVE",
          "confidence": 0.92,
          "clarity": 0.88,
          "linguistic_strength": 0.75
        }
      ]
    }
  ]
}

RULES:
- If no stable behavior exists, return an empty behaviors list
- Do NOT invent context
- Do NOT include explanations or extra fields
- Always return valid JSON
"""

# Test cases covering all intent types and edge cases
TEST_CASES = [
    # PREFERENCE tests
    {
        "id": 1,
        "prompt": "I prefer dark mode for my IDE",
        "expected": {
            "intent": "PREFERENCE",
            "target": "dark mode",
            "context": "IDE",
            "polarity": "POSITIVE",
            "strength_range": (0.6, 0.8)  # Normal preference
        }
    },
    {
        "id": 2,
        "prompt": "I strongly prefer Python for backend development",
        "expected": {
            "intent": "PREFERENCE",
            "target": "Python",
            "context": "backend",
            "polarity": "POSITIVE",
            "strength_range": (0.8, 1.0)  # Strong language
        }
    },
    {
        "id": 3,
        "prompt": "I like JavaScript",
        "expected": {
            "intent": "PREFERENCE",
            "target": "JavaScript",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.6, 0.8)
        }
    },
    {
        "id": 4,
        "prompt": "I don't like using light themes",
        "expected": {
            "intent": "PREFERENCE",
            "target": "light themes",
            "context": "general",
            "polarity": "NEGATIVE",
            "strength_range": (0.6, 0.8)
        }
    },
    
    # CONSTRAINT tests
    {
        "id": 5,
        "prompt": "I cannot work with Java due to company policy",
        "expected": {
            "intent": "CONSTRAINT",
            "target": "Java",
            "context": "work",
            "polarity": "NEGATIVE",
            "strength_range": (0.8, 1.0)  # Strong constraint
        }
    },
    {
        "id": 6,
        "prompt": "I avoid writing CSS directly",
        "expected": {
            "intent": "CONSTRAINT",
            "target": "CSS",
            "context": "general",
            "polarity": "NEGATIVE",
            "strength_range": (0.6, 0.8)
        }
    },
    
    # HABIT tests
    {
        "id": 7,
        "prompt": "I always write unit tests for new features",
        "expected": {
            "intent": "HABIT",
            "target": "unit tests",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.8, 1.0)  # "always" = strong
        }
    },
    {
        "id": 8,
        "prompt": "I usually work late at night",
        "expected": {
            "intent": "HABIT",
            "target": "late night work",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.6, 0.8)  # "usually" = normal
        }
    },
    
    # SKILL tests
    {
        "id": 9,
        "prompt": "I'm experienced with React and Redux",
        "expected": {
            "intent": "SKILL",
            "target": "React and Redux",
            "context": "frontend",
            "polarity": "POSITIVE",
            "strength_range": (0.6, 0.8)
        }
    },
    {
        "id": 10,
        "prompt": "I'm proficient in Docker containerization",
        "expected": {
            "intent": "SKILL",
            "target": "Docker",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.7, 0.9)
        }
    },
    
    # COMMUNICATION tests
    {
        "id": 11,
        "prompt": "I prefer brief, concise answers without long explanations",
        "expected": {
            "intent": "COMMUNICATION",
            "target": "brief answers",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.6, 0.8)
        }
    },
    {
        "id": 12,
        "prompt": "I want code examples in all responses",
        "expected": {
            "intent": "COMMUNICATION",
            "target": "code examples",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.7, 0.9)
        }
    },
    
    # Edge cases
    {
        "id": 13,
        "prompt": "I might prefer TypeScript over JavaScript sometimes",
        "expected": {
            "intent": "PREFERENCE",
            "target": "TypeScript",
            "context": "general",
            "polarity": "POSITIVE",
            "strength_range": (0.3, 0.5)  # Very weak/uncertain
        }
    },
    {
        "id": 14,
        "prompt": "I prefer dark mode in general but light mode for reading documentation",
        "expected_count": 2,  # Should extract TWO behaviors
        "expected": [
            {
                "intent": "PREFERENCE",
                "target": "dark mode",
                "context": "general",
                "polarity": "POSITIVE"
            },
            {
                "intent": "PREFERENCE",
                "target": "light mode",
                "context": "documentation",
                "polarity": "POSITIVE"
            }
        ]
    },
    {
        "id": 15,
        "prompt": "What is Python?",  # Not a behavior - should return empty
        "expected": {
            "should_be_empty": True
        }
    }
]


def extract_canonical_behavior(prompt: str) -> Dict:
    """
    Call GPT-4 with the new canonical extraction prompt.
    
    Returns:
        Parsed JSON response with canonical fields
    """
    try:
        start_time = time()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": CANONICAL_EXTRACTION_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        extraction_time_ms = int((time() - start_time) * 1000)
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        logger.info(f"Extraction completed in {extraction_time_ms}ms")
        return parsed
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"error": str(e)}


def validate_field(field_name: str, actual: any, expected: any, test_id: int) -> bool:
    """Validate a single field matches expectations."""
    if actual == expected:
        return True
    else:
        logger.warning(f"Test {test_id} - {field_name} mismatch: expected '{expected}', got '{actual}'")
        return False


def validate_strength_range(actual: float, expected_range: tuple, test_id: int) -> bool:
    """Validate strength falls within expected range."""
    min_val, max_val = expected_range
    if min_val <= actual <= max_val:
        return True
    else:
        logger.warning(f"Test {test_id} - strength out of range: expected {expected_range}, got {actual:.2f}")
        return False


def run_poc_tests():
    """Run all POC test cases and calculate accuracy."""
    print("\n" + "="*80)
    print("  PROOF OF CONCEPT: Canonical Behavior Extraction Test")
    print("="*80 + "\n")
    
    results = {
        "test_run_timestamp": datetime.now().isoformat(),
        "total_tests": len(TEST_CASES),
        "test_results": [],
        "accuracy": {
            "intent": {"correct": 0, "total": 0},
            "target": {"correct": 0, "total": 0},
            "context": {"correct": 0, "total": 0},
            "polarity": {"correct": 0, "total": 0},
            "strength": {"correct": 0, "total": 0}
        }
    }
    
    for test_case in TEST_CASES:
        test_id = test_case["id"]
        prompt = test_case["prompt"]
        expected = test_case["expected"]
        
        print(f"\n--- Test {test_id} ---")
        print(f"Prompt: '{prompt}'")
        
        # Extract canonical behavior
        extracted = extract_canonical_behavior(prompt)
        
        if "error" in extracted:
            print(f"❌ EXTRACTION FAILED: {extracted['error']}")
            results["test_results"].append({
                "test_id": test_id,
                "prompt": prompt,
                "status": "FAILED",
                "error": extracted['error']
            })
            continue
        
        # Handle empty behavior case (Test 15)
        if isinstance(expected, dict) and expected.get("should_be_empty"):
            behaviors = extracted.get("segments", [{}])[0].get("behaviors", [])
            if len(behaviors) == 0:
                print("✅ Correctly returned empty behaviors list")
                results["test_results"].append({
                    "test_id": test_id,
                    "prompt": prompt,
                    "status": "PASSED",
                    "note": "Empty list validation"
                })
            else:
                print(f"❌ Expected empty, got {len(behaviors)} behaviors")
                results["test_results"].append({
                    "test_id": test_id,
                    "prompt": prompt,
                    "status": "FAILED",
                    "note": f"Expected empty, got {len(behaviors)} behaviors"
                })
            continue
        
        # Handle multiple behaviors case (Test 14)
        if "expected_count" in test_case:
            behaviors = extracted.get("segments", [{}])[0].get("behaviors", [])
            if len(behaviors) == test_case["expected_count"]:
                print(f"✅ Correctly extracted {len(behaviors)} behaviors")
                results["test_results"].append({
                    "test_id": test_id,
                    "prompt": prompt,
                    "status": "PASSED",
                    "extracted_count": len(behaviors),
                    "behaviors": behaviors
                })
            else:
                print(f"❌ Expected {test_case['expected_count']}, got {len(behaviors)} behaviors")
                results["test_results"].append({
                    "test_id": test_id,
                    "prompt": prompt,
                    "status": "FAILED",
                    "expected_count": test_case["expected_count"],
                    "actual_count": len(behaviors)
                })
            continue
        
        # Standard single behavior validation
        try:
            behavior = extracted["segments"][0]["behaviors"][0]
            
            # Use first expected if it's a list (Test 14 fallback)
            if isinstance(expected, list):
                expected = expected[0]
            
            test_result = {
                "test_id": test_id,
                "prompt": prompt,
                "extracted": behavior,
                "validations": {}
            }
            
            # Validate each field
            intent_valid = validate_field("intent", behavior.get("intent"), expected["intent"], test_id)
            target_valid = validate_field("target", behavior.get("target"), expected["target"], test_id)
            context_valid = validate_field("context", behavior.get("context"), expected["context"], test_id)
            polarity_valid = validate_field("polarity", behavior.get("polarity"), expected["polarity"], test_id)
            
            # Handle both 'strength' and 'linguistic_strength' field names
            strength_value = behavior.get("linguistic_strength") or behavior.get("strength", 0.0)
            strength_valid = validate_strength_range(
                strength_value,
                expected["strength_range"],
                test_id
            )
            
            # Update accuracy counters
            results["accuracy"]["intent"]["total"] += 1
            results["accuracy"]["intent"]["correct"] += int(intent_valid)
            
            results["accuracy"]["target"]["total"] += 1
            results["accuracy"]["target"]["correct"] += int(target_valid)
            
            results["accuracy"]["context"]["total"] += 1
            results["accuracy"]["context"]["correct"] += int(context_valid)
            
            results["accuracy"]["polarity"]["total"] += 1
            results["accuracy"]["polarity"]["correct"] += int(polarity_valid)
            
            results["accuracy"]["strength"]["total"] += 1
            results["accuracy"]["strength"]["correct"] += int(strength_valid)
            
            # Test status
            all_valid = all([intent_valid, target_valid, context_valid, polarity_valid, strength_valid])
            test_result["status"] = "PASSED" if all_valid else "PARTIAL"
            test_result["validations"] = {
                "intent": intent_valid,
                "target": target_valid,
                "context": context_valid,
                "polarity": polarity_valid,
                "strength": strength_valid
            }
            
            if all_valid:
                print("✅ All fields validated successfully")
            else:
                print("⚠️  Some fields failed validation")
            
            print(f"Extracted: intent={behavior.get('intent')}, target={behavior.get('target')}, "
                  f"context={behavior.get('context')}, polarity={behavior.get('polarity')}, "
                  f"strength={strength_value:.2f}")
            
            results["test_results"].append(test_result)
            
        except (KeyError, IndexError) as e:
            print(f"❌ Failed to parse extracted behavior: {e}")
            results["test_results"].append({
                "test_id": test_id,
                "prompt": prompt,
                "status": "FAILED",
                "error": f"Parse error: {e}",
                "raw_response": extracted
            })
    
    # Calculate final accuracy percentages
    print("\n" + "="*80)
    print("  ACCURACY RESULTS")
    print("="*80 + "\n")
    
    for field, stats in results["accuracy"].items():
        if stats["total"] > 0:
            percentage = (stats["correct"] / stats["total"]) * 100
            status = "✅" if percentage >= 90 else "⚠️" if percentage >= 80 else "❌"
            print(f"{status} {field.upper()}: {stats['correct']}/{stats['total']} ({percentage:.1f}%)")
            results["accuracy"][field]["percentage"] = percentage
    
    # Overall accuracy
    total_validations = sum(s["total"] for s in results["accuracy"].values())
    total_correct = sum(s["correct"] for s in results["accuracy"].values())
    overall_percentage = (total_correct / total_validations) * 100 if total_validations > 0 else 0
    
    print(f"\n{'✅' if overall_percentage >= 90 else '❌'} OVERALL: {total_correct}/{total_validations} ({overall_percentage:.1f}%)")
    results["overall_accuracy"] = overall_percentage
    
    # Save results
    output_file = f"test results/canonical_extraction_poc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("test results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
    
    # Final verdict
    print("\n" + "="*80)
    if overall_percentage >= 90:
        print("  ✅ POC SUCCESSFUL - Ready for implementation")
    elif overall_percentage >= 80:
        print("  ⚠️  POC NEEDS REFINEMENT - Review failed cases and adjust prompt")
    else:
        print("  ❌ POC FAILED - Major prompt redesign needed")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    run_poc_tests()
