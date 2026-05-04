"""
Comprehensive E2E Test - Canonical Behavior System
Tests across multiple domains with various prompt patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import get_user_behaviors, search_similar_behaviors
from services.openAiClient import embed_text
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCase:
    def __init__(self, name: str, prompt: str, expected_behaviors: int, 
                 expected_intents: list, expected_targets: list):
        self.name = name
        self.prompt = prompt
        self.expected_behaviors = expected_behaviors
        self.expected_intents = expected_intents
        self.expected_targets = expected_targets
        self.result = None
        self.passed = False


# ========================================================================
# TEST CASES - MULTI-DOMAIN COMPREHENSIVE COVERAGE
# ========================================================================

TEST_CASES = [
    # ===== PROGRAMMING DOMAIN =====
    TestCase(
        name="PROG-1: Multiple preferences in one prompt",
        prompt="I prefer dark mode for my IDE. I like using Python for backend development. I enjoy writing tests.",
        expected_behaviors=3,
        expected_intents=["PREFERENCE", "PREFERENCE", "PREFERENCE"],
        expected_targets=["dark_mode", "Python", "tests"]
    ),
    
    TestCase(
        name="PROG-2: Constraints and restrictions",
        prompt="I cannot use Java due to company policy. I must avoid jQuery in new projects.",
        expected_behaviors=2,
        expected_intents=["CONSTRAINT", "CONSTRAINT"],
        expected_targets=["Java", "jQuery"]
    ),
    
    TestCase(
        name="PROG-3: Skills and capabilities",
        prompt="I am proficient in React and TypeScript. I have experience with Docker containerization.",
        expected_behaviors=2,
        expected_intents=["SKILL", "SKILL"],
        expected_targets=["React", "Docker"]
    ),
    
    TestCase(
        name="PROG-4: Communication preferences",
        prompt="Please provide code examples when explaining concepts. Keep responses concise and to the point.",
        expected_behaviors=2,
        expected_intents=["COMMUNICATION", "COMMUNICATION"],
        expected_targets=["code_examples", "concise_responses"]
    ),
    
    TestCase(
        name="PROG-5: Context-specific behaviors",
        prompt="I prefer functional programming for frontend. I prefer object-oriented programming for backend.",
        expected_behaviors=2,
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["functional_programming", "object_oriented"]
    ),
    
    # ===== FOOD DOMAIN =====
    TestCase(
        name="FOOD-1: Dietary preferences",
        prompt="I love spicy food. I prefer vegetarian options when available.",
        expected_behaviors=2,
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["spicy_food", "vegetarian"]
    ),
    
    TestCase(
        name="FOOD-2: Allergies and restrictions",
        prompt="I cannot eat dairy products. I must avoid shellfish due to allergies.",
        expected_behaviors=2,
        expected_intents=["CONSTRAINT", "CONSTRAINT"],
        expected_targets=["dairy", "shellfish"]
    ),
    
    TestCase(
        name="FOOD-3: Mixed polarity",
        prompt="I enjoy Indian cuisine but I dislike overly sweet desserts.",
        expected_behaviors=2,
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["Indian_cuisine", "sweet_desserts"]
    ),
    
    # ===== HEALTH & FITNESS =====
    TestCase(
        name="HEALTH-1: Exercise habits",
        prompt="I usually exercise in the morning. I tend to do yoga before breakfast.",
        expected_behaviors=2,
        expected_intents=["HABIT", "HABIT"],
        expected_targets=["exercise", "yoga"]
    ),
    
    TestCase(
        name="HEALTH-2: Health constraints",
        prompt="I cannot do high-impact exercises due to knee injury. I need to avoid caffeine after 2 PM.",
        expected_behaviors=2,
        expected_intents=["CONSTRAINT", "CONSTRAINT"],
        expected_targets=["high_impact_exercises", "caffeine"]
    ),
    
    # ===== WORK & PRODUCTIVITY =====
    TestCase(
        name="WORK-1: Work preferences",
        prompt="I prefer working remotely. I like having meetings in the afternoon rather than morning.",
        expected_behaviors=2,
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["remote_work", "afternoon_meetings"]
    ),
    
    TestCase(
        name="WORK-2: Work habits",
        prompt="I always review my tasks at the start of the day. I regularly take breaks every hour.",
        expected_behaviors=2,
        expected_intents=["HABIT", "HABIT"],
        expected_targets=["task_review", "breaks"]
    ),
    
    # ===== DUPLICATE DETECTION TESTS =====
    TestCase(
        name="DUP-1: Exact wording variations",
        prompt="I prefer dark mode. I like dark mode. I enjoy dark mode.",
        expected_behaviors=1,  # Should be deduplicated
        expected_intents=["PREFERENCE"],
        expected_targets=["dark_mode"]
    ),
    
    TestCase(
        name="DUP-2: General vs specific context",
        prompt="I prefer Python. I prefer Python for data analysis.",
        expected_behaviors=1,  # General should subsume specific
        expected_intents=["PREFERENCE"],
        expected_targets=["Python"]
    ),
    
    TestCase(
        name="DUP-3: Different specific contexts",
        prompt="I prefer TypeScript for frontend. I prefer TypeScript for backend.",
        expected_behaviors=2,  # Different contexts should create separate behaviors
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["TypeScript", "TypeScript"]
    ),
    
    # ===== CONFLICT DETECTION TESTS =====
    TestCase(
        name="CONFLICT-1: Opposite polarity",
        prompt="I love Python. I cannot use Python at work.",
        expected_behaviors=2,  # Both stored, conflict flagged
        expected_intents=["PREFERENCE", "CONSTRAINT"],
        expected_targets=["Python", "Python"]
    ),
    
    TestCase(
        name="CONFLICT-2: Contradictory preferences",
        prompt="I prefer dark mode. Actually, I prefer light mode now.",
        expected_behaviors=1,  # Should handle as update/supersede
        expected_intents=["PREFERENCE"],
        expected_targets=["light_mode"]  # Most recent wins
    ),
    
    # ===== EDGE CASES =====
    TestCase(
        name="EDGE-1: Weak/uncertain language",
        prompt="I might like dark mode. I sometimes use Python.",
        expected_behaviors=2,  # Should still extract but with lower strength
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["dark_mode", "Python"]
    ),
    
    TestCase(
        name="EDGE-2: Compound preferences",
        prompt="I prefer using VS Code with Vim keybindings and dark theme.",
        expected_behaviors=1,  # Single compound preference
        expected_intents=["PREFERENCE"],
        expected_targets=["VS_Code"]  # Primary target
    ),
    
    TestCase(
        name="EDGE-3: Temporal context",
        prompt="I prefer coffee in the morning. I prefer tea in the evening.",
        expected_behaviors=2,  # Different temporal contexts
        expected_intents=["PREFERENCE", "PREFERENCE"],
        expected_targets=["coffee", "tea"]
    ),
]


def run_test_case(test_case: TestCase) -> bool:
    """
    Execute a single test case and validate results
    """
    # Create unique user ID for this test to avoid accumulation
    test_user_id = f"test_{test_case.name.replace(' ', '_').replace(':', '_')}_{datetime.now().strftime('%H%M%S%f')}"
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_case.name}")
    logger.info(f"USER: {test_user_id}")
    logger.info(f"PROMPT: {test_case.prompt}")
    logger.info(f"{'='*70}")
    
    try:
        # Step 1: Extract behaviors from prompt
        extraction_result = run_behavior_extraction(test_case.prompt)
        
        if not extraction_result.success:
            logger.error(f"❌ Extraction failed: {extraction_result.error}")
            test_case.result = f"Extraction failed: {extraction_result.error}"
            return False
        
        # Step 2: Store behaviors
        stored_behaviors = store_behavior(extraction_result, test_user_id)
        
        # Get stored behaviors from database
        behaviors = get_user_behaviors(test_user_id)
        
        # Log extracted behaviors
        logger.info(f"\nExtracted {len(behaviors)} behavior(s):")
        for idx, behavior in enumerate(behaviors, 1):
            logger.info(
                f"  {idx}. [{behavior.get('intent', 'N/A')}] "
                f"{behavior.get('target', 'N/A')} "
                f"(context: {behavior.get('context', 'N/A')}, "
                f"polarity: {behavior.get('polarity', 'N/A')})"
            )
            logger.info(f"     Text: {behavior.get('behavior_text', 'N/A')[:80]}...")
        
        # Validation
        passed = True
        validation_results = []
        
        # Check behavior count
        if len(behaviors) != test_case.expected_behaviors:
            passed = False
            validation_results.append(
                f"Behavior count mismatch: expected {test_case.expected_behaviors}, "
                f"got {len(behaviors)}"
            )
        else:
            validation_results.append(
                f"✓ Behavior count correct: {len(behaviors)}"
            )
        
        # Check intents (if available)
        actual_intents = [b.get('intent') for b in behaviors if b.get('intent')]
        if actual_intents:
            # Check if all expected intents are present (order may vary)
            missing_intents = set(test_case.expected_intents) - set(actual_intents)
            if missing_intents:
                passed = False
                validation_results.append(
                    f"Missing intents: {missing_intents}"
                )
            else:
                validation_results.append(
                    f"✓ Intents correct: {actual_intents}"
                )
        
        # Check targets (if available)
        actual_targets = [b.get('target') for b in behaviors if b.get('target')]
        if actual_targets:
            # Flexible matching - normalize spaces/underscores and case
            def normalize_target(t):
                return str(t).lower().replace('_', ' ').replace('-', ' ').strip()
            
            normalized_actual = [normalize_target(t) for t in actual_targets]
            normalized_expected = [normalize_target(t) for t in test_case.expected_targets]
            
            # Check if all expected targets are present (flexible matching)
            targets_found = sum(
                1 for expected in normalized_expected
                if any(expected in actual or actual in expected 
                       for actual in normalized_actual)
            )
            
            if targets_found < len(test_case.expected_targets):
                passed = False
                validation_results.append(
                    f"Target mismatch: expected {test_case.expected_targets}, "
                    f"got {actual_targets}"
                )
            else:
                validation_results.append(
                    f"✓ Targets match: {actual_targets}"
                )
        
        # Log validation results
        logger.info(f"\nValidation:")
        for result in validation_results:
            logger.info(f"  {result}")
        
        test_case.result = "\n".join(validation_results)
        test_case.passed = passed
        
        if passed:
            logger.info(f"\n✅ TEST PASSED")
        else:
            logger.info(f"\n❌ TEST FAILED")
        
        return passed
    
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}", exc_info=True)
        test_case.result = f"Exception: {str(e)}"
        return False


def run_all_tests():
    """
    Execute all test cases and generate summary report
    """
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE CANONICAL BEHAVIOR SYSTEM TEST")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    results = []
    passed_count = 0
    
    for test_case in TEST_CASES:
        try:
            passed = run_test_case(test_case)
            if passed:
                passed_count += 1
            results.append({
                'name': test_case.name,
                'passed': passed,
                'result': test_case.result,
                'prompt': test_case.prompt
            })
        except Exception as e:
            logger.error(f"Test case failed with exception: {e}")
            results.append({
                'name': test_case.name,
                'passed': False,
                'result': f"Exception: {str(e)}",
                'prompt': test_case.prompt
            })
    
    # Generate summary report
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY REPORT")
    logger.info("="*70)
    
    # Group by category
    categories = {}
    for result in results:
        category = result['name'].split('-')[0]
        if category not in categories:
            categories[category] = {'passed': 0, 'total': 0}
        categories[category]['total'] += 1
        if result['passed']:
            categories[category]['passed'] += 1
    
    logger.info("\nResults by Category:")
    for category, stats in sorted(categories.items()):
        percentage = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger.info(
            f"  {category}: {stats['passed']}/{stats['total']} "
            f"({percentage:.1f}%)"
        )
    
    logger.info(f"\nOverall: {passed_count}/{len(TEST_CASES)} tests passed "
                f"({passed_count/len(TEST_CASES)*100:.1f}%)")
    
    # Detailed results
    logger.info("\n" + "="*70)
    logger.info("DETAILED RESULTS")
    logger.info("="*70)
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        logger.info(f"\n{status} - {result['name']}")
        logger.info(f"  {result['result']}")
    
    # Save results to file
    output_file = f"test results/comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(TEST_CASES),
                'passed': passed_count,
                'failed': len(TEST_CASES) - passed_count,
                'pass_rate': passed_count / len(TEST_CASES) * 100
            },
            'by_category': categories,
            'results': results
        }, f, indent=2)
    
    logger.info(f"\n📝 Detailed results saved to: {output_file}")
    
    return passed_count == len(TEST_CASES)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
