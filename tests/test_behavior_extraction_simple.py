"""
Behavior Extraction Performance Test Suite (Simple Output for Documentation)
============================================================================
Tests the GPT-4.1-mini behavior extraction model and outputs clean results
suitable for documentation and logging.

Run with: python -m tests.test_behavior_extraction_simple
Save to file: python -m tests.test_behavior_extraction_simple > test_results.txt
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import uuid

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.openAiClient import extract_behavior


def generate_long_prompt(target_length: int) -> str:
    """Generate a realistic long prompt with multiple behaviors"""
    base_prompts = [
        "I'm a senior software engineer specializing in distributed systems and microservices architecture. ",
        "I prefer detailed technical explanations with code examples and architectural diagrams. ",
        "I'm allergic to shellfish and lactose intolerant, so please avoid those in any food-related examples. ",
        "I work in the PST timezone and usually code during late evening hours between 8 PM and 2 AM. ",
        "I have a preference for TypeScript over JavaScript and use React with Next.js for frontend development. ",
        "I like responses that are structured with clear headings, bullet points, and numbered steps. ",
        "I'm currently working on implementing a real-time collaborative editing feature similar to Google Docs. ",
        "I prefer functional programming patterns over object-oriented approaches when applicable. ",
        "I use VS Code as my primary editor with Vim keybindings enabled. ",
        "I'm experienced with PostgreSQL and Redis but new to MongoDB and DynamoDB. ",
        "I appreciate when explanations start with high-level concepts before diving into implementation details. ",
        "I'm color-blind (deuteranopia), so please avoid using red-green color distinctions in visual examples. ",
        "I prefer asynchronous communication and detailed documentation over synchronous meetings. ",
        "I'm working on improving my DevOps skills, particularly with Kubernetes and Terraform. ",
        "I like to see performance benchmarks and trade-offs when comparing different solutions. "
    ]
    
    prompt = ""
    while len(prompt) < target_length:
        for sentence in base_prompts:
            prompt += sentence
            if len(prompt) >= target_length:
                break
    
    return prompt[:target_length].strip()


def run_extraction_test(test_name: str, prompt: str, test_number: int, total_tests: int) -> Dict[str, Any]:
    """Run a single extraction test and return results"""
    
    print("\n" + "="*100)
    print(f"TEST {test_number}/{total_tests}: {test_name}")
    print("="*100)
    
    print(f"\nPROMPT ({len(prompt)} characters):")
    print("-" * 100)
    print(prompt)
    print("-" * 100)
    
    # Run extraction
    result = extract_behavior(prompt)
    
    # Parse results
    success = result.get("success", False)
    error = result.get("error")
    metadata = result.get("metadata", {})
    segments = result.get("segments", [])
    
    # Display metadata
    print("\nMETADATA:")
    print(f"  Success: {success}")
    print(f"  Extraction Time: {metadata.get('extraction_time_ms', 0)} ms")
    print(f"  Tokens Used: {metadata.get('tokens_used', 0)} tokens")
    print(f"  Prompt Length: {metadata.get('prompt_length', len(prompt))} characters")
    
    if error:
        print(f"  Error: {error}")
        print("\n" + "="*100 + "\n")
        return {
            "test_name": test_name,
            "prompt_length": len(prompt),
            "success": False,
            "error": error,
            "extraction_time_ms": metadata.get('extraction_time_ms', 0),
            "tokens_used": metadata.get('tokens_used', 0),
            "segments_count": 0,
            "behaviors_count": 0
        }
    
    # Display extracted behaviors
    total_behaviors = 0
    print(f"\nEXTRACTED SEGMENTS: {len(segments)}")
    print("-" * 100)
    
    for i, segment in enumerate(segments, 1):
        segment_text = segment.get("text", "")
        behaviors = segment.get("behaviors", [])
        total_behaviors += len(behaviors)
        
        print(f"\nSEGMENT {i}:")
        print(f"  Text: \"{segment_text}\"")
        print(f"  Behaviors Found: {len(behaviors)}")
        
        if behaviors:
            print()
            for j, behavior in enumerate(behaviors, 1):
                description = behavior.get("description", "")
                confidence = behavior.get("confidence", 0.0)
                clarity = behavior.get("clarity", 0.0)
                linguistic_strength = behavior.get("linguistic_strength", 0.0)
                
                print(f"  [{j}] {description}")
                print(f"      - Confidence:          {confidence:.3f}")
                print(f"      - Clarity:             {clarity:.3f}")
                print(f"      - Linguistic Strength: {linguistic_strength:.3f}")
        else:
            print("      (No behaviors detected in this segment)")
    
    print("\n" + "-" * 100)
    print(f"TOTAL BEHAVIORS EXTRACTED: {total_behaviors}")
    
    # Calculate averages
    avg_confidence = 0.0
    avg_clarity = 0.0
    avg_linguistic_strength = 0.0
    
    if total_behaviors > 0:
        total_confidence = 0
        total_clarity = 0
        total_linguistic_strength = 0
        
        for segment in segments:
            for behavior in segment.get("behaviors", []):
                total_confidence += behavior.get("confidence", 0.0)
                total_clarity += behavior.get("clarity", 0.0)
                total_linguistic_strength += behavior.get("linguistic_strength", 0.0)
        
        avg_confidence = total_confidence / total_behaviors
        avg_clarity = total_clarity / total_behaviors
        avg_linguistic_strength = total_linguistic_strength / total_behaviors
        
        print(f"AVERAGE CONFIDENCE: {avg_confidence:.3f}")
        print(f"AVERAGE CLARITY: {avg_clarity:.3f}")
        print(f"AVERAGE LINGUISTIC STRENGTH: {avg_linguistic_strength:.3f}")
    
    print("\n" + "="*100 + "\n")
    
    return {
        "test_name": test_name,
        "prompt_length": len(prompt),
        "success": success,
        "error": None,
        "extraction_time_ms": metadata.get('extraction_time_ms', 0),
        "tokens_used": metadata.get('tokens_used', 0),
        "segments_count": len(segments),
        "behaviors_count": total_behaviors,
        "avg_confidence": avg_confidence,
        "avg_clarity": avg_clarity,
        "avg_linguistic_strength": avg_linguistic_strength,
        "raw_result": result
    }


def print_summary_table(results: List[Dict[str, Any]]):
    """Print summary statistics table"""
    
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100 + "\n")
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}\n")
    
    # Table header
    print("-" * 160)
    print(f"{'Test Name':<35} | {'Prompt':<7} | {'Time (ms)':<10} | {'Tokens':<7} | {'Segments':<8} | {'Behaviors':<9} | {'Avg Conf':<8} | {'Avg Clarity':<11} | {'Avg Ling.Str':<12}")
    print("-" * 160)
    
    # Table rows
    for result in results:
        name = result["test_name"][:33]
        prompt_len = f"{result['prompt_length']}ch"
        time_ms = f"{result['extraction_time_ms']}"
        tokens = f"{result['tokens_used']}"
        segments = f"{result['segments_count']}"
        behaviors = f"{result['behaviors_count']}"
        conf = f"{result.get('avg_confidence', 0):.3f}" if result['success'] else "N/A"
        clar = f"{result.get('avg_clarity', 0):.3f}" if result['success'] else "N/A"
        ling_str = f"{result.get('avg_linguistic_strength', 0):.3f}" if result['success'] else "N/A"
        
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"{status} {name:<31} | {prompt_len:<7} | {time_ms:<10} | {tokens:<7} | {segments:<8} | {behaviors:<9} | {conf:<8} | {clar:<11} | {ling_str:<12}")
    
    print("-" * 160)
    
    # Calculate and display averages
    if successful_tests:
        avg_time = sum(r["extraction_time_ms"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["tokens_used"] for r in successful_tests) / len(successful_tests)
        avg_segments = sum(r["segments_count"] for r in successful_tests) / len(successful_tests)
        avg_behaviors = sum(r["behaviors_count"] for r in successful_tests) / len(successful_tests)
        avg_conf = sum(r.get("avg_confidence", 0) for r in successful_tests) / len(successful_tests)
        avg_clar = sum(r.get("avg_clarity", 0) for r in successful_tests) / len(successful_tests)
        avg_ling_str = sum(r.get("avg_linguistic_strength", 0) for r in successful_tests) / len(successful_tests)
        
        print(f"\nOVERALL AVERAGES (Successful Tests Only):")
        print(f"  - Response Time: {avg_time:.2f} ms")
        print(f"  - Token Usage: {avg_tokens:.2f} tokens")
        print(f"  - Segments per Prompt: {avg_segments:.2f}")
        print(f"  - Behaviors per Prompt: {avg_behaviors:.2f}")
        print(f"  - Average Confidence Score: {avg_conf:.3f}")
        print(f"  - Average Clarity Score: {avg_clar:.3f}")
        print(f"  - Average Linguistic Strength: {avg_ling_str:.3f}")
        
        # Token cost estimation (GPT-4 mini pricing)
        total_tokens = sum(r["tokens_used"] for r in successful_tests)
        # Approximate cost: $0.150 per 1M input tokens, $0.600 per 1M output tokens
        # Assuming 70% input, 30% output
        estimated_cost = (total_tokens * 0.7 * 0.150 / 1_000_000) + (total_tokens * 0.3 * 0.600 / 1_000_000)
        
        print(f"\nCOST ESTIMATION:")
        print(f"  - Total Tokens Used: {total_tokens}")
        print(f"  - Estimated Cost: ${estimated_cost:.6f}")
        print(f"  - Cost per Request: ${estimated_cost / len(successful_tests):.6f}")


def main():
    """Main test execution function"""
    
    print("="*100)
    print("BEHAVIOR EXTRACTION PERFORMANCE TEST SUITE")
    print("="*100)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: GPT-4.1-mini")
    print(f"Embedding Model: text-embedding-3-large")
    print("="*100)
    
    # Define test cases
    test_cases = [
        {
            "name": "Short Single Preference",
            "prompt": "I prefer concise answers.",
        },
        {
            "name": "Dual Behavior (Allergy + Preference)",
            "prompt": "I'm allergic to peanuts and I prefer code examples over long explanations.",
        },
        {
            "name": "Medium - Student Context",
            "prompt": "Hi, I'm working on a React project for my university assignment. I prefer getting code examples with explanations rather than just theory. Also, I work better with visual diagrams when explaining complex concepts. Can you help me understand JWT authentication?",
        },
        {
            "name": "Medium - Work Style",
            "prompt": "I'm building a Python API with FastAPI. I'm in EST timezone so I usually work evenings. I prefer step-by-step instructions. Also, I'm using VS Code as my editor.",
        },
        {
            "name": "Question (No Behavior Expected)",
            "prompt": "What's the difference between let and const in JavaScript?",
        },
        {
            "name": "Complex Multi-Behavior",
            "prompt": "Hello! I'm a software engineering student working on my final year research project about user behavior management in LLMs. I prefer detailed technical explanations with code examples when learning new concepts. I'm allergic to dairy products, so when you give food examples, please avoid those. I work in UTC+5:30 timezone and usually code late at night. I like when responses include diagrams or visual representations for architecture decisions. I'm currently implementing a vector database solution using ChromaDB. Could you explain how to optimize similarity search performance?",
        },
        {
            "name": "Edge - Very Short",
            "prompt": "Use simple words.",
        },
        {
            "name": "Edge - Command Only (No Behavior)",
            "prompt": "Create a Python function to sort a list.",
        },
        {
            "name": "Edge - Temporary State",
            "prompt": "I'm tired today and need help with this bug quickly.",
        },
        {
            "name": "Long Prompt (~1000 chars)",
            "prompt": generate_long_prompt(1000),
        },
        {
            "name": "Edge - Multiple Preferences Explicit",
            "prompt": "When explaining technical concepts, I prefer: 1) Start with a simple analogy, 2) Provide code examples in Python or TypeScript, 3) Include performance considerations, 4) Show common pitfalls to avoid, 5) Give me links to official documentation. I'm a visual learner, so diagrams help me a lot. I also prefer dark mode code examples with syntax highlighting.",
        },
        {
            "name": "Edge - Mixed Request + Behavior",
            "prompt": "Can you review my React code? By the way, I prefer functional components over class components and I like using TypeScript for type safety.",
        },
    ]
    
    # Run all tests
    results = []
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        result = run_extraction_test(
            test_name=test_case["name"],
            prompt=test_case["prompt"],
            test_number=i,
            total_tests=total_tests
        )
        results.append(result)
    
    # Print summary
    print_summary_table(results)
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")

    
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save results to JSON file
    output_file = f"behavior_extraction_model_test_results_{datetime_str}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4.1-mini",
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r["success"]]),
            "failed_tests": len([r for r in results if not r["success"]]),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_file}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user\n")
    except Exception as e:
        print(f"\n\nTest failed with error: {str(e)}\n")
        import traceback
        traceback.print_exc()
