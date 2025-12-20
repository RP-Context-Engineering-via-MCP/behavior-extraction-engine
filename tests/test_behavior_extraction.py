"""
Behavior Extraction Performance Test Suite
==========================================
Tests the GPT-4.1-mini behavior extraction model with various prompt lengths
and complexity levels to measure response time, token usage, and extraction quality.

Run with: python -m tests.test_behavior_extraction
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.openAiClient import extract_behavior


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")


def print_subheader(text: str):
    """Print a formatted subheader"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*len(text)}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(key: str, value: Any):
    """Print key-value info"""
    print(f"{Colors.BLUE}{key}:{Colors.END} {value}")


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


def run_extraction_test(test_name: str, prompt: str) -> Dict[str, Any]:
    """
    Run a single extraction test and return results
    
    Args:
        test_name: Name of the test case
        prompt: User prompt to extract behaviors from
        
    Returns:
        Dictionary containing test results and metrics
    """
    print_subheader(f"Test Case: {test_name}")
    print_info("Prompt Length", f"{len(prompt)} characters")
    print_info("Prompt Preview", f"{prompt[:100]}..." if len(prompt) > 100 else prompt)
    
    # Run extraction
    result = extract_behavior(prompt)
    
    # Parse results
    success = result.get("success", False)
    error = result.get("error")
    metadata = result.get("metadata", {})
    segments = result.get("segments", [])
    
    # Display metadata
    print_info("Extraction Time", f"{metadata.get('extraction_time_ms', 0)} ms")
    print_info("Tokens Used", f"{metadata.get('tokens_used', 0)} tokens")
    print_info("Success", f"{Colors.GREEN}Yes{Colors.END}" if success else f"{Colors.RED}No{Colors.END}")
    
    if error:
        print_error(f"Error: {error}")
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
    print(f"\n{Colors.YELLOW}Extracted Segments: {len(segments)}{Colors.END}")
    
    for i, segment in enumerate(segments, 1):
        segment_text = segment.get("text", "")
        behaviors = segment.get("behaviors", [])
        total_behaviors += len(behaviors)
        
        print(f"\n  {Colors.BOLD}Segment {i}:{Colors.END}")
        print(f"  Text: {Colors.CYAN}\"{segment_text[:80]}{'...' if len(segment_text) > 80 else ''}\"{Colors.END}")
        print(f"  Behaviors: {len(behaviors)}")
        
        for j, behavior in enumerate(behaviors, 1):
            description = behavior.get("description", "")
            confidence = behavior.get("confidence", 0.0)
            clarity = behavior.get("clarity", 0.0)
            
            print(f"\n    {Colors.GREEN}[{j}] {description}{Colors.END}")
            print(f"        • Confidence: {confidence:.2f} {get_score_indicator(confidence)}")
            print(f"        • Clarity:    {clarity:.2f} {get_score_indicator(clarity)}")
    
    print(f"\n{Colors.BOLD}Total Behaviors Extracted: {total_behaviors}{Colors.END}")
    
    return {
        "test_name": test_name,
        "prompt_length": len(prompt),
        "success": success,
        "error": None,
        "extraction_time_ms": metadata.get('extraction_time_ms', 0),
        "tokens_used": metadata.get('tokens_used', 0),
        "segments_count": len(segments),
        "behaviors_count": total_behaviors,
        "avg_confidence": calculate_avg_confidence(segments),
        "avg_clarity": calculate_avg_clarity(segments)
    }


def get_score_indicator(score: float) -> str:
    """Return a visual indicator for score quality"""
    if score >= 0.8:
        return f"{Colors.GREEN}█████{Colors.END}"
    elif score >= 0.6:
        return f"{Colors.GREEN}████{Colors.END}{Colors.YELLOW}░{Colors.END}"
    elif score >= 0.4:
        return f"{Colors.YELLOW}███░░{Colors.END}"
    elif score >= 0.2:
        return f"{Colors.RED}██░░░{Colors.END}"
    else:
        return f"{Colors.RED}█░░░░{Colors.END}"


def calculate_avg_confidence(segments: List[Dict]) -> float:
    """Calculate average confidence across all behaviors"""
    total_confidence = 0
    total_behaviors = 0
    
    for segment in segments:
        for behavior in segment.get("behaviors", []):
            total_confidence += behavior.get("confidence", 0.0)
            total_behaviors += 1
    
    return total_confidence / total_behaviors if total_behaviors > 0 else 0.0


def calculate_avg_clarity(segments: List[Dict]) -> float:
    """Calculate average clarity across all behaviors"""
    total_clarity = 0
    total_behaviors = 0
    
    for segment in segments:
        for behavior in segment.get("behaviors", []):
            total_clarity += behavior.get("clarity", 0.0)
            total_behaviors += 1
    
    return total_clarity / total_behaviors if total_behaviors > 0 else 0.0


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics for all tests"""
    print_header("TEST SUMMARY")
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print_info("Total Tests", len(results))
    print_success(f"Successful: {len(successful_tests)}")
    if failed_tests:
        print_error(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        print("\n")
        print(f"{Colors.BOLD}Performance Metrics:{Colors.END}")
        print(f"{'─'*80}")
        print(f"{'Test Name':<30} {'Prompt':<8} {'Time':<10} {'Tokens':<8} {'Behaviors':<10} {'Conf':<6} {'Clarity':<6}")
        print(f"{'─'*80}")
        
        for result in results:
            name = result["test_name"][:28]
            prompt_len = f"{result['prompt_length']}ch"
            time_ms = f"{result['extraction_time_ms']}ms"
            tokens = f"{result['tokens_used']}t"
            behaviors = f"{result['behaviors_count']}b"
            conf = f"{result.get('avg_confidence', 0):.2f}" if result['success'] else "N/A"
            clar = f"{result.get('avg_clarity', 0):.2f}" if result['success'] else "N/A"
            
            status = f"{Colors.GREEN}✓{Colors.END}" if result['success'] else f"{Colors.RED}✗{Colors.END}"
            print(f"{status} {name:<28} {prompt_len:<8} {time_ms:<10} {tokens:<8} {behaviors:<10} {conf:<6} {clar:<6}")
        
        print(f"{'─'*80}")
        
        # Calculate averages
        avg_time = sum(r["extraction_time_ms"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["tokens_used"] for r in successful_tests) / len(successful_tests)
        avg_behaviors = sum(r["behaviors_count"] for r in successful_tests) / len(successful_tests)
        avg_conf = sum(r.get("avg_confidence", 0) for r in successful_tests) / len(successful_tests)
        avg_clar = sum(r.get("avg_clarity", 0) for r in successful_tests) / len(successful_tests)
        
        print(f"\n{Colors.BOLD}Averages:{Colors.END}")
        print(f"  • Response Time: {avg_time:.2f} ms")
        print(f"  • Token Usage: {avg_tokens:.2f} tokens")
        print(f"  • Behaviors per Prompt: {avg_behaviors:.2f}")
        print(f"  • Average Confidence: {avg_conf:.2f}")
        print(f"  • Average Clarity: {avg_clar:.2f}")


def main():
    """Main test execution function"""
    print_header("BEHAVIOR EXTRACTION PERFORMANCE TEST SUITE")
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
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
            "name": "Edge - Temporary State (Low Confidence Expected)",
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
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(test_cases)}]{Colors.END}")
        result = run_extraction_test(
            test_name=test_case["name"],
            prompt=test_case["prompt"]
        )
        results.append(result)
        print()  # Add spacing between tests
    
    # Print summary
    print_summary(results)
    
    print(f"\n{Colors.BOLD}Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}\n")
    except Exception as e:
        print(f"\n\n{Colors.RED}Test failed with error: {str(e)}{Colors.END}\n")
        raise
