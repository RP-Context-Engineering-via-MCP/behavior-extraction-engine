"""
Prompt to Persistence Flow Test Suite
======================================
Tests the complete end-to-end flow from raw user prompt to database-ready behaviors.

Flow:
1. Raw Prompt → GPT Extraction (with timing)
2. Extracted Behaviors → Credibility Calculation + Filtering
3. Final StoredBehavior Objects → Ready for Database (with timing)

Run with: python -m tests.test_prompt_to_persistence_flow
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.extractor import run_behavior_extraction, store_behavior
from models.behavior import ExtractionResult, StoredBehavior


def format_time(ms: float) -> str:
    """Format milliseconds with appropriate unit"""
    if ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms/1000:.2f} s"


def run_full_flow_test(test_name: str, prompt: str, test_number: int, total_tests: int) -> Dict[str, Any]:
    """
    Run complete flow test: prompt → extraction → storage preparation
    
    Returns detailed results with timing for each step
    """
    
    print("\n" + "="*120)
    print(f"TEST {test_number}/{total_tests}: {test_name}")
    print("="*120)
    
    print(f"\n📝 RAW PROMPT ({len(prompt)} characters):")
    print("-" * 120)
    print(prompt)
    print("-" * 120)
    
    # ========================================
    # STEP 1: Extract behaviors from prompt
    # ========================================
    print("\n⏱️  STEP 1: EXTRACTING BEHAVIORS FROM PROMPT...")
    step1_start = time.time()
    
    extraction_result = run_behavior_extraction(prompt)
    
    step1_duration_ms = (time.time() - step1_start) * 1000
    print(f"✓ Extraction completed in {format_time(step1_duration_ms)}")
    
    if not extraction_result.success:
        print(f"\n❌ Extraction failed: {extraction_result.error}")
        print("\n" + "="*120 + "\n")
        return {
            "test_name": test_name,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "success": False,
            "error": extraction_result.error,
            "step1_extraction_time_ms": step1_duration_ms,
            "step2_storage_prep_time_ms": 0,
            "total_time_ms": step1_duration_ms,
            "extracted_behaviors_count": 0,
            "stored_behaviors_count": 0,
            "filtered_behaviors_count": 0,
        }
    
    # Display extraction results
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"  - Segments Found: {len(extraction_result.segments)}")
    print(f"  - GPT Extraction Time: {format_time(extraction_result.extraction_time)}")
    
    extracted_behaviors_count = sum(len(seg.behaviors) for seg in extraction_result.segments)
    print(f"  - Total Behaviors Extracted: {extracted_behaviors_count}")
    
    # Display detailed extraction results
    print(f"\n🔍 EXTRACTED SEGMENTS & BEHAVIORS:")
    print("-" * 120)
    
    for i, segment in enumerate(extraction_result.segments, 1):
        print(f"\n  SEGMENT {i}:")
        print(f"    Text: \"{segment.text}\"")
        print(f"    Behaviors: {len(segment.behaviors)}")
        
        if segment.behaviors:
            for j, behavior in enumerate(segment.behaviors, 1):
                print(f"\n    [{j}] {behavior.description}")
                print(f"        • Confidence:          {behavior.confidence:.3f}")
                print(f"        • Clarity:             {behavior.clarity:.3f}")
                print(f"        • Linguistic Strength: {behavior.linguistic_strength:.3f}")
                print(f"        • Extracted At:        {behavior.extracted_at}")
        else:
            print("        (No behaviors in this segment)")
    
    # ========================================
    # STEP 2: Prepare behaviors for storage
    # ========================================
    print("\n" + "-" * 120)
    print("\n⏱️  STEP 2: PREPARING BEHAVIORS FOR DATABASE STORAGE...")
    print("    (Calculating credibility, filtering low-quality, generating embeddings)")
    
    step2_start = time.time()
    
    stored_behaviors = store_behavior(extraction_result)
    
    step2_duration_ms = (time.time() - step2_start) * 1000
    print(f"✓ Storage preparation completed in {format_time(step2_duration_ms)}")
    
    filtered_count = extracted_behaviors_count - len(stored_behaviors)
    
    print(f"\n📦 STORAGE PREPARATION RESULTS:")
    print(f"  - Behaviors Ready for DB: {len(stored_behaviors)}")
    print(f"  - Behaviors Filtered Out: {filtered_count}")
    
    if filtered_count > 0:
        print(f"  - Filter Reason: Low credibility (below threshold)")
    
    # Display final database-ready behaviors
    print(f"\n💾 FINAL DATABASE-READY BEHAVIORS:")
    print("-" * 120)
    
    if stored_behaviors:
        for i, stored_behavior in enumerate(stored_behaviors, 1):
            print(f"\n  BEHAVIOR {i}:")
            print(f"    ID:                    {stored_behavior.behavior_id}")
            print(f"    Text:                  \"{stored_behavior.behavior_text}\"")
            print(f"    Credibility:           {stored_behavior.credibility:.4f}")
            print(f"    Confidence:            {stored_behavior.extraction_confidence:.3f}")
            print(f"    Clarity:               {stored_behavior.clarity_score:.3f}")
            print(f"    Linguistic Strength:   {stored_behavior.linguistic_strength:.3f}")
            print(f"    Decay Rate:            {stored_behavior.decay_rate}")
            print(f"    Reinforcement Count:   {stored_behavior.reinforcement_count}")
            print(f"    Created At:            {datetime.fromtimestamp(stored_behavior.created_at).isoformat()}")
            print(f"    Last Seen At:          {datetime.fromtimestamp(stored_behavior.last_seen_at).isoformat()}")
            print(f"    Session ID:            {stored_behavior.session_id}")
            print(f"    Embedding Dimension:   {len(stored_behavior.embedding) if stored_behavior.embedding else 0}")
            print(f"    Prompt History IDs:    {stored_behavior.prompt_history_ids if stored_behavior.prompt_history_ids else '[]'}")
    else:
        print("  (No behaviors passed credibility threshold for storage)")
    
    # ========================================
    # SUMMARY
    # ========================================
    total_time_ms = step1_duration_ms + step2_duration_ms
    
    print("\n" + "-" * 120)
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"  - Step 1 (Extraction):         {format_time(step1_duration_ms)}")
    print(f"    └─ GPT Processing:           {format_time(extraction_result.extraction_time)}")
    print(f"    └─ Validation & Conversion:  {format_time(step1_duration_ms - extraction_result.extraction_time)}")
    print(f"  - Step 2 (Storage Prep):       {format_time(step2_duration_ms)}")
    print(f"    └─ Per Behavior (avg):       {format_time(step2_duration_ms / extracted_behaviors_count) if extracted_behaviors_count > 0 else '0 ms'}")
    print(f"  - TOTAL TIME:                  {format_time(total_time_ms)}")
    
    print("\n" + "="*120 + "\n")
    
    # Prepare detailed result for JSON export
    return {
        "test_name": test_name,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "success": True,
        "error": None,
        
        # Timing
        "step1_extraction_time_ms": step1_duration_ms,
        "gpt_processing_time_ms": extraction_result.extraction_time,
        "validation_time_ms": step1_duration_ms - extraction_result.extraction_time,
        "step2_storage_prep_time_ms": step2_duration_ms,
        "storage_prep_per_behavior_ms": step2_duration_ms / extracted_behaviors_count if extracted_behaviors_count > 0 else 0,
        "total_time_ms": total_time_ms,
        
        # Counts
        "segments_count": len(extraction_result.segments),
        "extracted_behaviors_count": extracted_behaviors_count,
        "stored_behaviors_count": len(stored_behaviors),
        "filtered_behaviors_count": filtered_count,
        
        # Detailed data
        "extracted_segments": [
            {
                "text": seg.text,
                "behaviors": [
                    {
                        "description": b.description,
                        "confidence": b.confidence,
                        "clarity": b.clarity,
                        "linguistic_strength": b.linguistic_strength,
                        "extracted_at": b.extracted_at
                    }
                    for b in seg.behaviors
                ]
            }
            for seg in extraction_result.segments
        ],
        
        "stored_behaviors": [
            {
                "behavior_id": sb.behavior_id,
                "behavior_text": sb.behavior_text,
                "credibility": sb.credibility,
                "extraction_confidence": sb.extraction_confidence,
                "clarity_score": sb.clarity_score,
                "linguistic_strength": sb.linguistic_strength,
                "decay_rate": sb.decay_rate,
                "reinforcement_count": sb.reinforcement_count,
                "created_at": sb.created_at,
                "last_seen_at": sb.last_seen_at,
                "session_id": sb.session_id,
                "embedding": sb.embedding if sb.embedding else [],
                "prompt_history_ids": sb.prompt_history_ids
            }
            for sb in stored_behaviors
        ]
    }


def print_summary_table(results: List[Dict[str, Any]]):
    """Print summary statistics table"""
    
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120 + "\n")
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}\n")
    
    # Table header
    print("-" * 160)
    print(f"{'Test Name':<30} | {'Prompt':<8} | {'Extract':<10} | {'Prepare':<10} | {'Total':<10} | {'Extracted':<10} | {'Stored':<8} | {'Filtered':<9}")
    print("-" * 160)
    
    # Table rows
    for result in results:
        name = result["test_name"][:28]
        prompt_len = f"{result['prompt_length']}ch"
        extract_time = format_time(result.get('step1_extraction_time_ms', 0))
        prepare_time = format_time(result.get('step2_storage_prep_time_ms', 0))
        total_time = format_time(result.get('total_time_ms', 0))
        extracted = f"{result.get('extracted_behaviors_count', 0)}"
        stored = f"{result.get('stored_behaviors_count', 0)}"
        filtered = f"{result.get('filtered_behaviors_count', 0)}"
        
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"{status} {name:<26} | {prompt_len:<8} | {extract_time:<10} | {prepare_time:<10} | {total_time:<10} | {extracted:<10} | {stored:<8} | {filtered:<9}")
    
    print("-" * 160)
    
    # Calculate and display averages
    if successful_tests:
        avg_extract = sum(r["step1_extraction_time_ms"] for r in successful_tests) / len(successful_tests)
        avg_prepare = sum(r["step2_storage_prep_time_ms"] for r in successful_tests) / len(successful_tests)
        avg_total = sum(r["total_time_ms"] for r in successful_tests) / len(successful_tests)
        avg_extracted = sum(r["extracted_behaviors_count"] for r in successful_tests) / len(successful_tests)
        avg_stored = sum(r["stored_behaviors_count"] for r in successful_tests) / len(successful_tests)
        avg_filtered = sum(r["filtered_behaviors_count"] for r in successful_tests) / len(successful_tests)
        
        filter_rate = (sum(r["filtered_behaviors_count"] for r in successful_tests) / 
                      sum(r["extracted_behaviors_count"] for r in successful_tests) * 100) if sum(r["extracted_behaviors_count"] for r in successful_tests) > 0 else 0
        
        print(f"\nOVERALL AVERAGES (Successful Tests):")
        print(f"  - Extraction Time:       {format_time(avg_extract)}")
        print(f"  - Storage Prep Time:     {format_time(avg_prepare)}")
        print(f"  - Total Time:            {format_time(avg_total)}")
        print(f"  - Behaviors Extracted:   {avg_extracted:.2f} per prompt")
        print(f"  - Behaviors Stored:      {avg_stored:.2f} per prompt")
        print(f"  - Behaviors Filtered:    {avg_filtered:.2f} per prompt ({filter_rate:.1f}% filter rate)")
        
        print(f"\nQUALITY METRICS:")
        total_extracted = sum(r["extracted_behaviors_count"] for r in successful_tests)
        total_stored = sum(r["stored_behaviors_count"] for r in successful_tests)
        storage_rate = (total_stored / total_extracted * 100) if total_extracted > 0 else 0
        print(f"  - Total Behaviors Extracted:  {total_extracted}")
        print(f"  - Total Behaviors Stored:     {total_stored}")
        print(f"  - Storage Rate:               {storage_rate:.1f}% (passed credibility threshold)")


def main():
    """Main test execution function"""
    
    print("="*120)
    print("PROMPT TO PERSISTENCE FLOW TEST SUITE")
    print("="*120)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: GPT-4.1-mini")
    print(f"Embedding Model: text-embedding-3-large")
    print(f"\nFlow: Raw Prompt → GPT Extraction → Credibility Calculation → Database-Ready Behaviors")
    print("="*120)
    
    # Define test cases (variety of scenarios)
    test_cases = [
        {
            "name": "Simple Single Preference",
            "prompt": "I prefer dark mode for all interfaces.",
        },
        {
            "name": "Strong Commitment (Allergy)",
            "prompt": "I'm severely allergic to peanuts and tree nuts.",
        },
        {
            "name": "Multiple Preferences",
            "prompt": "I prefer Python over JavaScript for backend work. I like detailed code comments. I use VS Code as my editor.",
        },
        {
            "name": "Mixed Strength Behaviors",
            "prompt": "I definitely prefer TypeScript for large projects. I sometimes use Jest for testing. I'm thinking about learning GraphQL.",
        },
        {
            "name": "Weak Uncertain Language",
            "prompt": "I might like using React. Maybe I should try Vue sometime. I guess I prefer functional components.",
        },
        {
            "name": "Technical Preferences",
            "prompt": "I prefer detailed technical explanations with code examples. I like responses structured with headings and bullet points. I work better with visual diagrams for complex architectures.",
        },
        {
            "name": "Personal Context",
            "prompt": "I'm a software engineering student working on my final year project about LLMs. I'm allergic to dairy and shellfish. I work in UTC+5:30 timezone and usually code late at night between 11 PM and 3 AM.",
        },
        {
            "name": "No Clear Behaviors",
            "prompt": "Can you explain how JWT authentication works? What are the security implications?",
        },
    ]
    
    # Run all tests
    results = []
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        result = run_full_flow_test(
            test_name=test_case["name"],
            prompt=test_case["prompt"],
            test_number=i,
            total_tests=total_tests
        )
        results.append(result)
    
    # Print summary
    print_summary_table(results)
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120 + "\n")
    
    # Save results to JSON file
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prompt_to_persistence_flow_{datetime_str}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4.1-mini",
            "embedding_model": "text-embedding-3-large",
            "test_type": "prompt_to_persistence_flow",
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r["success"]]),
            "failed_tests": len([r for r in results if not r["success"]]),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Detailed results saved to: {output_file}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}\n")
        import traceback
        traceback.print_exc()
