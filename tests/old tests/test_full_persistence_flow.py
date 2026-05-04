"""
Full Flow Test: Prompt → Extraction → Credibility → Database Persistence
==========================================================================
Tests the complete end-to-end pipeline with actual database insertion and verification.

This test will:
1. Send a prompt to the extraction system
2. Extract behaviors with GPT-4
3. Calculate credibility scores and filter low-quality behaviors
4. Generate embeddings and insert into database (via store_behavior())
5. Query database to verify data was stored
6. Display detailed logs at each step

Run with: python -m tests.test_full_persistence_flow
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.extractor import run_behavior_extraction, store_behavior
from services.behaviorRepository import insert_behavior, search_similar_behaviors
from db.connection import get_db_connection
from config.configurations import SAMPLE_USERID

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_section(title: str, char: str = "="):
    """Print a formatted section header"""
    print("\n" + char * 100)
    print(f"  {title}")
    print(char * 100 + "\n")


def verify_database_storage(user_id: str, behavior_ids: list) -> dict:
    """
    Query database to verify behaviors were actually stored
    
    Args:
        user_id: User identifier
        behavior_ids: List of behavior IDs that should be in database
        
    Returns:
        dict with verification results
    """
    print_section("STEP 3: VERIFYING DATABASE STORAGE", "=")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Query all behaviors for this user
                cur.execute(
                    """
                    SELECT behavior_id, behavior_text, credibility, 
                           extraction_confidence, clarity_score, linguistic_strength,
                           created_at, session_id
                    FROM behaviors
                    WHERE user_id = %s
                    ORDER BY created_at DESC;
                    """,
                    (user_id,)
                )
                
                results = cur.fetchall()
                
                print(f"✓ Database query successful")
                print(f"✓ Found {len(results)} behaviors for user: {user_id}\n")
                
                if results:
                    print("📊 STORED BEHAVIORS IN DATABASE:")
                    print("-" * 100)
                    
                    for i, row in enumerate(results, 1):
                        behavior_id, text, cred, conf, clarity, ling_str, created, session = row
                        
                        print(f"\n  BEHAVIOR {i}:")
                        print(f"    ID:                  {behavior_id}")
                        print(f"    Text:                \"{text}\"")
                        print(f"    Credibility:         {cred:.4f}")
                        print(f"    Confidence:          {conf:.3f}")
                        print(f"    Clarity:             {clarity:.3f}")
                        print(f"    Linguistic Strength: {ling_str:.3f}")
                        print(f"    Created At:          {datetime.fromtimestamp(created).isoformat()}")
                        print(f"    Session ID:          {session}")
                        
                        # Check if this was one of the behaviors we just inserted
                        if behavior_id in behavior_ids:
                            print(f"    ✓ VERIFIED: This behavior was just inserted")
                    
                    print("\n" + "-" * 100)
                else:
                    print("⚠️  No behaviors found in database")
                
                return {
                    "success": True,
                    "stored_count": len(results),
                    "verified_ids": [row[0] for row in results],
                    "error": None
                }
                
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        print(f"\n❌ Database verification failed: {str(e)}")
        return {
            "success": False,
            "stored_count": 0,
            "verified_ids": [],
            "error": str(e)
        }


def test_full_flow(prompt: str, user_id: str):
    """
    Execute complete flow: Extract → Store → Verify
    
    Args:
        prompt: User's natural language input
        user_id: User identifier
    """
    
    print("\n" + "=" * 100)
    print("  FULL PERSISTENCE FLOW TEST")
    print("=" * 100)
    print(f"\n  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  User ID: {user_id}")
    print(f"  Prompt:  \"{prompt}\"")
    print("\n" + "=" * 100)
    
    # ============================================
    # STEP 1: Extract behaviors from prompt
    # ============================================
    print_section("STEP 1: EXTRACTING BEHAVIORS FROM PROMPT", "=")
    print(f"Prompt: \"{prompt}\"\n")
    
    try:
        extraction_result = run_behavior_extraction(prompt)
        
        if not extraction_result.success:
            print(f"❌ Extraction failed: {extraction_result.error}")
            return {
                "success": False,
                "error": f"Extraction failed: {extraction_result.error}"
            }
        
        extracted_count = sum(len(seg.behaviors) for seg in extraction_result.segments)
        print(f"✓ Extraction completed successfully")
        print(f"✓ Segments found: {len(extraction_result.segments)}")
        print(f"✓ Behaviors extracted: {extracted_count}")
        print(f"✓ Extraction time: {extraction_result.extraction_time:.2f}ms\n")
        
        # Display extracted behaviors
        print("📋 EXTRACTED BEHAVIORS:")
        print("-" * 100)
        for i, segment in enumerate(extraction_result.segments, 1):
            print(f"\n  Segment {i}: \"{segment.text}\"")
            if segment.behaviors:
                for j, behavior in enumerate(segment.behaviors, 1):
                    print(f"    [{j}] {behavior.description}")
                    print(f"        Confidence:          {behavior.confidence:.3f}")
                    print(f"        Clarity:             {behavior.clarity:.3f}")
                    print(f"        Linguistic Strength: {behavior.linguistic_strength:.3f}")
            else:
                print("    (No behaviors in this segment)")
        print("\n" + "-" * 100)
        
    except Exception as e:
        logger.exception("Extraction failed")
        print(f"\n❌ Extraction error: {str(e)}")
        return {"success": False, "error": f"Extraction error: {str(e)}"}
    
    # ============================================
    # STEP 2: Calculate credibility & filter
    # ============================================
    print_section("STEP 2: CALCULATING CREDIBILITY & FILTERING", "=")
    
    try:
        stored_behaviors = store_behavior(extraction_result, user_id)
        
        filtered_count = extracted_count - len(stored_behaviors)
        
        print(f"✓ Credibility calculation completed")
        print(f"✓ Behaviors passing threshold: {len(stored_behaviors)}")
        print(f"✓ Behaviors filtered out: {filtered_count}\n")
        
        if not stored_behaviors:
            print("⚠️  No behaviors passed credibility threshold")
            print("   All behaviors were filtered as low quality")
            return {
                "success": True,
                "extracted_count": extracted_count,
                "stored_count": 0,
                "filtered_count": filtered_count,
                "message": "All behaviors filtered due to low credibility"
            }
        
        print("💾 BEHAVIORS READY FOR STORAGE:")
        print("-" * 100)
        for i, behavior in enumerate(stored_behaviors, 1):
            print(f"\n  BEHAVIOR {i}:")
            print(f"    ID:                  {behavior.behavior_id}")
            print(f"    Text:                \"{behavior.behavior_text}\"")
            print(f"    Credibility:         {behavior.credibility:.4f}")
            print(f"    Confidence:          {behavior.extraction_confidence:.3f}")
            print(f"    Clarity:             {behavior.clarity_score:.3f}")
            print(f"    Linguistic Strength: {behavior.linguistic_strength:.3f}")
            print(f"    Embedding:           3072")
        print("\n" + "-" * 100)
        
        # Note: Behaviors are already inserted into database by store_behavior()
        # Extract behavior IDs for verification
        inserted_ids = [behavior.behavior_id for behavior in stored_behaviors]
        
        print(f"\n✓ Database insertion completed by store_behavior()")
        print(f"✓ Behaviors stored: {len(inserted_ids)}")
        
    except Exception as e:
        logger.exception("Storage preparation failed")
        print(f"\n❌ Storage preparation error: {str(e)}")
        return {"success": False, "error": f"Storage preparation error: {str(e)}"}
    
    # ============================================
    # STEP 3: Verify storage
    # ============================================
    verification = verify_database_storage(user_id, inserted_ids)
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print_section("FINAL SUMMARY", "=")
    
    print(f"📊 FLOW EXECUTION SUMMARY:")
    print(f"  • Prompt:                    \"{prompt}\"")
    print(f"  • User ID:                   {user_id}")
    print(f"  • Behaviors Extracted:       {extracted_count}")
    print(f"  • Behaviors Filtered:        {filtered_count}")
    print(f"  • Behaviors Stored:          {len(inserted_ids)}")
    print(f"  • Behaviors Verified in DB:  {verification['stored_count']}")
    
    if len(inserted_ids) == verification['stored_count']:
        print(f"\n✅ SUCCESS: All {len(inserted_ids)} behaviors verified in database!")
    else:
        print(f"\n⚠️  WARNING: Mismatch between stored ({len(inserted_ids)}) and verified ({verification['stored_count']})")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100 + "\n")
    
    return {
        "success": True,
        "extracted_count": extracted_count,
        "stored_count": len(inserted_ids),
        "verified_count": verification['stored_count'],
        "filtered_count": filtered_count,
        "behavior_ids": inserted_ids
    }


def main():
    """Main test execution"""
    
    print("\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 98 + "║")
    print("║" + "BEHAVIOR EXTRACTION & PERSISTENCE DEMO".center(98) + "║")
    print("║" + "Prompt → Extraction → Credibility → Database Persistence → Verification".center(98) + "║")
    print("║" + " " * 98 + "║")
    print("╚" + "=" * 98 + "╝")
    print("\n")
    
    # Get user input
    print("=" * 100)
    print("  INTERACTIVE PROMPT INPUT")
    print("=" * 100)
    print("\nPlease enter your prompt (describe your preferences, constraints, or behaviors):")
    print("Example: 'I prefer Python for backend. I'm allergic to peanuts. I like detailed comments.'\n")
    
    try:
        user_prompt = input("Your Prompt: ").strip()
        
        if not user_prompt:
            print("\n❌ Error: Prompt cannot be empty")
            return 1
        
        print(f"\n✓ Received prompt ({len(user_prompt)} characters)")
        
        # Use sample user ID
        TEST_USER_ID = SAMPLE_USERID
        print(f"✓ Using User ID: {TEST_USER_ID}")
        
        # Run the full flow
        result = test_full_flow(user_prompt, TEST_USER_ID)
        
        if result.get("success") and result.get("stored_count", 0) > 0:
            print("\n" + "=" * 100)
            print("  ✅ DEMO COMPLETED SUCCESSFULLY")
            print("=" * 100)
            print(f"\n  📊 Summary:")
            print(f"     • Behaviors Extracted: {result.get('extracted_count', 0)}")
            print(f"     • Behaviors Stored:    {result.get('stored_count', 0)}")
            print(f"     • Behaviors Verified:  {result.get('verified_count', 0)}")
            print(f"\n  💡 You can now check the database to see the stored behaviors!")
            print("=" * 100 + "\n")
            return 0
        elif result.get("success") and result.get("stored_count", 0) == 0:
            print("\n⚠️  No behaviors were stored (all filtered or extraction failed)")
            return 1
        else:
            print(f"\n❌ DEMO FAILED: {result.get('error', 'Unknown error')}\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user\n")
        return 130
        
    except Exception as e:
        logger.exception("Demo failed with unexpected error")
        print(f"\n❌ DEMO FAILED: {str(e)}\n")
        return 1


if __name__ == "__main__":
    exit(main())
