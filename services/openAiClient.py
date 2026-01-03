import json
from time import time
from typing import List
from openai import AzureOpenAI
from config.configurations import(
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT_MODEL,
    EMBED_MODEL,
)
from models.behavior import ConflictAnalysisResult, ConflictAnalysisType
import logging

logger = logging.getLogger(__name__)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    timeout=30
    )

MAX_PROMPT_LENGTH = 8000 
MIN_PROMPT_LENGTH = 3

def extract_behavior(prompt: str) -> dict[str, any]:
    
    if not prompt or not prompt.strip():
        return {
            "segments": [],
            "success": False,
            "error": "Prompt cannot be empty",
            "metadata": {"prompt_length": 0, "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    prompt = prompt.strip()
    
    if len(prompt) < MIN_PROMPT_LENGTH:
        return {
            "segments": [],
            "success": False,
            "error": f"Prompt too short (min {MIN_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "segments": [],
            "success": False,
            "error": f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }
    

    system_prompt="""You are a behavior canonicalization engine. Your task is to extract ONLY long-term, 
    reusable user behaviors and represent them in a normalized, machine-reasonable form. A behavior MUST be stable across time. 
    Do NOT extract temporary states or one-time requests.
    ---
    FOR EACH BEHAVIOR, YOU MUST PRODUCE A CANONICAL FORM WITH THESE FIELDS:
    
    1. intent (choose ONE that best fits):
       - PREFERENCE → likes, prefers, enjoys, favors, interested in
       - CONSTRAINT → cannot, avoids, allergic to, restricted from, forbidden
       - HABIT → usually, always, regularly, tends to, routinely
       - SKILL → experienced with, proficient in, knows, capable of
       - COMMUNICATION → prefers brief answers, wants examples, needs context
    
    2. target (CRITICAL - must be CONCISE and NOUN-LIKE):
       ✓ GOOD: "Python", "dark mode", "spicy food", "morning exercise"
       ✗ BAD: "writing CSS directly", "using Python for backend", "eating spicy food always"
       - Extract the CORE NOUN or noun phrase (1-3 words maximum)
       - Remove verbs, articles, and modifiers
       - Examples across domains:
         * Programming: "Python", "dark mode", "unit tests", "Git", "REST APIs"
         * Food: "spicy food", "dairy", "vegetables", "coffee", "sushi"
         * Work: "remote work", "morning meetings", "email", "Slack", "presentations"
         * Health: "morning exercise", "yoga", "meditation", "early sleep"
         * Entertainment: "jazz music", "sci-fi books", "documentaries", "hiking"
    
    3. context (optional scope where behavior applies):
       - Programming: "IDE", "frontend", "backend", "testing", "code review"
       - Work: "work", "meetings", "presentations", "team collaboration"
       - Time: "morning", "night", "weekends", "weekdays"
       - Environment: "home", "office", "gym", "outdoors"
       - If no specific context, use "general"
       - DO NOT invent context - only extract if explicitly mentioned
    
    4. polarity (behavioral direction):
       - POSITIVE → likes, prefers, wants, enjoys, seeks
       - NEGATIVE → dislikes, avoids, cannot, restricts, rejects
    
    5. confidence, clarity, linguistic_strength (all 0.0-1.0):
       - confidence: How certain you are this is a stable behavior (not a question or temporary state)
       - clarity: How clear and unambiguous the statement is
       - linguistic_strength: Intensity of user's language
         * Strong indicators → 0.8-1.0: "strongly", "always", "never", "absolutely", "definitely"
         * Normal preference → 0.6-0.8: "prefer", "like", "usually", "generally"
         * Mild → 0.4-0.6: "tend to", "somewhat", "kind of", "sometimes"
         * Weak/uncertain → <0.4: "might", "maybe", "could", "possibly"
    ---
    OUTPUT FORMAT (STRICT JSON - use these EXACT field names):
    
    {
      "segments": [
        {
          "text": "original segment text",
          "behaviors": [
            {
              "description": "concise human-readable summary (e.g., 'prefers Python for backend')",
              "intent": "PREFERENCE",
              "target": "Python",
              "context": "backend",
              "polarity": "POSITIVE",
              "confidence": 0.92,
              "clarity": 0.88,
              "linguistic_strength": 0.75
            }
          ]
        }
      ]
    }
    
    CRITICAL RULES:
    - Target must be CONCISE (1-3 words) - the noun, not the whole phrase
    - Use field name "linguistic_strength" (NOT "strength")
    - If no stable behavior exists, return empty behaviors list
    - Do NOT invent context if not mentioned
    - Do NOT include extra fields or explanations
    - All scores must be between 0.0 and 1.0
    
    MULTI-DOMAIN EXAMPLES:
    
    Input: "I'm vegetarian and cannot eat meat"
    Output: {"intent": "CONSTRAINT", "target": "meat", "context": "general", "polarity": "NEGATIVE", "linguistic_strength": 0.9}
    
    Input: "I prefer working from home in the mornings"
    Output: {"intent": "PREFERENCE", "target": "remote work", "context": "morning", "polarity": "POSITIVE", "linguistic_strength": 0.7}
    
    Input: "I always do yoga before breakfast"
    Output: {"intent": "HABIT", "target": "yoga", "context": "morning", "polarity": "POSITIVE", "linguistic_strength": 0.85}
    
    Input: "I'm experienced with AWS cloud infrastructure"
    Output: {"intent": "SKILL", "target": "AWS", "context": "cloud infrastructure", "polarity": "POSITIVE", "linguistic_strength": 0.75}
"""
    start_time = time()
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
            max_tokens=5500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"}
            )
        extraction_time_ms = int((time() - start_time) * 1000)
        token_used = response.usage.total_tokens if response.usage else 0
        content = response.choices[0].message.content

        if not content:
            return {
                "segments": [],
                "success": False,
                "error": "Empty response from GPT",
                "metadata": {
                    "prompt_length": len(prompt),
                    "extraction_time_ms": extraction_time_ms,
                    "tokens_used": token_used
                }
            }   
           
        result = json.loads(content)

        return {
            "segments": result["segments"],
            "success": True,
            "error": None,
            "metadata": {
                "prompt_length": len(prompt),
                "extraction_time_ms": round(extraction_time_ms, 2),
                "tokens_used": token_used
            }
        }
    
    except json.JSONDecodeError as e:
        extraction_time_ms = (time() - start_time) * 1000
        return {
            "segments": [],
            "success": False,
            "error": f"Failed to parse GPT response as JSON: {str(e)}",
            "metadata": {
                "prompt_length": len(prompt),
                "extraction_time_ms": round(extraction_time_ms, 2),
                "tokens_used": 0
            }
        }
    except Exception as e:
        extraction_time_ms = (time() - start_time) * 1000
        error_type = type(e).__name__
        return {
            "segments": [],
            "success": False,
            "error": f"{error_type}: {str(e)}",
            "metadata": {
                "prompt_length": len(prompt),
                "extraction_time_ms": round(extraction_time_ms, 2),
                "tokens_used": 0
            }
        }


# text embedding for singel text
def embed_text(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for embedding")
    
    try:
        response = client.embeddings.create(
            model= EMBED_MODEL,
            input= text
        )

        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Embedding error: {str(e)}")
    
    
    
# text embedding for multiple texts for efficiency
def embed_batch(texts: List[str]) -> List[List[float]]:

    if not texts:
        raise ValueError("Text list cannot be empty")
    
    # Validate all texts
    cleaned_texts = []
    for idx, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Text at index {idx} is empty")
        cleaned_texts.append(text.strip())

    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=cleaned_texts
        )
        # IMPORTANT: Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    
    except Exception as e:
        raise Exception(f"Batch embedding error: {str(e)}")


def analyze_conflict(
    behavior_1_text: str,
    behavior_2_text: str,
    distance: float
) -> ConflictAnalysisResult:
    """
    Use GPT-4 to analyze if two behaviors conflict or are compatible.
    
    This function is called when behaviors are semantically close (distance 0.15-0.40)
    but it's unclear if they contradict each other or can coexist.
    
    Args:
        behavior_1_text: First behavior description
        behavior_2_text: Second behavior description
        distance: Semantic distance between behaviors (0.15-0.40 typically)
        
    Returns:
        ConflictAnalysisResult with conflict_type, explanation, and confidence
        
    Raises:
        Exception: If LLM call fails or response is invalid
        
    Example:
        >>> analyze_conflict("prefers Python for backend", "prefers JavaScript for frontend", 0.22)
        ConflictAnalysisResult(conflict_type=COMPATIBLE, explanation="Different contexts", confidence=0.95)
    """
    if not behavior_1_text or not behavior_2_text:
        raise ValueError("Both behavior texts must be non-empty")
    
    system_prompt = """You are a behavior conflict analyzer. Your task is to determine if two user behaviors conflict, are compatible, or depend on context.

CONFLICT: Behaviors directly contradict each other and cannot both be true simultaneously.
Examples:
- "prefers dark mode" vs "prefers light mode"
- "prefers Python for programming" vs "prefers JavaScript for programming" (same domain, no context)
- "vegetarian diet" vs "eats meat regularly"

COMPATIBLE: Behaviors can coexist without contradiction.
Examples:
- "prefers Python for backend" vs "prefers JavaScript for frontend" (different contexts)
- "likes morning workouts" vs "likes evening reading" (different activities)
- "prefers concise code" vs "prefers detailed comments" (complementary)

CONTEXT_DEPENDENT: Relationship depends on additional context not specified.
Examples:
- "prefers working alone" vs "enjoys team collaboration" (might be task-dependent)
- "likes fast food" vs "health-conscious eater" (might be frequency-dependent)

ANALYSIS GUIDELINES:
1. Consider domain specificity (same domain = more likely to conflict)
2. Consider temporal context (habits at different times can coexist)
3. Consider intensity (strong preferences vs mild likes)
4. Consider scope (general vs specific contexts)
5. Default to COMPATIBLE if behaviors can coexist in any reasonable scenario

OUTPUT FORMAT (strict JSON):
{
  "conflict_type": "CONFLICT" | "COMPATIBLE" | "CONTEXT_DEPENDENT",
  "explanation": "2-3 sentence explanation of your reasoning",
  "confidence": 0.0-1.0
}

Be conservative: only classify as CONFLICT if behaviors genuinely cannot both be true."""

    user_prompt = f"""Analyze these two user behaviors:

Behavior 1: "{behavior_1_text}"
Behavior 2: "{behavior_2_text}"
Semantic distance: {distance:.3f}

Do these behaviors conflict, are they compatible, or is it context-dependent?"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temperature for consistent analysis
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            raise Exception("Empty response from GPT-4")
        
        result = json.loads(content)
        
        # Validate required fields
        if "conflict_type" not in result:
            raise Exception("Missing 'conflict_type' in GPT response")
        if "explanation" not in result:
            raise Exception("Missing 'explanation' in GPT response")
        if "confidence" not in result:
            raise Exception("Missing 'confidence' in GPT response")
        
        # Parse and validate conflict type
        conflict_type_str = result["conflict_type"].upper()
        try:
            conflict_type = ConflictAnalysisType(conflict_type_str)
        except ValueError:
            logger.warning(f"Invalid conflict_type '{conflict_type_str}', defaulting to CONTEXT_DEPENDENT")
            conflict_type = ConflictAnalysisType.CONTEXT_DEPENDENT
        
        # Validate confidence is in range
        confidence = float(result["confidence"])
        if not (0.0 <= confidence <= 1.0):
            logger.warning(f"Confidence {confidence} out of range, clamping to [0,1]")
            confidence = max(0.0, min(1.0, confidence))
        
        logger.info(
            f"Conflict analysis: '{behavior_1_text[:50]}...' vs '{behavior_2_text[:50]}...' "
            f"-> {conflict_type.value} (confidence: {confidence:.2f})"
        )
        
        return ConflictAnalysisResult(
            conflict_type=conflict_type,
            explanation=result["explanation"],
            confidence=confidence
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conflict analysis response: {str(e)}")
        raise Exception(f"Invalid JSON response from conflict analyzer: {str(e)}")
    except Exception as e:
        logger.error(f"Conflict analysis failed: {str(e)}")
        raise Exception(f"Conflict analysis error: {str(e)}")
        