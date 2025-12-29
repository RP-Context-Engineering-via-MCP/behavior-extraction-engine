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
    

    system_prompt="""You are a behavior extraction assistant. Your task is to identify user behaviors, preferences, constraints, and stable patterns from natural language input. Extract only long-term, reusable behaviors that describe enduring tendencies or preferences.

WHAT TO EXTRACT (stable patterns only):
- Personal preferences (e.g., “I like X”, “I prefer Y over Z”)
- Constraints and limitations (e.g., allergies, technical restrictions)
- Work styles and habits (e.g., “I code late at night”)
- Communication preferences (e.g., “keep it brief”, “use examples”)
- Domain expertise or skill-level indicators
- Repeated or strongly phrased tendencies that imply stable traits

WHAT NOT TO EXTRACT (non-stable or temporary statements):
- One-time requests (“send this email”)
- Questions (“what is X?”)
- Temporary or momentary states (“I’m tired today”)
- Context-specific statements that do not indicate a repeating behavior
- Hypothetical or uncertain statements unless they still imply a preference

SEGMENTATION RULES:
- Break the input prompt into semantic segments (sentences or meaningful clauses).
- Each segment may contain zero, one, or multiple behaviors.
- If no stable behaviors exist in a segment, return an empty behaviors array.

BEHAVIOR FIELDS TO EXTRACT:
For each behavior, extract:
- "description": A concise behavioral summary (e.g., “prefers code examples”)
- "confidence": 0.0–1.0 → How certain it is that the user expressed a stable behavior
- "clarity": 0.0–1.0 → How explicit and unambiguous the behavior is
- "linguistic_strength": 0.0–1.0 → How strong or committed the user’s language is

LINGUISTIC STRENGTH GUIDELINES:
Assign linguistic_strength based on the strength of wording:
- Strong commitments (“always”, “definitely”, “I strongly prefer”) → 0.8–1.0
- Consistent tendencies (“usually”, “mostly”, “I prefer”) → 0.6–0.79
- Mild preferences (“I like”, “I enjoy”) → 0.4–0.59
- Weak or uncertain language (“sometimes”, “maybe”, “I think I prefer”) → 0.2–0.39
- Extremely uncertain or hedged language → 0.0–0.19

OUTPUT FORMAT (strict):
Return JSON ONLY in the following structure:

{
  "segments": [
    {
      "text": "the segment text",
      "behaviors": [
        {
          "description": "short behavior description",
          "confidence": float,
          "clarity": float,
          "linguistic_strength": float
        }
      ]
    }
  ]
}

RULES FOR OUTPUT:
- Follow the JSON structure exactly.
- Do not include explanations or markdown.
- All floating-point numbers must be between 0.0 and 1.0.
- If a segment contains no extractable stable behaviors, return an empty behaviors array.
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
        