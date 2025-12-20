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
    

    system_prompt="""You are a behavior extraction assistant. Your task is to identify user behaviors, preferences, constraints, and patterns from natural language.

WHAT TO EXTRACT:
- Personal preferences (e.g., "I like X", "I prefer Y over Z")
- Constraints and limitations (e.g., allergies, technical restrictions)
- Work styles and habits (e.g., "I code late at night")
- Communication preferences (e.g., "keep it brief", "use examples")
- Domain expertise or knowledge level
- Repeated patterns that indicate stable traits

WHAT NOT TO EXTRACT:
- One-time requests ("send this email")
- Questions ("what is X?")
- Temporary states ("I'm tired today")
- Context-specific details that don't reveal lasting patterns

OUTPUT FORMAT:
Break the prompt into semantic segments. For each segment, extract 0 or more behaviors.
Return JSON in this exact structure:
{
  "segments": [
    {
      "text": "the segment text",
      "behaviors": [
        {
          "description": "short behavior description (e.g., 'prefers code examples')",
          "confidence": 0.0-1.0,
          "clarity": 0.0-1.0,
          "linguistic_strength": 0.0-1.0
        }
      ]
    }
  ]
}
RULES:
- Each segment should be a meaningful chunk of the prompt (sentence or clause)
- Extract ONLY stable, reusable behaviors (not temporary requests)
- Confidence: How certain this is a real, stable behavior (not temporary) 0.0-1.0
- Clarity: How clearly/explicitly the behavior was stated 0.0-1.0
- LINGUISTIC STRENGTH GUIDELINES: Assign linguistic_strength based on the strength of wording:
- Strong commitments (“always”, “definitely”, “I strongly prefer”) → 0.8–1.0
- Consistent tendencies (“usually”, “mostly”, “I prefer”) → 0.6–0.79
- Mild preferences (“I like”, “I enjoy”) → 0.4–0.59
- Weak or uncertain language (“sometimes”, “maybe”, “I think I prefer”) → 0.2–0.39
- Extremely uncertain or hedged language → 0.0–0.19
- If no behaviors detected in a segment, return empty behaviors array
- Return ONLY valid JSON, no markdown or explanations
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
        
        