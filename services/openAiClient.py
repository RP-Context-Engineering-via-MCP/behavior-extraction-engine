import json
from time import time
from typing import Any, Dict, List
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from config.configurations import(
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT_MODEL,
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

# Local embedding model — all-MiniLM-L6-v2 (384 dimensions)
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

MAX_PROMPT_LENGTH = 8000 
MIN_PROMPT_LENGTH = 3

def extract_behavior(prompt: str) -> Dict[str, Any]:
    
    if not prompt or not prompt.strip():
        return {
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": "Prompt cannot be empty",
            "metadata": {"prompt_length": 0, "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    prompt = prompt.strip()
    
    if len(prompt) < MIN_PROMPT_LENGTH:
        return {
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": f"Prompt too short (min {MIN_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }
    

    system_prompt="""You are a behavior canonicalization engine. Your task is to extract ONLY long-term, 
    reusable user behaviors and represent them in a normalized, machine-reasonable form. A behavior MUST be stable across time. 
    Do NOT extract temporary states or one-time requests.
    ---
    FOR EACH BEHAVIOR, YOU MUST PRODUCE A CANONICAL FORM WITH THESE FIELDS:
    
    1. intent (choose ONE that best fits - ordered by precedence):
       - CONSTRAINT → Hard rules: cannot, must not, avoids, allergic to, restricted from, forbidden, never
         ⚠️ CONSTRAINT behaviors are HARD RULES that can override other intent types!
         Examples: "never use eval()", "cannot eat gluten", "must not work weekends"
       - PREFERENCE → Soft desires: likes, prefers, enjoys, favors, interested in
       - HABIT → Frequency patterns: usually, always, regularly, tends to, routinely
       - SKILL → Capabilities: experienced with, proficient in, knows, capable of, expert in
       - COMMUNICATION → Interaction style: prefers brief answers, wants examples, needs context
    
    2. target (CRITICAL - must be CONCISE, NOUN-LIKE, and CANONICALLY NAMED):
       ✓ GOOD: "Python", "dark mode", "spicy food", "morning exercise"
       ✗ BAD: "writing CSS directly", "using Python for backend", "eating spicy food always"
       
       CANONICALIZATION RULES (VERY IMPORTANT):
       - Always use the FULL, STANDARD, MOST WIDELY RECOGNIZED name
       - NEVER use abbreviations, acronyms, or shorthand for the target
       - Convert all variations to the canonical form:
         
         Programming Languages & Technologies:
         * JS, js → "JavaScript"
         * TS, ts → "TypeScript"  
         * PY, py → "Python"
         * C# → "C Sharp"
         * CPP, cpp, C++ → "C Plus Plus"
         * RB, rb → "Ruby"
         * Go, golang → "Go"
         * K8s, k8 → "Kubernetes"
         * DB, db → "database"
         * SQL, sql → "SQL"
         * NoSQL, nosql → "NoSQL"
         * API, api → "API"
         * REST, rest → "REST API"
         * GraphQL, gql → "GraphQL"
         * HTML, html → "HTML"
         * CSS, css → "CSS"
         * SCSS, scss → "SCSS"
         
         General:
         * TDD, tdd → "test-driven development"
         * OOP, oop → "object-oriented programming"
         * FP, fp → "functional programming"
         * CI/CD, cicd → "CI/CD"
         * PR, pr (code context) → "pull request"
         * WFH, wfh → "remote work"
         * AM, am → "morning"
         * PM, pm → "afternoon"
       
       - Extract the CORE NOUN or noun phrase (1-3 words maximum)
       - Remove verbs, articles, and modifiers
       - Use lowercase for common nouns, proper case for proper nouns
    
    3. context (optional scope where behavior applies):
       - Programming: "IDE", "frontend", "backend", "testing", "code review", "debugging"
       - Work: "work", "meetings", "presentations", "team collaboration"
       - Time: "morning", "night", "weekends", "weekdays"
       - Environment: "home", "office", "gym", "outdoors"
       - If no specific context, use "general"
       - DO NOT invent context - only extract if explicitly mentioned
    
    4. polarity (behavioral direction):
       - POSITIVE → likes, prefers, wants, enjoys, seeks, uses, enables
       - NEGATIVE → dislikes, avoids, cannot, restricts, rejects, disables, never
    
    5. confidence, clarity, linguistic_strength (all 0.0-1.0):
       - confidence: How certain you are this is a stable behavior (not a question or temporary state)
       - clarity: How clear and unambiguous the statement is
       - linguistic_strength: How strongly the user expresses this behavior.
         This is the DOMINANT signal for credibility — score it carefully.
         * Maximum (0.85-1.00): "absolutely", "always", "never", "must", "cannot",
                                "definitely", "without exception", "I refuse to"
         * Strong   (0.65-0.85): "strongly prefer", "really love", "hate", "I will not"
         * Normal   (0.45-0.65): "prefer", "like", "use", "do" (no modifier)
         * Mild     (0.30-0.45): "tend to", "usually", "generally", "often"
         * Weak     (0.15-0.30): "sometimes", "occasionally", "kind of", "sort of",
                                 "somewhat", "when I remember"
         * Very weak (0.00-0.20): "might", "maybe", "could", "possibly", "perhaps",
                                  "would like to", "hope to", "someday",
                                  "at some point", "if I have time"

         ⚠️ Conditional capability statements such as "I can use X if required",
            "I'm able to work with Y if needed", "I could do Z if it's required"
            score below 0.20 — they describe fallback ability, not a preference.
         ⚠️ CONSTRAINT intent should always have linguistic_strength ≥ 0.85.
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
              "linguistic_strength": 0.55
            }
          ]
        }
      ],
      "profile_signals": {
        "intents": {"PROBLEM_SOLVING": 0.85, "LEARNING": 0.40},
        "interests": {"PROGRAMMING": 0.80},
        "behavior_level": "INTERMEDIATE",
        "signals": {"CODE_FOCUSED": 0.75},
        "complexity": 0.65,
        "consistency": 0.5
      }
    }
    
    CRITICAL RULES:
    - Target must be CONCISE (1-3 words) - the noun, not the whole phrase
    - Target must use CANONICAL/FULL form - NEVER abbreviations (JavaScript not JS)
    - Use field name "linguistic_strength" (NOT "strength")
    - If no stable behavior exists, return empty behaviors list
    - Do NOT invent context if not mentioned
    - Do NOT include extra fields or explanations
    - All scores must be between 0.0 and 1.0
    - CONSTRAINT behaviors represent hard rules and should have high linguistic_strength
    - profile_signals MUST always be included with at least one intent and one interest
    
    ⚠️ COMPARATIVE STATEMENTS: For "X over Y" or "X instead of Y" statements:
    - Extract ONLY the PREFERRED option (X) with POSITIVE polarity
    - Do NOT extract the rejected option (Y) as a separate behavior
    - Examples:
      * "I prefer TypeScript over JavaScript" → Extract ONLY TypeScript POSITIVE
      * "I like Angular instead of React" → Extract ONLY Angular POSITIVE
      * "I don't like React, prefer Angular" → Extract ONLY Angular POSITIVE
    - This prevents creating multiple conflicting behaviors from a single preference statement
    
    ⚠️ SLEEP SCHEDULE CONFLICTS: Behaviors about sleep timing and daily rhythm patterns
    (night owl, morning person, early riser, staying up late, etc.) MUST use the unified
    target "sleep schedule" — regardless of wording — so the system can detect when two
    opposing sleep patterns exist for the same user.
    Use POLARITY to encode the direction of the pattern:
      - POSITIVE → night-skewed pattern: stays up late, night owl, up past midnight, goes to bed after midnight
      - NEGATIVE → morning-skewed pattern: morning person, early bird, in bed by X pm, must sleep early
    Context should be "general" (do not use "morning" or "night" as context for sleep schedule — these are the polarity).

    Examples:
      Input:  "I'm a night owl — I stay up past midnight every day"
      Output: {intent: "HABIT", target: "sleep schedule", context: "general", polarity: "POSITIVE", linguistic_strength: 0.85}

      Input:  "I've become a morning person — I'm in bed by 10pm and up at 5am"
      Output: {intent: "CONSTRAINT", target: "sleep schedule", context: "general", polarity: "NEGATIVE", linguistic_strength: 0.85}

      Input:  "My sleep specialist told me I must be asleep by 10pm"
      Output: {intent: "CONSTRAINT", target: "sleep schedule", context: "general", polarity: "NEGATIVE", linguistic_strength: 0.9}

    This ensures "night owl" (POSITIVE) and "morning person" (NEGATIVE) share the same
    target so the conflict detection pipeline can recognize the contradiction.

    MULTI-DOMAIN EXAMPLES:

    Input: "I'm vegetarian and cannot eat meat"
    Output: {"intent": "CONSTRAINT", "target": "meat", "context": "general", "polarity": "NEGATIVE", "linguistic_strength": 0.9}

    Input: "I prefer working from home in the mornings"
    Output: {"intent": "PREFERENCE", "target": "remote work", "context": "morning", "polarity": "POSITIVE", "linguistic_strength": 0.7}

    Input: "I always do yoga before breakfast"
    Output: {"intent": "HABIT", "target": "yoga", "context": "morning", "polarity": "POSITIVE", "linguistic_strength": 0.85}
    
    Input: "I'm experienced with AWS cloud infrastructure"
    Output: {"intent": "SKILL", "target": "AWS", "context": "cloud infrastructure", "polarity": "POSITIVE", "linguistic_strength": 0.75}
    
    Input: "I like JS for frontend development"
    Output: {"intent": "PREFERENCE", "target": "JavaScript", "context": "frontend", "polarity": "POSITIVE", "linguistic_strength": 0.65}
    ⚠️ Note: "JS" was normalized to "JavaScript"
    
    Input: "Never use eval() in production code"
    Output: {"intent": "CONSTRAINT", "target": "eval function", "context": "production", "polarity": "NEGATIVE", "linguistic_strength": 0.95}
    ⚠️ Note: "Never" indicates CONSTRAINT with high linguistic_strength
    
    Input: "I prefer TypeScript over JavaScript for frontend"
    Output: {"intent": "PREFERENCE", "target": "TypeScript", "context": "frontend", "polarity": "POSITIVE", "linguistic_strength": 0.7}
    ⚠️ Note: Comparative statement - only extract the PREFERRED option (TypeScript), not the rejected one
    
    Input: "I like Angular instead of React"
    Output: {"intent": "PREFERENCE", "target": "Angular", "context": "general", "polarity": "POSITIVE", "linguistic_strength": 0.65}
    ⚠️ Note: Extract only the preferred choice (Angular)
    
    Input: "Maybe I should try using JavaScript for backend"
    Output: {"intent": "PREFERENCE", "target": "JavaScript", "context": "backend", "polarity": "POSITIVE", "confidence": 0.40, "clarity": 0.45, "linguistic_strength": 0.10}
    ⚠️ Note: Hedged tentative statement — linguistic_strength near minimum (very-weak band).

---
TASK 2: PROFILE SIGNAL EXTRACTION (for Profile Service)
---
Additionally, analyze the ENTIRE prompt to extract "profile_signals" — a holistic view of the user's
behavioral patterns for profile matching. This uses DIFFERENT vocabularies than canonical behaviors.

PROFILE SIGNAL VOCABULARIES:

intents (user's PURPOSE - select all that apply with confidence 0.0-1.0):
  - LEARNING → User wants to understand a concept deeply
  - TASK_COMPLETION → User wants something done, result-focused
  - PROBLEM_SOLVING → User is debugging or solving technical issues
  - EXPLORATION → User is brainstorming or generating ideas
  - GUIDANCE → User is seeking advice or personal direction
  - ENGAGEMENT → Casual, fun, low-commitment interaction

interests (topic AREAS user engages with - select all that apply with confidence 0.0-1.0):
  - AI → Artificial intelligence, machine learning
  - DATA_SCIENCE → Data analysis, statistics
  - WRITING → Drafting, editing, summarizing
  - PROGRAMMING → Coding, debugging, algorithms
  - CREATIVE → Stories, scripts, ideation
  - HEALTH → Well-being, diet, exercise
  - PERSONAL_GROWTH → Career, life guidance
  - ENTERTAINMENT → Games, quizzes, leisure

behavior_level (one of):
  - BEGINNER → Simple questions, needs explanation
  - INTERMEDIATE → Knows basics, building skills
  - ADVANCED → Expert-level, sophisticated queries

signals (preferred RESPONSE style - select all that apply with confidence 0.0-1.0):
  - DEEP_REASONING → Open-ended curiosity-driven queries
  - DETAILED_EXPLANATION → User wants thorough explanation
  - CODE_FOCUSED → Response expected to contain code
  - STEP_BY_STEP → User wants progressive breakdown
  - QUICK_ANSWER → User wants concise response
  - CREATIVE_OUTPUT → User wants generated creative content
  - EMPATHETIC_RESPONSE → User needs emotional tone
  - ITERATIVE_REFINEMENT → User expects multiple turn refinement

complexity (float 0.0-1.0):
  Prompt complexity based on length, constraints, multi-step nature, and technical depth.

consistency (float 0.0-1.0):
  Set to 0.5 as default (actual consistency is calculated across sessions).

PROFILE_SIGNALS OUTPUT (include in JSON response):
"profile_signals": {
    "intents": {"PROBLEM_SOLVING": 0.85, "LEARNING": 0.40},
    "interests": {"PROGRAMMING": 0.80, "AI": 0.45},
    "behavior_level": "ADVANCED",
    "signals": {"CODE_FOCUSED": 0.75, "DETAILED_EXPLANATION": 0.60},
    "complexity": 0.78,
    "consistency": 0.5
}

⚠️ RULES FOR PROFILE_SIGNALS:
- Must include at least one intent and one interest
- All confidence scores must be 0.0-1.0
- behavior_level must be exactly one of: BEGINNER, INTERMEDIATE, ADVANCED
- Profile signals are extracted from the ENTIRE prompt, not per-segment
"""
    start_time = time()
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, 
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
                "profile_signals": None,
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
            "segments": result.get("segments", []),
            "profile_signals": result.get("profile_signals"),
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
            "profile_signals": None,
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
            "profile_signals": None,
            "success": False,
            "error": f"{error_type}: {str(e)}",
            "metadata": {
                "prompt_length": len(prompt),
                "extraction_time_ms": round(extraction_time_ms, 2),
                "tokens_used": 0
            }
        }


def extract_behavior_with_history(prompt: str, recent_history: List[dict]) -> Dict[str, Any]:
    """
    Extract behaviors from a prompt AND enrich it to a standalone query using conversation history.
    
    This performs two simultaneous tasks:
    1. Contextual Query Rewriting - converts prompt to standalone query for similarity search
    2. Behavior Canonicalization - extracts long-term behaviors
    
    Args:
        prompt: User's latest prompt (may contain references like "it", "that", "the above")
        recent_history: List of recent conversation messages [{"role": "user"/"assistant", "text": "..."}]
        
    Returns:
        dict with:
            - standalone_query: str - The enriched standalone version of the prompt
            - segments: list - Extracted behavior segments
            - profile_signals: dict - Profile signals for Profile Service
            - success: bool
            - error: Optional[str]
            - metadata: dict with timing and token info
    """
    if not prompt or not prompt.strip():
        return {
            "standalone_query": None,
            "standalone_queries": None,
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": "Prompt cannot be empty",
            "metadata": {"prompt_length": 0, "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    prompt = prompt.strip()
    
    if len(prompt) < MIN_PROMPT_LENGTH:
        return {
            "standalone_query": None,
            "standalone_queries": None,
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": f"Prompt too short (min {MIN_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "standalone_query": None,
            "standalone_queries": None,
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)",
            "metadata": {"prompt_length": len(prompt), "extraction_time_ms": 0, "tokens_used": 0}
        }

    system_prompt = """You are performing THREE tasks on user input:
    
    ⚠️ CRITICAL: Extract behaviors ONLY from 'LATEST PROMPT', NOT from 'RECENT HISTORY'!
    Recent history is ONLY for query rewriting (TASK 2), NOT for behavior extraction!
    
    TASK 1: BEHAVIOR EXTRACTION FROM LATEST PROMPT ONLY
    ---
    Extract ONLY long-term, reusable user behaviors from the 'LATEST PROMPT' and represent them in a normalized, machine-reasonable form.
    
    ⚠️ STRICT RULE: Do NOT extract behaviors from 'RECENT HISTORY'! 
    - History messages have ALREADY been processed by the system
    - Extracting from history causes false reinforcement of existing behaviors
    - ONLY analyze the 'LATEST PROMPT' for NEW behaviors
    
    A behavior MUST be stable across time. 
    Do NOT extract temporary states, questions, or one-time requests.
    
    Examples of what NOT to extract:
    - Questions: "which is better?", "what do you recommend?"
    - Requests: "tell me about...", "explain..."
    - Temporary states: "I'm hungry right now", "currently working on..."
    
    TASK 2: SEMANTIC PROBE GENERATION (Multi-Probe HyDE Transformation)
    ---
    Look at 'RECENT HISTORY' and 'LATEST PROMPT'. You must convert the user's intent into a SET of 'Semantic Search Probes' to be used as `standalone_queries`.

    ⚠️ MULTI-PROBE REQUIREMENT (CRITICAL — fixes abstract↔concrete asymmetry):
    A single probe cannot cover broad/multi-faceted queries (e.g., "how do I deploy this?"
    spans Docker, Kubernetes, cloud, CI/CD).  Emit 1 to 3 probes that together cover the
    DIFFERENT FACETS of what the user is asking about.  Each probe should be a separate
    short behavioral statement targeting one concrete sub-topic.

    Rules for EACH probe:
    - Must NOT be a question.  Must NOT be a full sentence.
    - Must be a short declarative behavior segment (action verb + concrete noun).
    - Must use CONCRETE domain vocabulary that plausibly describes stored behaviors.
    - Must NOT use filler words: "specific", "certain", "particular", "various", "some".
    - Each probe should target a DIFFERENT facet of the query (don't emit near-duplicates).

    How many probes:
    - Narrow query (1 facet)         → 1 probe
    - Multi-faceted query (2 facets) → 2 probes
    - Broad query (3+ facets)        → 3 probes (hard maximum)

    Also emit `standalone_query` (singular) — set it equal to the FIRST probe in the list
    for backwards compatibility with logging.

    Examples of Multi-Probe Sets:
    - User: "Which Python web framework should I use for a new API project?"
        standalone_queries: [
          "prefers Python for backend development",
          "uses FastAPI for REST API"
        ]
    - User: "How should I configure the appearance settings of my IDE?"
        standalone_queries: [
          "prefers dark mode in IDE",
          "uses high contrast themes for accessibility",
          "dislikes bright white backgrounds"
        ]
    - User: "What quality assurance practices should I follow for this new feature?"
        standalone_queries: [
          "writes unit tests for code",
          "practices test-driven development",
          "performs code reviews before merging"
        ]
    - User: "I need to build a modern web frontend for a new project"
        standalone_queries: [
          "uses React for frontend applications",
          "prefers TypeScript for JavaScript projects",
          "uses Tailwind CSS for styling"
        ]
    - User: "How should I set up the deployment pipeline for this new microservice?"
        standalone_queries: [
          "uses Docker and Kubernetes for deployment",
          "uses Terraform for infrastructure as code",
          "prefers AWS for cloud deployments"
        ]
    - User: "What should I consider when planning my weekly meals?"
        standalone_queries: [
          "avoids meat and dairy in diet",
          "prefers organic and locally sourced food"
        ]
    - User: "How do I make my technical knowledge more accessible to teammates?"
        standalone_queries: [
          "prefers written documentation over verbal explanations",
          "adds comments to complex code sections",
          "maintains personal knowledge base"
        ]

    ❌ BAD probes (abstract filler words — will FAIL vector search):
    - "has specific food preferences" ← "specific" matches nothing
    - "selects TV entertainment" ← too compressed, no domain vocabulary
    - "uses specific music genre for focus" ← "specific genre" adds nothing
    - "has device and app preferences" ← no concrete terms like dark mode, cloud, settings
    - "shops electronics with specific method" ← "specific method" is empty

    ❌ BAD multi-probe sets (probes are near-duplicates — wastes a probe slot):
    - ["uses Python for backend", "prefers Python for server-side"]  ← same facet, different words
    - ["writes unit tests", "writes tests for code"]                 ← same facet
    ---
    FOR EACH BEHAVIOR, YOU MUST PRODUCE A CANONICAL FORM WITH THESE FIELDS:
    
    1. intent (choose ONE that best fits - ordered by precedence):
       - CONSTRAINT → Hard rules: cannot, must not, avoids, allergic to, restricted from, forbidden, never
         ⚠️ CONSTRAINT behaviors are HARD RULES that can override other intent types!
         Examples: "never use eval()", "cannot eat gluten", "must not work weekends"
       - PREFERENCE → Soft desires: likes, prefers, enjoys, favors, interested in
       - HABIT → Frequency patterns: usually, always, regularly, tends to, routinely
       - SKILL → Capabilities: experienced with, proficient in, knows, capable of, expert in
       - COMMUNICATION → Interaction style: prefers brief answers, wants examples, needs context
    
    2. target (CRITICAL - must be CONCISE, NOUN-LIKE, and CANONICALLY NAMED):
       ✓ GOOD: "Python", "dark mode", "spicy food", "morning exercise", "lactose"
       ✗ BAD: "writing CSS directly", "using Python for backend", "eating spicy food always", "lactose intolerance"
       
       CANONICALIZATION RULES (VERY IMPORTANT):
       - Always use the FULL, STANDARD, MOST WIDELY RECOGNIZED name
       - NEVER use abbreviations, acronyms, or shorthand for the target
       - Convert all variations to the canonical form:
         
         Programming Languages & Technologies:
         * JS, js → "JavaScript"
         * TS, ts → "TypeScript"  
         * PY, py → "Python"
         * C# → "C Sharp"
         * CPP, cpp, C++ → "C Plus Plus"
         * RB, rb → "Ruby"
         * Go, golang → "Go"
         * K8s, k8 → "Kubernetes"
         * DB, db → "database"
         * SQL, sql → "SQL"
         * NoSQL, nosql → "NoSQL"
         * API, api → "API"
         * REST, rest → "REST API"
         * GraphQL, gql → "GraphQL"
         * HTML, html → "HTML"
         * CSS, css → "CSS"
         * SCSS, scss → "SCSS"
         
         General:
         * TDD, tdd → "test-driven development"
         * OOP, oop → "object-oriented programming"
         * FP, fp → "functional programming"
         * CI/CD, cicd → "CI/CD"
         * PR, pr (code context) → "pull request"
         * WFH, wfh → "remote work"
         * AM, am → "morning"
         * PM, pm → "afternoon"
       
       - Extract the CORE NOUN or noun phrase (1-3 words maximum)
       - Remove verbs, articles, and modifiers
       - Use lowercase for common nouns, proper case for proper nouns
    
    3. context (optional scope where behavior applies):
       - Programming: "IDE", "frontend", "backend", "testing", "code review", "debugging"
       - Work: "work", "meetings", "presentations", "team collaboration"
       - Time: "morning", "night", "weekends", "weekdays"
       - Environment: "home", "office", "gym", "outdoors"
       - If no specific context, use "general"
       - DO NOT invent context - only extract if explicitly mentioned
    
    4. polarity (behavioral direction):
       - POSITIVE → likes, prefers, wants, enjoys, seeks, uses, enables
       - NEGATIVE → dislikes, avoids, cannot, restricts, rejects, disables, never
    
    5. confidence, clarity, linguistic_strength (all 0.0-1.0):
       - confidence: How certain you are this is a stable behavior (not a question or temporary state)
       - clarity: How clear and unambiguous the statement is
       - linguistic_strength: How strongly the user expresses this behavior.
         This is the DOMINANT signal for credibility — score it carefully.
         * Maximum (0.85-1.00): "absolutely", "always", "never", "must", "cannot",
                                "definitely", "without exception", "I refuse to"
         * Strong   (0.65-0.85): "strongly prefer", "really love", "hate", "I will not"
         * Normal   (0.45-0.65): "prefer", "like", "use", "do" (no modifier)
         * Mild     (0.30-0.45): "tend to", "usually", "generally", "often"
         * Weak     (0.15-0.30): "sometimes", "occasionally", "kind of", "sort of",
                                 "somewhat", "when I remember"
         * Very weak (0.00-0.20): "might", "maybe", "could", "possibly", "perhaps",
                                  "would like to", "hope to", "someday",
                                  "at some point", "if I have time"

         ⚠️ Conditional capability statements such as "I can use X if required",
            "I'm able to work with Y if needed", "I could do Z if it's required"
            score below 0.20 — they describe fallback ability, not a preference.
         ⚠️ CONSTRAINT intent should always have linguistic_strength ≥ 0.85.
    ---
    OUTPUT FORMAT (STRICT JSON - use these EXACT field names):

    {
      "standalone_query": "The first probe — kept as a string for backwards compatibility (e.g., 'prefers Python for backend')",
      "standalone_queries": [
        "prefers Python for backend development",
        "uses FastAPI for REST API"
      ],
      "required_intents": ["CONSTRAINT", "PREFERENCE"],
      "segments": [
        {
          "text": "original segment text FROM LATEST PROMPT ONLY",
          "behaviors": [
            {
              "description": "concise human-readable summary (e.g., 'prefers Python for backend')",
              "intent": "PREFERENCE",
              "target": "Python",
              "context": "backend",
              "polarity": "POSITIVE",
              "confidence": 0.92,
              "clarity": 0.88,
              "linguistic_strength": 0.55
            }
          ]
        }
      ],
      "profile_signals": {
        "intents": {"PROBLEM_SOLVING": 0.85, "LEARNING": 0.40},
        "interests": {"PROGRAMMING": 0.80},
        "behavior_level": "INTERMEDIATE",
        "signals": {"CODE_FOCUSED": 0.75},
        "complexity": 0.65,
        "consistency": 0.5
      }
    }
    
    PROFILE_SIGNALS EXTRACTION (TASK 3):
    Additionally, analyze the LATEST PROMPT to extract "profile_signals" — a holistic view of the user's
    behavioral patterns for profile matching. Use these vocabularies:
    
    intents (purpose - 0.0-1.0): LEARNING, TASK_COMPLETION, PROBLEM_SOLVING, EXPLORATION, GUIDANCE, ENGAGEMENT
    interests (topics - 0.0-1.0): AI, DATA_SCIENCE, WRITING, PROGRAMMING, CREATIVE, HEALTH, PERSONAL_GROWTH, ENTERTAINMENT
    behavior_level: BEGINNER | INTERMEDIATE | ADVANCED
    signals (response style - 0.0-1.0): DEEP_REASONING, DETAILED_EXPLANATION, CODE_FOCUSED, STEP_BY_STEP, QUICK_ANSWER, CREATIVE_OUTPUT, EMPATHETIC_RESPONSE, ITERATIVE_REFINEMENT
    complexity: 0.0-1.0 (prompt complexity)
    consistency: 0.5 (default)
    
    REQUIRED_INTENTS RULES:
    - Predict which behavior intent types are RELEVANT to the user's query for retrieval
    - This determines which stored behaviors should be searched
    - Use the standalone_query to decide what intent types matter:
      * Food/diet queries → ["CONSTRAINT", "PREFERENCE"] (allergies + food preferences)
      * Coding queries → ["PREFERENCE", "SKILL", "CONSTRAINT"] (tools, skills, restrictions)
      * Routine/schedule queries → ["HABIT", "CONSTRAINT"] (routines + restrictions)
      * Communication queries → ["COMMUNICATION", "PREFERENCE"]
      * General queries → ["PREFERENCE", "CONSTRAINT"] (safe default)
    - ALWAYS include "CONSTRAINT" — constraints (allergies, restrictions) are safety-critical
    - Return 2-3 intent types maximum
    
    ⚠️ CRITICAL RULES:
    - Extract behaviors ONLY from 'LATEST PROMPT' - NEVER from 'RECENT HISTORY'!
    - If LATEST PROMPT is a question or has no behaviors, return empty segments list []
    - standalone_queries is MANDATORY - 1 to 3 short behavioral probes (action verb + concrete noun), NOT questions.
    - standalone_query (singular, kept for backwards compatibility) MUST equal standalone_queries[0].
    - Each probe MUST use concrete domain vocabulary — NEVER use "specific", "certain", "particular" as they fail vector search.
    - Probes within standalone_queries MUST target DIFFERENT facets of the query (no near-duplicates).
    - Target must be CONCISE (1-3 words) - the noun, not the whole phrase
    - Target must use CANONICAL/FULL form - NEVER abbreviations (JavaScript not JS)
    - Use field name "linguistic_strength" (NOT "strength")
    - If no stable behavior exists in LATEST PROMPT, return empty behaviors list
    - Do NOT invent context if not mentioned
    - Do NOT include extra fields or explanations
    - All scores must be between 0.0 and 1.0
    - CONSTRAINT behaviors represent hard rules and should have high linguistic_strength
    - Do NOT extract background context as behaviors (e.g., "I'm a software engineer" is not a behavior)
    
    ⚠️ COMPARATIVE STATEMENTS: For "X over Y" or "X instead of Y" statements:
    - Extract ONLY the PREFERRED option (X) with POSITIVE polarity
    - Do NOT extract the rejected option (Y) as a separate behavior
    - This prevents creating multiple conflicting behaviors from a single preference statement
    
    ⚠️ CRITICAL EXAMPLE - Behavior Extraction with History:
    
    RECENT HISTORY:
    [USER]: "I like healthy breakfast options like oatmeal and fruits."
    [ASSISTANT]: "Great! Oatmeal with berries, or a banana smoothie are excellent choices."
    
    LATEST PROMPT: "among above what is the sweetest food"
    
    CORRECT Output:
    {
      "standalone_query": "prefers sweet food",
      "standalone_queries": ["prefers sweet food", "enjoys oatmeal and fruits as breakfast"],
      "segments": []
    }

    ⚠️ WHY segments is empty:
    - Latest prompt is a QUESTION, not a behavior statement
    - "I like healthy breakfast..." is in HISTORY and was ALREADY PROCESSED - DO NOT extract it again!

    WRONG Output (DO NOT DO THIS):
    {
      "standalone_query": "Which is the sweetest food among oatmeal and fruits?",
      "standalone_queries": ["Which is the sweetest food..."],
      "segments": [{"text": "I like healthy breakfast options...", "behaviors": [...]}]  ❌ WRONG! Question + history extraction!
    }
    """

    # Build conversation context
    user_content = "RECENT HISTORY:\n"
    if recent_history and len(recent_history) > 0:
        for msg in recent_history:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            user_content += f'[{role.upper()}]: "{text}"\n'
    else:
        user_content += "[No prior history]\n"
    
    user_content += f'\nLATEST PROMPT:\n{prompt}'

    start_time = time()
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0, 
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
                "standalone_query": None,
                "segments": [],
                "profile_signals": None,
                "success": False,
                "error": "Empty response from GPT",
                "metadata": {
                    "prompt_length": len(prompt),
                    "extraction_time_ms": extraction_time_ms,
                    "tokens_used": token_used
                }
            }   
           
        result = json.loads(content)

        from config.configurations import MAX_STANDALONE_QUERIES

        # ----- standalone_queries (multi-probe) ---------------------------------
        # Prefer the new list field; fall back to the legacy single-string field;
        # final fallback is the raw prompt (avoids hard failure on retrieval).
        raw_queries = result.get("standalone_queries")
        standalone_queries: list[str] = []
        if isinstance(raw_queries, list):
            seen = set()
            for q in raw_queries:
                if not isinstance(q, str):
                    continue
                q_clean = q.strip()
                if not q_clean:
                    continue
                # Dedupe case-insensitively to filter out near-misses
                key = q_clean.lower()
                if key in seen:
                    continue
                seen.add(key)
                standalone_queries.append(q_clean)

        legacy_query = result.get("standalone_query")
        if isinstance(legacy_query, str) and legacy_query.strip():
            legacy_clean = legacy_query.strip()
            if legacy_clean.lower() not in {q.lower() for q in standalone_queries}:
                standalone_queries.insert(0, legacy_clean)

        if not standalone_queries:
            logger.warning("LLM did not provide any standalone probe, falling back to raw prompt")
            standalone_queries = [prompt]

        # Cap to MAX_STANDALONE_QUERIES
        standalone_queries = standalone_queries[:MAX_STANDALONE_QUERIES]
        standalone_query = standalone_queries[0]

        # Extract required_intents with safe default
        required_intents = result.get("required_intents", ["PREFERENCE", "CONSTRAINT"])
        # Validate intent values
        valid_intents = {"PREFERENCE", "CONSTRAINT", "HABIT", "SKILL", "COMMUNICATION"}
        required_intents = [i for i in required_intents if i in valid_intents]
        if not required_intents:
            required_intents = ["PREFERENCE", "CONSTRAINT"]

        return {
            "standalone_query": standalone_query,
            "standalone_queries": standalone_queries,
            "required_intents": required_intents,
            "segments": result.get("segments", []),
            "profile_signals": result.get("profile_signals"),
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
            "standalone_query": None,
            "standalone_queries": None,
            "segments": [],
            "profile_signals": None,
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
            "standalone_query": None,
            "standalone_queries": None,
            "segments": [],
            "profile_signals": None,
            "success": False,
            "error": f"{error_type}: {str(e)}",
            "metadata": {
                "prompt_length": len(prompt),
                "extraction_time_ms": round(extraction_time_ms, 2),
                "tokens_used": 0
            }
        }


# text embedding for single text (local MiniLM model — 384 dimensions)
def embed_text(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for embedding")
    
    try:
        embedding = _embedding_model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Embedding error: {str(e)}")
    
    
    
# text embedding for multiple texts for efficiency (local MiniLM model — 384 dimensions)
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
        embeddings = _embedding_model.encode(cleaned_texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]
    
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

INTENT TYPES AND THEIR CONFLICT RULES:
- CONSTRAINT: Hard rules (cannot, never, must not) - can conflict with ANY other intent type
- PREFERENCE: Soft desires (likes, prefers) - typically only conflicts with other PREFERENCE on same target
- HABIT: Frequency patterns (usually, always) - conflicts with other HABIT on same activity
- SKILL: Capabilities - rarely conflicts
- COMMUNICATION: Interaction style - conflicts with other COMMUNICATION styles

⚠️ CRITICAL: CONSTRAINT behaviors are HARD RULES. A CONSTRAINT can override a PREFERENCE!
Example: "never use eval()" (CONSTRAINT) conflicts with "prefers using eval for dynamic code" (PREFERENCE)

CONFLICT: Behaviors directly contradict each other and cannot both be true simultaneously.
⚠️ BE DECISIVE: If behaviors are in the SAME DOMAIN and SAME/SIMILAR CONTEXT with competing preferences, classify as CONFLICT.

Examples of CONFLICT:
- "prefers dark mode" vs "prefers light mode" (same target, opposite preference) → CONFLICT
- "prefers TypeScript for frontend" vs "prefers JavaScript for frontend" (same domain, same context, competing tools) → CONFLICT
- "prefers Python for backend" vs "prefers Ruby for backend" (same domain, same context, competing languages) → CONFLICT
- "never uses Python" (CONSTRAINT) vs "prefers Python for scripting" (PREFERENCE) → CONFLICT (cross-intent)
- "cannot eat gluten" vs "enjoys bread" → CONFLICT
- "vegetarian diet" vs "eats meat regularly" → CONFLICT

COMPATIBLE: Behaviors can coexist without contradiction.
Examples:
- "prefers Python for backend" vs "prefers JavaScript for frontend" (DIFFERENT contexts: backend vs frontend) → COMPATIBLE
- "prefers Python for backend" vs "prefers Python for data analysis" (SAME target, DIFFERENT contexts) → COMPATIBLE
- "likes morning workouts" vs "likes evening reading" (different activities, different times) → COMPATIBLE
- "prefers concise code" vs "prefers detailed comments" (complementary practices) → COMPATIBLE
- "skilled in Python" vs "prefers Python" (same direction, different intents, no conflict) → COMPATIBLE
- "prefers pair programming for complex features" vs "enjoys mentoring junior developers" (related but different practices) → COMPATIBLE

CONTEXT_DEPENDENT: Use this SPARINGLY - only when truly ambiguous.
Examples:
- "prefers working alone" vs "enjoys team collaboration" (might be task-dependent) → CONTEXT_DEPENDENT
- "likes fast food" vs "health-conscious eater" (might be frequency-dependent) → CONTEXT_DEPENDENT

ANALYSIS GUIDELINES (STRICT):
1. Check intent types first - CONSTRAINT vs other intent is more likely to CONFLICT
2. Same domain + Same context + Competing preferences = CONFLICT (be decisive!)
3. Same domain + Different contexts (e.g., backend vs frontend) = COMPATIBLE
4. Temporal contexts are usually compatible (morning vs evening)
5. Context specialization matters: "Python for backend" vs "Python for data analysis" = COMPATIBLE (different use cases)
6. DO NOT use CONTEXT_DEPENDENT for clear domain conflicts - default to CONFLICT for same-domain competing preferences
7. Only use CONTEXT_DEPENDENT when the relationship is genuinely ambiguous and depends on unstated factors

OUTPUT FORMAT (strict JSON):
{
  "conflict_type": "CONFLICT" | "COMPATIBLE" | "CONTEXT_DEPENDENT",
  "explanation": "2-3 sentence explanation of your reasoning",
  "confidence": 0.0-1.0
}

DECISION RULES:
- Same domain + Same context + Competing preferences → CONFLICT (don't overthink it!)
- Different contexts OR different domains → COMPATIBLE
- CONSTRAINT vs any other intent on same target → CONFLICT
- Only use CONTEXT_DEPENDENT if truly ambiguous (rare!)
- When in doubt between CONFLICT and CONTEXT_DEPENDENT for competing preferences → Choose CONFLICT"""

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
        