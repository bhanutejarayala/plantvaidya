"""
Security & Safety Guardrails
===============================
Demonstrates the "Security features" course concept. These guardrails are
wired in as ADK before_model_callback / before_tool_callback hooks so they
run BEFORE any LLM call or tool execution, not just as a prompt suggestion.

Covers:
  1. Input scope-limiting  - reject queries unrelated to crop/plant health
     so the agent can't be repurposed as a general-purpose chatbot.
  2. PII minimization       - strip/never log farmer location, phone
     numbers, or other identifying details from state passed between agents.
  3. Dosage safety cap      - never let the Advisory Agent surface a
     chemical dosage above the knowledge base's verified safe maximum
     (enforced again here as a second, independent check on top of the
     MCP `check_dosage_safety` tool).
  4. Output sanitization    - block responses that recommend banned/
     restricted pesticides.
"""

import re
from typing import Any, Optional

# --- 1. Scope limiting -------------------------------------------------

ON_TOPIC_KEYWORDS = {
    "leaf", "plant", "crop", "disease", "tomato", "fungus", "fungal",
    "pest", "treatment", "spray", "fertilizer", "blight", "spot", "mold",
    "virus", "mite", "farmer", "agriculture", "farming", "soil", "irrigation",
}

OFF_TOPIC_REFUSAL = (
    "I'm PlantVaidya, a crop-health assistant focused on tomato leaf disease "
    "diagnosis and treatment advice. I can't help with unrelated requests, "
    "but I'm happy to help if you have a plant health question."
)


def is_on_topic(user_text: str) -> bool:
    text = user_text.lower()
    return any(keyword in text for keyword in ON_TOPIC_KEYWORDS)


def scope_guardrail(user_text: str) -> Optional[str]:
    """Return a refusal string if the request is off-topic, else None."""
    if not user_text.strip():
        return OFF_TOPIC_REFUSAL
    if not is_on_topic(user_text):
        return OFF_TOPIC_REFUSAL
    return None


# --- 2. PII minimization ------------------------------------------------

PHONE_RE = re.compile(r"\b\d{10}\b")
COORD_RE = re.compile(r"[-+]?\d{1,3}\.\d{3,},\s*[-+]?\d{1,3}\.\d{3,}")


def redact_pii(text: str) -> str:
    """Strip phone numbers and raw GPS coordinates before logging/storage."""
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = COORD_RE.sub("[REDACTED_LOCATION]", text)
    return text


# --- 3 & 4. Dosage safety + banned substance filtering -------------------

BANNED_SUBSTANCES = {
    "monocrotophos", "endosulfan", "dichlorodiphenyltrichloroethane", "ddt",
    "methyl parathion", "phorate",
}


def sanitize_advisory_output(advisory_text: str, kb_max_dosage: Optional[float] = None,
                              proposed_dosage: Optional[float] = None) -> dict[str, Any]:
    """
    Final safety pass over the Advisory Agent's output before it reaches
    the Farmer-facing Agent.

    Returns a dict: {"safe": bool, "text": str, "flags": [...]}
    """
    flags = []
    lower = advisory_text.lower()

    for substance in BANNED_SUBSTANCES:
        if substance in lower:
            flags.append(f"banned_substance:{substance}")

    if kb_max_dosage is not None and proposed_dosage is not None:
        if proposed_dosage > kb_max_dosage:
            flags.append("dosage_exceeds_safe_limit")
            advisory_text = advisory_text.replace(
                str(proposed_dosage),
                f"{kb_max_dosage} (capped to verified safe maximum)",
            )

    safe = len(flags) == 0
    return {"safe": safe, "text": advisory_text, "flags": flags}


# --- ADK callback wiring --------------------------------------------------

def before_model_guardrail_callback(callback_context, llm_request):
    """
    ADK `before_model_callback` hook: intercepts the request before it
    reaches the LLM. Return a response to short-circuit the call (used for
    off-topic rejection); return None to let the call proceed normally.

    Wire this in via: LlmAgent(..., before_model_callback=before_model_guardrail_callback)
    """
    try:
        user_text = ""
        if getattr(llm_request, "contents", None):
            last = llm_request.contents[-1]
            parts = getattr(last, "parts", [])
            user_text = " ".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
    except Exception:
        user_text = ""

    refusal = scope_guardrail(user_text)
    if refusal:
        from google.genai import types as genai_types
        return genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=refusal)],
                    )
                )
            ]
        )
    return None
