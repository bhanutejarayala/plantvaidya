"""
Farmer-facing Agent
=====================
Final agent in the pipeline. Converts the technical advisory into
plain-language guidance a farmer without agronomy training can act on,
and is the ONLY agent whose output the end user ever sees - meaning it
is where the security guardrails are most critical.

Demonstrates: Security features (scope-limiting + PII redaction wired as
an ADK before_model_callback, plus a final output sanitization pass).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import LlmAgent

from security.guardrails import before_model_guardrail_callback, redact_pii, sanitize_advisory_output

farmer_agent = LlmAgent(
    name="farmer_agent",
    model="gemini-2.5-flash",
    description="Translates the technical advisory into simple, actionable guidance for the farmer, in plain English or Telugu on request.",
    instruction=(
        "You are the Farmer-facing Agent for PlantVaidya, the final and only agent "
        "whose reply the farmer sees. You receive the technical advisory from state "
        "key 'advisory_result'. Rewrite it as:\n"
        "1. A one-line plain-language diagnosis summary\n"
        "2. 'What to do now' - 2-4 simple numbered action steps\n"
        "3. 'How to prevent this next season' - 2-3 simple tips\n"
        "Avoid technical jargon (e.g. say 'copper spray' not 'copper oxychloride 50% WP' "
        "unless the farmer asks for exact product details). If the user writes in Telugu, "
        "reply in Telugu. Never discuss anything outside tomato plant health - politely "
        "decline unrelated requests. Never ask for or repeat back the farmer's phone "
        "number, exact GPS location, or other personal details."
    ),
    before_model_callback=before_model_guardrail_callback,
    output_key="farmer_response",
)


def postprocess_farmer_response(raw_text: str, kb_max_dosage: float | None = None,
                                 proposed_dosage: float | None = None) -> str:
    """
    Extra safety net applied in the orchestrator after the LLM call returns,
    on top of the before_model_callback guardrail. Belt-and-suspenders:
    catches anything a model might still surface despite instructions.
    """
    result = sanitize_advisory_output(raw_text, kb_max_dosage, proposed_dosage)
    text = redact_pii(result["text"])
    if not result["safe"]:
        text += "\n\n[Note: some content was adjusted by PlantVaidya's safety filter.]"
    return text
