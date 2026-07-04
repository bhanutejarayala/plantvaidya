"""
No-LLM Smoke Test
====================
Exercises the Diagnosis classifier and the MCP knowledge-base tools
directly, with no Gemini API key required. Useful for quickly verifying
the non-LLM parts of the pipeline (model + MCP server + guardrails) work
before wiring up the full agent pipeline, and for graders who don't want
to provision an API key just to sanity-check the repo.

Usage:
    python demo/no_llm_smoke_test.py path/to/leaf_image.jpg
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.classifier import TomatoDiseaseClassifier
from security.guardrails import scope_guardrail, redact_pii, sanitize_advisory_output

# Import the knowledge base lookup logic directly (bypassing the MCP stdio
# transport, since this script is about testing logic, not transport).
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server"))
import server as kb_server  # noqa: E402


def main(image_path: str):
    print("=" * 60)
    print("1) DIAGNOSIS AGENT LOGIC")
    print("=" * 60)
    clf = TomatoDiseaseClassifier()
    result = clf.predict(image_path)
    print(f"Mode: {result.mode}")
    print(f"Disease: {result.disease}")
    print(f"Confidence: {result.confidence:.2%}")

    print("\n" + "=" * 60)
    print("2) ADVISORY AGENT LOGIC (MCP tools called directly)")
    print("=" * 60)
    treatment = kb_server.get_treatment(result.disease)
    print(json.dumps(treatment, indent=2))

    proposed_dosage = 4.0  # intentionally over-limit, to demonstrate the guardrail
    safety = kb_server.check_dosage_safety(result.disease, proposed_dosage)
    print(f"\nDosage safety check (proposed {proposed_dosage}g/L): {json.dumps(safety, indent=2)}")

    print("\n" + "=" * 60)
    print("3) SECURITY GUARDRAILS")
    print("=" * 60)
    print("On-topic check ('my tomato leaves have spots'):",
          scope_guardrail("my tomato leaves have spots") or "PASSED (on-topic)")
    print("On-topic check ('what's the capital of France'):",
          scope_guardrail("what's the capital of France"))
    print("PII redaction test:", redact_pii("call me at 9876543210 or find me at 17.385000, 78.486700"))

    sanitized = sanitize_advisory_output(
        f"Apply copper spray at {proposed_dosage}g/litre for {treatment.get('common_name', 'this disease')}.",
        kb_max_dosage=treatment.get("max_safe_copper_dosage_g_per_litre"),
        proposed_dosage=proposed_dosage,
    )
    print("Sanitized advisory output:", json.dumps(sanitized, indent=2))

    print("\nAll non-LLM components ran successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo/no_llm_smoke_test.py path/to/leaf_image.jpg")
        sys.exit(1)
    main(sys.argv[1])
