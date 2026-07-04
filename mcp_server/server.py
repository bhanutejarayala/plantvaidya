"""
PlantVaidya MCP Server
========================
Exposes a static tomato-disease knowledge base as MCP tools so that
the Advisory Agent can look up treatment plans without hardcoding
domain knowledge inside the agent/model itself.

Run standalone for testing:
    python mcp_server/server.py

The Advisory Agent (agents/advisory_agent.py) connects to this server
over stdio using google-adk's MCPToolset + StdioConnectionParams.

Tools exposed:
    - list_diseases()                -> list of all known disease keys
    - get_treatment(disease_name)     -> full treatment/prevention record
    - check_dosage_safety(disease_name, proposed_dosage_g_per_litre)
                                       -> safety check against max safe dosage
"""

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

with open(KB_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE: dict[str, Any] = json.load(f)

mcp = FastMCP("plantvaidya-knowledge-base")


@mcp.tool()
def list_diseases() -> list[str]:
    """Return the list of tomato disease keys this knowledge base covers."""
    return list(KNOWLEDGE_BASE.keys())


@mcp.tool()
def get_treatment(disease_name: str) -> dict[str, Any]:
    """
    Look up the treatment plan for a diagnosed tomato disease.

    Args:
        disease_name: The disease class label as output by the diagnosis
            model (e.g. "Tomato_Early_blight"). Matching is case-insensitive
            and tolerant of spaces vs underscores.

    Returns:
        A dict with common_name, cause, severity, symptoms, treatment steps,
        prevention steps, and a max safe copper-fungicide dosage used by the
        security guardrail. Returns an error dict if the disease is unknown.
    """
    normalized = disease_name.strip().replace(" ", "_")
    for key in KNOWLEDGE_BASE:
        if key.lower() == normalized.lower():
            return {"disease_key": key, **KNOWLEDGE_BASE[key]}

    # fuzzy fallback: substring match
    for key in KNOWLEDGE_BASE:
        if normalized.lower() in key.lower() or key.lower() in normalized.lower():
            return {"disease_key": key, **KNOWLEDGE_BASE[key]}

    return {
        "error": f"No record found for '{disease_name}'.",
        "available_diseases": list(KNOWLEDGE_BASE.keys()),
    }


@mcp.tool()
def check_dosage_safety(disease_name: str, proposed_dosage_g_per_litre: float) -> dict[str, Any]:
    """
    Security/safety guardrail tool: validate a proposed chemical dosage
    against the knowledge base's maximum safe dosage before it is ever
    surfaced to a farmer.

    Args:
        disease_name: The disease being treated.
        proposed_dosage_g_per_litre: Dosage the Advisory Agent wants to recommend.

    Returns:
        A dict indicating whether the dosage is safe, and the capped/safe
        value to use if it exceeds the maximum.
    """
    record = get_treatment(disease_name)
    if "error" in record:
        return {"safe": False, "reason": "Unknown disease; cannot verify dosage."}

    max_safe = record.get("max_safe_copper_dosage_g_per_litre", 0)
    if max_safe == 0:
        return {
            "safe": proposed_dosage_g_per_litre == 0,
            "reason": "This condition has no approved chemical dosage (viral/pest issue).",
            "recommended_dosage_g_per_litre": 0,
        }

    is_safe = proposed_dosage_g_per_litre <= max_safe
    return {
        "safe": is_safe,
        "max_safe_dosage_g_per_litre": max_safe,
        "recommended_dosage_g_per_litre": min(proposed_dosage_g_per_litre, max_safe),
        "reason": "Within safe limits." if is_safe else "Exceeds safe limit; capped to maximum.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
