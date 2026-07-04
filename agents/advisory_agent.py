"""
Advisory Agent
================
Second agent in the pipeline. Takes the Diagnosis Agent's output and looks
up a verified treatment plan via the PlantVaidya MCP server (a separate
process exposing the tomato-disease knowledge base as tools), rather than
letting the LLM hallucinate agrochemical advice from parametric memory.

Demonstrates: MCP Server integration, Security features (dosage-safety
tool call is mandatory before any dosage can be recommended).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "mcp_server", "server.py")

plantvaidya_mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python3",
            args=[_SERVER_SCRIPT],
        ),
        timeout=15.0,
    ),
)

advisory_agent = LlmAgent(
    name="advisory_agent",
    model="gemini-2.5-flash",
    description="Looks up verified treatment and dosage-safety information for a diagnosed tomato disease via the PlantVaidya MCP knowledge-base server.",
    instruction=(
        "You are the Advisory Agent for PlantVaidya. You receive a diagnosed "
        "disease label from the Diagnosis Agent (see state key 'diagnosis_result'). "
        "Steps you MUST follow in order:\n"
        "1. Call get_treatment(disease_name) on the MCP server to fetch the verified "
        "treatment plan, cause, symptoms, and prevention steps.\n"
        "2. If the treatment involves a chemical dosage, call "
        "check_dosage_safety(disease_name, proposed_dosage_g_per_litre) to verify it "
        "is within the safe limit BEFORE including it in your answer. Never state a "
        "dosage that has not passed this safety check.\n"
        "3. Produce a structured advisory: disease common name, severity, cause, "
        "treatment steps, prevention steps, and the safety-checked dosage.\n"
        "Never invent treatment information that did not come from the MCP tools."
    ),
    tools=[plantvaidya_mcp_toolset],
    output_key="advisory_result",
)
