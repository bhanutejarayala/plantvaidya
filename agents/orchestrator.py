"""
Orchestrator
=============
Wires Diagnosis -> Advisory -> Farmer-facing agents into a single
SequentialAgent pipeline (ADK's built-in multi-agent workflow primitive).
Each sub-agent writes its output to session state via `output_key`, and
the next sub-agent reads the previous agent's state key from its own
instruction context automatically.

This module exposes `root_agent`, the entry point ADK's CLI/web runner
and the demo script both use.

Demonstrates: Multi-agent system (ADK) - the core required concept.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import SequentialAgent

from agents.diagnosis_agent import diagnosis_agent
from agents.advisory_agent import advisory_agent
from agents.farmer_agent import farmer_agent

root_agent = SequentialAgent(
    name="plantvaidya_pipeline",
    description=(
        "PlantVaidya: a three-agent tomato leaf disease pipeline. "
        "Diagnosis Agent (vision classifier) -> Advisory Agent (MCP knowledge "
        "base lookup) -> Farmer-facing Agent (plain-language + safety guardrails)."
    ),
    sub_agents=[diagnosis_agent, advisory_agent, farmer_agent],
)
