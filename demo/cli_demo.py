"""
PlantVaidya CLI Demo
=======================
Runs the full Diagnosis -> Advisory -> Farmer-facing pipeline on a single
leaf image and prints each agent's output, so you can see the multi-agent
hand-off happening step by step. This is also what you'd screen-record
for the submission video's demo segment.

Usage:
    export GOOGLE_API_KEY=your_key_here
    python demo/cli_demo.py path/to/leaf_image.jpg

If GOOGLE_API_KEY is not set, run demo/no_llm_smoke_test.py instead, which
exercises the classifier + MCP server directly without needing Gemini.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.orchestrator import root_agent


async def run_pipeline(image_path: str):
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set. Export it first, e.g.:")
        print("  export GOOGLE_API_KEY=your_key_here")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    runner = InMemoryRunner(agent=root_agent, app_name="plantvaidya")
    session = await runner.session_service.create_session(
        app_name="plantvaidya", user_id="demo_farmer"
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Diagnose the tomato leaf image at path: {image_path}")],
    )

    print(f"\n{'='*60}\nPlantVaidya Pipeline Run\n{'='*60}")
    print(f"Input image: {image_path}\n")

    async for event in runner.run_async(
        user_id="demo_farmer", session_id=session.id, new_message=user_message
    ):
        if event.author and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
            if text.strip():
                print(f"\n--- {event.author} ---\n{text.strip()}")

    print(f"\n{'='*60}\nDone.\n{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo/cli_demo.py path/to/leaf_image.jpg")
        sys.exit(1)
    asyncio.run(run_pipeline(sys.argv[1]))
