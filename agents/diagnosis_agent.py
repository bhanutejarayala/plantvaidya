"""
Diagnosis Agent
================
First agent in the pipeline. Its only job is to look at a leaf image and
return a disease label + confidence, using the existing fine-tuned
MobileNetV2+CBAM classifier as a tool. It deliberately does NOT give
advice - that's the Advisory Agent's job - keeping each agent single
-responsibility, which is easier to test, debug, and reason about.

Demonstrates: Multi-agent system (ADK), Agent skills (classifier packaged
as a reusable tool/skill any agent in the system can call).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from model.classifier import TomatoDiseaseClassifier

_classifier = TomatoDiseaseClassifier()


def diagnose_leaf_image(image_path: str) -> dict:
    """
    Run the tomato leaf disease classifier on an uploaded image.

    Args:
        image_path: Absolute or relative path to the leaf photo (jpg/png).

    Returns:
        dict with keys: disease (str, model class label), confidence
        (float, 0-1), mode ("real" if trained weights were used, "demo"
        if running with the fallback pseudo-classifier).
    """
    result = _classifier.predict(image_path)
    return {
        "disease": result.disease,
        "confidence": result.confidence,
        "mode": result.mode,
    }


diagnosis_tool = FunctionTool(func=diagnose_leaf_image)

diagnosis_agent = LlmAgent(
    name="diagnosis_agent",
    model="gemini-2.5-flash",
    description="Diagnoses tomato leaf diseases from photos using a fine-tuned MobileNetV2+CBAM classifier.",
    instruction=(
        "You are the Diagnosis Agent for PlantVaidya, a tomato crop-health system. "
        "When given an image path, call the diagnose_leaf_image tool exactly once. "
        "Report back ONLY the structured result: disease label, confidence score, "
        "and inference mode. Do not offer treatment advice - that is handled by "
        "another agent downstream. Keep your response terse and factual."
    ),
    tools=[diagnosis_tool],
    output_key="diagnosis_result",
)
