# PlantVaidya: A Safety-First Multi-Agent System for Tomato Disease Diagnosis

**Track: Agents for Good**

## The problem

Tomato is one of the most widely grown vegetable crops in India, and it's also one of the most disease-prone. Smallholder farmers routinely lose a significant share of their harvest to fungal, bacterial, and viral leaf diseases that are treatable if caught early — but agricultural extension officers are scarce relative to the number of farms, so diagnostic delay is common. When farmers do turn to informal advice or generic AI chatbots, the results can be actively harmful: a hallucinated pesticide dosage is not a minor inconvenience, it's a food-safety and health risk.

This is a problem agents are unusually well suited to, but only if they're built with the right guardrails. A single LLM call given a photo and asked "what do I do?" will confidently improvise an answer that sounds authoritative and may be wrong. The interesting engineering problem isn't "can an LLM describe tomato diseases" (it can) — it's "can we build a system that never lets the LLM's fluency substitute for verified fact when the fact matters."

## Why agents, specifically

PlantVaidya is deliberately split into three single-responsibility agents rather than one agent with three tools, because each stage has a fundamentally different trust model:

- **Vision is not a language task.** Disease classification from a leaf photo is handled by a trained CNN (MobileNetV2 backbone with a CBAM attention module), not by asking an LLM to describe an image. The Diagnosis Agent's only job is to call this classifier and report the result — it has no authority to add commentary or advice.
- **Domain knowledge should never come from parametric memory.** The Advisory Agent is not allowed to state a treatment or a dosage unless it retrieved it from the PlantVaidya MCP server, a standalone process serving a verified knowledge base. This is enforced through the agent's instructions and structurally through tool-use — the treatment data physically doesn't exist in the agent's context until the MCP tool returns it.
- **The user-facing surface is the highest-risk surface.** The Farmer-facing Agent is the only agent whose text a real person ever reads, so it is where the security guardrails live: an ADK `before_model_callback` hook rejects off-topic requests before they ever reach the LLM, and a post-processing pass redacts PII and re-verifies that no dosage exceeds the safety-checked maximum.

Keeping these concerns in separate agents means each one is independently testable, and a bug in the Farmer-facing Agent's phrasing can't accidentally corrupt the diagnosis logic or vice versa.

## Architecture

The pipeline is a `SequentialAgent` (ADK's built-in multi-agent orchestration primitive) chaining three `LlmAgent`s:

**Farmer uploads a leaf photo → Diagnosis Agent → Advisory Agent → Farmer-facing Agent → Farmer receives plain-language advice**

1. **Diagnosis Agent** wraps the trained classifier as an ADK `FunctionTool`. Given an image path, it calls the model exactly once and reports back a structured result: disease label, confidence score, and inference mode (real weights vs. demo fallback). It does not offer advice.

2. **MCP Server** is a separate process (built with FastMCP) exposing the tomato-disease knowledge base as three tools: `list_diseases`, `get_treatment(disease_name)`, and `check_dosage_safety(disease_name, proposed_dosage)`. The knowledge base covers ten disease classes with cause, symptoms, severity, treatment steps, prevention steps, and a maximum safe copper-fungicide dosage per condition.

3. **Advisory Agent** connects to the MCP server over stdio using ADK's `MCPToolset` + `StdioConnectionParams`. Its instructions require it to call `get_treatment` before saying anything about the disease, and to call `check_dosage_safety` before stating any numeric dosage — this is the second of two independent safety checks (the first being the guardrail's own hard cap, described below).

4. **Farmer-facing Agent** rewrites the technical advisory as a one-line diagnosis summary, 2-4 numbered action steps, and 2-3 prevention tips, avoiding jargon and responding in Telugu if the farmer writes in Telugu. It carries a `before_model_callback` guardrail (`security/guardrails.py`) that intercepts the request before the LLM call: if the user's message doesn't relate to plant/crop health, the agent short-circuits with a scope-refusal instead of letting the model attempt an answer. A second, independent post-processing pass (`postprocess_farmer_response`) redacts phone numbers and raw GPS coordinates, checks for banned-substance mentions, and re-caps any dosage that slipped past the first check.

This "check it twice, in two different layers" pattern for dosage safety is intentional: guardrails that live only in a system prompt are advisory, not enforced. The dosage cap here is enforced once as a tool contract (the MCP `check_dosage_safety` tool) and once as a deterministic Python post-processing step that runs regardless of what the LLM decided to say.

## Implementation details

- **Model**: MobileNetV2 backbone fine-tuned on the standard PlantVillage tomato subset (10 classes), with a CBAM (channel + spatial attention) block inserted to help the network focus on lesion regions rather than background leaf texture. This architecture choice is drawn directly from prior published work on this exact classification task.
- **Demo mode**: because trained model weights are large binary artifacts unsuited to a git repo, `model/classifier.py` ships with a deterministic fallback: if no weights file is present at the configured path, the classifier hashes the input image to produce a stable, plausible prediction so the entire multi-agent pipeline — MCP calls, guardrails, and all — is runnable and gradeable out of the box, with zero setup beyond `pip install -r requirements.txt`. Dropping trained weights into `model/weights/` switches it to real inference with no code changes.
- **MCP server**: rather than hardcoding the knowledge base into the Advisory Agent's system prompt (which would count as "the LLM just knows this," undermining the whole point), the knowledge base is served by an independent process the agent must actively query. This is the literal MCP concept from the course: a tool server the agent connects to, not a static prompt.
- **Testing**: `demo/no_llm_smoke_test.py` exercises the classifier, the MCP tool logic, and the guardrail functions directly — including deliberately proposing an over-limit dosage to verify the safety cap actually engages — without requiring a Gemini API key, so the non-LLM engineering can be verified independently of the agent orchestration layer.

## What I'd build next

Given more time, the natural extensions are: (1) replacing the static knowledge base with a live regional agri-extension API so treatment advice accounts for current weather (e.g. delaying a spray before rain), (2) adding a lightweight Concierge-style reminder agent that follows up with the farmer on a spray schedule, and (3) extending the same three-agent scaffolding to other crops (maize, chili) by swapping only the Diagnosis Agent's underlying classifier and the MCP knowledge base — the Advisory and Farmer-facing agents, and the guardrail architecture, are already crop-agnostic.

## Closing

PlantVaidya's real contribution isn't the tomato classifier — that's existing, published work. It's the pattern around it: treating an LLM as an interface and reasoning layer, never as a source of truth for anything where being wrong has real-world consequences, and enforcing that boundary structurally (tool calls, MCP servers, deterministic post-processing) rather than hoping a system prompt holds.
