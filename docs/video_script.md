# PlantVaidya — Video Script (target: 4:30–5:00)

Record with screen share (terminal + a couple of slides/diagram) and your voice.
Suggested tool: OBS Studio or just Loom/Zoom screen record. Upload unlisted or public to YouTube.

---

## 1. Problem statement (0:00–0:45)

**[Face cam or title slide: "PlantVaidya"]**

> "Tomato is one of India's most widely grown vegetable crops, and one of the most disease-prone. Smallholder farmers lose a big share of every harvest to leaf diseases that are treatable — if caught early. But agricultural extension officers are scarce, and when farmers turn to generic AI chatbots for help, the advice can be wrong in ways that actually matter — like a hallucinated pesticide dosage. That's the problem PlantVaidya solves: fast, accurate diagnosis with treatment advice you can actually trust."

## 2. Why agents (0:45–1:30)

**[Show the architecture diagram — docs/architecture.svg]**

> "A single LLM call given a leaf photo and asked 'what do I do' will confidently make things up. So instead of one agent, I built three, each with a narrow job and a different trust model. The Diagnosis Agent only classifies the image — it doesn't give advice. The Advisory Agent is not allowed to state a treatment unless it retrieved it from a separate MCP server holding a verified knowledge base — never from the model's memory. And the Farmer-facing Agent, the only one the user actually sees, carries the security guardrails: it rejects off-topic requests, redacts personal info, and enforces a hard safety cap on any chemical dosage before it reaches the farmer."

## 3. Architecture walkthrough (1:30–2:30)

**[Point at each box in the diagram as you talk]**

> "Here's the flow. A farmer uploads a leaf photo. The Diagnosis Agent wraps a fine-tuned MobileNetV2 with a CBAM attention module — this is existing work I published on tomato leaf classification — as an ADK tool, and returns a disease label and confidence score. That gets handed to the Advisory Agent, which connects over MCP to a standalone knowledge-base server exposing three tools: list diseases, get treatment, and check dosage safety. The Advisory Agent is instructed to call get_treatment before saying anything about the disease, and to call check_dosage_safety before stating any numeric dosage. Finally, the Farmer-facing Agent rewrites all of that into plain language — a one-line diagnosis, a few numbered action steps, and prevention tips — in English or Telugu."

> "All three are chained together with ADK's SequentialAgent, which is the multi-agent orchestration primitive from the course."

## 4. Demo (2:30–4:00)

**[Switch to terminal]**

> "Let me show it running."

```bash
python demo/no_llm_smoke_test.py sample_images/early_blight.jpg
```

> "This exercises the classifier and the MCP knowledge base tools directly — you can see the diagnosis, the full treatment record pulled from the MCP server, and here's the security guardrail in action: I deliberately proposed an over-limit dosage, and you can see check_dosage_safety caps it back down to the verified maximum."

**[Then run, or show a pre-recorded clip of:]**

```bash
export GOOGLE_API_KEY=...
python demo/cli_demo.py sample_images/early_blight.jpg
```

> "And here's the full three-agent pipeline running end to end with real Gemini calls — you can see each agent's output stream in as it hands off to the next: Diagnosis, then Advisory, then the final farmer-facing message."

**[Optionally show an off-topic query being rejected, to demonstrate the guardrail]**

> "And if I ask it something unrelated — say, 'what's the capital of France' — it declines, because the Farmer-facing Agent's guardrail rejects anything outside plant health before the request even reaches the model."

## 5. The build (4:00–4:45)

> "This is built on Google's Agent Development Kit, with a FastMCP server for the knowledge base, running on Gemini 2.5 Flash. The classifier is a fine-tuned MobileNetV2 with CBAM attention, from my earlier published research. Everything's on GitHub with a full README, and the repo runs out of the box in a demo mode — no GPU or API key needed — for anyone grading it, and switches to real model inference the moment you drop in trained weights."

## 6. Close (4:45–5:00)

> "PlantVaidya's real contribution isn't the classifier — that already existed. It's the pattern around it: never letting the LLM be the source of truth for something where being wrong has real consequences, and enforcing that structurally instead of just hoping a prompt holds. Thanks for watching."

---

### Recording checklist
- [ ] Screen record terminal + diagram, 1080p
- [ ] Keep total runtime under 5:00 (hard requirement)
- [ ] Upload to YouTube (public or unlisted — must be publicly viewable, no login wall)
- [ ] Attach the YouTube link + a cover image in the Kaggle Writeup's Media Gallery
