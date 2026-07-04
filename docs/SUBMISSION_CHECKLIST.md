# Submission Checklist — deadline: July 6, 2026, 11:59 PM PT

## Code (done — in this repo)
- [x] Multi-agent system (ADK SequentialAgent, 3 sub-agents)
- [x] MCP server (FastMCP, 3 tools)
- [x] Security features (scope guardrail, PII redaction, dosage safety cap)
- [x] Agent skills (classifier packaged as a reusable tool)
- [x] README with architecture, setup instructions, diagram
- [x] Runs out of the box in demo mode (no GPU/API key needed to verify)
- [x] No API keys hardcoded anywhere (.env.example only)
- [x] Tested: syntax-checked, smoke-tested, agents import and wire correctly

## What YOU still need to do

1. **Push to GitHub** (public repo)
   - `git init && git add . && git commit -m "PlantVaidya capstone submission"`
   - Create a public GitHub repo, push
   - Double check `.env` is NOT committed (`.gitignore` already excludes it)

2. **(Optional but recommended) Drop in your real model weights**
   - Export your trained MobileNetV2+CBAM tomato model from Kaggle as `.keras`
   - Save it locally as `model/weights/mobilenetv2_cbam_tomato.keras`
   - This flips the classifier from demo mode to real inference — worth doing since you already have this trained model

3. **Get a Gemini API key**
   - https://aistudio.google.com/apikey (free tier is enough for a demo run)
   - `export GOOGLE_API_KEY=...`

4. **Grab 2-3 sample tomato leaf images** to demo with
   - Use images from your own tomato disease dataset/paper if you have them locally
   - Or a few PlantVillage tomato images

5. **Run the demo yourself once** before recording, to make sure it works on your machine:
   ```bash
   pip install -r requirements.txt
   python demo/no_llm_smoke_test.py your_image.jpg
   python demo/cli_demo.py your_image.jpg
   ```

6. **Record the video** (script in `docs/video_script.md`, under 5 minutes, upload to YouTube)

7. **Write the Kaggle Writeup**
   - Draft is in `docs/kaggle_writeup_draft.md` (1,167 words, well under the 2,500 limit)
   - Copy it into a new Kaggle Writeup, select "Agents for Good" track
   - Attach: cover image (screenshot of the architecture diagram works well), the video, and the GitHub link

8. **Submit** before July 6, 11:59 PM PT — don't leave it as a draft, it must be formally submitted to be judged.

## If you're short on time
Steps 2 (real weights) is the only truly optional one — demo mode is fully functional and honestly documented as such, so skipping it does not break your submission. Everything else is required.
