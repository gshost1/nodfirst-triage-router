# Droid Build Brief — OIS Workshop Day (MLH + DigitalOcean) — "Nod First Triage Router"
Mon Oct 12 2026 · The Midway, 900 Marin St SF · doors 9:00, demos 10:00, build 10:45, lunch 12:00, wrap 2:00 (demo before 2pm) · https://luma.com/OIS
Builder: Georgiy (George) Shostakovych, solo, Nod First (nodfirst.com). Build tonight, polish on site.

## Sponsors & partners (verified unless flagged)
- **Major League Hacking (MLH)**, host. Source: https://luma.com/OIS
- **DigitalOcean**, the platform (Gradient / Serverless Inference, Inference Router). Docs: https://docs.digitalocean.com/products/inference/how-to/si-endpoints/ , https://docs.digitalocean.com/products/inference/how-to/use-inference-router/
- **Nous Research**, featured Hermes agent and models. Repo: https://github.com/NousResearch/hermes-agent (exists; README not fetched in detail [UNVERIFIED usage])
- **Hugging Face**: Luma asks attendees to create an account. Main summit (Oct 13) also lists Nous, HF, OpenHands, NVIDIA, DO.
- Judging, prizes, and a Devpost/MLH submission: **[UNVERIFIED]**. Luma shows a workshop day, not a stated competition. Check the MLH setup email.

## Stack / rules found
- Encouraged paths (Luma): (1) deploy a Hermes agent on DO, (2) route across models with the DO Inference Router, (3) serve or fine-tune an open model.
- DO inference is OpenAI-compatible at `https://inference.do-ai.run/v1`. Header: `Authorization: Bearer $MODEL_ACCESS_KEY` (`doo-v1-…`). Router models use `"model": "router:<name>"`. Get the list with `GET /v1/models`.
- Rules on fresh repos or on-site-only building: **[UNVERIFIED]**. Standard MLH practice is fine with prep, but keep a clean commit history and say it was prepped Sunday night.
- Free only. Use the DO credit given on site. Don't put a card on file beyond what signup requires.

## Project: Nod First Triage Router (distinct angle = *model routing by caregiving risk*)
An adult child texts or types a message about their parent ("Mom skipped her 8am metformin and sounds confused"). A classifier step labels it **routine / logistics / urgent-health**. It routes to:
- routine (mood check-in, chit-chat) → cheap small open model
- logistics (appointments, sibling task split) → mid model + tools
- urgent-health (confusion, falls, chest pain, med errors) → strongest model + safety prompt ("call 911/doctor" escalation, never diagnose)
The UI shows each reply with a **route badge**: model used, latency, est. cost. A dashboard tallies how much was saved compared with sending everything to the big model. Uses Hermes (Nous) as the agent/tool-calling model where available. The DO Inference Router is the primary path, and a client-side router is the fallback.

### Shared core (same code as the Akamai brief)
```python
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ["LLM_MODEL"]
```

## Env vars
```
LLM_BASE_URL=https://inference.do-ai.run/v1
LLM_API_KEY=doo-v1-...            # DO Model Access Key
LLM_MODEL=router:nodfirst         # or a concrete model id from /v1/models
MODEL_SMALL=<small id>  MODEL_MID=<mid id>  MODEL_LARGE=<hermes/large id>   # client-side fallback router
ROUTER_MODE=do|client|mock
PORT=8000
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install openai fastapi uvicorn pydantic python-dotenv
curl -s $LLM_BASE_URL/models -H "Authorization: Bearer $LLM_API_KEY" | jq '.data[].id'
# Pick ids, fill .env. Create router "nodfirst" in Control Panel → INFERENCE → Inference Router (per DO docs)
```

## Droid prompt (paste step by step)
1. "Create a Python repo `nodfirst-agent` with a package `core/` containing `llm.py`: an OpenAI-SDK client built only from env vars LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, with timeout 30s, 2 retries, and a `MOCK` mode (LLM_BASE_URL=mock) that returns canned replies. No provider-specific code."
2. "Add `core/agent.py`: a caregiver agent with a system prompt (warm, concise, never diagnoses, escalates emergencies to 911/doctor). Add tools with simple JSON schemas: `log_med(med, time, taken:bool)`, `add_appointment(title, datetime)`, `assign_sibling_task(sibling, task)`, `get_parent_summary()`. In-memory/SQLite store seeded with demo parent 'Mom (Ludmila, 81)', siblings Anna and Pavel. Tool-calling must degrade gracefully: if the model returns no tool_calls, parse a JSON block, else plain reply."
3. "Add `router/`: `classify(msg)` → routine|logistics|urgent using a cheap model call plus a keyword safety net (falls, chest pain, confusion, missed dose, can't reach). ROUTER_MODE=do sends to `router:nodfirst`. ROUTER_MODE=client maps the class → MODEL_SMALL/MID/LARGE. Record model, latency_ms, tokens, est_cost (cost table in config) per call."
4. "Add FastAPI `app.py` with POST /chat, GET /stats, and a single static `index.html` chat UI that shows a colored route badge per reply and a stats panel (calls per tier, avg latency, $ saved vs all-LARGE)."
5. "Add `scripts/demo.py` that replays 5 scripted messages (one routine, two logistics, two urgent) and prints a table. Add README with env setup, architecture diagram (ascii), and a 'Built at OIS Workshop Day with DigitalOcean + Nous Hermes' section."
6. "Add `tests/test_router.py` covering the keyword safety net so urgent messages never go to the small model."

## MVP (tonight)
Core + client-side router + chat UI + demo script, working in MOCK and against DO.
## Stretch (on site)
- Real DO Inference Router config (screenshot it for the demo)
- Hermes agent from the Nous repo deployed on a DO Droplet/App Platform, using the same tools
- Daily "sibling digest" summary on the cheap model
- Eval: 20 labeled messages → routing accuracy + cost chart

## 2-minute demo script
0:00 Problem: "40M+ Americans care for a parent, many remotely. Every message isn't equal: a mood check-in and 'Mom fell' shouldn't cost the same or get the same model." (Verify the stat before saying it, or drop it.)
0:20 Type routine message → green badge, small model, ~$0.0001.
0:40 "Book Mom's cardiology for Thursday, Pavel drives" → tools fire, appointment and sibling task appear.
1:05 "Mom sounds confused and missed her metformin" → red badge, large/Hermes model, escalation guidance.
1:30 Stats panel: X% cheaper than all-large, urgent recall 100% on test set.
1:45 "One env var swaps providers. All open models on DigitalOcean." Close with nodfirst.com.

## Submission checklist
- [ ] Public GitHub repo, README, .env.example (no keys committed!)
- [ ] Demo before 2pm to mentors/MLH staff
- [ ] Devpost/MLH form: **[UNVERIFIED]**. Check the setup email or ask staff at 10am
- [ ] 1–2 min screen recording as backup (QuickTime/OBS) in case Wi-Fi dies
- [ ] Leave by ~2pm for 525 Market (Akamai deadline 4pm)

## Curveballs
- **Key/credit arrives late:** run ROUTER_MODE=mock, or use your own DO key created tonight (inference is pay-per-token. Stop if it needs payment beyond free credit).
- **Model ids differ from the plan:** ids come only from `/v1/models` into env. Never hardcode them.
- **Hermes not on DO serverless:** use whatever Nous/Hermes id is listed. Otherwise frame it as "Hermes-ready" and use the largest open model listed.
- **Router UI/API unavailable:** client-side router (already built). Say so honestly.
- **Tool-calling flaky:** JSON-block fallback parser, plus keep tools to 4.
- **Wi-Fi:** phone hotspot, mock mode, recorded video.
- **Fresh repo / on-site-only rule:** if staff require it, create a new repo on site and port the code in visible commits. Disclose the prep.
- **Team requirement:** none known. Solo is fine at MLH events typically [UNVERIFIED].
- **Required sponsor usage:** DO inference is the core path. Show DO in the demo explicitly.
- **Safety:** disclaimer in the UI: "not medical advice."
