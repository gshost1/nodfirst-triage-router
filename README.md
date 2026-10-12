# Nod First Triage Router

Every message an adult child sends about an aging parent is not equally urgent.
This local demo classifies each caregiving message as **routine**, **logistics**,
or **urgent-health**, then routes it to a differently sized model, so a mood
check-in and "Mom fell" do not get the same model or the same cost.

Urgent messages bypass every router in every mode and go straight to the
configured large model with a non-diagnostic escalation ("call 911 or Mom's
doctor"). The agent never diagnoses.

**Not medical advice.** If a message could be an emergency, call 911 now.

Built for the OIS Workshop Day (MLH + DigitalOcean) brief at
[nodfirst.com](https://nodfirst.com). Local-first MVP: FastAPI, one static
vanilla-JS page, a provider-neutral OpenAI-SDK client, SQLite for the demo
store, and an optional DigitalOcean Inference Router path.

## Status: what is verified and what is not

| Item | Status |
| --- | --- |
| Local app (MOCK routing, tools, `/chat`, `/stats`, static UI) | Working; tested locally with MOCK doubles and temporary SQLite files |
| Five-message demo CLI (`scripts/demo.py`) | Working offline (MOCK rehearsal) |
| DigitalOcean model list (`GET /v1/models`) | Verified 2026-10-12: HTTP 200, 82 model IDs, no Nous Hermes model, no speech-to-text model (`qwen3-tts-voicedesign` is text-to-speech only) |
| `nodfirst` Inference Router created in the DO Control Panel | **Done** (2026-10-12): routine -> `glm-5.3-flash`, logistics -> `glm-5.3`, urgent-health -> `qwen3.8-max`, fallback `glm-5.3-flash` |
| Full chat flow in `ROUTER_MODE=do` | **Verified 2026-10-12, latest run clean** (see "Live rehearsal results" below): all five scripted messages classified as expected, both urgent messages served by `qwen3.8-max` via the direct-large bypass, routine/logistics served by `glm-5.3-flash`/`glm-5.3` through `router:nodfirst`, every turn under 4 s, 12 calls, estimated spend $0.006 |
| Live bugs found and fixed during rehearsal | Thinking tiers returned empty text until `reasoning_effort=none`; summary tools overrode urgent replies; `qwen3.8-max` called a lookup tool on "Mom had a fall"; `glm-5.3` asked for a time instead of booking. All fixed and re-verified live |
| DigitalOcean text-to-speech (`qwen3-tts-voicedesign`) | Verified 2026-10-12: `POST /digest/speak` returned a 26 s WAV through the app; inputs over ~400 chars timed out, so the digest is capped at 320 |
| Botmail alert and digest email | Verified 2026-10-12: urgent alert and digest both delivered (Botmail events queued, sent, delivered) |
| Inbound webhook (`POST /inbound`) | Verified locally with JSON and Twilio-shaped form posts; no carrier account attached |
| Nous Hermes model or agent used/deployed | **No.** This app is Hermes-ready only |
| GitHub repository | Public: <https://github.com/gshost1/nodfirst-triage-router> |
| Spend | Under $0.10 total across all live rehearsals and probes on 2026-10-12; costs shown anywhere in the app are estimates |

## Architecture

```text
              caregiver message (browser chat, or scripts/demo.py)
                                  |
                                  v
              +---------------------------------------+
              |  FastAPI  POST /chat  (127.0.0.1)     |
              |  + POST /inbound (SMS/email gateway)  |
              |  + GET /thread, GET /alerts, /ack     |
              +---------------------------------------+
                                  |
                                  v
              +---------------------------------------+
              | classify the message as routine,      |
              | logistics, or urgent; an urgent       |
              | keyword safety net (falls, chest      |
              | pain, confusion, missed dose, can't   |
              | reach the parent) OVERRIDES a         |
              | lower-risk label                      |
              +---------------------------------------+
                                  |
                                  v
  +-------------------------------------------------------------+
  | routing, chosen by ROUTER_MODE                              |
  |                                                             |
  |  mock    canned metadata. No network, no DigitalOcean call  |
  |  client  class -> MODEL_SMALL / MODEL_MID / MODEL_LARGE     |
  |  do      routine + logistics -> router:nodfirst             |
  |          urgent -> bypass every router -> MODEL_LARGE       |
  +-------------------------------------------------------------+
                                  |
                                  v
              +---------------------------------------+
              | caregiver agent + five tools          |
              | (SQLite demo store, git-ignored)      |
              +---------------------------------------+
                                  |
                                  v
              +---------------------------------------+
              | provider-neutral OpenAI-SDK client    |
              | (core/llm.py)                         |
              |   MOCK: canned replies, offline       |
              |   LIVE: DigitalOcean endpoint         |
              +---------------------------------------+
                                  |
                                  v
       reply + route + model + latency + estimated cost + tool actions
                                  |
                                  v
       route badge, tool-actions panel, stats panel (estimated savings)
```

MOCK never reaches DigitalOcean. Live modes reach the DigitalOcean
OpenAI-compatible endpoint; `ROUTER_MODE=do` also routes through the
DigitalOcean Inference Router.

## Setup (Python 3.12 + uv)

Python **3.12** is required (`pyproject.toml` pins `>=3.12,<3.13`); the project
uses [uv](https://docs.astral.sh/uv/) to manage the interpreter and the
dependencies declared in `pyproject.toml`/`uv.lock`.

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (creates .venv)
uv sync

# 3. Create your local environment file and fill in your own values
cp .env.example .env
```

`.env` is git-ignored. Put your own DigitalOcean Model Access Key in it and
never commit or share it. `.env.example` holds placeholders only.

### Environment variables

| Variable | Purpose | Value for this repo |
| --- | --- | --- |
| `LLM_BASE_URL` | OpenAI-compatible endpoint. `mock` returns canned replies with no network access | `mock` for offline use, `https://inference.do-ai.run/v1` for DigitalOcean |
| `LLM_API_KEY` | Provider credential (DO Model Access Key). Never printed, logged, or committed | placeholder in `.env.example`; real value only in `.env` |
| `LLM_MODEL` | Model used when `ROUTER_MODE=do`, i.e. `router:nodfirst`, or a concrete ID from `GET /v1/models` | `router:nodfirst` |
| `MODEL_SMALL` | Tier for routine classification/replies in client mode | `glm-5.3-flash` |
| `MODEL_MID` | Tier for logistics in client mode | `glm-5.3` |
| `MODEL_LARGE` | Tier for urgent/direct replies in every mode | `qwen3.8-max` |
| `ROUTER_MODE` | `do`, `client`, or `mock` | `mock` (offline) |
| `PORT` | Loopback port for the local app | `8000` |
| `CARE_DB` | Optional SQLite path override (default `nodfirst.sqlite3`, git-ignored) | unset |
| `DO_ROUTER_MODEL` | Optional Inference Router name override (default `router:nodfirst`) | unset |
| `LLM_REASONING_EFFORT` | Sent on every real request. The GLM and Qwen tiers are thinking models; without it their reasoning uses the whole completion budget and the text comes back empty (verified 2026-10-12). Set to `""` to omit | `none` (default) |
| `INBOUND_CONTACTS` | Optional `number-or-email=Name` pairs for `POST /inbound` sender mapping | unset |
| `INBOUND_SECRET` | Optional shared secret required in `X-Inbound-Token` on `POST /inbound` | unset |
| `TTS_MODEL` / `TTS_VOICE` | Optional text-to-speech model id and voice design for "Read digest aloud" | defaults `qwen3-tts-voicedesign`, a warm calm adult woman |
| `ALERT_EMAIL_TO` | Optional comma-separated recipients for urgent alert email | unset (alerts stay local) |
| `BOTMAIL_API_KEY` / `BOTMAIL_KEY_FILE` | Botmail key, inline or as a file path. Never printed, logged, or committed | unset |

Model IDs come from the already verified `GET /v1/models` response and live in
configuration, never hardcoded in runtime routing code. The selection rationale
is in [`MODEL_SELECTION.md`](MODEL_SELECTION.md). Catalog availability does not
prove native tool-calling support for an individual model, and no live
tool-capability check has been made.

### Model pricing used for cost estimates

Costs in the UI, the demo table, and `/stats` are **estimates**, computed with
the published DigitalOcean Serverless Inference rates for the configured tier
IDs (`core/pricing.py`).

| Setting | Model ID | USD / 1M input tokens | USD / 1M output tokens |
| --- | --- | --- | --- |
| `MODEL_SMALL` | `glm-5.3-flash` | $0.15 | $0.50 |
| `MODEL_MID` | `glm-5.3` | $1.40 | $4.40 |
| `MODEL_LARGE` | `qwen3.8-max` | $2.00 | $6.00 |

Source: <https://docs.digitalocean.com/products/inference/details/pricing/>
(page updated 2026-10-09; retrieved 2026-10-12). These are published list
prices, not a bill; the only live inference traffic sent is the single 62-token
router smoke request recorded above. Models without a published entry are
reported as unpriced instead of guessed.

## Run the app locally

The app binds to loopback only.

```bash
# Offline rehearsal, no credential needed
ROUTER_MODE=mock LLM_BASE_URL=mock uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The page shows the chat composer, a colored route
badge per reply, the tool-actions panel, the stats panel, a Replay control, and
the "not medical advice" disclaimer. The page calls only the local API; it never
holds a key or contacts a provider.

### Shared sibling thread and urgent alerts

The chat is one thread shared by the siblings, not a per-tab session:

- **Writing as** picks the sender (You, Anna, Pavel). The sender is stored with
  each message and prefixed into the model's view of the thread, so the agent
  can tell "Pavel: she took it at 9" from Anna's question before it.
- **History.** The last 8 stored messages are sent as context on every turn
  (`HISTORY_TURNS` in `core/chat.py`). The classifier still sees only the bare
  new message, so routing never depends on who wrote it.
- **Reload keeps the thread.** `GET /thread` returns the roster and recent
  messages; the page renders them on load.
- **Urgent alerts.** Every urgent-health turn also writes an alert
  (`GET /alerts`). The "Urgent alerts for the siblings" panel lists them with
  the escalation note until a sibling clicks Acknowledge
  (`POST /alerts/{id}/ack`).
- **Outbound alert email (optional).** Set `ALERT_EMAIL_TO` (comma-separated)
  and a Botmail key (`BOTMAIL_API_KEY`, or `BOTMAIL_KEY_FILE` pointing at the
  key file) and every urgent alert is also emailed through
  [Botmail](https://botmail.pro) (`POST /v1/send`, idempotent per alert).
  Delivery is best effort: a failure or timeout is recorded on the alert as
  `email_status=failed` and shown in the panel, and the chat turn still
  succeeds. The key is never logged or echoed. With nothing configured the
  panel says "Email delivery: off" and alerts stay local. Verified once on
  2026-10-12 with a real send to the owner's own address (HTTP 202).

All of it lives in the same git-ignored SQLite file as the tools' data.

### Daily digest

`summarize_day` is the fifth tool. Its text is built entirely from the store
(last 24 hours of thread activity by class and sender, urgent alerts and how
many are unacknowledged, medications taken/missed, appointments, sibling
tasks); the model never paraphrases it, so nothing in the digest is invented.

- **Ask for today's digest** in the UI sends a routine message, so the request
  goes to the small model while the content comes from the store.
- `GET /digest` returns the same digest with no model call at all.
- **Email digest to siblings** (`POST /digest/send`) mails it to
  `ALERT_EMAIL_TO` through Botmail when alert email is configured; otherwise
  it reports why not. No scheduler is included: a cron hitting
  `POST /digest/send` once a day is the whole "daily" part.

### Voice: speak to it, hear the digest

- **Voice input** is the browser's own speech recognition (the microphone
  button next to the composer). DigitalOcean has no speech-to-text route:
  verified 2026-10-12, `POST /v1/audio/transcriptions` is 404 for every model
  and `nemotron-3-nano-omni` rejects audio content parts. So the transcript is
  produced locally and goes through the same triage as typed text. Nothing is
  sent until the recognizer stops.
- **Voice output** is DigitalOcean text-to-speech: **Read digest aloud** calls
  `POST /digest/speak`, which sends the digest text to `qwen3-tts-voicedesign`
  on `/v1/audio/speech` (same endpoint and key as the chat models; `TTS_MODEL`
  and `TTS_VOICE` override the model and voice design) and streams the WAV
  back to an audio player. Verified live: a 26 second WAV of the digest.
  Inputs are capped at 320 characters because longer ones time out on the
  gateway. In MOCK mode, or without a key, the endpoint answers 503 and the
  page falls back to the browser's speech synthesis and says so.

### Voice mode: a spoken conversation

**🎙 Voice mode** in the thread header runs a hands-free loop: listen (browser
speech recognition) → triage and answer (`POST /chat` with `voice: true`, which
asks for one or two plain spoken sentences and caps the reply at 120 tokens)
→ speak the reply (`POST /speak`, DigitalOcean `qwen3-tts-voicedesign`, or the
browser voice if you pick it in the selector or the cloud voice is off) →
listen again, until you press Stop. A status strip shows Listening, Thinking,
and Speaking, and a typing bubble appears while the model works.

Questions about recorded data are answered directly: "Who is driving Mom
today?" calls `get_parent_summary` and replies with the sibling task, not the
whole summary (verified live 2026-10-12: "the only driving I have on record is
Pavel taking Mom to cardiology on Thursday"). The store-text override now
applies only when you ask for a summary, digest, or recap.

Timing on DigitalOcean: 3 to 6 s for the answer plus 5 to 17 s for the cloud
voice, so a voice turn is roughly 10 to 20 s. Choose **Browser voice** for an
instant reply when the cloud voice is too slow for the room.

### Inbound webhook (SMS or email gateway)

`POST /inbound` lets a real text or email enter the same triage. No carrier
account is needed to demo it; curl stands in for the gateway:

```bash
# Generic JSON: the reply comes back as the chat-turn JSON
curl -s -X POST http://127.0.0.1:8000/inbound -H 'Content-Type: application/json' \
  -d '{"from": "+15551234567", "text": "Mom missed her 8am metformin and sounds confused"}'
```

```bash
# Twilio-shaped form (From, Body): the reply comes back as TwiML, so a
# gateway pointed at this URL texts the answer straight back
curl -s -X POST http://127.0.0.1:8000/inbound --data-urlencode "From=+15551234567" \
  --data-urlencode "Body=Book Mom's cardiology for Thursday, Pavel drives"
```

- `INBOUND_CONTACTS` maps a number or address to a sibling name
  (`+15551234567=Anna,pavel@example.com=Pavel`). An unknown sender is shown
  by its number or address, never as a sibling, so nobody is impersonated.
- `INBOUND_SECRET`, when set, must arrive in the `X-Inbound-Token` header
  (constant-time compare); otherwise the endpoint returns 401. The app is
  loopback-only, so exposing it to a gateway means a tunnel plus the secret.
- Blank, over-long (2000 chars), or non-object bodies are rejected with 422.
- Inbound content is data: it goes through the classifier and the agent like
  a typed message and never changes configuration.

Routing modes:

| Mode | What it does | When to use it |
| --- | --- | --- |
| `mock` | Canned replies and routing metadata, no network at all | Offline rehearsal, tests, demos without a credential |
| `client` | Maps routine/logistics/urgent to `MODEL_SMALL`/`MODEL_MID`/`MODEL_LARGE` on the DO endpoint | Documented fallback when the Inference Router is unavailable |
| `do` | Routine/logistics go through `router:nodfirst` (the router picks the serving model); urgent bypasses it to `MODEL_LARGE` | The intended DigitalOcean path; the router exists and one routine request has been smoke-verified |

## Five-message demo (offline)

```bash
uv run python scripts/demo.py
```

The CLI replays the approved five-message script (one routine, two logistics,
two urgent) through the same `ChatService` the API uses and prints a table, the
class counts, and each reply. It **always** runs in MOCK: it builds the canned
client directly, does not read `.env`, and makes no provider request.

Export the tier IDs to show the configured model IDs and estimated costs in the
table:

```bash
MODEL_SMALL=glm-5.3-flash MODEL_MID=glm-5.3 MODEL_LARGE=qwen3.8-max \
  ROUTER_MODE=mock LLM_BASE_URL=mock uv run python scripts/demo.py
```

```text
#  Class      Route              Model          Latency   Est. cost  Tools                                 Message
-  ---------  -----------------  -------------  --------  ---------  ------------------------------------  --------------------------------------------
1  routine    mock               glm-5.3-flash  0.005 ms  $0.000045  -                                     How did Mom's visit with her neighbor go ...
2  logistics  mock               glm-5.3        0.016 ms  $0.001056  add_appointment, assign_sibling_task  Book Mom's cardiology for Thursday, Pavel...
3  logistics  mock               glm-5.3        0.002 ms  $0.000418  -                                     Can you schedule Mom's pharmacy refill de...
4  urgent     mock:direct-large  qwen3.8-max    0.002 ms  $0.000688  -                                     Mom had a fall in the kitchen this morning.
5  urgent     mock:direct-large  qwen3.8-max    0.002 ms  $0.000694  -                                     Mom says she has chest pain and I can't r...

Counts: routine 1, logistics 2, urgent 2 (5 messages)
```

Options: `--db <path>` selects the SQLite store (default `$CARE_DB` or
`nodfirst.sqlite3`). The same five messages drive the UI's Replay control, so
the CLI and the page cannot drift apart.

The demo always reports MOCK routes (`mock`, `mock:direct-large`) and labels the
cost column as an estimate. Latency in MOCK mode measures local canned work, not
inference.

## Live rehearsal results (2026-10-12, `ROUTER_MODE=do`)

The five scripted messages through the real DigitalOcean path, after the fixes
listed in the status table. Latency is the whole turn (classifier plus agent).

| # | Message | Expected | Got | Served model | Path | Latency | Tools |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | How did Mom's visit with her neighbor go yesterday? | routine | routine | `glm-5.3-flash` | `router:nodfirst`, route `routine` | 3.4 s | none |
| 2 | Book Mom's cardiology for Thursday, Pavel drives | logistics | logistics | `glm-5.3` | `router:nodfirst`, route `logistics` | 3.7 s | `add_appointment` |
| 3 | Can you schedule Mom's pharmacy refill delivery for Friday? | logistics | logistics | `glm-5.3-flash` | `router:nodfirst`, route `routine` (the router's own choice) | 3.2 s | `add_appointment` |
| 4 | Mom had a fall in the kitchen this morning. | urgent | urgent | `qwen3.8-max` | direct, router bypassed | 2.7 s | none |
| 5 | Mom says she has chest pain and I can't reach her doctor. | urgent | urgent | `qwen3.8-max` | direct, router bypassed | 3.7 s | none |

Totals: 12 inference calls (5 classifier on the small tier, 5 router, 2 large),
estimated spend $0.006, estimated savings versus sending everything to the
large tier $0.006. Both urgent replies open with "call 911 now"; the routine
reply says honestly that nothing is recorded about the visit.

Earlier runs the same day were not clean, and the fixes are in the git history:
an empty routine reply (thinking model, reasoning budget), a misclassified
routine message (same cause, in the 16-token classifier), an urgent reply
replaced by a digest, and a 112 s logistics turn that did not reproduce.

## Two-minute walkthrough

Prepare one of these two servers, then open <http://127.0.0.1:8000>:

```bash
# Offline (what you can always demo)
ROUTER_MODE=mock LLM_BASE_URL=mock uv run uvicorn app:app --host 127.0.0.1 --port 8000

# DigitalOcean: the nodfirst router exists and the key is configured (full flow verified, see above)
ROUTER_MODE=do uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

| Time | Action | What to show / say |
| --- | --- | --- |
| 0:00–0:15 | Point at the disclaimer and the empty badge/stats panels | "Every message about my mom isn't equal: a check-in and a fall shouldn't get the same model or the same cost." |
| 0:15–0:35 | Type `How did Mom's visit with her neighbor go yesterday?` | Routine, green badge, small tier, estimated cost near zero |
| 0:35–1:00 | Type `Book Mom's cardiology for Thursday, Pavel drives` | Logistics, blue badge, mid tier; the tool panel shows `add_appointment` and `assign_sibling_task` |
| 1:00–1:25 | Type `Mom had a fall in the kitchen this morning.` | Urgent, red badge, direct to the large tier, bypassing every router; the reply says call 911 or Mom's doctor and never diagnoses |
| 1:25–1:50 | Open the stats panel | Calls per tier, average latency, estimated spend, and estimated savings versus pricing the same tokens at the large-model rate |
| 1:50–2:00 | Name the mode out loud, then close | See the honesty lines below |

State the mode you are actually running:

- **MOCK** is an offline rehearsal: canned replies, no DigitalOcean call, so do
  not present it as live DO usage.
- **`ROUTER_MODE=do`** shows the DO inference path: routine and logistics go
  through `router:nodfirst`, which selects the serving model (the badge shows the
  provider-reported model and selected route when the API returns them); urgent
  messages never reach the router.
- **`ROUTER_MODE=client`** is the fallback when `nodfirst` is unavailable or
  unauthorized: it sends the same messages to the tier model IDs directly on the
  DigitalOcean endpoint.

The full flow in `ROUTER_MODE=do` has been verified (see "Live rehearsal
results"). If your key lacks router scope, or the router is unavailable in your
environment, demo `ROUTER_MODE=client` or MOCK and say so instead of claiming the
router path works.

Avoid unsupported numbers: this MVP has no evaluated accuracy figure and no
benchmark, so do not quote savings percentages or urgent-recall rates. The stats
panel shows measured call counts, measured local latency, and *estimated* cost.

## Submission checklist (owner actions)

These are the user-owned items from the build brief. This MVP did not perform
any of them, and no external submission, public repository, or recording was
created.

- [ ] **Publish the public GitHub repository** — pending your explicit
      authorization. The local repo has **no remote configured** and nothing has
      been pushed.
- [ ] **Demo before 2pm** — the workshop day wraps at 2pm, so demo to
      mentors/MLH staff before then.
- [ ] **Confirm the MLH/Devpost form** — a form, judging, and prizes are
      **unverified**. Check the MLH setup email or ask event staff; do not assume
      a competition or judging process exists.
- [ ] **Record a 1–2 minute screen recording as a backup** in case the network
      or the live path fails on site.
- Note (your schedule, not an event rule): leave at approximately 2pm for the
  Akamai deadline. No external submission or recording was created on your
  behalf.

Do not add unverified caregiver statistics, event rules, judging/prize details,
or accuracy claims. Earlier notes floated a caregiver-population statistic and a
perfect urgent-recall rate; neither is verified or evaluated here, so neither
appears in this README or in the UI.

## OIS, DigitalOcean, and Nous Hermes

- **OIS Workshop Day (MLH + DigitalOcean)** is the event this project was built
  for.
- **DigitalOcean Serverless Inference** is the model endpoint: an
  OpenAI-compatible API at `https://inference.do-ai.run/v1`. Verified live
  requests are `GET /v1/models` (HTTP 200, 81 model IDs) and one low-token chat
  completion through `router:nodfirst` (see the status table above).
- **DigitalOcean Inference Router**: in `ROUTER_MODE=do`, routine and logistics
  traffic is sent as `router:nodfirst` and the router chooses the serving model;
  urgent traffic bypasses the router entirely. The router was created in the DO
  Control Panel (2026-10-12), and one low-token routine request has been verified
  end to end: HTTP 200, serving model `glm-5.3-flash`, selected route `routine`.
  This verifies only that this one configured routine request works and that the
  configured key carries router scope; urgent/bypass routing, the other two
  configured routes, and a full application chat flow in `ROUTER_MODE=do` remain
  unverified. The code path also remains unit-tested against a test double.
- **Nous Hermes**: no Hermes model appeared in the verified model list, and
  deploying a Hermes agent is outside this MVP. The app is **Hermes-ready**: it
  speaks the standard OpenAI-compatible chat-completions and tool-calling
  surface behind a provider-neutral client, so a Hermes model or agent can be
  dropped in through environment configuration. **No Nous Hermes model or agent
  was used or deployed in this MVP.**

## Not in this MVP

Hermes deployment to a Droplet/App Platform, a scheduler for the digest, SMS
delivery, a live carrier/gateway account for the inbound webhook, cloud deployment, load
testing, a labeled accuracy evaluation, and the 20-message routing evaluation.
The app is prepped for those, not shipping them.

## Tests and checks

```bash
uv run pytest -q
uv run mypy app.py core router scripts
uv run ruff check .
```

Every test runs in MOCK mode with temporary SQLite files; no test issues a
provider request. `tests/test_demo.py` covers the five-message mix, the offline
guarantee, and consistency with the UI replay script.

## Repository hygiene

- Local-only: no Git remote is configured and nothing is pushed.
- `.env`, `*.sqlite3`, and `*.db` are git-ignored; only `.env.example`
  (placeholders) is tracked. `BRIEF.md` stays in place and unchanged.
- The app binds to loopback only, and the browser page holds no credentials.
