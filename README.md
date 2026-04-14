# Aviation Voice Assistant

Multilingual (EN/FR) airport chatbot with a full speech pipeline.

**Pipeline:** VAD → LID → ASR → NER → Slot Fill → Diarization → Claude agent

---

## Summary

A production-ready voice assistant for airport / flight queries, supporting English and French.

**What it does:**
- Accepts audio (WAV/MP3/OGG/WebM) via REST upload or real-time WebSocket stream
- Strips silence with Silero VAD, detects the language, and transcribes with Cohere (batch) or Deepgram Nova-3 (streaming), falling back to OpenAI Whisper
- Extracts aviation entities: airports (IATA codes, city names, full names), airlines, flight numbers, dates, cabin class, passenger count — across 156 airports and 40 airlines worldwide including Africa and South America
- Fills structured flight-intent slots (origin, destination, dates, cabin, pax count) and identifies the user's intent (book, status, baggage, check-in, lounge, etc.)
- Optionally diarizes multi-speaker audio with pyannote.audio (runs locally, free)
- Passes the parsed intent + conversation history to a Claude claude-sonnet-4-6 dialogue agent with tool use (flight search, status, baggage rules, airport info) and prompt caching

**Tech stack:** FastAPI · spaCy · Silero VAD (ONNX) · Cohere Transcribe · Deepgram SDK v6 · OpenAI Whisper · pyannote.audio · Anthropic Claude · structlog · Prometheus

**Test coverage:** 75 tests, all passing — NLP unit tests require no API keys; API integration tests use mocks.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure keys
cp .env.example .env
# Edit .env with your API keys (see API Keys section below)

# 3. Run the server
uvicorn app.main:app --reload

# 4. Test
python tests/smoke_test.py          # offline NLP tests, no keys needed
pytest tests/ -v                    # full test suite (mocked)
```

---

## API Keys & Billing

### Cohere — Batch Transcription (primary ASR)

**Key:** `COHERE_API_KEY`  
**Get it:** https://dashboard.cohere.com/api-keys

| Tier | Price | Limits | Notes |
|---|---|---|---|
| **Trial** | Free | 10 calls/min, 1 000 calls/month | No credit card required |
| **Production** | ~$0.008 / min of audio | Unlimited | Pay-as-you-go |

> Cohere Transcribe is optimised for batch accuracy over latency. Best for uploaded audio files.  
> Pricing URL: https://cohere.com/pricing

---

### Deepgram — Streaming ASR (real-time primary)

**Key:** `DEEPGRAM_API_KEY`  
**Get it:** https://console.deepgram.com/signup

| Tier | Price | Limits | Notes |
|---|---|---|---|
| **Pay As You Go** | Free until $200 credit used | Unlimited requests | $200 free credit on signup, no expiry |
| **Nova-3 Streaming** | $0.0059 / min | — | After free credit |
| **Nova-3 Batch** | $0.0043 / min | — | After free credit |
| **Growth** | Volume discounts | — | Contact sales |

> Free $200 credit is enough for ~560 hours of streaming — ample for development and testing.  
> Pricing URL: https://deepgram.com/pricing

---

### OpenAI — Whisper (fallback ASR)

**Key:** `OPENAI_API_KEY`  
**Get it:** https://platform.openai.com/api-keys

| Tier | Price | Limits | Notes |
|---|---|---|---|
| **Free trial** | $5 credit (new accounts) | Expires after 3 months | One-time only |
| **Whisper-1** | $0.006 / min of audio | No hard limit | Maps to large-v3 on the API |
| **Tier 1–5** | Same rate, higher RPM | RPM increases with spend | Rate limits raised automatically |

> Only used as fallback when Cohere or Deepgram fail. Low usage expected in normal operation.  
> Pricing URL: https://openai.com/api/pricing

---

### Anthropic — Claude Dialogue Agent

**Key:** `ANTHROPIC_API_KEY`  
**Get it:** https://console.anthropic.com/settings/keys

| Tier | Price | Limits | Notes |
|---|---|---|---|
| **Free** | None — pay-as-you-go only | — | No free tier |
| **claude-sonnet-4-6 input** | $3.00 / M tokens | — | System prompt cached after first call |
| **claude-sonnet-4-6 output** | $15.00 / M tokens | — | Typical reply ~200 tokens |
| **Prompt cache write** | $3.75 / M tokens | 5-min TTL | One-time per cold cache |
| **Prompt cache read** | $0.30 / M tokens | — | 10x cheaper than uncached input |

**Estimated cost per conversation turn** (system prompt cached):  
~200 input tokens + ~200 output tokens ≈ **$0.003 per turn**

> This project uses prompt caching on the system prompt — after the first request the ~500-token system prompt costs $0.30/M instead of $3.00/M.  
> Pricing URL: https://www.anthropic.com/pricing

---

### HuggingFace — Pyannote Speaker Diarization

**Key:** `HUGGINGFACE_TOKEN`  
**Get it:** https://huggingface.co/settings/tokens (Read token)

| Tier | Price | Notes |
|---|---|---|
| **Free** | $0 | Model runs **locally** — no per-call cost |

**One-time setup required:**
1. Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept dependency license: https://huggingface.co/pyannote/segmentation-3.0
3. Add `HUGGINGFACE_TOKEN=hf_...` to `.env`

> Model weights (~300 MB) are downloaded once to `~/.cache/huggingface/` and run locally.  
> No API calls after the initial download. GPU optional — runs on CPU.

---

## Cost Summary

| Provider | Role | Free option | Paid rate |
|---|---|---|---|
| Cohere | Batch ASR | Trial: 1 000 calls/month | ~$0.008/min |
| Deepgram | Streaming ASR | $200 free credit | $0.0059/min |
| OpenAI Whisper | ASR fallback | $5 trial credit | $0.006/min |
| Anthropic Claude | Dialogue agent | None | ~$0.003/turn |
| HuggingFace | Diarization | **Free forever** | $0 |

**Rough cost for 1 000 voice queries** (avg 30 s audio, 3 turns each):
- ASR: 1 000 × 0.5 min × $0.008 = **$4.00**
- Agent: 1 000 × 3 turns × $0.003 = **$9.00**
- Diarization: **$0.00**
- **Total ≈ $13 per 1 000 queries**

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcribe` | Batch upload — transcript + NER + slots |
| `POST` | `/v1/audio/query` | Batch upload + Claude reply |
| `WS` | `/v1/chat/stream` | Real-time streaming voice chat |
| `GET` | `/health` | Provider liveness check |

### Example — batch query
```bash
curl -X POST http://localhost:8000/v1/audio/query \
  -F "file=@recording.wav"
```

### Example — health check
```bash
curl http://localhost:8000/health
```

### Multi-turn conversation via session_id

Each call to `/v1/audio/query` returns a `session_id`. Pass it back on subsequent requests to continue the same conversation — Claude will remember the context (origin, destination, dates already confirmed, etc.).

```bash
# Turn 1 — no session_id, one is auto-generated
curl -X POST http://localhost:8000/v1/audio/query \
  -F "file=@turn1.wav"
# → {"session_id": "bfc951b5-...", "reply": "Here are flights from CDG to MNL..."}

# Turn 2 — pass the session_id back
curl -X POST http://localhost:8000/v1/audio/query \
  -F "file=@turn2.wav" \
  -F "session_id=bfc951b5-..."
# → {"reply": "Confirmed — AF 384 on April 20, business class. Shall I proceed?"}
```

> **TODO** — Multi-turn UX improvements needed:
> - Auto-carry `session_id` between turns in the Swagger UI (currently requires manual copy-paste)
> - Add a thin web frontend (or CLI tool) that stores `session_id` in memory and sends it automatically
> - Consider cookie or header-based session tracking for browser clients
> - Add session expiry and cleanup (sessions currently live in memory indefinitely)

---

## Deployment

### Option A — Local (Swagger UI)

```bash
uvicorn app.main:app --reload --port 8001
```
Open **http://localhost:8001/docs** for the interactive API.

---

### Option B — Share publicly via ngrok (no deployment needed)

Expose your local server to the internet with a public URL:

```bash
# 1. Install ngrok (one-time)
pip install pyngrok

# 2. In one terminal — start the server
uvicorn app.main:app --port 8001

# 3. In another terminal — open a tunnel
python -c "from pyngrok import ngrok; t = ngrok.connect(8001); print(t.public_url)"
```

You get a URL like `https://abc123.ngrok-free.app` — shareable, accessible from any device, no server required.

---

### Option C — Deploy to Railway (permanent, free tier)

Railway auto-deploys from GitHub using the existing `Dockerfile`:

1. Push the repo to GitHub
2. Go to **railway.app** → New Project → Deploy from GitHub → select the repo
3. Add your API keys as environment variables in the Railway dashboard (same names as `.env`)
4. Railway builds the Docker image and gives you a public URL

The free tier gives you **$5/month credit** — enough for light testing.

> **Note:** Make sure `.env` is in `.gitignore` before pushing — API keys must be set as Railway environment variables, never committed to the repo.

---

## Testing

```bash
# Level 1 — no deps, no keys (IATA vocab only)
python tests/smoke_test.py

# Level 2 — NLP tests (spaCy + dateparser, no keys)
pip install dateparser spacy && python -m spacy download en_core_web_sm
python tests/smoke_test.py
pytest tests/test_iata_vocab.py tests/test_slot_filler.py tests/test_date_normalizer.py -v

# Level 3 — API tests, mocked (full requirements, no keys)
pip install -r requirements.txt
pytest tests/test_api.py -v

# Level 4 — live server (real keys in .env)
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/v1/audio/query -F "file=@audio.wav"
```

---

## Fixes Applied During Setup

Issues encountered and resolved while getting the test suite to pass on Python 3.13 with Deepgram SDK v6.1.1:

| Error | Cause | Fix |
|---|---|---|
| `ImportError: DeepgramClientOptions` | Renamed to `DeepgramClientEnvironment` in SDK v6.1.1 | Rewrote `deepgram_provider.py` for the new Fern-generated API — options now passed as kwargs, no option objects |
| `ModuleNotFoundError: audioop` | `audioop` removed from Python 3.13 stdlib | `pip install audioop-lts` (added to `requirements.txt`) |
| `ModuleNotFoundError: torchaudio` | Silero VAD loads at app startup via lifespan; `torchaudio` not installed | Patched `get_vad/ner/slot_filler/asr_router` in `app.main` inside the pytest fixture so heavy singletons are not loaded during mocked tests |
| `TypeError: MagicMock can't be awaited` | `/health` endpoint calls `await router.health()` but mock wasn't async | Replaced with `AsyncMock(return_value={...})` in `test_health` |
| `ValidationError: entities — Input should be a valid dict` | `TranscribeResponse.entities` typed as `dict[str, list[str]]` but slot filler returns `list[dict]` | Changed field type to `list[dict]` in `audio.py` |
| Intent `LOUNGE_ACCESS` misclassified as `FLIGHT_STATUS` | `"gate"` keyword was in both intent pattern lists; `flight_status` matched first | Removed `"gate"` from `flight_status` keywords in `slot_filler.py` |
| `"Ninoy Aquino"` not resolved to MNL | Short-form logic dropped the first word only (`"Aquino International Airport"`), missing `"Ninoy Aquino"` | Added suffix-stripping pass in `build_spacy_patterns()` to strip `"International Airport"` etc. |
| `"first class"` resolved to economy | spaCy labels `"first"` as ORDINAL, blocking the cabin keyword scan via `seen_spans` | Moved `seen_spans.add()` to after the `label is not None` check in `ner.py` |
| French cabin `"affaires"` not recognised | French cabin terms absent from `_CABIN_KEYWORDS` | Added `"affaires"`, `"économique"`, `"première"` to the keyword set in `ner.py` |
| `FileNotFoundError: ffprobe` on audio upload | pydub calls `ffprobe` for media info even on WAV files; not installed on Windows | Replaced pydub with `soundfile` (primary) + `imageio_ffmpeg` subprocess (fallback) in `vad.py` — ffprobe never required |
| `422 No speech detected` with valid M4A file | MP4/M4A container stores metadata at end of file; ffmpeg can't decode it from stdin pipe (no seeking) | Switched ffmpeg input from `pipe:0` to a temp file so ffmpeg can seek |
| `AttributeError: 'AsyncClientV2' has no attribute 'transcribe'` | Cohere SDK v6 moved transcription to `client.audio.transcriptions.create()` | Updated `cohere_provider.py` to use the new method |
| Cohere `404 model 'transcribe-v1' not found` | Wrong model ID — Cohere renamed the model | Changed to `cohere-transcribe-03-2026` in `config.py` |
| Cohere `400 language is required` | Cohere Transcribe requires language, no auto-detection | Always pass `language` (defaults to `"en"` if not provided) |
| Cohere `400 unsupported file extension (got: mp4)` | VAD outputs WAV but original M4A filename extension was passed to ASR | Hardcoded `audio_format="wav"` after VAD processing in `audio.py` |
| Deepgram `400 keywords not supported for Nova-3` | Nova-3 renamed `keywords` parameter to `keyterm` | Replaced all `keywords` with `keyterm` in `deepgram_provider.py` |
| `date_normalize_failed` for "April twentieth" | dateparser doesn't parse written ordinals from ASR output | Added ordinal-word → digit conversion (`_normalise_ordinals`) before dateparser |
| `invalid language 'string'` from Swagger UI | Swagger placeholder text `"string"` passed as language hint to ASR providers | Added sanitisation in `audio.py` to reject placeholder values |

---

## Architecture

```
Audio input
    |
    v
VAD  (Silero)               strip silence, reject no-speech frames
    |
    v
LID  (langdetect / lingua)  detect language: "en" or "fr"
    |
    v
ASR  Batch:    Cohere Transcribe  -->  Whisper (fallback)
     Stream:   Deepgram Nova-3    -->  Whisper (fallback)
    |
    v
NER  (spaCy + IATA EntityRuler)
     AIRPORT  AIRLINE  FLIGHT_NO  DATE  TIME  CABIN  PAX
    |
    v
Slot Filler
     origin  destination  departure_date  return_date
     cabin_class  passenger_count  airline_pref  flight_number
    |
    v
Speaker Diarization  (pyannote.audio 3.1)   optional
    |
    v
Claude Dialogue Manager  (claude-sonnet-4-6)
     tool use: search_flights, get_flight_status, get_baggage_rules, get_airport_info
     prompt caching on system prompt
     multi-turn history per session_id
```

---

## Project Structure

```
TRAVEL/
├── app/
│   ├── main.py
│   ├── core/           config, logging
│   ├── asr/            VAD, LID, Cohere, Deepgram, Whisper, router
│   ├── nlp/            IATA vocab, NER, slot filler, date normalizer
│   ├── diarization/    pyannote speaker diarization
│   ├── agent/          Claude dialogue manager, prompts, tools
│   └── api/v1/         REST + WebSocket endpoints
├── data/
│   └── iata_codes.json 156 airports, 40 airlines
├── tests/
│   ├── smoke_test.py
│   ├── test_iata_vocab.py
│   ├── test_slot_filler.py
│   ├── test_date_normalizer.py
│   └── test_api.py
├── .env.example
├── requirements.txt
└── Dockerfile
```
