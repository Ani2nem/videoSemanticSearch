# Video Insight Engine

A modular, async video-understanding pipeline that extracts features from video frames and audio clips, then consolidates them into structured metadata using an LLM.

---

## Prerequisites

- **Python 3.11+**
- **pip** (or a virtual-environment manager such as `uv` or `conda`)
- **ffmpeg** — required by faster-whisper for audio decoding

  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg
  ```

---

## Installation

```bash
cd video-insight-engine
pip install -r requirements.txt
```

> Some packages (PaddleOCR, ultralytics, faster-whisper) download model weights on first run. Ensure you have an internet connection.

---

## Generate Protobuf Files

Run this **once** from the project root before starting any server or test:

```bash
python3 -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/mcp.proto
```

This generates `mcp_pb2.py` and `mcp_pb2_grpc.py` in the project root.

---

## Environment Variables

`run_demo.py` loads a `.env` file automatically via **python-dotenv**, so you can store credentials without exporting them in every shell session.

Create `video-insight-engine/.env`:

```
OPENAI_API_KEY=sk-...
```

Alternatively, export the variable manually as usual:

```bash
export OPENAI_API_KEY=sk-...
```

The `.env` file takes precedence if both are set. Never commit `.env` to version control.

---

## Running MCP Servers

### Running all servers (full pipeline)

Open **five separate terminals**, one per server:

```bash
# Terminal 1 — Hair Color (port 50051)
python3 -m mcps.hair_color.server

# Terminal 2 — Body Build (port 50052)
python3 -m mcps.body_build.server

# Terminal 3 — People Count (port 50053)
python3 -m mcps.people_count.server

# Terminal 4 — Captions / OCR (port 50054)
python3 -m mcps.captions.server

# Terminal 5 — Audio Transcription (port 50055)
python3 -m mcps.audio.server
```

### Running only the servers you need

**Servers are optional.** The `ExtractionAgent` issues all five MCP calls concurrently with a 0.5 s connection timeout. Any server that is offline is detected immediately and recorded as `"Server Offline"` in the bundle — the pipeline continues with whatever servers are available.

This means you can start only the MCP you are actively developing and still run the full demo:

```bash
# Work on hair_color only — all other MCPs will show "Server Offline"
python3 -m mcps.hair_color.server
```

The `ExtractionBundle` returned will have `partial_flags.hair_color = True` and all other flags set to `False` with `extraction_errors[name] = "Server Offline"`. The consolidation step (if enabled) receives and handles the partial data gracefully.

---

## Running the Full Demo

### Prepare sample files

`run_demo.py` **automatically scans the `samples/` directory** for all images with `.jpg`, `.jpeg`, or `.png` extensions (case-insensitive) and processes them as a batch. You do not need to rename files or edit the script.

```
samples/                  ← drop any number of images here
    IMG_0086.JPG
    brad.jpeg
    frame.png
    clip.wav              ← single shared audio file for all images
```

`clip.wav` is the only required filename. Images can have any name.

### Configure your API key

Create a `.env` file in the project root and add your OpenAI key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

`run_demo.py` uses **python-dotenv** to load this file automatically at startup — no `export` or shell configuration is required. The key is read once when the script launches and passed directly to the `ConsolidationAgent`.

> **Dry Run Safety:** If you want to test the extraction pipeline without spending any OpenAI credits, set `DRY_RUN = True` in `run_demo.py` before running. The consolidation step is skipped entirely and no API call is made. See the [Dry Run Mode](#dry-run-mode) section for details.

### Run modes

| Command | Behaviour |
|---|---|
| `python run_demo.py` | Live pipeline — calls GPT-4o-mini (requires `OPENAI_API_KEY`) |
| `python run_demo.py --mock` | Mock consolidation — no LLM call, no API key needed |
| `python run_demo.py` with `DRY_RUN = True` | Extraction only — see Dry Run section below |

```bash
# Live run (requires OPENAI_API_KEY in environment or .env)
python run_demo.py

# Mock consolidation — description is the raw stringified bundle
python run_demo.py --mock
```

For every image the demo prints:
- A `Processing [n/total]: filename` header
- Per-MCP extraction status (`OK` / `FAILED` / `Server Offline`)
- The full raw `ExtractionBundle` as JSON
- The final `ConsolidatedOutput` summary (description, tags, confidence)

---

## Dry Run Mode

Set `DRY_RUN = True` at the top of `run_demo.py` to run **extraction only** — the `ConsolidationAgent` is never instantiated and no OpenAI call is made. This is useful for:

- Verifying that your local MCP servers are returning correct data
- Inspecting raw feature payloads from all five extractors before writing a prompt
- Running the full batch on many images cheaply during development

```python
# run_demo.py — line ~62
DRY_RUN = True   # ← set to False to enable consolidation
```

With `DRY_RUN = True` the output for each image looks like:

```
────────────────────────────────────────────────────────────
  Processing [1/3]: brad.jpeg
────────────────────────────────────────────────────────────

  [1/2] Extraction  →  source_id='brad'
        hair_color     : OK
        body_build     : OK
        people_count   : OK
        captions       : Server Offline
        audio          : Server Offline

  Raw MCP data for 'brad.jpeg':
  { ... full ExtractionBundle JSON ... }

  [DRY RUN] Skipping consolidation for 'brad.jpeg'.
```

To re-enable consolidation, set `DRY_RUN = False` and choose a mode (`--mock` or live).

---

## Running the Tests

```bash
pytest tests/
```

To run a specific suite:

```bash
pytest tests/test_mcps.py -v              # MCP server integration tests
pytest tests/test_extraction_agent.py -v  # Circuit breaker & retry unit tests
pytest tests/test_end_to_end.py -v        # Full pipeline with mocked LLM
```

> `test_mcps.py` starts in-process gRPC servers on offset ports (50151–50155) so it does not conflict with running MCP servers.

---

## Project Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │           ExtractionAgent            │
                          │                                      │
                          │  ┌──────────────────────────────┐   │
  input                   │  │  asyncio.gather (fan-out)     │   │
  ──────►  [source_id,    │  │                              │   │
            image_path,   │  │  hair_color  ──► port 50051  │   │
            audio_path]   │  │  body_build  ──► port 50052  │   │  ExtractionBundle
                          │  │  people_count──► port 50053  │──►│ ──────────────────►
                          │  │  captions    ──► port 50054  │   │
                          │  │  audio       ──► port 50055  │   │
                          │  └──────────────────────────────┘   │
                          │                                      │
                          │  • asyncio.Semaphore (max 8 inputs) │
                          │  • Per-MCP timeout: 0.5 s           │
                          │  • Offline server → "Server Offline"│
                          │    in extraction_errors (soft fail) │
                          │  • Per-MCP retry (2 retries, exp.   │
                          │    backoff 100 ms → 300 ms)         │
                          │  • Per-MCP circuit breaker          │
                          │    (open after 5 consecutive fails, │
                          │     half-open probe after 30 s)     │
                          │  • Soft deps: partial failure ok     │
                          │  • Hard deps: raises ExtractionError │
                          │    (unless server is offline)        │
                          └─────────────────────────────────────┘
                                            │
                                            │ ExtractionBundle
                                            ▼
                          ┌─────────────────────────────────────┐
                          │         ConsolidationAgent           │
                          │                                      │
                          │  1. Serialise bundle → prompt        │
                          │  2. Call GPT-4o-mini via LangChain   │
                          │  3. Parse + validate JSON response   │
                          │     into ConsolidatedOutput          │
                          │                                      │
                          │  MockConsolidationAgent (--mock):    │
                          │  • Skips LLM entirely                │
                          │  • description = stringified bundle  │
                          │  • No OPENAI_API_KEY required        │
                          └─────────────────────────────────────┘
                                            │
                                            │ ConsolidatedOutput
                                            ▼
                               {description, tags,
                                confidence_score,
                                dominant_features,
                                partial_result,
                                prompt_version}
```

### Key design decisions

| Concern | Choice |
|---|---|
| IPC protocol | gRPC (protobuf) — typed, efficient, language-agnostic |
| Async runtime | `grpc.aio` — all channels and stubs are async |
| Resilience | Per-MCP circuit breaker + exponential-backoff retries |
| Offline detection | 0.5 s RPC timeout; `UNAVAILABLE`/`DEADLINE_EXCEEDED` → soft `"Server Offline"` |
| Schema validation | Pydantic v2 — strict field types, coercion, validators |
| LLM integration | LangChain `ChatOpenAI` — easy model swapping |
| Mock LLM | `MockConsolidationAgent` — full pipeline without API key |
| Credentials | `python-dotenv` — `.env` file loaded automatically |
| Batch processing | `run_demo.py` scans `samples/` for all images automatically |
| Dry run | `DRY_RUN = True` in `run_demo.py` — extraction only, no LLM |
| Test strategy | In-process server fixtures + `AsyncMock` patches — no live services needed |
