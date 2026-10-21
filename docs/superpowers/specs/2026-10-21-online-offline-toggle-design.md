# Online/offline session toggle — design

2026-10-21 · for the 2026-10-27 workshop ("LO-FI: a LOcal, personal FInance
voice AI bot"). Author: Claude session with vipyne; decisions below are
labeled theirs or builder's call.

## Intent

The offline demo is honest but hardware-bound: lookups take 30–70 s on the
presenter's machine and attendees' laptops vary wildly. The workshop's
abstract already promises the counter-move — "with just a few lines of code
changed, your local bot _can_ use any hosted LLM service. You control when
it has access to local, potentially sensitive documents." This project makes
that promise a first-class, impressive demo: every session is either
**offline** (today's fully-local split stack) or **online** (hosted
everything — fast, good), chosen on the page before Connect, with the
privacy boundary enforced at the file level.

## Goals

1. **Per-session online/offline toggle** on the client page, chosen before
   Connect. Offline is the default. A fresh session is one or the other;
   no mid-session switching (decided: the user — "simpler and safer").
2. **Online stack = zero local models**: Deepgram STT + OpenAI LLM +
   Cartesia TTS, cascade architecture. Online the LLM holds ONE retrieval
   tool — `semantic_search` over the public db (builder's call, see below);
   snappy lookups are the point.
3. **Exactly two stories: offline and online.** The legacy
   `LOFI_LLM_ENGINE=hosted` hybrid (hosted front + local STT/TTS/analyst)
   is DELETED (decided: the user — "too complicated"): the `hosted` branch
   in `services.py`, `hosted_variant(prompt)`, `_preflight_hosted_llm`,
   the `LOFI_HOSTED_*` config knobs, and the env-engine fields' old
   meaning in `lofi-config`. `tests/test_services_llm_engine.py` is
   rewritten for the new mode selection; the banner machinery is re-keyed
   to session mode (its split/cascade HOSTED_DETAIL texts go away in favor
   of one online text).
4. **`./lofi.sh start --online-only`** is in scope (decided: the user): a
   weak laptop with keys and a browser runs the online demo with zero
   model downloads. Preflights become mode-aware — online-only skips the
   Ollama/STT/RAG gates and checks keys (+ `lofi-public.db` presence,
   warning only) instead; an offline session attempted against an
   online-only server fails at connect with a clear message.
5. **File-level privacy boundary**: `data/` splits into `data/private/`
   (transactions CSV, passphrase canary) and `data/public/` (IRS pubs).
   Two indexes are built: `lofi.db` over all of `data/` (offline mounts
   this) and `lofi-public.db` over `data/public/` only (online mounts this
   file and nothing else). The online session never opens the private db —
   there is no row filtering to get wrong.
6. **Config split**: `.env` holds exactly the three hosted keys
   (`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`) and
   `services.py` stops rejecting keys there. `.lofirc` takes over the old
   `.env` role: all `LOFI_*` settings, what `configure.sh` reads/writes
   (decided: the user).
7. **SmallWebRTC becomes the default browser transport**, WebSocket the
   automatic fallback (WSL2 and anything else that can't do UDP), MoQ stays
   `?transport=moq` opt-in.
8. **Two-way Spanish works online**: multilingual Deepgram + hosted LLM +
   Spanish-capable Cartesia voice make real conversation practice an online
   demo beat. Offline keeps the existing teach-only wait-lesson flourish.
9. **Explainer content** lands in DEMO.md and the README: why
   SmallWebRTC/WebSocket, echo cancellation via the browser, and a
   rule-of-thumb formula for what model size a machine can run.

## Non-goals

- Mid-session stack switching (rejected for sharp edges; revisit post-workshop).
- Anthropic/other LLM providers on stage. OpenAI only for now; the
  OpenAI-compatible client keeps the door open (an OpenAI key is required
  regardless for embeddings).
- The offline ⚡fast/🧠accurate analyst toggle — that is existing punch
  item 2, tracked separately.
- Any MoQ work.

## Anti-goals (failure even if it "works")

- An online session that can read `data/private/` content by any path:
  opening `lofi.db`, a tool pointed at the wrong db, a shared retrieval
  cache. The private db path must never be constructed in online mode.
- A session whose mode is ambiguous on screen. Online must be LOUD: the
  banner (re-labeled `⚠️ ONLINE MODE`) shows, and the What's Running
  panel names the online services and `lofi-public.db`. Naming duality is
  offline/online everywhere — "hosted" implied the whole bot is hosted and
  is retired (`HOSTED_INTRO` → `ONLINE_INTRO`, paired with
  `OFFLINE_INTRO`).
- Keys leaking: into `.lofirc`, into logs, into `lofi-config` messages
  sent to the client, or into git (`.env` must be gitignored — verify).
- Breaking the offline demo. Offline is the talk's thesis; online is the
  encore. All 253 tests stay green; the offline trio (passphrase / records
  question / chitchat) still passes by voice.

## Constraints

- **Workshop is 2026-10-27.** Working beats elegant; every piece must be
  demoable independently so partial completion still improves the talk.
- **The ralph tunables loop is in flight** on this repo (centralizing
  prompts/model defaults into `tunables.py`). New tunables introduced here
  (online model ids, voice id, lesson language) go into `tunables.py` if it
  exists by build time; do not edit files ralph's plan owns while its loop
  is running. Coordinate: this project starts after the tunables loop
  completes or is paused at a gate.
- Pipecat pinned `>=1.13,<1.14`. `DeepgramSTTService`,
  `CartesiaTTSService`, `OpenAILLMService` all ship in 1.13 (verified via
  context hub); each needs its pipecat extra (`pipecat-ai[deepgram]`,
  `[cartesia]`; openai client is already a dep of the hosted path — verify).
- Windows/WSL2: QUIC and WebRTC UDP cannot cross the Windows↔WSL2
  boundary (measured, 2026-10-13) — WebSocket fallback must remain
  automatic, not opt-in. Native Windows (PowerShell route) should get
  SmallWebRTC (aiortc ships Windows wheels — verify at build time).
- Never commit/push — the user does that.
- One README per repo, top level only.

## Decided design (the how, as decided)

### Config files

- `.env` → exactly the three keys. `config.py` loads `.lofirc` first (the
  defaults), then `.env` (the keys); both optional; shell env wins over
  both, matching today's precedence. `LOFI_ENV_FILE` override continues to
  work (points at the `.lofirc`-role file; builder may add
  `LOFI_KEYS_FILE` if symmetry is needed — builder's call).
- `.lofirc`: current `env.example` content. `lofi.sh`'s
  `load_env_defaults` reads it; `configure.sh` reads/writes it (backup
  `.lofirc.bak`). `env.example` is replaced by `lofirc.example` plus a
  three-line `env.example` showing the key names with empty values
  (builder's call on exact naming, but examples must exist for both files).
- `services.py`: the "keys never go in .env" rejection and
  `hosted_key_in_dotenv_only()` are removed/inverted — `.env` is now
  exactly where keys go. Migration note in README. `.env` and `.lofirc`
  both gitignored (verify `.env` already is; add `.lofirc`).

### Data + indexes

- `data/private/`: `Transactions_*.csv`, `passphrase-smoke.txt`.
  `data/public/`: `irs-pub-17.pdf`, `irs-pub-334.pdf`. Root `data/` files
  are treated as private (conservative default — builder implements,
  README documents).
- `./lofi.sh index-rag` builds both dbs in one run:
  - `models/rag-index/lofi.db` — all of `data/`, nomic-embed-text via
    Ollama (unchanged behavior, unchanged file).
  - `models/rag-index/lofi-public.db` — `data/public/` only, embedded with
    OpenAI `text-embedding-3-small` (my call, accepted: Anthropic has no
    embeddings API and the OpenAI key is present anyway). Requires
    `OPENAI_API_KEY` and network; when absent, print a clear notice and
    skip the public build (offline index still builds).
- `ingest.py` gains data-dir + embedder + db-path parameters (it already
  walks recursively via `rglob`). Incremental hashing continues to work
  per-db.

### Session mode plumbing

- Client: a visible two-state control near Connect — `🔌 offline` /
  `☁️ online` — default offline, disabled while connected. The chosen mode
  rides the connect request (mechanism left to builder: query param or
  request body on the runner's start/offer/ws endpoints — whatever reaches
  the server before `build_pipeline` runs; `?mode=online` in the page URL
  must also work for shareability).
- Server: `build_pipeline(transport, mode)` (or equivalent) selects the
  stack per connection. Online mode fails fast at connect with a clear
  client-visible error when a required key is missing. A missing
  `lofi-public.db` does NOT block connect (builder's call): the session
  degrades to chitchat-only with a loud "no public index — run
  ./lofi.sh index-rag with OPENAI_API_KEY set" line in the panel, matching
  the offline no-index precedent. Falling back to `lofi.db` is forbidden
  (anti-goal).
- `./lofi.sh start --online-only`: skips the Ollama wait and model pulls;
  bot preflights become mode-aware (online-only checks the three keys,
  warns on missing public db, and skips `_preflight_llm/_stt/_rag`);
  offline connect attempts against an online-only server fail at connect
  with a clear message.
- Online pipeline: local Silero VAD stays (it is turn-taking plumbing, not
  a data-touching model; README says so), Deepgram STT (multilingual
  model, e.g. nova-3 `multi` — builder verifies the 1.13 settings shape),
  OpenAI LLM (default model a tunable, suggested `gpt-4o-mini` for
  latency; cascade system prompt variant), ONE tool — `semantic_search`
  bound to `lofi-public.db` with the OpenAI embedder (builder's call: the
  other three tools query the transactions table, empty by design in the
  public db; the online prompt states it has public reference documents
  and no personal records, so the bot SAYS the boundary out loud when
  asked a records question), Cartesia TTS (Spanish-capable voice id as a
  tunable).
- `lofi-config` message gains the mode; the What's Running panel renders
  hosted entries (role colors already exist); the hosted banner text
  extends to cover full-stack online ("conversation AND retrieved public
  documents go to OpenAI/Deepgram/Cartesia; your private records db is not
  mounted").

### Transport default

- `pickTransport()` defaults to `webrtc`; on offer failure the client
  falls back to `websocket` automatically (visible in the transport label,
  which already right-justifies). `?transport=` override keeps working.
  `aiortc` moves from the `webrtc` extra into base deps so `uv sync` is
  enough (builder verifies wheel availability on macOS/Linux/Windows
  first; if any platform lacks wheels, keep the extra and fall back to
  making `./lofi.sh start` warn).

### Language learning

- Offline: unchanged (wait-lesson, teach-only, split only).
- Online: when the lesson language is set, the online system prompt
  invites two-way practice in that language; Deepgram multilingual +
  Cartesia voice handle the audio. No watchdog machinery online (lookups
  are fast; there is no dead air to fill).

### Explainers (DEMO.md + README)

- Why SmallWebRTC vs WebSocket vs MoQ (UDP vs TCP, WSL2 boundary, AEC).
- Echo cancellation: browser `getUserMedia` gives it free; headphones
  route exists for the no-browser path.
- Model-size budget rule of thumb: a Q4-quantized model needs roughly
  `params/2` GB of RAM/VRAM for weights (7B→~4.5 GB, 14B→~9 GB,
  27B→~17.7 GB measured) plus 1–3 GB for KV cache/overhead; unified-memory
  Macs share that with the OS, so leave a third of RAM free; the STT/TTS
  models add ~2 GB. Presented as a table in the README with this repo's
  measured numbers.

## Demo beats this unlocks (for DEMO.md)

1. Offline passphrase + Whole Foods chain (existing).
2. Flip to online, reconnect: same tax question answers in ~2 s; the
   banner is loud; ask for the passphrase — it genuinely cannot know, and
   says so (the online prompt names its boundary: public documents only).
3. Flip offline again: passphrase answers. File-level privacy, visible.
4. Online + Spanish: hold a short two-way exchange while reviewing the
   (public) budget docs — the abstract's "practice your broken Spanish"
   line, fulfilled.

## Left to the builder

- Exact mode-passing mechanism through the pipecat runner endpoints
  (webrtc `/api/offer`, websocket `/ws-client`, MoQ serve mode) — must
  work for at least webrtc + websocket; MoQ may stay offline-only.
- `ingest.py` parameterization details and how `./lofi.sh index-rag`
  sequences the two builds; `./lofi.sh status` should show both indexes.
- Deepgram/Cartesia settings shapes on pipecat 1.13 (use the
  pipecat-context-hub MCP per repo rule), model/voice defaults as
  tunables.
- Online cascade prompt: start from the existing cascade system prompt,
  add the hosted/public-data framing and optional language-practice line.
  Prompt text is measured, not vibed — keep changes minimal and run the
  chitchat/records trio against the hosted model once wired.
- Test strategy: unit tests for config-file split, two-db ingest, mode
  plumbing, and online tool binding (mock the hosted services; no keys in
  CI). Keep 253 green.
- `.lofirc` migration ergonomics: on startup, if `.env` contains `LOFI_*`
  keys (the old world), print a one-line pointer to the rename.

## Open items already owned elsewhere

- ⚡/🧠 analyst toggle (punch item 2) — offline lever, after ralph.
- Windows end-to-end retest — after this lands (it changes the default
  transport story that retest must cover).
