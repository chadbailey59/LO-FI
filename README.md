# lofi — a local, fully-offline Pipecat voice bot

> WIP but isn't everything?

A voice bot that runs Key-Free & **100% offline**. Speech-to-text, the language model, 
and text-to-speech are all local services, and the "transport†" is your machine's own
audio hardware — the microphone and speakers.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) **≥ 1.13**: the
split architecture rides `LLMWithBackend`/`BackendLLMWorker`, which first
shipped in that release.
This repo is meant to double as a clear, reproducible **example** of how to
wire up a fully-local Pipecat bot.

The v1 personality is a **private financial thinking partner**: something you can
talk through money decisions with, out loud, knowing nothing you say leaves the
computer.

†Audio is hard and there are a few ways to handle it in this scenario. See "how do you
solve a problem like echo cancellation?"

---

## Requirements

- **Apple Silicon Mac recommended** — Whisper-MLX (the default STT) uses Apple's
  MLX framework and only runs there. Intel Macs and Linux work too: they default
  to CPU STT (`faster_whisper`) automatically. Windows works **natively
  (PowerShell) or via WSL2** — see
  [Running on Windows](#running-on-windows).
- ~15 GB free disk for the models
- Python **3.12** is pinned as 3.14 is too new for the ML wheels.
- **A plain `uv sync` needs no compiler** — everything in the base install comes
  from a prebuilt wheel. Several deps (onnxruntime, numba/llvmlite, cryptography)
  have already dropped Intel-mac wheels, so `pyproject.toml` pins those back to
  their last Intel-mac release under `[tool.uv]`. If `uv sync` ever starts
  building something from source, run `python3 scripts/check_wheels.py` to see
  which platform lost a wheel.
  The one exception is PyAudio, which has no macOS/Linux wheels — so it is an
  opt-in extra (`--extra local-audio`) needed only by the headphones front-end.
- **Pipecat is pinned `>=1.13,<1.14`** — the first release carrying the split
  architecture's building blocks (`LLMWithBackend`, `BackendLLMWorker`). Note:
  pipecat ≥ 1.8 drops Intel-mac wheels entirely (its core dep `loudness`
  ships none) — **Intel macs use the frozen [`intel-mac` branch](../../tree/intel-mac)**
  (pipecat 1.7, cascade + all three transports).
- **[uv](https://docs.astral.sh/uv/)** — Python package manager.
- **[Ollama](https://ollama.com/)** — serves the local LLM.
- **PortAudio** OR **any web browser** — PyAudio's native dependency / audio
  handling. Browser front-ends need neither PortAudio nor a compiler.

---

## Quickstart Setup

### 0. The [short short version](https://www.youtube.com/watch?v=5X4HYA-lB-U):

> [!NOTE]
> The first pull will take a few minutes to download the models.

#### Browser-based (for echo cancellation):
```bash
git clone git@github.com:vipyne/LO-FI.git && cd LO-FI
uv sync
bash scripts/run_ollama.sh
uv run python scripts/prefetch_models.py
```
Ctrl+C; then turn off wi-fi if you want to show off and then:

```bash
./lofi.sh start
```
Open http://localhost:7860 in any browser, click Connect & have a conversation.

Or...

#### PyAudio & headphones (for echo cancellation):

> [!IMPORTANT]
> Use headphones 🎧

```bash
git clone git@github.com:vipyne/LO-FI.git && cd LO-FI
brew install portaudio            # Debian: sudo apt install portaudio19-dev
uv sync --extra local-audio       # the extra adds PyAudio (needs PortAudio)
bash scripts/run_ollama.sh
uv run python scripts/prefetch_models.py
```
Ctrl+C; then turn off wi-fi if you want to show off and then:

```bash
./lofi.sh start -t headphones
```
Have a conversation.

## Setup

### 1. Clone, install system deps, and sync the environment

```bash
git clone git@github.com:vipyne/LO-FI.git && cd LO-FI
uv sync                       # creates .venv and installs everything (Python 3.12)
```

Starting from a bare Linux box or a fresh Windows WSL2 (Ubuntu) shell — no git,
no uv, no Ollama yet? This is the whole setup:

```bash
# deps
sudo apt-get update && sudo apt-get install -y git curl 
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env        # or restart the shell so `uv` is on PATH
curl -fsSL https://ollama.com/install.sh | sh

# code
git clone https://github.com/vipyne/LO-FI.git && cd LO-FI
uv sync
```

(On WSL2, clone into the Linux filesystem — `~/`, not `/mnt/c/…`.)

> [!NOTE]
> **On Windows?** The whole walkthrough — WSL2 install link, RAM-sized model
> picks, why WebSocket is the only transport that crosses the WSL2 boundary —
> lives in [Running on Windows](#running-on-windows).

That is all you need for the browser front-ends. For the headphones front-end
(`bot.py`), PyAudio has to compile against PortAudio, so install it and opt into
the extra:

```bash
brew install portaudio                # Debian: sudo apt install portaudio19-dev
uv sync --extra local-audio
```

> [!NOTE]
> `uv sync` uninstalls any extra you don't pass, so keep listing the ones you
> want: `uv sync --extra local-audio --extra piper`. (`./configure.sh -i` preserves
> whatever is already installed.)

Optionally copy the settings template (everything is optional — the bot runs with an
empty or absent `.lofirc`):

```bash
cp lofirc.example .lofirc
```

### 2. Fetch the models (the one-time online step)

Four model-backed components need weights. Two download from Hugging Face
(anonymously — none are gated); the LLM is pulled by Ollama. Silero VAD ships
*inside* the Pipecat package, so it downloads nothing.

All checkpoints are steered into **one directory** — `LOFI_MODEL_DIR`, default
`./models/` (gitignored) — so everything the bot needs lives next to the code.
Every engine follows it, so you can move the whole lot anywhere:

```bash
mv ./models /Volumes/T7/lofi-models
echo 'LOFI_MODEL_DIR=/Volumes/T7/lofi-models' >> .env
```

Absolute paths and `~` both work; relative paths resolve against the repo root,
not your shell's cwd. `config.py` and `scripts/model_dir.sh` implement the same
rules, so Python and the shell scripts always agree.

**One command fetches everything:**

```bash
uv run python scripts/prefetch_models.py
```

Pulls the Ollama models (`LOFI_LLM_MODEL` ~4.7 GB, `LOFI_RAG_LLM_MODEL`
~5.2 GB, plus `LOFI_EMBED_MODEL` ~0.3 GB) into `$LOFI_MODEL_DIR/ollama` —
starting a temporary `ollama serve`
if none is running — downloads the configured STT engine's weights
(Whisper-MLX `large-v3-turbo`, ~1.5 GB, into `$LOFI_MODEL_DIR/huggingface`;
faster-whisper/Moonshine on non-Apple-Silicon) and Kokoro's ONNX model +
voices (~350 MB) into `$LOFI_MODEL_DIR/kokoro`, and load-checks the bundled
Silero VAD (no download). Run this **once, while online**;
after it finishes the bot can run with Wi-Fi off.

Approximate total download: **~12 GB** (4.7 GB front LLM + 5.2 GB analyst +
0.3 GB embeddings + 1.5 GB Whisper + 0.35 GB Kokoro).

To serve the LLM for the bot (and re-pull `LOFI_LLM_MODEL` if needed):

```bash
bash scripts/run_ollama.sh
```

This relocates Ollama's model store to `./models/ollama`, starts `ollama serve`,
and keeps the server running in the foreground for the bot. Leave it running
(or re-run it) whenever you use the bot — it's the local LLM server.

### 3. Run

> [!IMPORTANT]
> Use headphones 🎧

With the Ollama server (`bash scripts/run_ollama.sh`) running:

```bash
uv run bot.py
```

The bot speaks a short greeting, then listens. Talk to it; it replies through your
speakers. Talk over it and it yields (barge-in). Press **Ctrl-C** to stop.

**One-command launch:** `./lofi.sh start` brings up the repo-local Ollama server
(if it isn't already running), prints the exact models in play, and serves the
MoQ browser bot — so you can skip the manual `run_ollama.sh` step
(`./lofi.sh stop` shuts down what it started). For the headphones front-end run
`uv run bot.py` directly — see
[echo cancellation](#how-do-you-solve-a-problem-like-echo-cancellation). Not sure
what your machine can handle? `./configure.sh` prints recommended STT/LLM/TTS cascades
sized to your hardware (add `-v` for the full hardware profile and per-slot model
catalogs ranked by fit, or `-i` to interactively pick a combo the script
sanity-checks against your hardware).

### 4. Run offline

Once the models are fetched:

1. Make sure the local Ollama server is running (`bash scripts/run_ollama.sh`).
2. **Turn off Wi-Fi / enable Airplane Mode.** (It won't use the internet if you don't turn off the internet. This is just showing off.)
3. `uv run bot.py` and hold a conversation.

With `LOFI_LOG_LEVEL=DEBUG` (the default) you can watch the logs and confirm no service
reaches out to the network after the warm-up.

---

## Running on Windows

Two routes, both verified end-to-end on real hardware (2026-10-14):

- **Native (simplest):** PowerShell + uv, no Linux layer, full machine RAM.
  MoQ isn't available natively (its `moq-ffi` library ships no Windows
  wheels) — use the WebSocket transport.
- **WSL2:** the bot runs in Linux, your browser/mic/speakers stay on Windows.
  You get the bash tooling (`./lofi.sh`, `./configure.sh`), at the cost of
  installing WSL and a default RAM cap.

Either way, **expect speech-to-text to be the slow part**: without Apple
Silicon, Whisper runs on the CPU. Two knobs if it drags:
`LOFI_FASTER_WHISPER_MODEL` (a smaller model than the `DISTIL_MEDIUM_EN`
default is faster and less accurate), or the tiny CPU-first engine:
`uv sync --extra moonshine` + `LOFI_STT_ENGINE=moonshine`.

### Route A: native Windows (PowerShell)

No admin needed. In PowerShell:

```powershell
# one-time tools
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Git.Git          # if you don't have git
winget install --id Ollama.Ollama    # the local LLM server

# code + env (uv fetches its own Python 3.12)
git clone https://github.com/vipyne/LO-FI.git
cd LO-FI
uv sync

# models (one-time, online)
uv run python scripts/prefetch_models.py

# run — the bash launcher (lofi.sh) doesn't run natively; start the bot directly
uv run python bot_browser.py --host localhost --port 7860
```

Then open **`http://localhost:7860`** in any browser,
Connect, allow the mic, talk — WebSocket is the default transport.
Env vars in PowerShell are session-scoped assignments, e.g.:

```powershell
$env:LOFI_LLM_MODEL = "llama3.2:1b"
$env:LOFI_ARCHITECTURE = "cascade"
```

Optional: `uv sync --extra webrtc`
enables `?transport=webrtc` — native loopback UDP works, unlike WSL2, so
WebRTC's Opus-encoded path is available here.

### Route B: WSL2

#### 1. Install WSL2 (one-time)

Follow Microsoft's guide: <https://learn.microsoft.com/windows/wsl/install>.
The short version: open an **admin** terminal (Win+X → "Terminal (Admin)"),
run `wsl --install`, reboot, and create your Ubuntu username/password when the
first launch asks. No special networking configuration is needed — the
WebSocket transport below works in WSL2's default (NAT) networking mode.

#### 2. Set up lofi inside WSL

Open Windows Terminal → the **Ubuntu** profile (the ˅ dropdown next to the
new-tab button), then run the bare-Linux setup block from
[Setup step 1](#1-clone-install-system-deps-and-sync-the-environment) — deps
(git, curl, uv, Ollama), clone, `uv sync`. Two WSL-specific rules:

- **Clone into the Linux filesystem** (`~/`), never `/mnt/c/…` — the Windows
  mount is slow and flaky for this workload.
- Everything below runs **inside the Ubuntu tab**; only the browser runs on
  Windows.

#### 3. Pick models that fit your RAM

WSL shares the machine's RAM, and every resident model has to fit —
`./configure.sh` profiles the machine and ranks model combos that fit (`-i`
to pick interactively). The default pair (qwen2.5:7b front + qwen3:8b
analyst) wants **16 GB+**. On a ~8 GB machine, drop to a
single small model in `cascade` mode before prefetching:

```bash
export LOFI_ARCHITECTURE=cascade LOFI_LLM_MODEL=llama3.2:1b LOFI_RAG_LLM_MODEL=llama3.2:1b
uv run python scripts/prefetch_models.py
./lofi.sh index-rag          # optional: build the RAG index
```

That tier is **proof-of-life, not accuracy** — it converses and exercises the
whole pipeline, but a 1B model fumbles records questions (measured: as a
front model it invents numbers on ~1 in 3 data questions, which is why the
small-machine recipe uses `cascade`, not the `split` default). Real
records-accuracy wants the 16 GB tier — the
[model table](#three-architectures-one-env-var) shows what each step up buys.

#### 4. Run

```bash
./lofi.sh start
```

(`lofi.sh` detects WSL and skips its browser auto-open — the right browser is
on the Windows side.) Then, in **a browser on Windows**, open:

```
http://localhost:7860
```

Click **Connect**, allow the mic, talk. WebSocket — the default transport —
is the only one that works here: MoQ's QUIC and WebRTC's media are both UDP,
which never crosses the Windows↔WSL boundary (verified in NAT and mirrored
modes, Hyper-V firewall open, raw-UDP probe). The WebSocket transport is pure
TCP and crosses the boundary the same way the web page itself does. Don't
force `?transport=moq` or `?transport=webrtc` here — each spends ~15–30 s
failing before the fallback chain lands back on websocket.

### Windows troubleshooting

- **WSL only sees about half your RAM** — that's its default cap, and it's
  what `./configure.sh` and Ollama see. A 16 GB machine shows ~8 GB inside
  Ubuntu. Raise it: edit `%UserProfile%\.wslconfig` on the Windows side
  (`vi /mnt/c/Users/<you>/.wslconfig` from bash works) to

  ```ini
  [wsl2]
  memory=12GB
  ```

  then `wsl --shutdown` in PowerShell and relaunch. With 12 GB the real
  16 GB-tier model pair usually fits.
- **Forgot your Ubuntu sudo password**: in PowerShell, `wsl -d Ubuntu -u root`,
  then `passwd <your-username>`, `exit`. (A once-a-day MOTD error in the root
  shell is harmless noise.)
- **First reply after startup is slow**: that's the model loading into RAM;
  `OLLAMA_KEEP_ALIVE=30m` keeps it resident afterwards.
- **A `(could not resolve models — run 'uv sync' and retry)` line at start**:
  cosmetic; the bot still starts.
- **Editing Windows-side files from bash**: they're under `/mnt/c/Users/<you>/…`
  (e.g. `vi /mnt/c/Users/<you>/.wslconfig`) — handy, but keep the repo itself
  in `~/`.

---

## How do you solve a problem like echo cancellation

### Use headphones

Because reasons, it's much closer to impossible than just impractical to get native 
macOS AEC (Acoustic Echo Cancellation) to work with pyaudio. Use headphones and 
the bot won't keep interrupting itself.

### Use the web browser's `getUserMedia`

Another fantastic workaround is to use a browser. Not the internet, just the web 
browser. Do this and 🎉, you have echo cancellation.

`./lofi.sh start` brings up Ollama and serves a local page (still
fully offline — the browser talks to the bot over loopback, no internet):

```bash
./lofi.sh start    # open http://localhost:7860 in any browser
```

The default browser transport is **WebSocket** — pure TCP, works in every
browser and across every boundary (including Windows↔WSL2, where UDP never
crosses). The same bot serves two more, forced with `?transport=moq` /
`?transport=webrtc`; a param-forced transport that fails to connect falls
back automatically (**moq → webrtc → websocket**):

- **MoQ over QUIC** (`?transport=moq`): media-over-QUIC via WebTransport,
  which Safari and Firefox don't ship yet (Chrome/Edge only).
- **SmallWebRTC** (`?transport=webrtc`, `uv sync --extra webrtc`): the
  classic voice-bot transport, UDP media over loopback.

---

## No secrets, no keys

In the default, 100% local configuration there are **no API keys** anywhere in
this project, and nowhere to put one:

- **Ollama** pulls the LLM from its own public registry and serves it locally.
- **Whisper-MLX and Kokoro** download anonymously from Hugging Face (none are
  gated); **Silero VAD** ships bundled with Pipecat.

`.lofirc`/`.env` are **config only** — model names, a voice, device indices,
cache paths. They are gitignored, but nothing secret ever belongs in `.lofirc`.
The single network event in the bot's entire lifecycle is the one-time,
anonymous model download in step 2.

---

## What's inside

| Component | Service | Notes |
|---|---|---|
| Speech-to-text | `WhisperSTTServiceMLX` *(default)* | Apple-Silicon-optimized Whisper via MLX. Alternatives via `LOFI_STT_ENGINE`: `faster_whisper` (CPU), `moonshine` (tiny CPU ONNX) |
| Language model | `qwen2.5:7b` front + `qwen3:8b` records analyst *(defaults)* via **Ollama** | Local, OpenAI-compatible endpoint; env-configurable. The split architecture's analyst backend (`LOFI_RAG_LLM_MODEL`) answers from your records. |
| Text-to-speech | `KokoroTTSService` *(default)* | Natural local neural voice (kokoro-onnx). Alternative via `LOFI_TTS_ENGINE`: `piper` |
| Turn-taking | Silero VAD | Barge-in / interruptions, fully local (bundled with Pipecat) |
| Transport | `WebSocket` (browser) *(default)* | free echo cancellation via `getUserMedia`; works in any browser. `?transport=moq` / `?transport=webrtc` opt into the others |
| Alternative transport | `LocalAudioTransport` | PyAudio mic + speaker I/O (requires headphones) |

---

## Retrieval (your transactions and documents)

The bot answers real questions about your own money — fully offline, no vector
database server. Drop `.txt`, `.md`, `.pdf`, or `.csv` files into `./data/`,
then:

```bash
ollama pull nomic-embed-text   # one-time, while online
./lofi.sh index-rag
```

### Two architectures, one env var

`LOFI_ARCHITECTURE` picks how retrieval and conversation share the work:

- **`split`** (the default) — a conversational front model (reasoning off)
  delegates records work to a reasoning analyst backend via Pipecat's
  `LLMWithBackend`/`BackendLLMWorker`. The front holds one domain-named
  builtin (`ask_financial_records`); the backend (`LOFI_RAG_LLM_MODEL`,
  reasoning on, native API) holds the four retrieval tools and answers from
  your records. The backend's thinking stays in the native API's `reasoning`
  field and never reaches TTS; its answers are written back into the
  conversation, so follow-ups work.
- **`cascade`** — one model does the talking and the retrieving. It
  holds four typed tools over the index; any question about transactions,
  spending, merchants, accounts, totals, or saved documents turns into tool
  calls, and the model answers only from what they return — with a short
  "Let me check your records." covering the lookup. The simplest architecture
  to read and the stable fallback.

(Two earlier architectures live in git history: `observer`, an analyst
listening to every turn in parallel, and split-v1, a generic delegate tool
whose routing recall capped at 0.51 on small models.)

Pick models for your hardware — same code, env vars
(`LOFI_ARCHITECTURE=split LOFI_RAG_LLM_MODEL=<tag> ./lofi.sh start`).
Measured on an internal eval (28 gold queries incl. multi-turn, M4 Pro;
observer rows are historical — the split-v2 eval stage is pending):

| Config | RAM | Answer accuracy | Full lookup, mean |
|---|---|---|---|
| observer (retired): `qwen2.5:7b` front + `qwen3:8b` analyst | ≥ 16 GB | **0.93** | ~20 s accurate mode* |
| observer (retired): `qwen2.5:7b` front + `nemotron-3-nano:4b` analyst | ≥ 16 GB (8 GB with smaller STT) | 0.79 | ~14 s accurate mode* |
| `qwen2.5:14b` cascade | ≥ 24 GB | 0.65 | ~8.2 s |
| `qwen2.5:7b` cascade | ≥ 16 GB | 0.43 | ~4.0 s |

*Latencies measured on a loaded machine — treat as a ranking, not a benchmark.

Reasoning models (deepseek-r1, qwen3 with thinking) are not supported as the
**pipeline** model — nothing strips thinking tokens before the voice. The
split backend is exactly where a reasoning model belongs: its thinking
stays in the native API's `reasoning` field and never reaches TTS.

An earlier version did RAG the way most tutorials do — chunk the CSV, embed the
chunks, retrieve by cosine similarity. It could not answer "how much did I
spend on groceries?" because the grocery rows were scattered across chunks and
nothing could add them up. Structured data wants typed queries, not similarity
search; that version lives in the git history.

The model never writes SQL itself — each tool takes whitelisted arguments and
builds a parameterized query, capped at five rows or chunks per result:

| Tool | For |
|---|---|
| `lookup` | "what did I pay Harborview?" — transaction id, or bm25 full-text over rows |
| `filter_rows` | "my five biggest expenses in March 2025" — one column/operator/value, ordered |
| `aggregate` | "how much did I spend on groceries in 2025" — count/sum/avg/min/max, transfers excluded unless asked |
| `semantic_search` | "what's in my tax notes?" — embedding KNN fused with full-text, over everything |

### One SQLite file

`./lofi.sh index-rag` builds a single database at `LOFI_RAG_DB` (default
`./models/rag-index/lofi.db`) holding typed transaction rows, an FTS5
full-text index, and a `sqlite-vec` vector table side by side. Ingest is
incremental: each file is hashed, unchanged files are skipped, edited files are
re-embedded alone, and deleted files are purged. Change the embedding model and
it rebuilds from scratch and says why.

CSV files are indexed row by row, and each row is also written out as a prose
sentence — `On February 5, 2025, spent $412.00 at Harborview Credit Union
(Auto Payment) from account VIP Plus Banking (...4821).` — so full-text and
embedding search hit
the same rows the typed columns do. Monthly per-category rollups are generated
alongside, so "what did March look like" has something to retrieve. Bank
statement exports work as-is.

`./lofi.sh status` shows the index and both model tags. After adding or
editing documents, re-run `./lofi.sh index-rag`. With no index the bot starts
without the tool and simply converses.

The older numpy index files (`chunks.jsonl`, `embeddings.npy`,
`manifest.json`) are inert — nothing reads them. They are left on disk rather
than deleted.

### Is it really reading my documents? The canary trick

Plant a fact the model cannot possibly know, and ask for it:

```bash
echo "The secret passphrase for the tax vault is BLUE PELICAN 47." > data/canary.txt
./lofi.sh index-rag
./lofi.sh start -t headphones
```

Ask *"what's the passphrase for the tax vault?"* — a correct answer can only
have come from retrieval, and in the browser UI the retrieved-context panel
prints one line per backend tool call (which tool, its arguments, how many rows
came back, how long it took) plus the composed answer. The same canary also
demonstrates the index lifecycle: delete `data/canary.txt` and ask again
without re-indexing — the bot still answers, because it reads the built index,
never `data/` itself. Re-run `./lofi.sh index-rag` and ask a third time — now
the passphrase is gone. Moral: the index only changes when you rebuild it.

For a negative control, `mv models/rag-index /tmp/` and restart: the bot starts
with no tool and answers ungrounded. To inspect the index directly,
`uv run python ingest.py stats`, or open
`models/rag-index/lofi.db` with any `sqlite3` client.

### Reference documents

`./data/` ships with two IRS publications so `semantic_search` has real tax
prose to retrieve next to your transactions:

- [Publication 17](https://www.irs.gov/pub/irs-pdf/p17.pdf) — Your Federal Income Tax (individuals)
- [Publication 334](https://www.irs.gov/pub/irs-pdf/p334.pdf) — Tax Guide for Small Business

Canonical sources for more:

- IRS forms, instructions, and publications: <https://www.irs.gov/forms-instructions>
- The Internal Revenue Code itself (26 U.S.C.): <https://uscode.house.gov/download/download.shtml>
- Treasury regulations (26 CFR): <https://www.ecfr.gov/current/title-26>

Plain-prose publications chunk well. Raw statute and regulation text does
not — deeply nested numbering scatters each rule across chunks — so `./data/`
carries pubs only. Drop in whatever reference documents fit your life
(leases, manuals, insurance policies) and re-run `./lofi.sh index-rag`.

---

## One place for all models: consolidate

HuggingFace defaults to `~/.cache/huggingface`, Ollama to `~/.ollama/models` —
models you pulled before lofi (or outside it) live there, and lofi uses them
from there automatically: the bot loads weights straight from whichever cache
has them, and `./lofi.sh status` prints the real path (its `downloaded`
section inventories every model on disk, with sizes).
`./lofi.sh consolidate` is an optional convenience on top — it adopts external
models into `LOFI_MODEL_DIR` with symlinks so one `ls -al ./models` browses
every model with its real path (and shows what a zip-and-go copy of the repo
would actually include):

```bash
./lofi.sh consolidate -n   # dry run: print what would be adopted
./lofi.sh consolidate      # create the links
```

- **HuggingFace, per model:** each model in your home cache that the lofi
  store lacks becomes a symlink inside `models/huggingface/hub/`, loadable
  through `LOFI_MODEL_DIR` like any owned model. A broken partial download in
  the lofi store is replaced by a link to a complete external copy.
- **Ollama, whole store:** models share content-addressed blobs, so adoption
  links `models/ollama -> ~/.ollama/models` when the lofi store is empty. If
  both stores contain models, consolidate reports the conflict and touches
  nothing (skipped while an ollama server is running).

Nothing outside the repo is ever moved, modified, or deleted; the only writes
are symlinks (plus removal of lofi-owned 0-byte `.incomplete` stubs).
`./lofi.sh status` labels adopted models `(borrowed → /real/path)`. Borrowed means borrowed: a re-download of that
model follows the symlink into your home cache, and zipping up the repo
excludes borrowed weights — delete the link and `ollama pull` /
`uv run python scripts/prefetch_models.py` to own a copy instead.

---

## Choosing local models for your machine

The workshop question: *what can this computer actually run?* The process:

1. **Profile the machine.** `./configure.sh` reads your RAM/CPU/GPU and prints
   STT/LLM/TTS combos sized to it (`-v` for the full report with per-slot model
   catalogs ranked by fit, `-i` to pick interactively). Rule of thumb: a
   quantized model takes roughly its parameter count in GB (a 7b ≈ 4-5 GB, a
   14b ≈ 9 GB), and *everything resident at once* has to fit — front model +
   analyst + embedder + STT + TTS. `run_ollama.sh` sets
   `OLLAMA_MAX_LOADED_MODELS=3` and `OLLAMA_KEEP_ALIVE=30m` so the working set
   stays loaded instead of thrashing.
2. **Measure, don't vibe.** The eval harness is the yardstick:
   `uv run python evals/run_evals.py --stage split` scores a candidate
   analyst on gold queries (accuracy, missed questions, latency);
   `--stage front` checks a candidate front model never invents numbers and
   defers records questions. `evals/sweep.sh` loops a list of candidates
   through both and prints a leaderboard — edit the candidate lists and run it
   against your own hardware.
3. **Read the right numbers.** `answer_accuracy` is the headline;
   `sentinel_on_data` is questions the analyst wrongly ignored;
   `silent_on_chitchat` catches an analyst that butts into small talk;
   `total_ms_mean` is what the wait feels like. A model that's 5 points more
   accurate but twice as slow is usually the wrong trade for live voice —
   that's what the fast/accurate toggle is for.
4. **Mind residency.** Swapping models mid-session costs a multi-second reload.
   The defaults keep fast mode on the same analyst weights (thinking off) for
   exactly this reason.

---

## Configuration

Every knob is an environment variable (read from `.lofirc` if present; hosted
API keys live in `.env`). All are optional — the shown value is the default.
See [`lofirc.example`](lofirc.example) for the copy-paste template.

**`LOFI_` means it's ours.** Anything read by this repo carries the prefix, so
you can tell at a glance what's safe to change and what belongs to someone else.
Exactly four variables are **external** — read by third-party software, keeping
their upstream names because renaming them would break the tool that reads them:

| External variable | Read by |
|---|---|
| `HF_HOME`, `HF_HUB_DISABLE_PROGRESS_BARS` | `huggingface_hub` |
| `OLLAMA_MODELS`, `OLLAMA_HOST` | the `ollama` binary |

If you already export one of those globally, it affects lofi too. Note that
`LOFI_OLLAMA_BASE_URL` is *ours* despite the name — it's the URL the bot dials,
not something ollama reads.

### Prompts and model defaults: `tunables.py`

Prompts are edited directly in [`tunables.py`](tunables.py) — there's no env
override for them; editing the file IS the mechanism. Model defaults also live
there, and `LOFI_*` env vars below still override them. Landmine comments mark
measured, byte-sensitive text — read them before editing.

| Variable | Default | What it does |
|---|---|---|
| `LOFI_LLM_MODEL` | `qwen2.5:7b` | Ollama tag for the conversation model (and, in `cascade`, the four retrieval tools). Same string `run_ollama.sh` pulls and the bot serves. Smaller machines: see the model table above. |
| `LOFI_ARCHITECTURE` | `split` | `split` (default) \| `cascade` — see [Two architectures](#two-architectures-one-env-var). |
| `LOFI_RAG_LLM_MODEL` | `qwen3:8b` | Ollama tag for the records analyst backend (`split`). Needs a model that emits real API tool calls — `nemotron-3-nano:4b` narrates them as JSON text and fails as the split backend. |
| `LOFI_RAG_FAST_LLM_MODEL` | `$LOFI_RAG_LLM_MODEL` | Analyst model for the UI's ⚡ fast mode (thinking always off there). A *distinct* fast model needs `OLLAMA_MAX_LOADED_MODELS=4` so nothing gets evicted. |
| `LOFI_WAIT_LESSON_LANGUAGE` | *(empty = off)* | `split` only. When set, the front teaches one phrase in that language while the analyst works. |
| `LOFI_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible Ollama endpoint (note the trailing `/v1`). The RAG embedder talks to the same host minus `/v1`. |
| `LOFI_EMBED_MODEL` | `nomic-embed-text` | Ollama model tag for RAG embeddings. Pull it with `ollama pull nomic-embed-text`. |
| `LOFI_RAG_DATA_DIR` | `./data` | The documents to index (`.txt`/`.md`/`.pdf`) — put files here, then `./lofi.sh index-rag`. Absolute, `~`, or relative-to-repo. |
| `LOFI_RAG_INDEX_DIR` | `$LOFI_MODEL_DIR/rag-index` | Directory the index lives in. |
| `LOFI_RAG_DB` | `$LOFI_RAG_INDEX_DIR/lofi.db` | The single SQLite file: typed rows, FTS5 full text, and `sqlite-vec` vectors. |
| `LOFI_RAG_TOP_K` | `4` | Legacy knob from the always-on injection era; nothing reads it now — the analyst's tools cap themselves at five results. |
| `LOFI_RAG_CHUNK_TOKENS` | `500` | Chunk budget in whitespace-split words. |
| `LOFI_RAG_CHUNK_OVERLAP` | `50` | Words repeated between consecutive chunks. |
| `LOFI_STT_ENGINE` | `whisper_mlx`* | STT engine `services.py` builds: `whisper_mlx`, `faster_whisper`, or `moonshine` (`uv sync --extra moonshine`). *Default is `faster_whisper` on non-Apple-Silicon machines. |
| `LOFI_WHISPER_MODEL` | `LARGE_V3_TURBO` | `MLXModel` member: `TINY`, `MEDIUM`, `LARGE_V3`, `LARGE_V3_TURBO`. Must match what you prefetched. |
| `LOFI_FASTER_WHISPER_MODEL` | `DISTIL_MEDIUM_EN` | faster-whisper model (when `LOFI_STT_ENGINE=faster_whisper`); downloads on first use. |
| `LOFI_MOONSHINE_MODEL` | `SMALL_STREAMING` | Moonshine model (when `LOFI_STT_ENGINE=moonshine`); downloads on first use. |
| `LOFI_TTS_ENGINE` | `kokoro` | TTS engine `services.py` builds: `kokoro` or `piper` (`uv sync --extra piper`; piper-tts is GPL-3.0). |
| `LOFI_KOKORO_VOICE` | `af_heart` | Kokoro voice id (e.g. `af_bella`, `am_michael`, `bf_emma`). |
| `LOFI_PIPER_VOICE` | `en_US-lessac-medium` | Piper voice id (when `LOFI_TTS_ENGINE=piper`); downloads (~60 MB) on first use into `./models/piper`. |
| `LOFI_INPUT_DEVICE_INDEX` | *(system default)* | PyAudio mic index. |
| `LOFI_OUTPUT_DEVICE_INDEX` | *(system default)* | PyAudio speaker index. |
| `LOFI_GREETING` | *"Hi. I'm your private, offline financial thinking partner…"* | Opening line spoken on startup. |
| `LOFI_GREETING_DELAY_SECS` | `1.0` | Delay before the greeting (lets the audio-out stream spin up). |
| `LOFI_MOQ_SUBSCRIBER_TIMEOUT` | `15.0` | Max seconds the MoQ bot holds its greeting for the browser's audio subscription (MoQ has no replay — speaking earlier drops the audio). On timeout it greets anyway. |
| `LOFI_MOQ_AUDIO_AHEAD_MS` | `500` | Max milliseconds of bot audio written ahead of real time over MoQ. TTS runs faster than real time; uncapped bursts overflow the browser's playback ring buffer and clip the start of every bot turn. Raise only if playback stutters mid-utterance. |
| `LOFI_LOG_LEVEL` | `DEBUG` | Loguru level for stderr. `DEBUG` surfaces each service's activity — handy for the offline check. |
| `LOFI_WEB_PORT` | `7860` | Port `bot_browser.py` serves on (used by `./lofi.sh start`). |
| `LOFI_OPEN_BROWSER` | `1` | `0` stops `./lofi.sh start` auto-opening the page. Auto-open is always skipped on WSL (the right browser is on the Windows side). |
| `LOFI_STATE_DIR` | `.lofi` | Where `./lofi.sh` records the PIDs of the processes it started — the only PIDs `./lofi.sh stop` will ever touch. Relative paths resolve against the repo root. |
| `LOFI_VAD_CONFIDENCE` | `0.7` | Silero speech-probability threshold (0–1) before audio counts as speech. |
| `LOFI_VAD_MIN_VOLUME` | `0.0` | Absolute-loudness gate. `0.0` disables it, which keeps turn detection level-independent across mics — raise toward `0.3`–`0.6` only if a noisy room false-triggers. |
| `LOFI_VAD_START_SECS` | `0.2` | Sustained speech before "user started speaking". |
| `LOFI_VAD_STOP_SECS` | `0.2` | Sustained silence before "user stopped speaking". |
| `LOFI_MODEL_DIR` | `./models` | **The one directory every model downloads into** — HF cache, Kokoro, Piper and Ollama all hang off it. Absolute, `~`, or relative-to-repo. Move it to relocate everything at once. |
| `HF_HOME` **[external]** | `$LOFI_MODEL_DIR/huggingface` | Hugging Face cache root (Whisper-MLX, faster-whisper, Moonshine). Set only to split HF out of the shared dir. *Advanced.* |
| `HF_HUB_DISABLE_PROGRESS_BARS` **[external]** | `1` | Silences HuggingFace download progress bars, which otherwise clutter the bot's logs. *Advanced.* |
| `LOFI_KOKORO_MODEL_PATH` | `$LOFI_MODEL_DIR/kokoro/kokoro-v1.0.onnx` | Kokoro ONNX model path. *Advanced.* |
| `LOFI_KOKORO_VOICES_PATH` | `$LOFI_MODEL_DIR/kokoro/voices-v1.0.bin` | Kokoro voices bundle path. *Advanced.* |
| `LOFI_PIPER_DOWNLOAD_DIR` | `$LOFI_MODEL_DIR/piper` | Where Piper voices download. *Advanced.* |
| `OLLAMA_MODELS` **[external]** | `$LOFI_MODEL_DIR/ollama` | Ollama store location (used by `run_ollama.sh`). Use an absolute path if you set it — ollama resolves relative paths against its own cwd. *Advanced.* |
| `OLLAMA_HOST` **[external]** | `127.0.0.1:11434` | Host the Ollama server binds to (used by `run_ollama.sh`). *Advanced.* |

Changing `LOFI_LLM_MODEL` swaps which local model answers; changing `LOFI_KOKORO_VOICE`
changes the voice you hear.

---

## Repository layout

Both bots share one offline brain (`pipeline.py` — the same STT → VAD → LLM →
TTS pipeline); they differ only in the transport. The STT/LLM/TTS services are
built in `services.py`, dispatched on `LOFI_STT_ENGINE` / `LOFI_TTS_ENGINE` —
so swapping engines (via `.lofirc` or `./configure.sh -i`) never touches a bot
file you may have customized.

```
lofi/
├── bot.py                    # CLI / headphones — LocalAudioTransport
├── bot_browser.py                # browser bot — websocket/moq/webrtc, serves client/dist
├── pipeline.py               # the one build_pipeline() every bot shares
├── services.py               # STT/LLM/TTS builders, engine-dispatched
├── config.py                 # env-driven settings, zero-config defaults
├── tunables.py               # every prompt + model defaults — THE file to edit
├── rag_agent.py              # the records analyst (native Ollama API, thinking on)
├── tools.py                  # the four retrieval tool schemas + Pipecat wiring
├── retrieval.py              # lookup / filter_rows / aggregate / semantic_search
├── store.py                  # the SQLite schema (rows + FTS5 + sqlite-vec)
├── ingest.py                 # extract → serialize → chunk → embed → index
├── silence_watchdog.py       # "I'm still working on it" during long lookups
├── spoken_text_filter.py     # TTS filter: "$3,000" → "three thousand dollars"
│
├── lofi.sh                   # start | stop | status | index-rag | consolidate | configure
├── configure.sh              # what can this machine handle? (-v full report, -i model picker)
│
├── scripts/
│   ├── model_dir.sh          # resolves LOFI_MODEL_DIR for the shell scripts
│   │                         #   (config.py does the same for Python)
│   ├── run_ollama.sh         # relocate Ollama store + serve + pull the LLM
│   ├── prefetch_models.py    # one-time online warm-up for every default model
│   ├── print_models.py       # the "what's running" panel / status lines
│   ├── check_wheels.py       # tripwire: every dep must have a wheel per platform
│   ├── check_audio.py        # diagnostic: raw mic input level meter
│   └── check_vad.py          # diagnostic: Silero VAD confidence/volume vs thresholds
│
├── client/                   # the browser UI (Vite + TypeScript; prebuilt in client/dist)
├── evals/                    # eval harness: run_evals.py, queries.jsonl, sweep.sh
├── tests/                    # offline pytest suite (no network, fake embedder)
│
├── lofirc.example            # documented config knobs (copy to .lofirc)
├── env.example               # the three hosted API key names (copy to .env)
├── .python-version           # 3.12
├── pyproject.toml            # uv project + pinned deps
├── uv.lock                   # locked dependency versions
│
└── models/                   # ALL checkpoints live here — $LOFI_MODEL_DIR,
    │                         # relocatable (gitignored; created by setup)
    ├── huggingface/          # Whisper-MLX + faster-whisper + Moonshine (HF cache)
    ├── kokoro/               # Kokoro onnx + voices
    ├── piper/                # Piper voices (if LOFI_TTS_ENGINE=piper)
    ├── rag-index/            # lofi.db — the RAG index (./lofi.sh index-rag)
    └── ollama/               # Ollama LLM store
```

---

## Not financial advice

The bot is a private *thinking partner*, not a licensed financial advisor. It has no
access to your real accounts and won't invent your numbers. For big, irreversible,
or high-stakes decisions, confirm with a qualified professional.
Ha, claude wrote this^ when I said I wanted to create a fully offline bot that I could 
talk to about my personal finances. But yes, _always_ consult a human after consulting
a bot.

## Emojis
claude did _not_ add enough/any emojis so: 
🎉🎊🥳🎈🎁🎀🌟✨💫⭐🌈🔥💥⚡☀️🌙🌛🌜🌞🪐🌍🌎🌏🌊🏔️⛰️🌋🗻🏕️🏖️🏜️🏝️🌅🌄🌇🌆🏙️🌃🌌🎆🎇🌠🌉🍀🌿🍃🌾🌵🌴🌳🌲🎄🌰🍄🌻🌺🌸🌼🌷🌹🥀💐🏵️🌊🐠🐟🐬🐳🐋🦈🐙🦑🦐🦞🦀🐚🐌🦋🐛🐝🐞🦗🕷️🦂🐢🐍🦎🦖🦕🐙🦭🦦🦥🐾🐕🐈🐇🐿️ 🦫🦃🐔🐓🐣🐤🐥🦆🦢🦅🦉🦚🦜🕊️🐧🐦🦩🦨🐘🦏🦛🐪🐫🦒🦓🐂🐃🐄🐎🐖🐏🐑🦙🐐🦌🐕‍🦺🐈‍⬛🦮🐩🐾🍎🍏🍐🍊🍋🍌🫐🍈🍒🍑🥭🍍🥥🥝🍅🍆🥑🥦🥬🥒🌶️ 🫑🌽🥕🫒🧄🧅🥔🍠🥐🥯🍞🥖🥨🧀🥚🍳🧈🥞🧇🥓🥩🍗🍖🌭🍔🍟🍕🫓🥪🥙🧆🌮🌯🫔🥗🥘🫕🥫🍝🍜🍲🍛🍣🍱🥟🦪🍤🍙🍚🍘🍥🥠🥮🍢🍡🍧🍨🍦🥧🧁🍰🎂🍮🍭🍬🍫🍿🍩🍪🌰🥜🍯🥛🍼☕🫖🍵🧃🥤🧋🍶🍺🍻🥂🍷🥃🍸🍹🧉🍾🧊🥄🍴🍽️🥣🥡🥢🧂⚽🏀🏈⚾🥎🎾🏐🏉🥏🎱🪀🏓🏸🏒🏑🥍🏏🪃🥅⛳🪁🏹🎣🤿🥊🥋🎽🛹🛼🛷⛸️ 🥌🎿⛷️ 🏂🪂🏋️ 🤼🤸⛹️ 🤺🤾🏌️ 🏇🧘🏄🏊🤽🚣🧗🚵🚴🏆🥇🥈🥉🏅🎖️ 🏵️ 🎗️ 🎫🎟️ 🎪🤹🎭🩰🎨🎬🎤🎧🎼🎹🥁🎷🎺🎸🪕🎻🎲♟️🎯🎳🎮🎰🧩
