"""config.py — env-driven settings for the fully-offline voice bot.

Single source of truth for every configurable knob. All values come from the
process environment (optionally populated from ``./.lofirc`` for settings and
``./.env`` for the hosted API keys), with sensible defaults so
``uv run bot.py`` works with **zero configuration**.

Two kinds of settings live here:

1. **Cache-dir vars** (``HF_HOME``, Kokoro/Piper paths, ``OLLAMA_MODELS``), all
   derived from ``LOFI_MODEL_DIR`` — the single directory every downloaded
   model lives under, which can be anywhere on the machine. These must be set
   *before* any model library is imported — Hugging Face freezes its cache root
   at ``huggingface_hub`` import time — so they are established here at *import
   time* via ``os.environ.setdefault(...)``. Importing this module therefore
   steers every engine's cache into that one tree (matching
   ``scripts/prefetch_models.py`` and ``scripts/model_dir.sh``, so the bot reads
   weights from exactly where the prefetch wrote them — the key to offline
   runs). It does NOT load any model, touch audio hardware, or hit the network.

   Because of the freeze-at-import behaviour, ``bot.py`` imports this module
   *before* it imports any ``pipecat`` service.

2. **Runtime settings** (model names, voice, device indices, greeting, log
   level). Exposed as getter functions read at *call time*, so they pick up
   whatever ``load_dotenv()`` in ``main()`` applied before the builders run.

The *defaults* for models, voices, STT/TTS engines, architecture and the
greeting live in ``tunables.py`` alongside every prompt — one file to edit. The
getters here read them as fallbacks, so ``LOFI_*`` env vars still override.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Model/voice/greeting defaults live in tunables.py — THE file users edit. The
# from-import keeps `config.DEFAULT_*` / `config.IS_APPLE_SILICON` working for
# services.py and the tests, and the getters below still let LOFI_* env vars win.
from tunables import (
    IS_APPLE_SILICON,
    DEFAULT_STT_ENGINE,
    DEFAULT_TTS_ENGINE,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_MOONSHINE_MODEL,
    DEFAULT_PIPER_VOICE,
    DEFAULT_KOKORO_VOICE,
    DEFAULT_LLM_MODEL,
    DEFAULT_RAG_LLM_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_ARCHITECTURE,
    DEFAULT_GREETING,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "./models"

# --- Repo-local cache defaults, established at import time ------------------
# Load the config files FIRST — before resolving LOFI_MODEL_DIR or
# setdefault-ing any cache var — so user overrides win over these defaults.
# Mirrors scripts/model_dir.sh (bash) and scripts/prefetch_models.py.

# Settings live in `.lofirc`, the three hosted API keys in `.env`.
SETTINGS_FILE_NAME = ".lofirc"
KEYS_FILE_NAME = ".env"
HOSTED_KEY_NAMES = ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY")


def _load_config_files(root: Path | None = None) -> None:
    """Load ``.lofirc`` (settings), then ``.env`` (keys), as dotenv defaults.

    Both loads are no-override, so the shell environment beats both files and
    ``.lofirc`` beats ``.env`` for a name in both. ``LOFI_ENV_FILE`` overrides
    which settings file loads (lofi.sh and configure.sh honor the same
    variable; tests point it at /dev/null for isolation); the keys file is
    always ``.env`` at the repo root.
    """
    root = Path(root) if root is not None else REPO_ROOT
    try:
        from dotenv import load_dotenv
    except Exception:  # python-dotenv is a dep, but never hard-fail on import
        return
    load_dotenv(os.environ.get("LOFI_ENV_FILE") or root / SETTINGS_FILE_NAME)
    load_dotenv(root / KEYS_FILE_NAME)


_load_config_files()


def _resolve_repo_path(raw: str) -> Path:
    """Shared resolution rules for every LOFI_* directory var: a leading ``~``
    is expanded, and relative paths resolve against the REPO ROOT rather than
    the process's cwd, so the value means the same thing no matter where the
    bot was launched from.
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    # normpath, not Path.resolve(): resolve() would follow symlinks (on macOS
    # /tmp/x comes back as /private/tmp/x), which surprises anyone who
    # deliberately points this at a symlinked disk and would also drift from
    # scripts/model_dir.sh, which cannot resolve symlinks for a path that does
    # not exist yet. Collapsing "." / ".." textually is all that is needed.
    return Path(os.path.normpath(path))


def _resolve_model_dir() -> Path:
    """Absolute path of the one directory holding every downloaded model.

    ``LOFI_MODEL_DIR`` (default ``./models``) can point anywhere on the machine
    — an external disk, a shared cache, whatever. ``scripts/model_dir.sh``
    implements the identical resolution rules for the shell scripts that cannot
    import this module.
    """
    return _resolve_repo_path(os.getenv("LOFI_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR)


MODELS_DIR = _resolve_model_dir()
KOKORO_DIR = MODELS_DIR / "kokoro"

# Every model store hangs off MODELS_DIR so one directory holds the lot. These
# are setdefault, not assignment: an explicitly-set HF_HOME (etc.) still wins.
#
# Hugging Face cache root — Whisper-MLX, faster-whisper AND Moonshine weights
# all land here. Steered into MODELS_DIR so a warmed-up tree is found offline
# instead of ~/.cache.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
# Silence HuggingFace's download progress bars ("Fetching N files", "Reconstruction
# complete") — they clutter the bot's logs on first-use model fetches. Weights are
# still downloaded; only the noisy tqdm output is suppressed.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Kokoro ONNX + voices bundle (also honored directly by build_tts via these vars).
os.environ.setdefault("LOFI_KOKORO_MODEL_PATH", str(KOKORO_DIR / "kokoro-v1.0.onnx"))
os.environ.setdefault("LOFI_KOKORO_VOICES_PATH", str(KOKORO_DIR / "voices-v1.0.bin"))
# Piper's voice download dir. Nothing external reads this — services.py passes
# it to PiperTTSService(download_dir=...) — but it is kept in the environment so
# configure.sh (which cannot import this module) probes the same directory.
os.environ.setdefault("LOFI_PIPER_DOWNLOAD_DIR", str(MODELS_DIR / "piper"))
# Ollama's store. Read by the `ollama` BINARY, not by any Python package: the bot
# only talks to the server over HTTP. scripts/run_ollama.sh is what actually
# exports it into `ollama serve`; setting it here keeps configure.sh and
# print_models.py reporting the same path the server uses.
os.environ.setdefault("OLLAMA_MODELS", str(MODELS_DIR / "ollama"))


# --- Legacy env names (everything we own gained a LOFI_ prefix) ------------
# Renaming is silent by nature: an old key in a config file is simply never
# read, and the bot comes up on defaults with no hint that the setting was
# dropped. Detect and say so instead.
_LEGACY_ENV_NAMES = (
    "STT_ENGINE", "TTS_ENGINE", "WHISPER_MODEL", "FASTER_WHISPER_MODEL",
    "MOONSHINE_MODEL", "KOKORO_VOICE", "KOKORO_MODEL_PATH", "KOKORO_VOICES_PATH",
    "PIPER_VOICE", "PIPER_DOWNLOAD_DIR", "LLM_MODEL", "OLLAMA_BASE_URL",
    "INPUT_DEVICE_INDEX", "OUTPUT_DEVICE_INDEX", "GREETING",
    "GREETING_DELAY_SECS", "LOG_LEVEL", "VAD_CONFIDENCE", "VAD_MIN_VOLUME",
    "VAD_START_SECS", "VAD_STOP_SECS", "WEB_PORT",
)


def _dotenv_keys(path: Path) -> list[str]:
    """Variable names assigned in a dotenv-style file; [] when unreadable."""
    try:
        if not path.is_file():
            return []
        lines = path.read_text().splitlines()
    except OSError:
        return []
    keys = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0].strip())
    return keys


def legacy_settings_in_env(root: Path | None = None) -> list[str]:
    """``LOFI_*`` settings sitting in ``.env``, which now holds keys only.

    Both files are dotenv loads, so the values still apply — the names are
    reported so one startup line can point them at ``.lofirc``.
    """
    root = Path(root) if root is not None else REPO_ROOT
    return [k for k in _dotenv_keys(root / KEYS_FILE_NAME) if k.startswith("LOFI_")]


def legacy_env_keys() -> list[str]:
    """Pre-prefix keys still present in ``./.lofirc`` or ``./.env``.

    Only the repo's own files are scanned, never the ambient environment: names
    like ``LOG_LEVEL`` and ``GREETING`` are generic enough that someone's shell
    profile could export them for entirely unrelated reasons, and warning about
    that would be noise. A key in a config file, by contrast, was meant for this bot.
    """
    found = []
    for name in (SETTINGS_FILE_NAME, KEYS_FILE_NAME):
        for key in _dotenv_keys(REPO_ROOT / name):
            if key in _LEGACY_ENV_NAMES and key not in found:
                found.append(key)
    return found


_legacy = legacy_env_keys()
if _legacy:
    print(
        "lofi: WARNING — these .env keys are no longer read; every setting this "
        "repo owns now takes a LOFI_ prefix (the unprefixed names left are the "
        "four read by huggingface_hub and ollama — see lofirc.example):\n"
        + "\n".join(f"    {k}  ->  LOFI_{k}" for k in _legacy),
        file=sys.stderr,
    )

_misplaced = legacy_settings_in_env()
if _misplaced:
    print(
        "lofi: settings belong in .lofirc now, .env holds only "
        f"{', '.join(HOSTED_KEY_NAMES)} — move from .env: {', '.join(_misplaced)} "
        "(they still apply)",
        file=sys.stderr,
    )


# --- Defaults for the runtime settings -------------------------------------
# Models, voices, STT/TTS engines, architecture and the greeting come from the
# tunables import above.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_RAG_DATA_DIR = "data"
DEFAULT_RAG_TOP_K = 4
DEFAULT_RAG_CHUNK_TOKENS = 500
DEFAULT_RAG_CHUNK_OVERLAP = 50
DEFAULT_STATE_DIR = ".lofi"
DEFAULT_GREETING_DELAY_SECS = 1.0
DEFAULT_MOQ_SUBSCRIBER_TIMEOUT = 15.0
DEFAULT_MOQ_AUDIO_AHEAD_MS = 500
DEFAULT_LOG_LEVEL = "DEBUG"

# --- Silero VAD tuning ------------------------------------------------------
# Pipecat gates speech on BOTH a neural confidence score AND an absolute EBU-R128
# loudness (`min_volume`). Silero's confidence is level-robust; the loudness gate
# is level-SENSITIVE and is what would otherwise force per-machine mic calibration.
# We default min_volume to 0.0 (gate OFF) so the bot trusts the neural score and
# "just works" across mics/machines regardless of input level. Raise it (e.g.
# 0.3-0.6) only if a loud/noisy room causes false triggers.
DEFAULT_VAD_CONFIDENCE = 0.7   # Silero speech probability required (0..1)
DEFAULT_VAD_MIN_VOLUME = 0.0   # absolute-loudness gate; 0 = disabled (portable)
DEFAULT_VAD_START_SECS = 0.2   # sustained speech before "user started"
DEFAULT_VAD_STOP_SECS = 0.2    # sustained silence before "user stopped"


def _get(name: str, default: str) -> str:
    """Return env var ``name`` (stripped), or ``default`` when unset/blank."""
    return os.getenv(name, "").strip() or default


def model_dir() -> str:
    """The one directory every downloaded model lives under (LOFI_MODEL_DIR).

    Resolved to an absolute path at import time; see ``_resolve_model_dir``.
    Point it anywhere — ``LOFI_MODEL_DIR=/Volumes/T7/lofi-models`` moves the
    HF cache, Kokoro, Piper and Ollama stores together.
    """
    return str(MODELS_DIR)


def stt_engine() -> str:
    """Which STT engine to build (LOFI_STT_ENGINE; default ``whisper_mlx`` on
    Apple Silicon, ``faster_whisper`` elsewhere).

    Options (see services.build_stt): ``whisper_mlx`` (Apple-GPU Whisper via MLX,
    multilingual), ``faster_whisper`` (CPU Whisper via CTranslate2 — the
    non-Apple-Silicon path), ``moonshine`` (tiny/fast CPU ONNX, English + a few
    languages; needs ``uv sync --extra moonshine``).
    """
    return _get("LOFI_STT_ENGINE", DEFAULT_STT_ENGINE)


def tts_engine() -> str:
    """Which TTS engine to build (LOFI_TTS_ENGINE, default ``kokoro``).

    Options (see services.build_tts): ``kokoro`` (ONNX, ~0.3 GB, 50+ voices),
    ``piper`` (in-process piper-tts, small fast voices; needs
    ``uv sync --extra piper``).
    """
    return _get("LOFI_TTS_ENGINE", DEFAULT_TTS_ENGINE)


def whisper_model() -> str:
    """``MLXModel`` member name for Whisper-MLX STT (default LARGE_V3_TURBO).

    Other members: TINY, MEDIUM, LARGE_V3. Must match a member the prefetch
    downloaded (``scripts/prefetch_models.py`` reads the same var).
    """
    return _get("LOFI_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


def faster_whisper_model() -> str:
    """``Model`` member name for faster-whisper STT (default DISTIL_MEDIUM_EN).

    Only used when LOFI_STT_ENGINE=faster_whisper. Members (pipecat
    ``services.whisper.stt.Model``): TINY, BASE, SMALL, MEDIUM, LARGE,
    LARGE_V3_TURBO, DISTIL_LARGE_V2, DISTIL_MEDIUM_EN (English-only).
    Weights download from Hugging Face on first use (cached under HF_HOME).
    """
    return _get("LOFI_FASTER_WHISPER_MODEL", DEFAULT_FASTER_WHISPER_MODEL)


def moonshine_model() -> str:
    """``Model`` member name for Moonshine STT (default SMALL_STREAMING).

    Only used when LOFI_STT_ENGINE=moonshine. Members (pipecat
    ``services.moonshine.stt.Model``): TINY, BASE, TINY_STREAMING,
    BASE_STREAMING, SMALL_STREAMING, MEDIUM_STREAMING. Weights download from
    the Moonshine hub on first use.
    """
    return _get("LOFI_MOONSHINE_MODEL", DEFAULT_MOONSHINE_MODEL)


def piper_voice() -> str:
    """Piper voice id (default ``en_US-lessac-medium``).

    Only used when LOFI_TTS_ENGINE=piper. Any id from the Piper voices collection
    (huggingface.co/rhasspy/piper-voices) works; the ~60 MB voice model
    downloads on first use into ``$LOFI_MODEL_DIR/piper/``.
    """
    return _get("LOFI_PIPER_VOICE", DEFAULT_PIPER_VOICE)


def piper_download_dir() -> str:
    """Directory Piper voices download into (LOFI_PIPER_DOWNLOAD_DIR).

    Defaults to ``$LOFI_MODEL_DIR/piper`` — every checkpoint the bot needs
    stays inside the one model directory, like the other engines.
    """
    return _get("LOFI_PIPER_DOWNLOAD_DIR", str(MODELS_DIR / "piper"))


def ollama_models_dir() -> str:
    """Directory Ollama keeps the LLM in (OLLAMA_MODELS).

    Defaults to ``$LOFI_MODEL_DIR/ollama``. Set by scripts/run_ollama.sh before
    it starts the server; exposed here so configure.sh and print_models.py can
    report the same path the server actually uses.
    """
    return _get("OLLAMA_MODELS", str(MODELS_DIR / "ollama"))


def llm_model() -> str:
    """Ollama model tag for the conversation LLM (default ``qwen2.5:7b``).

    The same string ``scripts/run_ollama.sh`` uses to ``ollama pull``, so the
    bot and the pull agree on which model is served.
    """
    return _get("LOFI_LLM_MODEL", DEFAULT_LLM_MODEL)


def ollama_base_url() -> str:
    """OpenAI-compatible Ollama endpoint (default ``http://localhost:11434/v1``).

    Note the trailing ``/v1``: the OpenAI-compat path, not the native API root.
    """
    return _get("LOFI_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def ollama_api_url() -> str:
    """Native Ollama API root (``/api/embed``, ``/api/tags``).

    Derived from ``ollama_base_url()`` by dropping the OpenAI-compat ``/v1``
    suffix — one host var covers both APIs.
    """
    return ollama_base_url().rstrip("/").removesuffix("/v1")


def embed_model() -> str:
    """Ollama model tag for RAG embeddings (LOFI_EMBED_MODEL, default
    ``nomic-embed-text``).

    Served by the same Ollama instance as the LLM; pull it with
    ``ollama pull nomic-embed-text``.
    """
    return _get("LOFI_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def rag_data_dir() -> str:
    """Directory of the documents to index (LOFI_RAG_DATA_DIR, default ``./data``).

    Drop ``.txt``/``.md``/``.pdf`` files here, then ``./lofi.sh index-rag``.
    Resolved like ``LOFI_MODEL_DIR``: absolute, ``~``, or relative to the repo
    root.
    """
    return str(_resolve_repo_path(_get("LOFI_RAG_DATA_DIR", DEFAULT_RAG_DATA_DIR)))


def rag_index_dir() -> str:
    """Where the built RAG index lives (LOFI_RAG_INDEX_DIR, default
    ``$LOFI_MODEL_DIR/rag-index``): chunks.jsonl + embeddings.npy +
    manifest.json, all written by ``./lofi.sh index-rag``.
    """
    return str(_resolve_repo_path(_get("LOFI_RAG_INDEX_DIR", str(MODELS_DIR / "rag-index"))))


def rag_llm_model() -> str:
    """Ollama tag for the records analyst (LOFI_RAG_LLM_MODEL, default
    ``qwen3:8b`` — the measured-best analyst, wants ~16GB with the front).

    The split backend needs a model that emits real API tool calls under the
    worker's composed prompt — nemotron-3-nano:4b narrates them as JSON text
    and answers nothing (measured 2026-10-16).
    """
    return _get("LOFI_RAG_LLM_MODEL", DEFAULT_RAG_LLM_MODEL)


def rag_fast_llm_model() -> str:
    """Ollama tag for the analyst's fast mode (LOFI_RAG_FAST_LLM_MODEL).

    Defaults to rag_llm_model() — same weights, thinking off, zero extra RAM.
    A distinct fast model needs OLLAMA_MAX_LOADED_MODELS=4 (front + analyst +
    fast analyst + embedder all resident).
    """
    return _get("LOFI_RAG_FAST_LLM_MODEL", "").strip() or rag_llm_model()


def wait_lesson_language() -> str:
    """Language for the wait-window mini-lesson (LOFI_WAIT_LESSON_LANGUAGE,
    default empty = off; split only).

    Set to a language name — e.g. Spanish, German — and while the analyst
    works the front teaches one short phrase related to the user's question.
    """
    return _get("LOFI_WAIT_LESSON_LANGUAGE", "").strip()


def architecture() -> str:
    """Bot architecture (LOFI_ARCHITECTURE, default ``split``).

    ``split`` (default): a conversational front model delegates records work
    to a reasoning analyst backend (pipecat LLMWithBackend/BackendLLMWorker)
    that runs the four retrieval tools.
    ``cascade``: one model holds the four retrieval tools directly — the
    simple, stable fallback.
    """
    value = _get("LOFI_ARCHITECTURE", DEFAULT_ARCHITECTURE).strip().lower()
    if value not in ("cascade", "split"):
        raise SystemExit(
            f"\n✖ Unknown LOFI_ARCHITECTURE '{value}' — valid: split (default), cascade\n"
        )
    return value


def rag_db_path() -> str:
    """SQLite file backing the RAG index (LOFI_RAG_DB, default
    ``$LOFI_RAG_INDEX_DIR/lofi.db``): typed transaction rows, FTS5, and
    sqlite-vec embeddings, all in one file.
    """
    return _get("LOFI_RAG_DB", str(Path(rag_index_dir()) / "lofi.db"))


def rag_top_k() -> int:
    """Retrieved chunks injected into context per user turn (LOFI_RAG_TOP_K, default 4)."""
    return _get_int("LOFI_RAG_TOP_K", DEFAULT_RAG_TOP_K)


def rag_chunk_tokens() -> int:
    """Chunk budget in whitespace-split words (LOFI_RAG_CHUNK_TOKENS, default 500)."""
    return _get_int("LOFI_RAG_CHUNK_TOKENS", DEFAULT_RAG_CHUNK_TOKENS)


def rag_chunk_overlap() -> int:
    """Words repeated between consecutive chunks (LOFI_RAG_CHUNK_OVERLAP, default 50)."""
    return _get_int("LOFI_RAG_CHUNK_OVERLAP", DEFAULT_RAG_CHUNK_OVERLAP)


def state_dir() -> str:
    """Where lofi.sh records the PIDs it owns (LOFI_STATE_DIR, default
    ``.lofi``). Resolved like ``LOFI_MODEL_DIR``; lofi.sh applies the same
    repo-relative rule. An ``ollama.pid`` in here means lofi started ollama.
    """
    return str(_resolve_repo_path(_get("LOFI_STATE_DIR", DEFAULT_STATE_DIR)))


def kokoro_voice() -> str:
    """Kokoro voice id (default ``af_heart``).

    Kokoro's shipped ``Settings.voice`` default is ``None`` (unsynthesizable),
    so the bot always supplies an explicit id.
    """
    return _get("LOFI_KOKORO_VOICE", DEFAULT_KOKORO_VOICE)


def kokoro_model_path() -> str:
    """Kokoro ONNX model path (default $LOFI_MODEL_DIR/kokoro/kokoro-v1.0.onnx)."""
    return _get("LOFI_KOKORO_MODEL_PATH", str(KOKORO_DIR / "kokoro-v1.0.onnx"))


def kokoro_voices_path() -> str:
    """Kokoro voices bundle path (default $LOFI_MODEL_DIR/kokoro/voices-v1.0.bin)."""
    return _get("LOFI_KOKORO_VOICES_PATH", str(KOKORO_DIR / "voices-v1.0.bin"))


def _device_index(name: str) -> int | None:
    """PyAudio device index from env ``name``; None (system default) if unset/blank."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def input_device_index() -> int | None:
    """Mic device index (LOFI_INPUT_DEVICE_INDEX); None = system default input."""
    return _device_index("LOFI_INPUT_DEVICE_INDEX")


def output_device_index() -> int | None:
    """Speaker device index (LOFI_OUTPUT_DEVICE_INDEX); None = system default output."""
    return _device_index("LOFI_OUTPUT_DEVICE_INDEX")


def greeting() -> str:
    """Opening line the bot speaks on startup (LOFI_GREETING)."""
    return _get("LOFI_GREETING", DEFAULT_GREETING)


def greeting_delay_secs() -> float:
    """Seconds to wait before speaking the greeting (LOFI_GREETING_DELAY_SECS).

    Gives the local audio-output stream time to spin up before the first frame.
    """
    raw = os.getenv("LOFI_GREETING_DELAY_SECS", "").strip()
    return float(raw) if raw else DEFAULT_GREETING_DELAY_SECS


def moq_subscriber_timeout() -> float:
    """Max seconds the MoQ bot holds its greeting for the browser's audio
    subscription (LOFI_MOQ_SUBSCRIBER_TIMEOUT).

    MoQ is live media with no replay — a greeting spoken before the browser
    subscribes is silently dropped. On timeout the bot greets anyway.
    """
    return _get_float("LOFI_MOQ_SUBSCRIBER_TIMEOUT", DEFAULT_MOQ_SUBSCRIBER_TIMEOUT)


def moq_audio_ahead_ms() -> int:
    """Max milliseconds of bot audio written ahead of real time over MoQ
    (LOFI_MOQ_AUDIO_AHEAD_MS).

    TTS generates faster than real time; without a tight cap the whole
    utterance bursts onto the wire at once and overflows the browser's
    playback ring buffer, which drops the oldest unplayed samples — the
    start of every bot turn. Must stay well under the browser ring's
    one-second capacity; raise only if playback stutters mid-utterance.
    """
    return _get_int("LOFI_MOQ_AUDIO_AHEAD_MS", DEFAULT_MOQ_AUDIO_AHEAD_MS)


def log_level() -> str:
    """Loguru level for stderr logging (LOFI_LOG_LEVEL, default DEBUG).

    DEBUG surfaces each service's activity — useful during the offline
    verification to confirm no service silently reaches the network.
    """
    return _get("LOFI_LOG_LEVEL", DEFAULT_LOG_LEVEL)


def _get_float(name: str, default: float) -> float:
    """Return env var ``name`` parsed as float, or ``default`` when unset/blank."""
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _get_int(name: str, default: int) -> int:
    """Return env var ``name`` parsed as int, or ``default`` when unset/blank."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def vad_confidence() -> float:
    """Silero speech-probability threshold (LOFI_VAD_CONFIDENCE, default 0.7)."""
    return _get_float("LOFI_VAD_CONFIDENCE", DEFAULT_VAD_CONFIDENCE)


def vad_min_volume() -> float:
    """Absolute-loudness gate (LOFI_VAD_MIN_VOLUME, default 0.0 = disabled).

    0.0 makes turn detection level-independent (portable across mics/machines).
    Raise toward 0.3-0.6 to reject low-level background noise on a loud setup.
    """
    return _get_float("LOFI_VAD_MIN_VOLUME", DEFAULT_VAD_MIN_VOLUME)


def vad_start_secs() -> float:
    """Sustained speech before 'user started speaking' (LOFI_VAD_START_SECS, default 0.2)."""
    return _get_float("LOFI_VAD_START_SECS", DEFAULT_VAD_START_SECS)


def vad_stop_secs() -> float:
    """Sustained silence before 'user stopped speaking' (LOFI_VAD_STOP_SECS, default 0.2)."""
    return _get_float("LOFI_VAD_STOP_SECS", DEFAULT_VAD_STOP_SECS)
