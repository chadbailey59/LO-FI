"""Fully-offline Pipecat voice bot — headphones / system-audio entry point.

Run with: `uv run bot.py`

The full-duplex pipeline itself is assembled by pipeline.py's build_pipeline(),
shared with the browser bots; this module owns only what is transport-specific.

Everything runs locally: LocalAudioTransport (mic + speakers), Silero VAD + Local
Smart Turn v3 for turn-taking, and the STT/LLM/TTS engines built by services.py
(defaults: Whisper-MLX, Ollama, Kokoro — swappable via LOFI_STT_ENGINE / LOFI_TTS_ENGINE in
.env or ./configure.sh -i). No cloud services, no API keys.

Importing this module has no side effects (no audio device access, no model loads) —
all hardware/model construction happens inside the builder functions and `main()`.
"""

import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

# Import config FIRST — before any pipecat/HF import. config sets HF_HOME (and the
# Kokoro cache paths) at import time, and Hugging Face freezes its cache root when
# huggingface_hub is imported, so this ordering is what makes the bot read the
# prefetched $LOFI_MODEL_DIR/huggingface weights offline.
import config

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

if TYPE_CHECKING:  # import for typing only — see build_transport() for why
    from pipecat.transports.local.audio import LocalAudioTransport

import store

# Pipeline assembly (VAD/STT/LLM/TTS/context) lives in pipeline.py, shared by all bots.
from pipeline import build_pipeline
from scripts.print_models import stt_repo_id
from services import stt_weights_location


def build_transport() -> "LocalAudioTransport":
    """Build the local audio transport (mic in + speaker out).

    - Device indices come from config (LOFI_INPUT_DEVICE_INDEX / LOFI_OUTPUT_DEVICE_INDEX;
      default: the system default input/output devices).

    PyAudio is imported here rather than at module scope on purpose. It is an
    opt-in extra (`local-audio`) because it has no macOS/Linux wheels and must
    compile against PortAudio — and bot_web.py / bot_moq.py import this module
    for its builders while getting audio from the browser instead. A top-level
    import would break those front-ends on machines without PortAudio, and would
    also give this module the import-time side effect its docstring disclaims.
    """
    try:
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
    except ImportError as e:
        # Pipecat logs its own hint here pointing at `pipecat-ai[local]`, which
        # is not how this repo installs it — override with the real command.
        sys.exit(
            f"bot.py needs PyAudio for mic/speaker access ({e}).\n"
            "  macOS:  brew install portaudio\n"
            "  Debian: sudo apt install portaudio19-dev\n"
            "  then:   uv sync --extra local-audio\n"
            "\n"
            "Or skip it and use a browser front-end, which needs no PortAudio and\n"
            "gives you echo cancellation for free:  ./lofi.sh start"
        )

    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        input_device_index=config.input_device_index(),
        output_device_index=config.output_device_index(),
    )
    return LocalAudioTransport(params)


async def _speak_greeting(worker: PipelineWorker) -> None:
    """Speak a short opening line shortly after the pipeline starts.

    `LocalAudioTransport` does NOT emit an `on_client_connected` event — that
    event only fires on networked transports (WebSocket/WebRTC/Daily/etc.), so
    hooking it here would silently never run. Instead, mirroring Pipecat's own
    `getting-started/01a-local-audio.py`, we wait briefly for the audio output
    stream to come up, then queue a `TTSSpeakFrame`.
    """
    await asyncio.sleep(config.greeting_delay_secs())
    await worker.queue_frames([TTSSpeakFrame(config.greeting())])


def _configure_logging() -> None:
    """Route Pipecat's loguru output to stderr at LOFI_LOG_LEVEL (default DEBUG).
    """
    logger.remove()
    logger.add(sys.stderr, level=config.log_level())


def _preflight_hosted_llm() -> None:
    """Fail fast when the hosted engine has no key or an unreachable endpoint."""
    if not config.hosted_api_key():
        if config.hosted_key_in_dotenv_only():
            sys.exit(
                "\n✖ Found an API key in .env — keys never go in .env (it gets backed up and shared)\n"
                "  Remove it from .env and export it in your shell instead:\n"
                "    export LOFI_HOSTED_API_KEY=sk-...\n"
            )
        sys.exit(
            "\n✖ LOFI_LLM_ENGINE=hosted needs an API key in the shell environment\n"
            "  Set LOFI_HOSTED_API_KEY (or OPENAI_API_KEY) — keys never go in .env\n"
        )
    base_url = config.hosted_base_url()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {config.hosted_api_key()}"},
    )

    # Never follow redirects with the key attached.
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=10) as resp:  # noqa: S310
            json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        sys.exit(
            f"\n✖ Cannot reach the hosted LLM endpoint at {base_url}\n"
            f"  Check LOFI_HOSTED_BASE_URL and your API key.\n"
            f"  Details: {exc}\n"
        )
    logger.info(f"LLM preflight OK: hosted endpoint reachable at {base_url}")


def _ollama_models_to_check() -> list[str]:
    """Models that must be pulled in the local Ollama: the front model when
    the engine is local, plus analyst (and fast analyst) when the architecture
    runs one — the analyst is always local, hosted front or not."""
    models: list[str] = []
    if config.llm_engine() == "ollama":
        models.append(config.llm_model())
    if config.architecture() == "split":
        for model in (config.rag_llm_model(), config.rag_fast_llm_model()):
            if model not in models:
                models.append(model)
    return models


def _preflight_llm(base_url: str) -> None:
    """Fail fast with a clear message if the LLM isn't usable.

    A missing Ollama server or un-pulled model otherwise fails silently mid-turn
    (the LLM call errors and nothing is spoken), which is confusing.
    """
    if config.llm_engine() == "hosted":
        _preflight_hosted_llm()
    to_check = _ollama_models_to_check()
    if not to_check:
        return
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 (localhost)
            models = json.load(resp).get("data", [])
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"\n✖ Cannot reach the local LLM server at {base_url}\n"
            f"  Start Ollama first:  ./scripts/run_ollama.sh   (or: ollama serve)\n"
            f"  Details: {exc}\n"
        )

    available = {str(m.get("id", "")) for m in models}
    for wanted in to_check:
        # Ollama reports tags like 'llama3:latest'; match the bare name too.
        if not any(wanted == m or wanted.split(":")[0] == m.split(":")[0] for m in available):
            listed = ", ".join(sorted(available)) or "(none)"
            sys.exit(
                f"\n✖ LLM model '{wanted}' is not available in Ollama at {base_url}\n"
                f"  Pull it:            ollama pull {wanted}\n"
                f"  Or point LOFI_LLM_MODEL / LOFI_RAG_LLM_MODEL / LOFI_RAG_FAST_LLM_MODEL\n"
                f"  at one you have: {listed}\n"
            )
    logger.info(f"LLM preflight OK: {', '.join(to_check)} available at {base_url}")


def _preflight_stt() -> None:
    """Fail fast when the configured STT model's weights are in NO local cache.

    Without this, the first utterance triggers a silent multi-GB Hugging Face
    download (progress bars are disabled in config.py) and the bot just seems
    deaf until it finishes. Weights found in any cache are fine as-is:
    services.build_stt() loads them from wherever they live.
    """
    repo_id = stt_repo_id()
    if repo_id is None:
        return
    location = stt_weights_location(repo_id)
    if location is None:
        sys.exit(
            f"\n✖ STT model '{repo_id}' has no weights in any local Hugging Face cache\n"
            f"  Without them the first utterance triggers a silent multi-GB download.\n"
            f"  Fetch them now:  uv run python scripts/prefetch_models.py\n"
        )
    kind, snapshot = location
    logger.info(f"STT preflight OK: '{repo_id}' weights present ({kind}): {snapshot}")


def _preflight_rag() -> None:
    """With an index the records agent embeds every semantic search, so the
    embed model must be pulled — check it upfront like the LLM.
    """
    store.preflight(Path(config.rag_db_path()), config.embed_model(), config.ollama_api_url())


async def main() -> None:
    """`uv run bot.py` entry point. It loads `.env` (config only — no
        secrets), assembles the pipeline, and hands the worker to a `WorkerRunner`,
        which manages the asyncio lifecycle and SIGINT/SIGTERM shutdown.
    """
    # Shell env wins over .env (matches config.py and lofi.sh).
    load_dotenv()
    _configure_logging()

    # Fail fast (with guidance) if the local LLM server/model isn't ready, rather
    # than silently producing no spoken reply when the first turn hits the LLM.
    _preflight_llm(config.ollama_base_url())
    _preflight_stt()
    _preflight_rag()

    transport = build_transport()
    built = build_pipeline(transport)

    worker = PipelineWorker(built.pipeline, params=PipelineParams())

    runner = WorkerRunner(handle_sigint=sys.platform != "win32")

    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), _speak_greeting(worker))


if __name__ == "__main__":
    asyncio.run(main())
