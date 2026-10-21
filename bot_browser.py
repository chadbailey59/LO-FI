"""Browser voice bot — same brain, echo cancellation free from the browser's getUserMedia.

Serves the lofi client (client/dist/) and offers three transports: WebSocket (the
client's default — pure TCP, crosses any boundary), SmallWebRTC (opt-in via
?transport=webrtc, needs `uv sync --extra webrtc`), and MoQ/QUIC (opt-in via
?transport=moq; the bot runs its own MoQ relay in serve mode with an
auto-generated localhost TLS cert — loopback QUIC, fully offline).

Run:
    ./lofi.sh start
    # opens http://localhost:7860 (the lofi client from client/dist/) — allow the mic, Connect.

Reuses the shared pipeline from pipeline.py (VADProcessor + Whisper/Ollama/Kokoro);
only transport construction and per-transport event-handler shapes live here.
"""

# config FIRST — sets HF_HOME / Kokoro cache paths before any pipecat/HF import (see config.py).
import config

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport
# moq-ffi ships no Windows wheels — on native Windows the MoQ transport is
# simply not offered and the bot serves websocket (+ webrtc when installed).
try:
    from pipecat.transports.moq.transport import MOQParams, MOQTransportClient

    # TEMP upstream-bug patch (present through pipecat 1.13.0): a barge-in's
    # reset_audio_pacing() nulls _publish_audio_clock during publish_audio's
    # pacing sleep; the += on wake raises TypeError (None + float) and the
    # frame is dropped ("Error writing audio frame"). Retry once with the
    # clock cleared so the write re-anchors. Remove once fixed in pipecat.
    _orig_publish_audio = MOQTransportClient.publish_audio

    async def _publish_audio_reanchor_on_race(self, audio: bytes):
        try:
            await _orig_publish_audio(self, audio)
        except TypeError:
            self._publish_audio_clock = None
            await _orig_publish_audio(self, audio)

    MOQTransportClient.publish_audio = _publish_audio_reanchor_on_race
except ImportError:
    MOQParams = None
from pipecat.workers.runner import WorkerRunner

import store
from bot import _configure_logging, _preflight_llm, _preflight_rag, _preflight_stt
from pipeline import build_pipeline
from scripts.print_models import model_entries

# Shell env wins over .env (matches config.py and lofi.sh).
load_dotenv()

CONFIG_MESSAGE_TYPE = "lofi-config"

CLIENT_DIST = Path(__file__).resolve().parent / "client" / "dist"


def _serve_client_dist(app) -> None:
    """Register the lofi frontend on the dev runner's FastAPI app.

    The runner has no option for a custom static dir (its `_setup_frontend_routes`
    hardcodes the pipecat-ai-prebuilt UI, now uninstalled — it logs one startup
    error about that and mounts nothing). Routes registered before `main()` win.

    The COOP/COEP headers make the page cross-origin isolated, which unlocks
    SharedArrayBuffer: the MoQ player then uses its one-second atomic audio
    ring instead of the 160ms postMessage fallback that drops the start of
    every bot utterance. Everything the page loads is same-origin, so the
    isolation costs nothing.
    """
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.middleware("http")
    async def cross_origin_isolation(request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return response

    app.mount("/assets", StaticFiles(directory=CLIENT_DIST / "assets"), name="client-assets")

    @app.get("/", include_in_schema=False)
    async def client_index():
        # no-cache: revalidate index.html on every load so a rebuilt dist's
        # hashed asset names reach the browser on a plain reload.
        return FileResponse(
            CLIENT_DIST / "index.html", headers={"Cache-Control": "no-cache"}
        )


def _lofi_config_message() -> dict:
    return {
        "type": CONFIG_MESSAGE_TYPE,
        "models": model_entries(),
        "ollama_host": config.ollama_base_url(),
        "ollama_started_by_lofi": (Path(config.state_dir()) / "ollama.pid").is_file(),
        "architecture": config.architecture(),
        "llm_engine": config.llm_engine(),
        "hosted_host": config.hosted_base_url() if config.llm_engine() == "hosted" else None,
        "rag": store.stats_summary(Path(config.rag_db_path()), config.rag_data_dir()),
    }

# MoQ transport with audio in/out. Serve mode (the bot is its own MoQ relay).
# audio_out_max_buffer_ms defaults to 25s, which lets a whole TTS utterance
# burst onto the wire at once and overflow the browser's playback ring buffer
# (dropping the start of every bot turn) — cap it near real time instead.
def build_transport_params() -> dict:
    """Transports the runner offers via /start: MoQ always; SmallWebRTC when
    aiortc is installed (uv sync --extra webrtc) — the fallback for Windows/WSL2
    (QUIC/UDP can't cross the Windows↔WSL boundary) and non-WebTransport browsers."""
    params = {}
    if MOQParams is not None:
        params["moq"] = lambda: MOQParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_max_buffer_ms=config.moq_audio_ahead_ms(),
        )
    else:
        logger.info("moq transport not offered — moq-ffi unavailable on this platform")
    # Plain WebSocket (pure TCP): the last-resort browser path — crosses any
    # boundary that forwards TCP, including Windows↔WSL2 where both QUIC and
    # WebRTC's UDP media (and aiortc's no-ICE-TCP) are dead ends.
    def websocket_params():
        from pipecat.frames.frames import InterruptionFrame, TextFrame
        from pipecat.serializers.protobuf import ProtobufFrameSerializer
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

        class JsClientProtobufSerializer(ProtobufFrameSerializer):
            """The JS websocket client parses only "audio" and "message" frame
            kinds — anything else logs "Unknown frame kind" in the browser, so
            skip text/transcription/interruption at the source. (Transcripts
            and bot text reach the client as RTVI "message" frames.)"""

            async def serialize(self, frame):
                # TranscriptionFrame subclasses TextFrame.
                if isinstance(frame, (TextFrame, InterruptionFrame)):
                    return None
                return await super().serialize(frame)

        return FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=JsClientProtobufSerializer(),
        )

    params["websocket"] = websocket_params

    try:
        import aiortc  # noqa: F401
    except ImportError:
        logger.info("webrtc transport not offered — enable with: uv sync --extra webrtc")
        return params
    from pipecat.transports.base_transport import TransportParams

    params["webrtc"] = lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )
    return params


transport_params = build_transport_params()


GREETING_SETTLE_SECS = 0.5


async def wait_for_audio_subscriber(
    transport, client_ready: asyncio.Event, mic_subscribed: asyncio.Event, timeout: float
) -> bool:
    """Hold the greeting until the browser can actually hear it, cleanly.

    MoQ is live media with no replay: audio written before the bot's audio
    track is open is dropped inside ``publish_audio``, and audio sent before
    the browser subscribes to that track is never delivered. The installed
    moq bindings (moq_rs 0.3.3) expose no subscriber signal — ``AudioProducer``
    has no ``used()`` — so this waits on the closest real signals instead:
    the track being open, the client's RTVI ``client-ready`` (sent only after
    the browser has wired up its audio pipeline), and the inbound mic
    subscription (``on_track_subscribed``). The last one matters for quality:
    subscribing to the mic spins up Opus decode, VAD, and the browser's echo
    canceller — mid-greeting that garbles a second of speech. Returns False
    if a signal is still missing after ``timeout`` seconds.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while transport._client._audio_out is None:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    for event in (client_ready, mic_subscribed):
        try:
            await asyncio.wait_for(event.wait(), max(deadline - loop.time(), 0))
        except TimeoutError:
            return False
    return True


def _transport_kind(runner_args) -> str:
    """Which transport a runner-args object selects: /api/offer hands
    SmallWebRTCRunnerArguments to bot(), /ws-client hands
    WebSocketRunnerArguments; everything else here is MoQ serve mode."""
    name = type(runner_args).__name__
    if name == "SmallWebRTCRunnerArguments":
        return "webrtc"
    if name == "WebSocketRunnerArguments":
        return "websocket"
    return "moq"


def _register_moq_handlers(transport, worker, client_ready, mic_subscribed) -> None:
    # NOTE: MoQ's handlers differ from SmallWebRTC's — they receive only the transport
    # (no client argument).
    @transport.event_handler("on_track_subscribed")
    async def on_track_subscribed(_transport, _track):
        mic_subscribed.set()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport):
        timeout = config.moq_subscriber_timeout()
        logger.info("MoQ client connected — waiting for its audio subscription")
        if not await wait_for_audio_subscriber(transport, client_ready, mic_subscribed, timeout):
            logger.warning(
                f"no audio subscriber confirmed after {timeout}s — greeting may be clipped"
            )
        await asyncio.sleep(GREETING_SETTLE_SECS)
        logger.info("greeting")
        await worker.queue_frames([TTSSpeakFrame(config.greeting())])

    # A browser tab closing surfaces as the peer's broadcast going away
    # (on_client_disconnected, new in Pipecat 1.7); the whole MoQ session ending
    # fires on_disconnected. End the bot session on either.
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport):
        logger.info("MoQ client disconnected — ending session")
        await worker.cancel()

    @transport.event_handler("on_disconnected")
    async def on_disconnected(_transport):
        logger.info("MoQ session ended — ending session")
        await worker.cancel()

    @transport.event_handler("on_error")
    async def on_error(_transport, message, _exception):
        logger.error(f"MoQ error: {message}")


def _register_webrtc_handlers(transport, worker) -> None:
    # SmallWebRTC and FastAPIWebsocket handlers receive (transport, client).
    # Audio flows as soon as the connection is up — no subscriber gate needed.
    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        await asyncio.sleep(GREETING_SETTLE_SECS)
        logger.info("WebRTC client connected — greeting")
        await worker.queue_frames([TTSSpeakFrame(config.greeting())])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("WebRTC client disconnected — ending session")
        await worker.cancel()


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Assemble and run the pipeline for one connected browser client."""
    built = build_pipeline(transport)
    worker = PipelineWorker(built.pipeline, params=PipelineParams())

    client_ready = asyncio.Event()
    mic_subscribed = asyncio.Event()

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        client_ready.set()
        logger.info("RTVI client ready — sending lofi-config")
        await rtvi.send_server_message(_lofi_config_message())

    if _transport_kind(runner_args) in ("webrtc", "websocket"):
        _register_webrtc_handlers(transport, worker)
    else:
        _register_moq_handlers(transport, worker, client_ready, mic_subscribed)

    # MOQInputTransport auto-connects to the relay when the pipeline starts, so we don't
    # dial transport.connect() here; we disconnect explicitly on shutdown.
    # SmallWebRTCTransport has no disconnect() — its peer connection closes itself.
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        disconnect = getattr(transport, "disconnect", None)
        if disconnect is not None:
            await disconnect()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point the Pipecat dev runner discovers and calls per connection."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    _configure_logging()
    _preflight_llm(config.ollama_base_url())
    _preflight_stt()
    _preflight_rag()

    from pipecat.runner.run import app, main

    _serve_client_dist(app)
    main()
