"""MoQ/QUIC transport support for the browser bot — opt-in via `?transport=moq`.

The bot runs its own MoQ relay in serve mode with an auto-generated localhost
TLS cert: loopback QUIC, fully offline. moq-ffi ships no Windows wheels, so on
native Windows this module offers no transport and `bot_browser.py` serves
SmallWebRTC/WebSocket only.

Holds everything MoQ-specific: the guarded bindings import, the publish_audio
pacing patch, the subscriber gate, transport params, and the event handlers.
`bot_browser.py` touches three names here —
:func:`moq_transport_params`, :func:`register_moq_handlers`, and
:func:`wait_for_audio_subscriber` (the gate, kept importable for tests).
"""

# config FIRST — sets HF_HOME / Kokoro cache paths before any pipecat/HF import (see config.py).
import config

import asyncio

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame

# moq-ffi ships no Windows wheels — on native Windows the MoQ transport is
# simply not offered and the bot serves websocket (+ webrtc when installed).
try:
    from pipecat.transports.moq.transport import MOQParams, MOQTransportClient

    # TEMP upstream-bug patch (present through pipecat 1.13.0): a barge-in's
    # reset_audio_pacing() nulls _publish_audio_clock during publish_audio's
    # pacing sleep; the += on wake raises TypeError (None + float) and the
    # frame is dropped ("Error writing audio frame"). Retry once with the
    # clock cleared so the write re-anchors. Remove once fixed in pipecat
    # (MOQ_FIX.md carries the PR prompt).
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


def moq_transport_params() -> dict:
    """The `moq` entry for the runner's transport table, `{}` when unavailable.

    Serve mode: the bot is its own relay. audio_out_max_buffer_ms defaults to
    25s, which lets a whole TTS utterance burst onto the wire at once and
    overflow the browser's playback ring buffer (dropping the start of every
    bot turn) — cap it near real time instead.
    """
    if MOQParams is None:
        logger.info("moq transport not offered — moq-ffi unavailable on this platform")
        return {}
    return {
        "moq": lambda: MOQParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_max_buffer_ms=config.moq_audio_ahead_ms(),
        )
    }


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


def register_moq_handlers(
    transport,
    worker,
    client_ready: asyncio.Event,
    mic_subscribed: asyncio.Event,
    greeting: str,
    settle_secs: float,
) -> None:
    """Wire one MoQ session's events. `greeting`/`settle_secs` come from
    `bot_browser` — the greeting flow is shared with webrtc/websocket."""

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
        await asyncio.sleep(settle_secs)
        logger.info("greeting")
        await worker.queue_frames([TTSSpeakFrame(greeting)])

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
