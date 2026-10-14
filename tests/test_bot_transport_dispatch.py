"""Tests for bot_moq's per-transport session wiring: MoQ keeps its
subscriber-gated greeting; SmallWebRTC greets on client-connected and ends the
session on disconnect."""

import asyncio

import bot_moq


class SmallWebRTCRunnerArguments:  # same class NAME the runner uses
    pass


class RunnerArguments:
    pass


def test_transport_kind_from_runner_args():
    assert bot_moq._transport_kind(SmallWebRTCRunnerArguments()) == "webrtc"
    assert bot_moq._transport_kind(RunnerArguments()) == "moq"


class FakeTransport:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def register(fn):
            self.handlers[name] = fn
            return fn

        return register


class FakeWorker:
    def __init__(self):
        self.queued = []
        self.cancelled = False

    async def queue_frames(self, frames):
        self.queued.extend(frames)

    async def cancel(self):
        self.cancelled = True


def test_webrtc_handlers_greet_and_end_session(monkeypatch):
    monkeypatch.setattr(bot_moq, "GREETING_SETTLE_SECS", 0)
    transport = FakeTransport()
    worker = FakeWorker()
    bot_moq._register_webrtc_handlers(transport, worker)

    assert set(transport.handlers) >= {"on_client_connected", "on_client_disconnected"}
    asyncio.run(transport.handlers["on_client_connected"](transport, object()))
    assert len(worker.queued) == 1
    asyncio.run(transport.handlers["on_client_disconnected"](transport, object()))
    assert worker.cancelled
