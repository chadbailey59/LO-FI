"""Tests for the tool wiring in build_pipeline: the conversation model is
offered `ask_financial_records` exactly when an index exists, and no tools at
all when it doesn't. Replaces tests/test_pipeline_rag.py.

The STT/LLM/TTS builders and the VAD stage are stubbed — this asserts the shape
of the context, not that models load.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from pipecat.processors.aggregators.llm_context import is_given
from pipecat.processors.frame_processor import FrameProcessor

import ingest
import pipeline
import tools

FIXTURES = Path(__file__).parent / "fixtures"
TRANSACTIONS_CSV = FIXTURES / "transactions-small.csv"


@pytest.fixture
def db(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(TRANSACTIONS_CSV, data_dir / "transactions.csv")
    shutil.copy(FIXTURES / "note.txt", data_dir / "note.txt")
    db_path = tmp_path / "lofi.db"
    ingest.ingest(data_dir, db_path, ingest.FakeEmbedder())
    return db_path


@pytest.fixture
def stub_services(monkeypatch):
    # Generic pipeline-shape tests run on cascade: split needs real LLM
    # services (LLMWithBackend wraps them) and has its own dedicated tests.
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    for name in ("build_stt", "build_llm", "build_tts", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: FrameProcessor())


def fake_transport():
    return SimpleNamespace(input=FrameProcessor, output=FrameProcessor)


def tool_names(context) -> list[str]:
    if not is_given(context.tools):
        return []
    return [tool.name for tool in context.tools.standard_tools]


def test_index_present_offers_the_four_retrieval_tools(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    assert sorted(tool_names(built.context)) == [
        "aggregate", "filter_rows", "lookup", "semantic_search",
    ]


def test_system_prompt_is_grounded_when_index_present(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    import datetime

    import store
    from prompts import financial_advisor

    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    content = built.context.messages[0]["content"]
    assert "Today's date is" in content
    first, last = store.date_range(db)
    assert f"The transaction records run from {first} to {last}." in content
    assert "The transaction categories are: " in content
    assert content.startswith(financial_advisor.BASE_PROMPT)
    pinned = financial_advisor.system_prompt(datetime.date(2026, 10, 12), (first, last))
    assert pinned.startswith(financial_advisor.BASE_PROMPT)
    assert pinned.endswith(f"Today's date is 2026-10-12. The transaction records run from {first} to {last}.")


def test_no_index_offers_no_tools(tmp_path, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(tmp_path / "missing" / "lofi.db"))
    built = pipeline.build_pipeline(fake_transport())
    assert tool_names(built.context) == []


def test_pipeline_still_builds_without_an_index(tmp_path, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(tmp_path / "missing" / "lofi.db"))
    built = pipeline.build_pipeline(fake_transport())
    assert built.pipeline is not None
    assert built.context.messages[0]["role"] == "system"


def test_built_bot_exposes_llm_and_tts(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    assert built.llm is not None
    assert built.tts is not None


class EventfulLLM(FrameProcessor):
    """FrameProcessor that, like a real LLM service, has the function-call event."""

    def __init__(self):
        super().__init__()
        self._register_event_handler("on_function_calls_started")


def test_filler_registered_on_function_calls_started(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    monkeypatch.setattr(pipeline, "build_llm", lambda *a, **kw: EventfulLLM())
    built = pipeline.build_pipeline(fake_transport())
    assert built.llm._event_handlers["on_function_calls_started"].handlers


def test_filler_phrase_is_short_spoken_text(db):
    assert pipeline.FILLER_PHRASE.endswith(".")
    assert "{" not in pipeline.FILLER_PHRASE


def test_filler_handler_pushes_busy_message_and_filler(db, monkeypatch, stub_services):
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

    class RecordingLLM(EventfulLLM):
        def __init__(self):
            super().__init__()
            self.frames = []

        async def push_frame(self, frame, *args, **kwargs):
            self.frames.append(frame)

    class RecordingTTS(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.queued = []

        async def queue_frame(self, frame, *args, **kwargs):
            self.queued.append(frame)

    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    monkeypatch.setattr(pipeline, "build_llm", lambda *a, **kw: RecordingLLM())
    monkeypatch.setattr(pipeline, "build_tts", RecordingTTS)
    built = pipeline.build_pipeline(fake_transport())
    handler = built.llm._event_handlers["on_function_calls_started"].handlers[0]
    asyncio.run(handler(built.llm, []))

    busy = [f for f in built.llm.frames if isinstance(f, RTVIServerMessageFrame)]
    assert busy and busy[0].data == {"type": pipeline.BUSY_MESSAGE_TYPE}
    assert [f.text for f in built.tts.queued if isinstance(f, TTSSpeakFrame)] == [
        pipeline.FILLER_PHRASE
    ]


def test_bot_entry_points_import_cleanly():
    import importlib

    for module in ("bot", "bot_moq"):
        importlib.import_module(module)


def test_llm_service_disables_reasoning():
    import services

    llm = services.build_llm()
    assert llm._settings.extra == {"extra_body": {"reasoning_effort": "none"}}


def test_invalid_architecture_exits_with_guidance(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    monkeypatch.setenv("LOFI_ARCHITECTURE", "sandwich")
    with pytest.raises(SystemExit):
        pipeline.build_pipeline(fake_transport())


def test_pipeline_includes_the_silence_watchdog(db, monkeypatch, stub_services):
    from silence_watchdog import SilenceWatchdog

    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    assert any(isinstance(p, SilenceWatchdog) for p in built.pipeline._processors)


def test_split_architecture_builds_llm_with_backend(db, monkeypatch):
    # Real LLM services (construction is offline); stub only STT/TTS/VAD.
    from pipecat.pipeline.llm_with_backend import LLMWithBackend

    from prompts import financial_advisor, records_analyst

    for name in ("build_stt", "build_tts", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda: FrameProcessor())
    monkeypatch.setenv("LOFI_ARCHITECTURE", "split")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())

    # The front holds no records tools — delegate is a builtin, not a context tool.
    assert tool_names(built.context) == []
    assert built.context.messages[0]["content"].startswith(financial_advisor.OFFLINE_INTRO)

    lwb = next(p for p in built.pipeline._processors if isinstance(p, LLMWithBackend))
    worker = lwb.backend
    assert sorted(tool_names(worker.context)) == [
        "aggregate", "filter_rows", "lookup", "semantic_search",
    ]
    backend_system = worker.context.messages[0]["content"]
    assert records_analyst.SYSTEM_PROMPT in backend_system
    assert "Today's date is" in backend_system


def test_split_frontend_gets_domain_named_delegate(db, monkeypatch):
    # qwen2.5:7b never calls a tool named "delegate" (measured 0%); renamed to
    # a domain term it calls reliably — the connector installs it accordingly.
    from pipecat.pipeline.llm_with_backend import LLMWithBackend

    for name in ("build_stt", "build_tts", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda: FrameProcessor())
    monkeypatch.setenv("LOFI_ARCHITECTURE", "split")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())

    lwb = next(p for p in built.pipeline._processors if isinstance(p, LLMWithBackend))
    builtin = lwb.frontend.get_llm_adapter().builtin_tools
    assert list(builtin) == ["ask_financial_records"]
    assert "delegate" not in builtin


def test_backend_transform_strips_think_and_is_awaitable(db):
    import asyncio

    from types import SimpleNamespace

    worker = pipeline.build_records_backend(str(db))
    transform = worker._transform_output
    out = SimpleNamespace(text="<think>hmm</think>You spent $40.00.")
    result = asyncio.run(transform(out))
    assert result.text == "You spent $40.00."
    only_think = SimpleNamespace(text="<think>all reasoning</think>")
    assert asyncio.run(transform(only_think)) is None


def test_split_pipeline_includes_backend_tool_panel(db, monkeypatch):
    for name in ("build_stt", "build_tts", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda: FrameProcessor())
    monkeypatch.setenv("LOFI_ARCHITECTURE", "split")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    assert any(isinstance(p, pipeline.BackendToolPanel) for p in built.pipeline._processors)


def test_cascade_pipeline_has_no_backend_tool_panel(db, monkeypatch, stub_services):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())
    assert not any(isinstance(p, pipeline.BackendToolPanel) for p in built.pipeline._processors)


def test_backend_tool_panel_renders_lofi_rag_message(db):
    import asyncio

    from pipecat.frames.frames import (
        ExternalFunctionCall,
        ExternalFunctionCallResultFrame,
        ExternalFunctionCallsStartedFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

    panel = pipeline.BackendToolPanel()
    pushed = []

    async def record(frame, *args, **kwargs):
        pushed.append(frame)

    panel.push_frame = record

    async def run():
        await panel.process_frame(
            ExternalFunctionCallsStartedFrame(
                function_calls=[ExternalFunctionCall("semantic_search", "call-1")]
            ),
            FrameDirection.DOWNSTREAM,
        )
        await panel.process_frame(
            ExternalFunctionCallResultFrame(
                function_name="semantic_search",
                tool_call_id="call-1",
                arguments={"query": "secret passphrase"},
                result={"results": [{"source": "x.txt", "page": None, "text": "hi"}]},
            ),
            FrameDirection.DOWNSTREAM,
        )

    asyncio.run(run())
    rag = [
        f for f in pushed
        if isinstance(f, RTVIServerMessageFrame) and f.data.get("type") == tools.RAG_MESSAGE_TYPE
    ]
    assert len(rag) == 1
    data = rag[0].data
    assert data["tool"] == "semantic_search"
    assert data["args"] == {"query": "secret passphrase"}
    assert data["count"] == 1
    assert data["chunks"] == [{"source": "x.txt", "page": None, "text": "hi"}]
    assert data["elapsed_ms"] >= 0


def test_spoken_backend_output_renders_without_relay_note():
    # Ollama maps developer-role notes to USER messages; the stock RELAY_NOTE
    # then re-triggers the relay on every later turn (measured: the front
    # repeats the old result instead of delegating the new question).
    from pipecat.pipeline.llm_with_backend import BACKEND_MESSAGE_PREFIX

    connector = pipeline.build_records_connector()

    spoken = SimpleNamespace(text="You spent $40.00.", prefers_spoken=True, is_thought=False)
    message = connector.render_output(spoken)
    assert message["content"] == f"{BACKEND_MESSAGE_PREFIX}You spent $40.00."
    assert "Relay this" not in message["content"]
    assert message["role"] == "developer"

    note = SimpleNamespace(text="checked rows", prefers_spoken=False, is_thought=False)
    assert "Relay this" not in connector.render_output(note)["content"]


def test_backend_transform_strips_markdown(db):
    import asyncio

    worker = pipeline.build_records_backend(str(db))
    out = SimpleNamespace(text="It is **BLUE MANATEE 436921**, in *passphrase-smoke.txt* (`data/`).")
    assert asyncio.run(worker._transform_output(out)).text == (
        "It is BLUE MANATEE 436921, in passphrase-smoke.txt (data/)."
    )


class RecordingTTSService(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.queued = []

    async def queue_frame(self, frame, *args, **kwargs):
        self.queued.append(frame)


def test_cascade_filler_is_spoken_but_kept_out_of_context(db, monkeypatch, stub_services):
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame

    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    monkeypatch.setattr(pipeline, "build_llm", lambda *a, **kw: EventfulLLM())
    monkeypatch.setattr(pipeline, "build_tts", RecordingTTSService)
    built = pipeline.build_pipeline(fake_transport())
    handler = built.llm._event_handlers["on_function_calls_started"].handlers[0]
    asyncio.run(handler(built.llm, []))

    fillers = [f for f in built.tts.queued if isinstance(f, TTSSpeakFrame)]
    assert [f.text for f in fillers] == [pipeline.FILLER_PHRASE]
    # Canned waiting lines in the context teach a small front model to answer
    # with waiting lines (and then invent results) instead of delegating.
    assert fillers[0].append_to_context is False


def test_split_speaks_no_canned_filler(db, monkeypatch):
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame

    for name in ("build_stt", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda: FrameProcessor())
    monkeypatch.setattr(pipeline, "build_tts", RecordingTTSService)
    monkeypatch.setenv("LOFI_ARCHITECTURE", "split")
    monkeypatch.setenv("LOFI_RAG_DB", str(db))
    built = pipeline.build_pipeline(fake_transport())

    front = built.llm.frontend
    handler = front._event_handlers["on_function_calls_started"].handlers[0]
    asyncio.run(handler(front, []))
    assert not any(isinstance(f, TTSSpeakFrame) for f in built.tts.queued)


def test_watchdog_reminder_skips_context():
    import asyncio

    from pipecat.frames.frames import TTSSpeakFrame

    from silence_watchdog import SilenceWatchdog

    watchdog = SilenceWatchdog(interval_secs=0.01, max_reminders=1)
    pushed = []

    async def record(frame, *args, **kwargs):
        pushed.append(frame)

    watchdog.push_frame = record

    async def run():
        await watchdog._remind()

    asyncio.run(run())
    reminders = [f for f in pushed if isinstance(f, TTSSpeakFrame)]
    assert reminders and all(f.append_to_context is False for f in reminders)
