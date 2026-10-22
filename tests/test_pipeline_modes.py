"""Tests for `build_pipeline(transport, mode)` — the two session stories.

Online mode is a cloud cascade over the PUBLIC index only. The privacy
guarantee is file-level, so these tests make the private db path and every
Ollama lookup raise: an online build that touches either one fails here.

Offline mode is the default, and anything that is not exactly "online" builds
it — a garbage mode never silently ships data to the cloud.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pipecat.processors.aggregators.llm_context import is_given
from pipecat.processors.frame_processor import FrameProcessor

import config
import ingest
import pipeline
import tools
import tunables

FIXTURES = Path(__file__).parent / "fixtures"
ONLINE_KEYS = ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY")


@pytest.fixture
def fake_keys(monkeypatch):
    """All three online keys present, obviously fake."""
    for name in ONLINE_KEYS:
        monkeypatch.setenv(name, f"fake-{name.lower()}")


@pytest.fixture
def public_db(tmp_path, monkeypatch):
    """A public index built with FakeEmbedder, pointed at by config."""
    data_dir = tmp_path / "public"
    data_dir.mkdir()
    (data_dir / "pub17.txt").write_text(
        "The standard deduction for a single filer is reported in Publication 17."
    )
    db_path = tmp_path / "lofi-public.db"
    ingest.ingest(data_dir, db_path, ingest.FakeEmbedder())
    monkeypatch.setenv("LOFI_PUBLIC_RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LOFI_PUBLIC_RAG_DB", str(db_path))
    return db_path


@pytest.fixture
def stub_online_services(monkeypatch):
    for name in ("build_online_stt", "build_online_llm", "build_online_tts"):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: FrameProcessor())
    monkeypatch.setattr(pipeline, "build_vad_processor", lambda *a, **kw: FrameProcessor())


@pytest.fixture
def stub_offline_services(monkeypatch):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    for name in ("build_stt", "build_llm", "build_tts", "build_vad_processor"):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: FrameProcessor())


@pytest.fixture
def no_private_db(monkeypatch):
    """Opening the private records db is a test failure."""

    def boom(*_args, **_kwargs):
        raise AssertionError("online pipeline reached for the private db")

    monkeypatch.setattr(config, "rag_db_path", boom)
    monkeypatch.setattr(config, "rag_data_dir", boom)


@pytest.fixture
def no_ollama(monkeypatch):
    """Any Ollama lookup from an online build is a test failure."""

    def boom(*_args, **_kwargs):
        raise AssertionError("online pipeline touched Ollama")

    monkeypatch.setattr(config, "ollama_base_url", boom)
    monkeypatch.setattr(config, "ollama_api_url", boom)


def fake_transport():
    return SimpleNamespace(input=FrameProcessor, output=FrameProcessor)


def tool_names(context) -> list[str]:
    if not is_given(context.tools):
        return []
    return [tool.name for tool in context.tools.standard_tools]


def test_online_pipeline_uses_public_db_only(
    public_db, fake_keys, stub_online_services, no_private_db
):
    built = pipeline.build_pipeline(fake_transport(), mode="online")
    assert built.pipeline is not None
    assert tool_names(built.context) == ["semantic_search"]


def test_online_tools_semantic_only(public_db):
    schemas = tools.build_public_tools(public_db, ingest.FakeEmbedder())
    assert [schema.name for schema in schemas] == ["semantic_search"]


def test_online_no_ollama_touch(
    public_db, fake_keys, stub_online_services, no_private_db, no_ollama
):
    built = pipeline.build_pipeline(fake_transport(), mode="online")
    assert built.llm is not None
    assert built.tts is not None


def test_online_prompt_is_the_online_story(
    public_db, fake_keys, stub_online_services, no_private_db
):
    built = pipeline.build_pipeline(fake_transport(), mode="online")
    prompt = built.context.messages[0]["content"]
    assert prompt.startswith(
        "You are a private financial thinking partner. This session is running in online mode"
    )
    assert "You run entirely offline" not in prompt
    assert "{language_line}" not in prompt


def test_online_prompt_invites_two_way_practice_when_language_set(monkeypatch):
    monkeypatch.setenv("LOFI_PRACTICE_LANGUAGE", "Spanish")
    with_language = tunables.online_system_prompt(language=config.practice_language())
    assert "Spanish" in with_language
    assert "{language}" not in with_language
    assert "Spanish" not in tunables.online_system_prompt()


def test_mode_default_and_garbage_offline(tmp_path, monkeypatch, stub_offline_services):
    monkeypatch.setenv("LOFI_RAG_DB", str(tmp_path / "missing" / "lofi.db"))
    for built in (
        pipeline.build_pipeline(fake_transport()),
        pipeline.build_pipeline(fake_transport(), mode="garbage"),
    ):
        prompt = built.context.messages[0]["content"]
        assert prompt.startswith(
            "You are a private financial thinking partner. You run entirely offline"
        )


def test_online_missing_public_db_degrades(
    tmp_path, monkeypatch, fake_keys, stub_online_services, no_private_db
):
    monkeypatch.setenv("LOFI_PUBLIC_RAG_DB", str(tmp_path / "missing" / "lofi-public.db"))
    monkeypatch.setenv("LOFI_PUBLIC_RAG_DATA_DIR", str(tmp_path / "public"))
    built = pipeline.build_pipeline(fake_transport(), mode="online")
    assert tool_names(built.context) == []
    assert pipeline.public_rag_summary() == (
        "no public index — run ./lofi.sh index-rag with OPENAI_API_KEY set"
    )


def test_public_rag_summary_reports_the_public_index(public_db):
    summary = pipeline.public_rag_summary()
    assert summary.startswith("index: ")
    assert str(public_db.parent) in summary or "public" in summary
