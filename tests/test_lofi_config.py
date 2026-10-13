import json

import config
from bot_moq import CONFIG_MESSAGE_TYPE, _lofi_config_message


def test_config_message_reports_llm_engine(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    assert _lofi_config_message()["llm_engine"] == "ollama"
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    assert _lofi_config_message()["llm_engine"] == "hosted"


def test_observer_architecture_adds_analyst_row(monkeypatch):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "observer")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "nemotron-3-nano:4b")
    message = _lofi_config_message()
    roles = [m["role"] for m in message["models"]]
    assert roles == ["STT", "LLM", "ANALYST", "TTS", "EMBED"]
    analyst = message["models"][roles.index("ANALYST")]
    assert "nemotron-3-nano:4b" in analyst["model"]


def test_hosted_engine_marks_llm_row(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.setenv("LOFI_HOSTED_BASE_URL", "https://api.example.com/v1")
    message = _lofi_config_message()
    llm = next(m for m in message["models"] if m["role"] == "LLM")
    assert "HOSTED" in llm["model"]
    assert "api.example.com" in llm["model"]


def test_rag_fast_llm_model_defaults_to_analyst_model(monkeypatch):
    monkeypatch.delenv("LOFI_RAG_FAST_LLM_MODEL", raising=False)
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:4b")
    assert config.rag_fast_llm_model() == "analyst:4b"
    monkeypatch.setenv("LOFI_RAG_FAST_LLM_MODEL", "tiny-fast:1b")
    assert config.rag_fast_llm_model() == "tiny-fast:1b"


def test_lofi_config_message_shape():
    message = _lofi_config_message()
    json.dumps(message)
    assert message["type"] == CONFIG_MESSAGE_TYPE == "lofi-config"
    assert [m["role"] for m in message["models"]] == [
        "STT",
        "LLM",
        "ANALYST",
        "TTS",
        "EMBED",
    ]
    for entry in message["models"]:
        assert entry["model"]
        assert entry["path"].lstrip().startswith("→")
    assert message["ollama_host"].startswith("http")
    assert isinstance(message["ollama_started_by_lofi"], bool)
    assert message["rag"].startswith(("index:", "no index"))
