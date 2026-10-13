"""Tests for the LLM engine dispatch in services.build_llm: local Ollama by
default, a hosted OpenAI-compatible endpoint when LOFI_LLM_ENGINE=hosted."""

import pytest

import config
import services


def test_default_engine_is_ollama(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    assert config.llm_engine() == "ollama"
    from pipecat.services.ollama.llm import OLLamaLLMService

    assert isinstance(services.build_llm(), OLLamaLLMService)


def test_unknown_engine_exits(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "azure")
    with pytest.raises(SystemExit):
        config.llm_engine()


def test_hosted_without_key_exits_with_guidance(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.delenv("LOFI_HOSTED_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        config, "_SHELL_KEY_SNAPSHOT", {"LOFI_HOSTED_API_KEY": "", "OPENAI_API_KEY": ""}
    )
    with pytest.raises(SystemExit) as excinfo:
        services.build_llm()
    assert "LOFI_HOSTED_API_KEY" in str(excinfo.value)


def test_hosted_builds_openai_compatible_service(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.setenv("LOFI_HOSTED_API_KEY", "sk-test")
    monkeypatch.setenv("LOFI_HOSTED_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LOFI_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(
        config, "_SHELL_KEY_SNAPSHOT", {"LOFI_HOSTED_API_KEY": "sk-test", "OPENAI_API_KEY": ""}
    )
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.openai.llm import OpenAILLMService

    llm = services.build_llm()
    assert isinstance(llm, OpenAILLMService)
    assert not isinstance(llm, OLLamaLLMService)


def test_hosted_key_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.delenv("LOFI_HOSTED_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr(
        config, "_SHELL_KEY_SNAPSHOT", {"LOFI_HOSTED_API_KEY": "", "OPENAI_API_KEY": "sk-env"}
    )
    assert config.hosted_api_key() == "sk-env"


def test_key_only_in_dotenv_is_refused(monkeypatch):
    # .env-sourced keys land in os.environ via load_dotenv but are absent from
    # the pre-dotenv shell snapshot — they must not be honored.
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.setenv("LOFI_HOSTED_API_KEY", "sk-from-dotenv")
    monkeypatch.setattr(
        config, "_SHELL_KEY_SNAPSHOT", {"LOFI_HOSTED_API_KEY": "", "OPENAI_API_KEY": ""}
    )
    assert config.hosted_api_key() == ""
    assert config.hosted_key_in_dotenv_only() is True
    with pytest.raises(SystemExit) as excinfo:
        services.build_llm()
    assert ".env" in str(excinfo.value)
