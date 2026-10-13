"""Tests for which models the LLM preflight checks against Ollama: the front
model when the engine is local, plus the analyst whenever the architecture
runs one — the analyst is always local, hosted front or not."""

import bot


def test_cascade_local_checks_front_only(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    monkeypatch.delenv("LOFI_RAG_LLM_MODEL", raising=False)
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    assert bot._ollama_models_to_check() == ["front:7b"]


def test_observer_adds_distinct_analyst(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    monkeypatch.setenv("LOFI_ARCHITECTURE", "observer")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:4b")
    assert bot._ollama_models_to_check() == ["front:7b", "analyst:4b"]


def test_observer_same_model_checked_once(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_ARCHITECTURE", "observer")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    assert bot._ollama_models_to_check() == ["front:7b"]


def test_hosted_front_still_checks_local_analyst(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_ENGINE", "hosted")
    monkeypatch.setenv("LOFI_ARCHITECTURE", "observer")
    monkeypatch.setenv("LOFI_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:4b")
    assert bot._ollama_models_to_check() == ["analyst:4b"]


def test_fast_analyst_model_is_checked_too(monkeypatch):
    monkeypatch.delenv("LOFI_LLM_ENGINE", raising=False)
    monkeypatch.setenv("LOFI_ARCHITECTURE", "observer")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:4b")
    monkeypatch.setenv("LOFI_RAG_FAST_LLM_MODEL", "tiny:1b")
    assert bot._ollama_models_to_check() == ["front:7b", "analyst:4b", "tiny:1b"]
