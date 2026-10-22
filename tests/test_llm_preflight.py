"""Tests for which models the LLM preflight checks against Ollama: the front
model, plus the analyst whenever the architecture runs one."""

import bot


def test_cascade_local_checks_front_only(monkeypatch):
    monkeypatch.delenv("LOFI_RAG_LLM_MODEL", raising=False)
    monkeypatch.setenv("LOFI_ARCHITECTURE", "cascade")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    assert bot._ollama_models_to_check() == ["front:7b"]


def test_split_adds_distinct_analyst(monkeypatch):
    monkeypatch.setenv("LOFI_ARCHITECTURE", "llmwithbackend")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:4b")
    assert bot._ollama_models_to_check() == ["front:7b", "analyst:4b"]


def test_split_same_model_checked_once(monkeypatch):
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_ARCHITECTURE", "llmwithbackend")
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    assert bot._ollama_models_to_check() == ["front:7b"]
