"""Offline tests for scripts/prefetch_models.py — the pure tag-list logic.

Network paths (ollama pull subprocess, HF snapshot downloads) are exercised by
the real one-time warm-up run, not here.
"""

from scripts.prefetch_models import ollama_tags


def test_ollama_tags_lists_llm_embed_then_analyst(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("LOFI_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "qwen3:8b")

    assert ollama_tags() == ["qwen2.5:14b", "nomic-embed-text", "qwen3:8b"]


def test_ollama_tags_dedupes_identical_tags(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_MODEL", "same:1b")
    monkeypatch.setenv("LOFI_EMBED_MODEL", "same:1b")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "same:1b")
    monkeypatch.delenv("LOFI_RAG_FAST_LLM_MODEL", raising=False)

    assert ollama_tags() == ["same:1b"]


def test_ollama_tags_include_distinct_fast_analyst(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_MODEL", "front:7b")
    monkeypatch.setenv("LOFI_EMBED_MODEL", "embed:1b")
    monkeypatch.setenv("LOFI_RAG_LLM_MODEL", "analyst:8b")
    monkeypatch.setenv("LOFI_RAG_FAST_LLM_MODEL", "tiny:1b")

    assert ollama_tags() == ["front:7b", "embed:1b", "analyst:8b", "tiny:1b"]
