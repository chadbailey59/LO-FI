"""Offline tests for scripts/prefetch_models.py — the pure tag-list logic.

Network paths (ollama pull subprocess, HF snapshot downloads) are exercised by
the real one-time warm-up run, not here.
"""

from scripts.prefetch_models import ollama_tags


def test_ollama_tags_lists_llm_then_embed_model(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("LOFI_EMBED_MODEL", "nomic-embed-text")

    assert ollama_tags() == ["qwen2.5:14b", "nomic-embed-text"]


def test_ollama_tags_dedupes_identical_tags(monkeypatch):
    monkeypatch.setenv("LOFI_LLM_MODEL", "same:1b")
    monkeypatch.setenv("LOFI_EMBED_MODEL", "same:1b")

    assert ollama_tags() == ["same:1b"]
