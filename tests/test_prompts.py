"""Tests for prompt composition: each front prompt opens with its own identity."""

import tunables

OFFLINE_IDENTITY = (
    "You are a private financial thinking partner. You run entirely offline"
)
ONLINE_IDENTITY = (
    "You are a private financial thinking partner. This session is running in online mode"
)


def test_offline_prompts_open_with_the_offline_identity():
    for prompt in (
        tunables.LOFI_OFFLINE_CASCADE_PROMPT,
        tunables.SPLIT_FRONT_PROMPT,
    ):
        assert prompt.startswith(OFFLINE_IDENTITY)


def test_online_prompt_opens_with_the_online_identity():
    assert tunables.LOFI_ONLINE_CASCADE_PROMPT.startswith(ONLINE_IDENTITY)
