"""Tests for prompt composition: each front prompt opens with its own identity."""

import tunables

OFFLINE_IDENTITY = (
    "You are a private financial thinking partner. You run entirely offline"
)
ONLINE_IDENTITY = (
    "You are a private financial thinking partner. This session is running in online mode"
)


def test_split_front_prompt_opens_with_the_offline_identity():
    assert tunables.SPLIT_FRONT_PROMPT.startswith(OFFLINE_IDENTITY)


def test_online_split_prompts_open_with_their_identities():
    assert tunables.LOFI_ONLINE_FRONT_PROMPT.startswith(ONLINE_IDENTITY)
    assert tunables.LOFI_ONLINE_ANALYST_PROMPT.startswith("You are a documents analyst")
