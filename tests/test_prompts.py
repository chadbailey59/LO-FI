"""Tests for prompt composition: offline-session prompts share one identity intro."""

import tunables


def test_prompts_share_the_offline_intro():
    for prompt in (
        tunables.BASE_PROMPT,
        tunables.SPLIT_FRONT_PROMPT,
    ):
        assert prompt.startswith(tunables.OFFLINE_INTRO)
