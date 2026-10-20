"""Tests for prompt composition: the hosted engine swaps the offline-identity
intro so the bot never claims to be offline while a hosted LLM converses."""

import tunables


def test_prompts_share_the_offline_intro():
    for prompt in (
        tunables.BASE_PROMPT,
        tunables.SPLIT_FRONT_PROMPT,
    ):
        assert prompt.startswith(tunables.OFFLINE_INTRO)


def test_hosted_variant_swaps_offline_claim_and_keeps_body():
    hosted = tunables.hosted_variant(tunables.SPLIT_FRONT_PROMPT)
    assert "entirely offline" not in hosted
    assert "records and documents stay" in hosted
    assert "A records analyst works alongside you" in hosted
