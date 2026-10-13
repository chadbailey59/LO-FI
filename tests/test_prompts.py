"""Tests for prompt composition: the hosted engine swaps the offline-identity
intro so the bot never claims to be offline while a hosted LLM converses."""

from prompts import financial_advisor


def test_prompts_share_the_offline_intro():
    for prompt in (
        financial_advisor.BASE_PROMPT,
        financial_advisor.FRONT_PROMPT,
        financial_advisor.OBSERVER_FRONT_PROMPT,
    ):
        assert prompt.startswith(financial_advisor.OFFLINE_INTRO)


def test_hosted_variant_swaps_offline_claim_and_keeps_body():
    hosted = financial_advisor.hosted_variant(financial_advisor.OBSERVER_FRONT_PROMPT)
    assert "entirely offline" not in hosted
    assert "records and documents stay" in hosted
    assert "A records analyst listens alongside you" in hosted
