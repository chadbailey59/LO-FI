"""Tests for the dead-air watchdog: arms when the user stops speaking, reminds
after the interval, disarms on bot text or the user speaking again."""

import asyncio

from pipecat.frames.frames import (
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from silence_watchdog import SilenceWatchdog


def reminders(down_frames, phrase):
    return [f for f in down_frames if isinstance(f, TTSSpeakFrame) and f.text == phrase]


def test_reminds_after_silence_and_repeats_up_to_cap():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=2)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[UserStoppedSpeakingFrame(), SleepFrame(0.2)],
        )
        return down

    down = asyncio.run(run())
    assert len(reminders(down, watchdog._phrase)) == 2


def test_bot_text_disarms():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=2)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                UserStoppedSpeakingFrame(),
                LLMTextFrame("Here is your answer."),
                SleepFrame(0.2),
            ],
        )
        return down

    down = asyncio.run(run())
    assert reminders(down, watchdog._phrase) == []


def test_spoken_answer_disarms():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=2)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                UserStoppedSpeakingFrame(),
                TTSSpeakFrame("You spent $40.00."),
                SleepFrame(0.2),
            ],
        )
        return down

    down = asyncio.run(run())
    assert reminders(down, watchdog._phrase) == []


def test_user_speaking_again_disarms():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=2)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                UserStoppedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                SleepFrame(0.2),
            ],
        )
        return down

    down = asyncio.run(run())
    assert reminders(down, watchdog._phrase) == []


def test_rearms_per_turn():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=1)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                UserStoppedSpeakingFrame(),
                SleepFrame(0.1),
                LLMTextFrame("answer one"),
                # System frames overtake queued data frames; give the disarm a
                # beat to land before re-arming, as real conversations do.
                SleepFrame(0.02),
                UserStoppedSpeakingFrame(),
                SleepFrame(0.1),
            ],
        )
        return down

    down = asyncio.run(run())
    assert len(reminders(down, watchdog._phrase)) == 2


def _rtvi(message_type):
    from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

    return RTVIServerMessageFrame(data={"type": message_type})


def test_front_chatter_does_not_disarm_while_busy():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=1)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                _rtvi("lofi-busy"),
                LLMTextFrame("Let me check your records."),
                SleepFrame(0.2),
            ],
        )
        return down

    down = asyncio.run(run())
    assert len(reminders(down, watchdog._phrase)) == 1


def test_idle_signal_disarms_busy_watch():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=1)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[_rtvi("lofi-busy"), _rtvi("lofi-idle"), SleepFrame(0.2)],
        )
        return down

    down = asyncio.run(run())
    assert reminders(down, watchdog._phrase) == []


def test_spoken_answer_clears_busy_watch():
    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=1)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[
                _rtvi("lofi-busy"),
                TTSSpeakFrame("You spent $40.00."),
                SleepFrame(0.2),
            ],
        )
        return down

    down = asyncio.run(run())
    assert reminders(down, watchdog._phrase) == []


def test_lesson_replaces_first_reminder_while_busy(monkeypatch):
    import silence_watchdog as sw

    monkeypatch.setattr(sw, "LESSON_FIRST_INTERVAL_SECS", 0.05)
    lessons = []

    async def lesson():
        lessons.append(1)

    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=2, lesson=lesson)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[_rtvi("lofi-busy"), SleepFrame(0.25)],
        )
        return down

    down = asyncio.run(run())
    assert lessons == [1]
    # Later reminders in the same wait stay canned.
    assert len(reminders(down, watchdog._phrase)) == 1


def test_lesson_skipped_when_not_busy(monkeypatch):
    import silence_watchdog as sw

    monkeypatch.setattr(sw, "LESSON_FIRST_INTERVAL_SECS", 0.05)
    lessons = []

    async def lesson():
        lessons.append(1)

    watchdog = SilenceWatchdog(interval_secs=0.05, max_reminders=1, lesson=lesson)

    async def run():
        down, _up = await run_test(
            watchdog,
            frames_to_send=[UserStoppedSpeakingFrame(), SleepFrame(0.2)],
        )
        return down

    down = asyncio.run(run())
    assert lessons == []
    assert len(reminders(down, watchdog._phrase)) == 1
