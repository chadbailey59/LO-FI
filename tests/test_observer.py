"""Tests for the observer architecture's RecordsObserver: triggers on new user
turns, speaks grounded answers, stays silent on the NOT_RECORDS sentinel and on
errors, writes spoken answers back into the context, cancels on barge-in."""

import asyncio

from pipecat.frames.frames import LLMContextFrame, TTSSpeakFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIClientMessageFrame,
    RTVIServerMessageFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

import config
import rag_agent
from observer_processor import MODE_MESSAGE_TYPE, RecordsObserver
from prompts import records_analyst

ANSWER = "You spent $40.00 at Paulie's Delicatessen on August 24, 2026."
TRACE = {"tool": "lookup", "args": {"query": "Paulie's"}, "count": 1, "elapsed_ms": 3.0}


def make_context(*user_turns):
    messages = [{"role": "system", "content": "front prompt"}]
    for turn in user_turns:
        messages.append({"role": "user", "content": turn})
    return LLMContext(messages=messages)


class FakeAgent:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def __call__(self, question, **kwargs):
        self.calls.append({"question": question, **kwargs})
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if callable(result):
            return await result()
        return result


def run_observer(context, agent, frames):
    observer = RecordsObserver(context, "/tmp/x.db", embedder=None, answer_fn=agent)

    async def run():
        down, _up = await run_test(observer, frames_to_send=frames)
        return down

    return asyncio.run(run())


def spoken(down):
    return [f.text for f in down if isinstance(f, TTSSpeakFrame)]


def routes(down):
    return [
        f for f in down
        if isinstance(f, RTVIServerMessageFrame) and f.data.get("type") == "lofi-rag"
    ]


def signals(down):
    return [
        f.data["type"] for f in down
        if isinstance(f, RTVIServerMessageFrame) and f.data.get("type") in ("lofi-busy", "lofi-idle")
    ]


def test_speaks_grounded_answer_and_updates_context():
    context = make_context("what did I buy at Paulie's?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[TRACE])])
    down = run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    assert spoken(down) == [ANSWER]
    assert len(routes(down)) == 1
    assert routes(down)[0].data["tool"] == "lookup"
    assert signals(down) == ["lofi-busy"]
    assert context.get_messages()[-1] == {"role": "assistant", "content": ANSWER}
    assert agent.calls[0]["question"] == "what did I buy at Paulie's?"
    assert agent.calls[0]["history"] == []
    assert agent.calls[0]["system_prompt"] == records_analyst.OBSERVER_SYSTEM_PROMPT


def test_passes_recent_history_to_analyst():
    context = LLMContext(
        messages=[
            {"role": "system", "content": "front prompt"},
            {"role": "user", "content": "what did I spend at Paulie's?"},
            {"role": "assistant", "content": "You spent $40.00 at Paulie's Delicatessen."},
            {"role": "user", "content": "what category was that?"},
        ]
    )
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    assert agent.calls[0]["question"] == "what category was that?"
    assert agent.calls[0]["history"] == [
        {"role": "user", "content": "what did I spend at Paulie's?"},
        {"role": "assistant", "content": "You spent $40.00 at Paulie's Delicatessen."},
    ]


def test_history_caps_at_six_turns_and_skips_non_text():
    messages = [{"role": "system", "content": "front prompt"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"question {i}?"})
        messages.append({"role": "assistant", "content": f"answer {i}."})
    messages.append({"role": "assistant", "content": [{"type": "image"}]})
    messages.append({"role": "user", "content": "latest question?"})
    context = LLMContext(messages=messages)
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    history = agent.calls[0]["history"]
    assert len(history) == 6
    assert history[0] == {"role": "user", "content": "question 5?"}
    assert history[-1] == {"role": "assistant", "content": "answer 7."}
    assert agent.calls[0]["question"] == "latest question?"


def test_zero_trace_answer_speaks_and_releases_busy():
    context = make_context("which store was the biggest one at?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    down = run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    assert spoken(down) == [ANSWER]
    assert signals(down) == ["lofi-busy", "lofi-idle"]


def test_sentinel_stays_silent():
    context = make_context("how are you today?")
    agent = FakeAgent([rag_agent.AgentResult(answer=records_analyst.NOT_RECORDS)])
    down = run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    assert spoken(down) == []
    assert routes(down) == []
    assert signals(down) == ["lofi-busy", "lofi-idle"]
    assert context.get_messages()[-1]["role"] == "user"


def test_error_stays_silent():
    context = make_context("what did I buy?")
    agent = FakeAgent([rag_agent.AgentResult(answer=rag_agent.FALLBACK_ANSWER, error="down")])
    down = run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])
    assert spoken(down) == []


def test_context_frame_forwards_immediately():
    context = make_context("what did I buy?")

    async def never():
        await asyncio.sleep(10)

    agent = FakeAgent([never])
    down = run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.05)])
    assert any(isinstance(f, LLMContextFrame) for f in down)


def mode_frame(mode):
    return RTVIClientMessageFrame(msg_id="m1", type=MODE_MESSAGE_TYPE, data={"mode": mode})


def mode_confirmations(down):
    return [
        f.data for f in down
        if isinstance(f, RTVIServerMessageFrame) and f.data.get("type") == MODE_MESSAGE_TYPE
    ]


def test_default_mode_is_accurate_thinking_analyst():
    context = make_context("what did I buy at Paulie's?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    run_observer(context, agent, [LLMContextFrame(context), SleepFrame(0.1)])

    assert agent.calls[0]["think"] is True
    assert agent.calls[0]["model"] == config.rag_llm_model()


def test_fast_mode_switches_analyst_to_no_think_fast_model(monkeypatch):
    monkeypatch.setenv("LOFI_RAG_FAST_LLM_MODEL", "tiny-fast:1b")
    context = make_context("what did I buy at Paulie's?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    down = run_observer(
        context,
        agent,
        [mode_frame("fast"), LLMContextFrame(context), SleepFrame(0.1)],
    )

    assert agent.calls[0]["think"] is False
    assert agent.calls[0]["model"] == "tiny-fast:1b"
    confirmations = mode_confirmations(down)
    assert confirmations and confirmations[0]["mode"] == "fast"
    assert confirmations[0]["analyst_model"] == "tiny-fast:1b"
    assert confirmations[0]["thinking"] is False


def test_mode_toggles_back_to_accurate():
    context = make_context("what did I buy at Paulie's?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    down = run_observer(
        context,
        agent,
        [mode_frame("fast"), mode_frame("accurate"), LLMContextFrame(context), SleepFrame(0.1)],
    )
    assert agent.calls[0]["think"] is True
    assert mode_confirmations(down)[-1]["mode"] == "accurate"


def test_unknown_client_messages_are_ignored_and_forwarded():
    context = make_context("what did I buy at Paulie's?")
    agent = FakeAgent([rag_agent.AgentResult(answer=ANSWER, traces=[])])
    other = RTVIClientMessageFrame(msg_id="m2", type="some-other-thing", data={"x": 1})
    down = run_observer(context, agent, [other, LLMContextFrame(context), SleepFrame(0.1)])

    assert any(isinstance(f, RTVIClientMessageFrame) for f in down)
    assert mode_confirmations(down) == []
    assert agent.calls[0]["think"] is True


def test_barge_in_cancels_pending_lookup():
    context = make_context("what did I buy?")

    async def slow():
        await asyncio.sleep(0.2)
        return rag_agent.AgentResult(answer=ANSWER, traces=[TRACE])

    agent = FakeAgent([slow])
    down = run_observer(
        context,
        agent,
        [LLMContextFrame(context), SleepFrame(0.05), UserStartedSpeakingFrame(), SleepFrame(0.3)],
    )
    assert spoken(down) == []


def test_new_turn_replaces_pending_lookup():
    context = make_context("first question?")

    async def slow():
        await asyncio.sleep(5)
        return rag_agent.AgentResult(answer="stale", traces=[])

    fresh = rag_agent.AgentResult(answer=ANSWER, traces=[])
    agent = FakeAgent([slow, fresh])
    context.add_message({"role": "user", "content": "second question?"})
    down = run_observer(
        context,
        agent,
        [LLMContextFrame(context), SleepFrame(0.05), LLMContextFrame(context), SleepFrame(0.15)],
    )
    assert spoken(down) == [ANSWER]
