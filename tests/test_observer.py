"""Tests for the observer architecture's RecordsObserver: triggers on new user
turns, speaks grounded answers, stays silent on the NOT_RECORDS sentinel and on
errors, writes spoken answers back into the context, cancels on barge-in."""

import asyncio

from pipecat.frames.frames import LLMContextFrame, TTSSpeakFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test

import rag_agent
from observer_processor import RecordsObserver
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
    assert agent.calls[0]["system_prompt"] == records_analyst.OBSERVER_SYSTEM_PROMPT


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
