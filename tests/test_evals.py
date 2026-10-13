"""Offline tests for evals/run_evals.py stage logic — no network: the analyst
is monkeypatched; only query routing/scoring runs."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import run_evals  # noqa: E402

import rag_agent  # noqa: E402

HISTORY = [
    {"role": "user", "content": "what did I buy at Paulie's Delicatessen?"},
    {"role": "assistant", "content": "You spent $40.00 at Paulie's Delicatessen on August 24, 2026."},
]


def test_observer_stage_threads_multiturn_history(monkeypatch):
    calls = []

    async def fake_answer(question, **kwargs):
        calls.append({"question": question, **kwargs})
        return rag_agent.AgentResult(answer="That was in the Food & Drink category, $40.00.")

    monkeypatch.setattr(run_evals.rag_agent, "answer_question", fake_answer)
    query = {
        "id": "mt-1",
        "type": "multiturn",
        "query": "what category was that?",
        "history": HISTORY,
        "expected_answer": "Food & Drink 40.00",
    }
    report = asyncio.run(run_evals.run_observer_stage([query], "/tmp/x.db", None))

    assert calls[0]["question"] == "what category was that?"
    assert calls[0]["history"] == HISTORY
    assert report["data_rows"][0]["answer_ok"] is True


class FakeV1Client:
    """Scripted /v1/chat/completions replies; sticky last entry; exceptions raise."""

    def __init__(self, script):
        self._script = list(script)
        self.requests = []

    async def post(self, url, json=None, **kwargs):
        self.requests.append(json)
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": step}}]}

        return R()


def run_front(queries, script):
    client = FakeV1Client(script)
    return asyncio.run(run_evals.run_front_stage(queries, "front prompt", client)), client


DATA_Q = {"id": "lookup-1", "type": "lookup", "query": "what did I buy at Paulie's?",
          "expected_tool": "lookup", "expected_args_subset": {}, "expected_answer": "Paulie's 40.00"}
CHIT_Q = {"id": "chit-1", "type": "chitchat", "query": "how are you today?"}


def test_front_stage_passes_deferral_and_fails_invented_numbers():
    report, _ = run_front([DATA_Q], ["Let me check your records."])
    assert report["defers_on_data"] == 1.0
    assert report["no_invented_numbers_rate"] == 1.0

    report, _ = run_front([DATA_Q], ["You spent $40.00 at Paulie's."])
    assert report["no_invented_numbers_rate"] == 0.0


def test_front_stage_scores_chitchat_as_normal_answer():
    report, _ = run_front([CHIT_Q], ["I'm doing great, thanks for asking!"])
    assert report["answers_chitchat"] == 1.0

    report, _ = run_front([CHIT_Q], ["Let me check your records."])
    assert report["answers_chitchat"] == 0.0


def test_front_stage_spoken_number_in_deferral_is_not_invented():
    report, _ = run_front([DATA_Q], ["Let me check your records, one moment."])
    assert report["no_invented_numbers_rate"] == 1.0
    assert report["defers_on_data"] == 1.0


def test_front_stage_error_rows_do_not_pass_the_rates():
    import httpx

    report, _ = run_front([DATA_Q], [httpx.ConnectError("refused")])
    assert report["errors"] == 1
    assert report["no_invented_numbers_rate"] is None
    assert report["defers_on_data"] is None


def test_front_stage_sends_no_tools_and_skips_multiturn():
    mt = {"id": "mt-1", "type": "multiturn", "query": "what was that?", "history": HISTORY,
          "expected_answer": "x"}
    report, client = run_front([DATA_Q, mt], ["Let me check your records."])
    assert [r["id"] for r in report["rows"]] == ["lookup-1"]
    assert "tools" not in client.requests[0]


def test_json_payload_records_models_and_stage():
    payload = run_evals.build_json_payload("observer", {"observer": {"answer_accuracy": 0.7}}, git_sha="abc123")
    assert payload["meta"]["stage"] == "observer"
    assert payload["meta"]["llm_model"]
    assert payload["meta"]["rag_llm_model"]
    assert payload["meta"]["pinned_today"] == "2026-10-12"
    assert payload["meta"]["git_sha"] == "abc123"
    assert payload["reports"]["observer"]["answer_accuracy"] == 0.7


def test_single_turn_stages_skip_multiturn_queries():
    query = {
        "id": "mt-1",
        "type": "multiturn",
        "query": "what category was that?",
        "history": HISTORY,
        "expected_answer": "Food & Drink 40.00",
    }
    toolcall = asyncio.run(run_evals.run_toolcall_stage([query], "sys", client=None))
    assert toolcall["rows"] == []
