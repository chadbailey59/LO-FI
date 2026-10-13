"""Tests for rag_agent.answer_question — the thinking half of the split
architecture. No network: every /api/chat round is served by a scripted fake
standing in for httpx.AsyncClient, while the four retrieval tools run for real
against a fixture SQLite db.
"""

import asyncio
import datetime
import json
import shutil
from pathlib import Path

import httpx
import pytest

import ingest
import rag_agent
import store
import tools

PINNED_TODAY = datetime.date(2026, 10, 12)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(FIXTURES / "transactions-small.csv", data_dir / "transactions.csv")
    shutil.copy(FIXTURES / "note.txt", data_dir / "note.txt")
    db_path = tmp_path / "lofi.db"
    ingest.ingest(data_dir, db_path, ingest.FakeEmbedder())
    return db_path


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Scripted /api/chat payloads; records every request body. The last entry
    is sticky, so a one-entry script answers every round the same way."""

    def __init__(self, script):
        self._script = list(script)
        self.requests = []
        self.urls = []

    async def post(self, url, json=None, **kwargs):
        self.urls.append(url)
        self.requests.append(json)
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return FakeResponse(step)


def tool_call_payload(name, arguments, content=""):
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    }


def content_payload(text):
    return {"message": {"role": "assistant", "content": text}}


def run(db_path, question, script, **kwargs):
    client = FakeClient(script)
    result = asyncio.run(
        rag_agent.answer_question(
            question,
            db_path=db_path,
            embedder=ingest.FakeEmbedder(),
            model="twin-test",
            api_url="http://localhost:11434",
            client=client,
            today=PINNED_TODAY,
            **kwargs,
        )
    )
    return result, client


def test_tool_defs_come_from_the_shared_params():
    assert [d["function"]["name"] for d in rag_agent.TOOL_DEFS] == list(tools.PARAMS)
    for d in rag_agent.TOOL_DEFS:
        assert d["function"]["parameters"]["properties"]


def test_request_thinks_by_default_and_never_streams(db):
    _result, client = run(db, "what did I pay?", [content_payload("Nothing.")])
    body = client.requests[0]
    assert body["think"] is True
    assert body["stream"] is False
    assert body["model"] == "twin-test"
    assert client.urls[0] == "http://localhost:11434/api/chat"


def test_question_is_grounded_with_date_range_and_categories(db):
    _result, client = run(db, "spend last August?", [content_payload("ok")])
    content = client.requests[0]["messages"][-1]["content"]
    assert content.startswith("Today's date is 2026-10-12.")
    first, last = store.date_range(db)
    assert f"The transaction records run from {first} to {last}." in content
    assert "The transaction categories are: " in content
    assert content.endswith("Question: spend last August?")


def test_history_threads_between_system_and_grounded_question(db):
    history = [
        {"role": "user", "content": "what did I spend at Paulie's?"},
        {"role": "assistant", "content": "You spent $40.00 at Paulie's Delicatessen."},
    ]
    _result, client = run(
        db, "what category was that?", [content_payload("Deli.")], history=history
    )
    messages = client.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1]["content"].endswith("Question: what category was that?")


def test_tool_round_trip_runs_real_retrieval(db):
    result, client = run(
        db,
        "what did I pay Harborview?",
        [
            tool_call_payload("lookup", {"query": "Harborview"}),
            content_payload("You paid Harborview $412.00 on March 3, 2025."),
        ],
    )
    assert result.error is None
    assert result.traces[0]["tool"] == "lookup"
    assert result.traces[0]["count"] >= 1
    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["rows"]


def test_semantic_search_populates_chunks(db):
    note_text = " ".join((FIXTURES / "note.txt").read_text().split())
    result, _client = run(
        db,
        "what's in my note?",
        [
            tool_call_payload("semantic_search", {"query": note_text}),
            content_payload("Your note says hello."),
        ],
    )
    assert result.chunks
    assert all("source" in c and "text" in c for c in result.chunks)


def test_bad_tool_call_gets_one_corrective_round(db):
    result, client = run(
        db,
        "what did I pay?",
        [
            tool_call_payload("sql_query", {"sql": "select 1"}),
            tool_call_payload("lookup", {"query": "Harborview"}),
            content_payload("Found it."),
        ],
    )
    correction = client.requests[1]["messages"][-1]
    assert correction["role"] == "tool"
    assert json.loads(correction["content"])["error"]
    assert result.answer == "Found it."


def test_second_bad_call_falls_back(db):
    result, _client = run(db, "what did I pay?", [tool_call_payload("sql_query", {"sql": "x"})])
    assert result.answer == rag_agent.FALLBACK_ANSWER
    assert result.error


def test_round_cap_forces_final_compose_without_tools(db):
    result, client = run(
        db,
        "what did I pay?",
        [
            tool_call_payload("lookup", {"query": "Harborview"}),
            tool_call_payload("lookup", {"query": "Whole Foods"}),
            content_payload("Done."),
        ],
        max_rounds=2,
    )
    assert "tools" in client.requests[0]
    assert "tools" in client.requests[1]
    assert "tools" not in client.requests[2]
    assert result.answer == "Done."


class FakeThinkRejectingClient(FakeClient):
    """400s any request with think=true, like Ollama serving a non-thinking model."""

    async def post(self, url, json=None, **kwargs):
        if json.get("think"):
            self.urls.append(url)
            self.requests.append(json)
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400, request=request, json={"error": "model does not support thinking"}
            )
            raise httpx.HTTPStatusError("Client error '400'", request=request, response=response)
        return await super().post(url, json=json, **kwargs)


def test_non_thinking_model_falls_back_to_think_off(db):
    client = FakeThinkRejectingClient([content_payload("You spent $40.00.")])
    result = asyncio.run(
        rag_agent.answer_question(
            "what did I pay?",
            db_path=db,
            embedder=ingest.FakeEmbedder(),
            model="twin-test",
            api_url="http://localhost:11434",
            client=client,
            today=PINNED_TODAY,
        )
    )
    assert result.answer == "You spent $40.00."
    assert result.error is None
    assert client.requests[0]["think"] is True
    assert client.requests[1]["think"] is False


def test_connect_error_returns_fallback(db):
    result, _client = run(db, "what did I pay?", [httpx.ConnectError("refused")])
    assert result.answer == rag_agent.FALLBACK_ANSWER
    assert result.error


def test_thinking_is_stripped_from_the_final_answer(db):
    result, _client = run(db, "hi", [content_payload("<think>hmm</think>You spent $40.00.")])
    assert result.answer == "You spent $40.00."


def test_cancelled_error_propagates_uncaught(db):
    with pytest.raises(asyncio.CancelledError):
        run(db, "what did I pay?", [asyncio.CancelledError()])
