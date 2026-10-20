"""The records agent: the thinking half of the split architecture.

``answer_question`` drives Ollama's native ``/api/chat`` with reasoning ON
(the same model tag the pipeline serves with reasoning off — one copy of
weights, two configs), executes the four typed retrieval tools against the
SQLite store, and composes one or two spoken sentences. The delegate tool
handler (tools.build_delegate_tool) speaks that answer verbatim.

Tool schemas come from tools.PARAMS — the same single source the cascade and
the evals use.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

import httpx

import config
import retrieval
import store
import tools
from tunables import (
    ANALYST_SYSTEM_PROMPT,
    FALLBACK_ANSWER,
    GROUNDING_CATEGORIES,
    GROUNDING_DATE_RANGE,
    GROUNDING_TODAY,
    QUESTION_PREFIX,
)

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
            },
        },
    }
    for name, spec in tools.PARAMS.items()
]

@dataclass
class AgentResult:
    answer: str
    traces: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    error: str | None = None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def grounded_question(question: str, db_path, today: datetime.date | None = None) -> str:
    """Question as the analyst sees it: today's date, the records' date range,
    and the category list prepended, keeping the system prompt byte-stable."""
    today = today or datetime.date.today()
    parts = [GROUNDING_TODAY.format(today=today.isoformat())]
    span = store.date_range(db_path)
    if span:
        parts.append(GROUNDING_DATE_RANGE.format(start=span[0], end=span[1]))
    names = store.categories(db_path)
    if names:
        parts.append(GROUNDING_CATEGORIES.format(categories=", ".join(names)))
    return " ".join(parts) + QUESTION_PREFIX + question


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _run_tool(name: str, args: dict, db_path, embedder) -> dict:
    if name == "lookup":
        return retrieval.lookup(db_path, **args)
    if name == "filter_rows":
        return retrieval.filter_rows(db_path, **args)
    if name == "aggregate":
        return retrieval.aggregate(db_path, **args)
    return retrieval.semantic_search(db_path, embedder=embedder, **args)


async def _execute(name: str, args, db_path, embedder) -> dict:
    if name not in tools.PARAMS:
        raise ValueError(f"unknown tool {name!r}; available: {', '.join(tools.PARAMS)}")
    if not isinstance(args, dict):
        raise ValueError(f"arguments for {name!r} must be a JSON object")
    try:
        return await asyncio.to_thread(_run_tool, name, args, db_path, embedder)
    except (TypeError, ValueError, KeyError, sqlite3.Error) as exc:
        raise ValueError(f"{name} rejected those arguments: {exc}") from exc


async def answer_question(
    question: str,
    *,
    db_path,
    embedder,
    model: str | None = None,
    api_url: str | None = None,
    client=None,
    max_rounds: int = 4,
    deadline_secs: float = 60.0,
    today: datetime.date | None = None,
    think: bool = True,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    model = model or config.rag_llm_model()
    api_url = api_url or config.ollama_api_url()
    url = f"{api_url.rstrip('/')}/api/chat"

    # History rides between system and question; grounding stays in the final
    # user message so the system prompt stays byte-stable for the KV cache.
    messages = [
        {"role": "system", "content": system_prompt or ANALYST_SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": grounded_question(question, db_path, today)},
    ]
    traces: list[dict] = []
    chunks: list[dict] = []
    timings = {"rounds": 0, "total_ms": 0.0, "llm_ms": 0.0, "tools_ms": 0.0}
    started = time.perf_counter()
    bad_calls = 0
    answer = ""
    error: str | None = None

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=deadline_secs)
    try:
        for round_index in range(max_rounds + 1):
            remaining = deadline_secs - (time.perf_counter() - started)
            if remaining <= 0:
                error = "records lookup ran out of time"
                break

            body = {
                "model": model,
                "messages": list(messages),
                "think": think,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.2},
            }
            if round_index < max_rounds:
                body["tools"] = TOOL_DEFS

            timings["rounds"] += 1
            llm_start = time.perf_counter()
            payload = None
            try:
                response = await asyncio.wait_for(http.post(url, json=body), timeout=remaining)
                response.raise_for_status()
                payload = response.json()
            except TimeoutError:
                error = "records lookup ran out of time"
            except httpx.HTTPStatusError as exc:
                # Non-thinking models 400 on think=true — drop thinking and retry.
                if think and exc.response.status_code == 400 and "think" in exc.response.text.lower():
                    think = False
                    remaining = deadline_secs - (time.perf_counter() - started)
                    try:
                        response = await asyncio.wait_for(
                            http.post(url, json={**body, "think": False}),
                            timeout=max(0.1, remaining),
                        )
                        response.raise_for_status()
                        payload = response.json()
                    except (TimeoutError, httpx.HTTPError) as retry_exc:
                        error = f"records model unreachable: {retry_exc!r} (after think-off retry for: {exc!r})"
                else:
                    error = f"records model unreachable: {exc!r}"
            except httpx.HTTPError as exc:
                error = f"records model unreachable: {exc!r}"
            timings["llm_ms"] += _elapsed_ms(llm_start)
            if payload is None:
                break

            message = payload.get("message", {})
            calls = message.get("tool_calls") or []
            if not calls or round_index >= max_rounds:
                answer = strip_think(message.get("content", ""))
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": message["tool_calls"],
                }
            )
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                args = tools._parse_arguments(function.get("arguments"))
                tool_start = time.perf_counter()
                try:
                    result = await _execute(name, args, db_path, embedder)
                except ValueError as exc:
                    bad_calls += 1
                    if bad_calls > 1:
                        error = f"records model kept calling {name!r} incorrectly: {exc}"
                        break
                    messages.append(
                        {"role": "tool", "content": json.dumps({"error": str(exc)})}
                    )
                    continue
                timings["tools_ms"] += _elapsed_ms(tool_start)
                traces.append(
                    {
                        "tool": name,
                        "args": args,
                        "count": tools._result_count(result),
                        "elapsed_ms": result.get("elapsed_ms", _elapsed_ms(tool_start)),
                    }
                )
                if name == "semantic_search":
                    chunks.extend(result.get("results", []))
                messages.append({"role": "tool", "content": json.dumps(result, default=str)})
            if error:
                break
    finally:
        if owns_client:
            await http.aclose()

    timings["total_ms"] = _elapsed_ms(started)
    if not answer:
        answer = FALLBACK_ANSWER
        error = error or "no answer composed"
    return AgentResult(answer=answer, traces=traces, timings=timings, chunks=chunks, error=error)
