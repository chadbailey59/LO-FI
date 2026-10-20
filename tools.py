"""The four retrieval tools the pipeline model calls directly.

`PARAMS` is the single source of truth for the tool schemas — `build_tools`
turns it into pipecat FunctionSchemas with bundled handlers, and the evals
build their OpenAI-format tool list from it. Each call pushes a `lofi-rag`
RTVI server message (route line: tool, args, count, elapsed) so the frontend
shows how an answer was found.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

import retrieval
from tunables import (
    DELEGATE_TOOL_NAME,
    LEGACY_DELEGATE_DESCRIPTION,
    NO_QUESTION_ANSWER,
)

RAG_MESSAGE_TYPE = "lofi-rag"
BUSY_MESSAGE_TYPE = "lofi-busy"
IDLE_MESSAGE_TYPE = "lofi-idle"

# Outer bound on one delegate call; the records agent's own deadline (60 s,
# reasoning on) fires first.
DELEGATE_TIMEOUT_SECS = 75.0

_COLUMNS = sorted(retrieval.COLUMN_SQL)
_OPERATORS = sorted(retrieval.OPERATORS)
_ORDER_BY = sorted(retrieval.ORDER_SQL)

_FILTER_ITEM = {
    "type": "object",
    "properties": {
        "column": {"type": "string", "enum": _COLUMNS},
        "operator": {"type": "string", "enum": _OPERATORS},
        "value": {"type": "string"},
        "value2": {"type": "string"},
    },
    "required": ["column", "operator", "value"],
}

# Per-tool and per-parameter descriptions are schema-structural (enums come from
# retrieval.COLUMN_SQL / OPERATORS / ORDER_SQL), so they live here; see tunables.py
# header.
PARAMS = {
    "lookup": {
        "description": (
            "Find specific transactions by merchant name, words from the "
            "statement, or an 18-digit transaction id."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": "Merchant name, words to search for, or a transaction id.",
            },
            "limit": {
                "type": "integer",
                "description": "How many transactions to return, at most 5.",
            },
        },
        "required": ["query"],
    },
    "filter_rows": {
        "description": (
            "List transactions matching a condition on a column, sorted by "
            "date or amount. Use for questions like the biggest expenses in a "
            "month. Add more conditions (a date range, a category) in filters."
        ),
        "properties": {
            "column": {"type": "string", "enum": _COLUMNS, "description": "Column to filter on."},
            "operator": {"type": "string", "enum": _OPERATORS, "description": "Comparison to apply."},
            "value": {
                "type": "string",
                "description": (
                    "Value to compare against. Dates may be a year, a year and "
                    "month like 2025-03, or a full date."
                ),
            },
            "value2": {
                "type": "string",
                "description": "Upper bound, for the between operator only.",
            },
            "filters": {
                "type": "array",
                "description": "Up to two more conditions, all of which must match.",
                "items": _FILTER_ITEM,
            },
            "order_by": {
                "type": "string",
                "enum": _ORDER_BY,
                "description": "Sort key; amount_abs ignores the sign.",
            },
            "descending": {"type": "boolean", "description": "Sort largest or newest first."},
            "limit": {"type": "integer", "description": "How many transactions to return, at most 5."},
        },
        "required": ["column", "operator", "value"],
    },
    "aggregate": {
        "description": (
            "Count transactions or total their amounts, optionally narrowed by "
            "up to three filters. Use for how much did I spend questions — a "
            "question naming a period needs a date filter."
        ),
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["count", "sum", "avg", "min", "max"],
                "description": "sum returns spent, received and net.",
            },
            "column": {
                "type": "string",
                "enum": _COLUMNS,
                "description": "Column to aggregate; usually amount.",
            },
            "filters": {
                "type": "array",
                "description": "Up to three conditions, all of which must match.",
                "items": _FILTER_ITEM,
            },
            "include_transfers": {
                "type": "boolean",
                "description": "Transfers are excluded from totals unless this is true.",
            },
        },
        "required": ["operation"],
    },
    "semantic_search": {
        "description": (
            "Search the user's saved documents and notes by meaning. Use for "
            "anything that is not a transaction lookup or total."
        ),
        "properties": {
            "query": {"type": "string", "description": "What to search the documents for."},
            "k": {"type": "integer", "description": "How many passages to return, at most 5."},
        },
        "required": ["query"],
    },
}


def _parse_arguments(raw) -> dict | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return dict(raw) if isinstance(raw, dict) else None


def _result_count(payload: dict) -> int:
    if "rows" in payload:
        return len(payload["rows"])
    if "results" in payload:
        return len(payload["results"])
    return payload.get("count", 0)


def build_backend_tools(db_path, embedder) -> list[FunctionSchema]:
    """The four schemas with plain handlers for a BackendLLMWorker: result in,
    result out — no RTVI side-channel (the worker already streams each call as
    a BackendToolCall event; the UI panel re-plumbs from those)."""

    funcs = {
        "lookup": retrieval.lookup,
        "filter_rows": retrieval.filter_rows,
        "aggregate": retrieval.aggregate,
        "semantic_search": lambda path, **kw: retrieval.semantic_search(path, embedder=embedder, **kw),
    }

    def make_handler(name, func):
        async def handler(params: FunctionCallParams) -> None:
            args = _parse_arguments(params.arguments)
            if args is None:
                await params.result_callback({"error": "arguments must be a JSON object"})
                return
            try:
                result = await asyncio.to_thread(func, db_path, **args)
            except (TypeError, ValueError, KeyError, sqlite3.Error) as exc:
                await params.result_callback({"error": str(exc)})
                return
            await params.result_callback(result)

        handler.__name__ = name
        return handler

    return [
        FunctionSchema(
            name=name,
            description=spec["description"],
            properties=spec["properties"],
            required=spec["required"],
            handler=make_handler(name, funcs[name]),
        )
        for name, spec in PARAMS.items()
    ]


def build_tools(db_path, embedder) -> list[FunctionSchema]:
    """Four FunctionSchemas with bundled handlers — pipecat registers them for
    any LLMContext that advertises the schemas."""

    funcs = {
        "lookup": retrieval.lookup,
        "filter_rows": retrieval.filter_rows,
        "aggregate": retrieval.aggregate,
        "semantic_search": lambda path, **kw: retrieval.semantic_search(path, embedder=embedder, **kw),
    }

    def make_handler(name, func):
        async def handler(params: FunctionCallParams) -> None:
            args = _parse_arguments(params.arguments)
            if args is None:
                await params.result_callback({"error": "arguments must be a JSON object"})
                return
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(func, db_path, **args)
            except (TypeError, ValueError, KeyError, sqlite3.Error) as exc:
                await params.result_callback({"error": str(exc)})
                return
            await params.llm.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": RAG_MESSAGE_TYPE,
                        "tool": name,
                        "args": args,
                        "count": _result_count(result),
                        "elapsed_ms": result.get(
                            "elapsed_ms", round((time.perf_counter() - started) * 1000, 2)
                        ),
                        "chunks": result.get("results", []) if name == "semantic_search" else [],
                    }
                )
            )
            await params.result_callback(result)

        handler.__name__ = name
        return handler

    return [
        FunctionSchema(
            name=name,
            description=spec["description"],
            properties=spec["properties"],
            required=spec["required"],
            handler=make_handler(name, funcs[name]),
        )
        for name, spec in PARAMS.items()
    ]


def build_delegate_tool(db_path, embedder) -> list[FunctionSchema]:
    """Legacy question-arg delegate (tests/evals); production split uses the
    no-arg BackendConnector tool. Handler runs the records agent (reasoning on,
    native API), pushes one `lofi-rag` route line per backend tool call, speaks
    the composed answer itself, and suppresses the pipeline model's follow-up
    completion — the front never composes over retrieved numbers."""
    import rag_agent

    async def speak_and_finish(params: FunctionCallParams, answer: str) -> None:
        await params.llm.push_frame(TTSSpeakFrame(answer))
        await params.result_callback(
            {"say_exactly": answer},
            properties=FunctionCallResultProperties(run_llm=False),
        )

    @tool_options(timeout_secs=DELEGATE_TIMEOUT_SECS)
    async def ask_financial_records(params: FunctionCallParams) -> None:
        args = _parse_arguments(params.arguments) or {}
        question = str(args.get("question") or "").strip()
        if not question:
            await speak_and_finish(params, NO_QUESTION_ANSWER)
            return

        result = await rag_agent.answer_question(question, db_path=db_path, embedder=embedder)
        for trace in result.traces:
            await params.llm.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": RAG_MESSAGE_TYPE,
                        "tool": trace["tool"],
                        "args": trace["args"],
                        "count": trace["count"],
                        "elapsed_ms": trace["elapsed_ms"],
                        "chunks": result.chunks if trace["tool"] == "semantic_search" else [],
                    }
                )
            )
        await speak_and_finish(params, result.answer)

    return [
        FunctionSchema(
            name=DELEGATE_TOOL_NAME,
            description=LEGACY_DELEGATE_DESCRIPTION,
            properties={
                "question": {
                    "type": "string",
                    "description": "The user's question, in their own words.",
                }
            },
            required=["question"],
            handler=ask_financial_records,
        )
    ]
