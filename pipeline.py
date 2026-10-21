"""Shared pipeline construction — one brain, every transport.

`build_pipeline(transport)` assembles the full-duplex pipeline used by every bot:

    transport.input() -> VAD -> STT -> user-context -> LLM -> TTS -> transport.output() -> assistant-context

Retrieval is no longer a pipeline stage. When an index exists the conversation
model is offered one tool, `ask_financial_records` (tools.py), whose handler
runs the records agent outside the pipeline.

Bot files keep only what differs per transport: transport construction, preflight,
greeting, and runner glue.
"""

import re
import time
from dataclasses import dataclass

# config FIRST — sets HF_HOME / Kokoro cache paths before any pipecat/HF import (see config.py).
import config

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallsStartedFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.base_transport import BaseTransport

import ingest
import store
import tools
from rag_agent import grounded_question, strip_think
import services
from services import build_backend_llm, build_llm, build_stt, build_tts
from silence_watchdog import SilenceWatchdog
from tunables import (
    ANALYST_SYSTEM_PROMPT,
    BACKEND_COMPACT_PAIRING_INSTRUCTION,
    COMPACT_PAIRING_INSTRUCTION,
    DELEGATE_TOOL_DESCRIPTION,
    DELEGATE_TOOL_NAME,
    FILLER_PHRASE,
    QUESTION_PREFIX,
    SPLIT_FRONT_PROMPT,
    WAIT_LESSON_NUDGE,
    hosted_variant,
    system_prompt,
)


@dataclass
class BuiltBot:
    pipeline: Pipeline
    context: LLMContext
    user_aggregator: LLMUserAggregator
    assistant_aggregator: LLMAssistantAggregator
    llm: object
    tts: object


def build_vad_processor() -> VADProcessor:
    return VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=config.vad_confidence(),
                min_volume=config.vad_min_volume(),
                start_secs=config.vad_start_secs(),
                stop_secs=config.vad_stop_secs(),
            )
        )
    )


BUSY_MESSAGE_TYPE = tools.BUSY_MESSAGE_TYPE

_missing_index_logged = False


def _index_ready(db_path) -> bool:
    global _missing_index_logged
    if store.has_index(db_path):
        return True
    if not _missing_index_logged:
        logger.info(f"no RAG index found at {db_path} — run ./lofi.sh index-rag")
        _missing_index_logged = True
    return False


def build_records_tools() -> list | None:
    """Cascade only: the four tools go straight on the conversation model."""
    db_path = config.rag_db_path()
    if config.architecture() == "split" or not _index_ready(db_path):
        return None
    embedder = ingest.OllamaEmbedder(config.ollama_api_url(), config.embed_model())
    return tools.build_tools(db_path, embedder)


def build_records_connector():
    """BackendConnector tuned for a small local front model.

    Measured on qwen2.5:7b: a tool literally named "delegate" is never called
    (the model narrates instead), and the stock frontier-sized pairing
    instruction suppresses calling entirely — a domain-named tool plus the
    compact instruction (and front temperature 0, see build_pipeline) delegates
    3/3. The tool description is domain-worded for the same reason (see
    DELEGATE_TOOL_DESCRIPTION).
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.frames.frames import LLMMessagesAppendFrame
    from pipecat.pipeline.llm_with_backend import (
        BACKEND_MESSAGE_PREFIX,
        BackendConnector,
        BackendError,
        BackendIdle,
        BackendOutput,
    )
    from pipecat.services.llm_service import FunctionCallParams

    class RecordsBackendConnector(BackendConnector):
        # Per-lull delivery state: something appended / something ran the front.
        _lull_appended = False
        _lull_ran_front = False
        @property
        def tools(self):
            async def handler(params: FunctionCallParams):
                await self.delegate(params)

            return [
                FunctionSchema(
                    name=DELEGATE_TOOL_NAME,
                    description=DELEGATE_TOOL_DESCRIPTION,
                    properties=self.request_strategy.tool_parameters,
                    required=self.request_strategy.tool_required,
                    handler=handler,
                ),
            ]

        # Ollama has no developer role, so backend notes reach the front as
        # USER messages — the stock per-message RELAY_NOTE then reads as a
        # standing user demand, and the 7B front re-relays the old result on
        # every later turn instead of delegating the new question (measured,
        # 2026-10-19 stuck session: stock note never delegates the follow-up;
        # bare "Backend:" message delegates; arrival-turn relay unaffected —
        # the compact pairing instruction already says to relay results once).
        def render_output(self, output):
            message = super().render_output(output)
            if message is not None and output.prefers_spoken and not output.is_thought:
                message = dict(message, content=f"{BACKEND_MESSAGE_PREFIX}{output.text}")
            return message

        # Analyst sometimes omits the spoken mark on its final answer; the
        # note then lands with run_llm=False and nothing runs the front — the
        # bot sits mute on a finished answer until the user speaks (measured
        # 2026-10-21 log: answer delivered silent 13s after the question,
        # spoken only after "You still there?" 63s later). When a lull ends
        # with appended output and no front run, nudge one: an empty append
        # with run_llm=True runs the front on the context it already holds,
        # waits out in-flight speech, and persists no message.
        async def deliver(self, frontend, event):
            await super().deliver(frontend, event)
            if isinstance(event, BackendOutput):
                if self.render_output(event) is not None:
                    self._lull_appended = True
                    if event.prefers_spoken:
                        self._lull_ran_front = True
            elif isinstance(event, BackendError):
                self._lull_ran_front = True
            elif isinstance(event, BackendIdle):
                if self._lull_appended and not self._lull_ran_front:
                    await self._append(
                        frontend, LLMMessagesAppendFrame(messages=[], run_llm=True)
                    )
                self._lull_appended = False
                self._lull_ran_front = False

    return RecordsBackendConnector(
        frontend_pairing_instruction=COMPACT_PAIRING_INSTRUCTION,
        # Post-delegate re-run narrates the handoff ("Let me check that for
        # you..."); those narrations accumulate in context as complete-looking
        # answers to records questions, and after two rounds the temp-0 7B
        # imitates them without the tool call (measured 2026-10-20,
        # evals --stage followup: live misses 3/3 with narrations, 1/3 without;
        # the arguably-conversational "Are you still there?" turn is the 1).
        # Off: the canned filler in build_pipeline speaks instead.
        respond_on_delegate=False,
    )


_MARKDOWN_EMPHASIS = re.compile(r"[*_]{1,3}([^*_]+?)[*_]{1,3}|`([^`]+)`")


def strip_markdown(text: str) -> str:
    """Drop emphasis/code markers the analyst sometimes adds — TTS reads them."""
    return _MARKDOWN_EMPHASIS.sub(lambda m: m.group(1) or m.group(2), text)


def build_records_backend(db_path):
    """The split architecture's backend: a BackendLLMWorker holding the four
    retrieval tools and the analyst prompt (+ per-session date grounding).
    Thinking never reaches TTS: the worker renders output as spoken-or-note,
    and transform_output strips any inline <think> blocks as a backstop."""
    from pipecat.workers.llm.backend_llm_worker import BackendLLMWorker

    embedder = ingest.OllamaEmbedder(config.ollama_api_url(), config.embed_model())
    # grounded_question keeps grounding out of the system prompt for the native
    # API's KV cache; here the backend context is built once per session, so
    # folding grounding into the system message is equally cache-stable.
    grounding = grounded_question("", db_path).rsplit(QUESTION_PREFIX.rstrip(" "), 1)[0]
    backend_system = f"{ANALYST_SYSTEM_PROMPT}\n\n{grounding}"

    async def _strip_think_output(output):
        text = strip_markdown(strip_think(output.text))
        if not text:
            return None
        output.text = text
        return output

    return BackendLLMWorker(
        llm=build_backend_llm(),
        context=LLMContext(
            messages=[{"role": "system", "content": backend_system}],
            tools=tools.build_backend_tools(db_path, embedder),
        ),
        transform_output=_strip_think_output,
        # Stock instruction inverts the >> mark on qwen3:8b — see the
        # BACKEND_COMPACT_PAIRING_INSTRUCTION comment in tunables.py.
        pairing_instruction=BACKEND_COMPACT_PAIRING_INSTRUCTION,
    )


class BackendToolPanel(FrameProcessor):
    """Mirrors the backend worker's tool calls into the UI's lofi-rag panel.

    The backend's handlers run outside the pipeline, so they can't push RTVI
    frames themselves; LLMWithBackend re-announces each call downstream as
    ExternalFunctionCall*Frames. Render each final result as the same lofi-rag
    message the cascade handlers send (tools.build_tools), and each call's
    start as a lofi-rag-pending message — an analyst turn runs 10-70 s, so the
    panel shows the lookup in flight instead of staying empty until it lands."""

    def __init__(self):
        super().__init__()
        self._started: dict[str, float] = {}

    async def _pending(self, tool: str, args: dict, active: bool) -> None:
        await self.push_frame(
            RTVIServerMessageFrame(
                data={
                    "type": tools.RAG_PENDING_MESSAGE_TYPE,
                    "tool": tool,
                    "args": args,
                    "active": active,
                }
            )
        )

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ExternalFunctionCallsStartedFrame):
            for call in frame.function_calls:
                self._started[call.tool_call_id] = time.perf_counter()
                # Started phase carries name + id only; args arrive in_progress.
                await self._pending(call.function_name, {}, True)
        elif isinstance(frame, ExternalFunctionCallInProgressFrame):
            args = frame.arguments if isinstance(frame.arguments, dict) else {}
            await self._pending(frame.function_name, args, True)
        elif isinstance(frame, ExternalFunctionCallCancelFrame):
            self._started.pop(frame.tool_call_id, None)
            await self._pending(frame.function_name, {}, False)
        elif isinstance(frame, ExternalFunctionCallResultFrame) and frame.is_final:
            started = self._started.pop(frame.tool_call_id, None)
            result = frame.result if isinstance(frame.result, dict) else {}
            args = frame.arguments if isinstance(frame.arguments, dict) else {}
            elapsed_ms = result.get(
                "elapsed_ms",
                round((time.perf_counter() - started) * 1000, 2) if started else 0,
            )
            await self.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": tools.RAG_MESSAGE_TYPE,
                        "tool": frame.function_name,
                        "args": args,
                        "count": tools._result_count(result),
                        "elapsed_ms": elapsed_ms,
                        "chunks": result.get("results", [])
                        if frame.function_name == "semantic_search"
                        else [],
                    }
                )
            )
        await self.push_frame(frame, direction)


def build_pipeline(transport: BaseTransport) -> BuiltBot:
    records_tools = build_records_tools()
    db_path = config.rag_db_path()
    splitting = config.architecture() == "split" and _index_ready(db_path)
    if splitting:
        # Load the analyst's weights now, in the background, so the first
        # delegation doesn't pay the cold load while the user waits.
        services.warm_analyst()
        # The framework appends its own delegation guidance to this.
        prompt = SPLIT_FRONT_PROMPT
    else:
        prompt = system_prompt(
            date_range=store.date_range(db_path) if records_tools else None,
            categories=store.categories(db_path) if records_tools else None,
        )
    if config.llm_engine() == "hosted":
        prompt = hosted_variant(prompt)
    context = (
        LLMContext(messages=[{"role": "system", "content": prompt}], tools=records_tools)
        if records_tools
        else LLMContext(messages=[{"role": "system", "content": prompt}])
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Split front runs at temperature 0: delegation reliability on a 7B
    # (measured 3/3 at 0, 2/3 at 0.7).
    front = build_llm(temperature=0 if splitting else None)
    if splitting:
        from pipecat.pipeline.llm_with_backend import LLMWithBackend

        llm = LLMWithBackend(
            frontend=front,
            backend=build_records_backend(db_path),
            connector=build_records_connector(),
        )
    else:
        llm = front
    tts = build_tts()

    @front.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await service.push_frame(RTVIServerMessageFrame(data={"type": BUSY_MESSAGE_TYPE}))
        # Model is silent during the tool call (split: respond_on_delegate off,
        # see build_records_connector), so speak the canned filler — out of the
        # context: waiting lines there teach the front to answer records
        # questions with waiting lines (measured 2026-10-19 and 2026-10-20).
        await tts.queue_frame(TTSSpeakFrame(FILLER_PHRASE, append_to_context=False))

    processors = [
        transport.input(),
        build_vad_processor(),
        build_stt(),
        user_aggregator,
    ]
    lesson_language = config.wait_lesson_language()
    lesson = None
    if splitting and lesson_language:
        from pipecat.frames.frames import LLMMessagesAppendFrame

        async def lesson():
            await llm.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": WAIT_LESSON_NUDGE.format(language=lesson_language),
                        }
                    ],
                    run_llm=True,
                )
            )

    pipeline = Pipeline(
        [
            *processors,
            llm,
            *([BackendToolPanel()] if splitting else []),
            SilenceWatchdog(lesson=lesson),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    return BuiltBot(
        pipeline=pipeline,
        context=context,
        user_aggregator=user_aggregator,
        assistant_aggregator=assistant_aggregator,
        llm=llm,
        tts=tts,
    )
