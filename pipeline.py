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
from prompts import records_analyst
from prompts.financial_advisor import (
    SPLIT_FRONT_PROMPT,
    hosted_variant,
    system_prompt,
)
from rag_agent import grounded_question, strip_think
from services import build_backend_llm, build_llm, build_stt, build_tts
from silence_watchdog import SilenceWatchdog


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


FILLER_PHRASE = "Let me check your records."
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


DELEGATE_NAME = "ask_financial_records"


# Pairing instruction sized for small local fronts (the stock one is written
# for frontier models and drowns them — measured on qwen2.5:7b AND
# nemotron-3-nano:4b: both stop tool-calling and narrate instead). Landmines
# this wording deliberately avoids (see prompts/financial_advisor.py): worked
# call examples, "never write X as text" clauses, and the word "Backend" —
# each measured to break one small model or the other. Relay reliability comes
# from the framework's per-message relay note.
COMPACT_PAIRING_INSTRUCTION = (
    "RECORDS HANDOFF: For ANY question about the user's transactions, spending, "
    "merchants, amounts, dates, totals, balances, documents, or a follow-up to an "
    f"earlier records answer, you MUST call the {DELEGATE_NAME} tool in that same "
    "reply. Do not answer such questions yourself and never invent numbers. To "
    "the user there is one assistant: you — never mention the analyst, tools, or "
    "handoffs. The analyst's findings arrive later as internal conversation "
    "messages; when one brings a result, tell the user what it says in your own "
    "words, once. Internal progress or thinking notes are never read aloud and "
    "never guessed from."
)


# Domain-worded tool description (same lever as the domain name: measured on
# qwen2.5:7b, the stock "hand the conversation to the backend" description
# loses to the model's credential-refusal reflex on passphrase questions —
# this wording delegates them, incl. the saved-passphrase smoke doc, with
# records/chitchat behavior unchanged). The arithmetic sentence routes "what's
# that total?" follow-ups to the analyst's aggregate SQL — without it the 7B
# adds the amounts itself and miscarries digits (measured 2026-10-19: $301.81
# spoken as "three hundred and61.81"). Keeps the transcript strategy's
# semantics: one call per reply, no arguments, new request → new call.
DOMAIN_TOOL_DESCRIPTION = (
    "Look up anything in the user's own financial records and saved documents: "
    "transactions, spending, income, accounts, merchants, amounts, dates, totals, "
    "balances, and anything they saved in documents or notes, including a "
    "passphrase or code they stored there. All arithmetic is its job too: a "
    "total, sum, difference, or average, even over amounts already said in "
    "this conversation. Call this as soon as any part of what "
    "the user asks needs it, and do the rest yourself. One call per reply, however "
    "many things the user asked for. It takes no arguments: the analyst reads the "
    "conversation itself. A request the user makes after your last call needs a "
    "call of its own, and so does anything the user says about work already handed "
    "over. Keep talking with the user while it works."
)


def build_records_connector():
    """BackendConnector tuned for a small local front model.

    Measured on qwen2.5:7b: a tool literally named "delegate" is never called
    (the model narrates instead), and the stock frontier-sized pairing
    instruction suppresses calling entirely — a domain-named tool plus the
    compact instruction (and front temperature 0, see build_pipeline) delegates
    3/3. The tool description is domain-worded for the same reason (see
    DOMAIN_TOOL_DESCRIPTION).
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.pipeline.llm_with_backend import BACKEND_MESSAGE_PREFIX, BackendConnector
    from pipecat.services.llm_service import FunctionCallParams

    class RecordsBackendConnector(BackendConnector):
        @property
        def tools(self):
            async def handler(params: FunctionCallParams):
                await self.delegate(params)

            return [
                FunctionSchema(
                    name=DELEGATE_NAME,
                    description=DOMAIN_TOOL_DESCRIPTION,
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

    return RecordsBackendConnector(
        frontend_pairing_instruction=COMPACT_PAIRING_INSTRUCTION
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
    grounding = grounded_question("", db_path).rsplit("\nQuestion:", 1)[0]
    backend_system = f"{records_analyst.SYSTEM_PROMPT}\n\n{grounding}"

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
    )


# Wait-window mini-lesson (LOFI_WAIT_LESSON_LANGUAGE, split only): while the
# analyst works, the watchdog's first reminder has the front teach one phrase
# tied to the user's question. Short and bounded — the analyst's answer queues
# right behind whatever the lesson generates.
WAIT_LESSON_NUDGE = (
    "The records lookup is still running; its result arrives soon. While the "
    "user waits, teach them one short beginner {language} phrase related to "
    "what they just asked: the phrase, how it sounds, and what it means — one "
    "sentence each, then stop. Do not mention the lookup or this note."
)


class BackendToolPanel(FrameProcessor):
    """Mirrors the backend worker's tool calls into the UI's lofi-rag panel.

    The backend's handlers run outside the pipeline, so they can't push RTVI
    frames themselves; LLMWithBackend re-announces each call downstream as
    ExternalFunctionCall*Frames. Render each final result as the same lofi-rag
    message the cascade handlers send (tools.build_tools)."""

    def __init__(self):
        super().__init__()
        self._started: dict[str, float] = {}

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ExternalFunctionCallsStartedFrame):
            for call in frame.function_calls:
                self._started[call.tool_call_id] = time.perf_counter()
        elif isinstance(frame, ExternalFunctionCallCancelFrame):
            self._started.pop(frame.tool_call_id, None)
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
        # Split: the front narrates its own handoff — a canned filler on top
        # is one of three stacked "let me check" lines, and in the context it
        # teaches the front to answer with waiting lines (measured 2026-10-19).
        # Cascade: the model is silent during the tool call, so speak it, but
        # keep it out of the context for the same reason.
        if not splitting:
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
