"""Shared pipeline construction — one brain, every transport.

`build_pipeline(transport)` assembles the full-duplex pipeline used by every bot:

    transport.input() -> VAD -> STT -> user-context -> LLM -> TTS -> transport.output() -> assistant-context

Retrieval is no longer a pipeline stage. When an index exists the conversation
model is offered one tool, `ask_financial_records` (tools.py), whose handler
runs the records agent outside the pipeline.

Bot files keep only what differs per transport: transport construction, preflight,
greeting, and runner glue.
"""

from dataclasses import dataclass

# config FIRST — sets HF_HOME / Kokoro cache paths before any pipecat/HF import (see config.py).
import config

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
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
# records/chitchat behavior unchanged). Keeps the transcript strategy's
# semantics: one call per reply, no arguments, new request → new call.
DOMAIN_TOOL_DESCRIPTION = (
    "Look up anything in the user's own financial records and saved documents: "
    "transactions, spending, income, accounts, merchants, amounts, dates, totals, "
    "balances, and anything they saved in documents or notes, including a "
    "passphrase or code they stored there. Call this as soon as any part of what "
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
    from pipecat.pipeline.llm_with_backend import BackendConnector
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

    return RecordsBackendConnector(
        frontend_pairing_instruction=COMPACT_PAIRING_INSTRUCTION
    )


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
        text = strip_think(output.text)
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
        await tts.queue_frame(TTSSpeakFrame(FILLER_PHRASE))

    processors = [
        transport.input(),
        build_vad_processor(),
        build_stt(),
        user_aggregator,
    ]
    pipeline = Pipeline(
        [
            *processors,
            llm,
            SilenceWatchdog(),
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
