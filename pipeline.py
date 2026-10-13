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
from observer_processor import RecordsObserver
from prompts.financial_advisor import FRONT_PROMPT, OBSERVER_FRONT_PROMPT, system_prompt
from services import build_llm, build_stt, build_tts
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
    db_path = config.rag_db_path()
    if config.architecture() == "observer" or not _index_ready(db_path):
        return None
    embedder = ingest.OllamaEmbedder(config.ollama_api_url(), config.embed_model())
    if config.architecture() == "split":
        return tools.build_delegate_tool(db_path, embedder)
    return tools.build_tools(db_path, embedder)


def build_pipeline(transport: BaseTransport) -> BuiltBot:
    records_tools = build_records_tools()
    db_path = config.rag_db_path()
    observing = config.architecture() == "observer" and _index_ready(db_path)
    if observing:
        prompt = OBSERVER_FRONT_PROMPT
    elif records_tools and config.architecture() == "split":
        # Grounding lives in the records agent's own prompt; the front only delegates.
        prompt = FRONT_PROMPT
    else:
        prompt = system_prompt(
            date_range=store.date_range(db_path) if records_tools else None,
            categories=store.categories(db_path) if records_tools else None,
        )
    context = (
        LLMContext(messages=[{"role": "system", "content": prompt}], tools=records_tools)
        if records_tools
        else LLMContext(messages=[{"role": "system", "content": prompt}])
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    llm = build_llm()
    tts = build_tts()

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await service.push_frame(RTVIServerMessageFrame(data={"type": BUSY_MESSAGE_TYPE}))
        await tts.queue_frame(TTSSpeakFrame(FILLER_PHRASE))

    processors = [
        transport.input(),
        build_vad_processor(),
        build_stt(),
        user_aggregator,
    ]
    if observing:
        processors.append(
            RecordsObserver(
                context,
                db_path,
                ingest.OllamaEmbedder(config.ollama_api_url(), config.embed_model()),
            )
        )
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
