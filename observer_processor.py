"""The observer architecture: the records agent listens to every user turn
itself (LOFI_ARCHITECTURE=observer).

No routing decision sits in front of retrieval — the analyst (reasoning on)
sees each user turn, decides relevance on its own, and either replies with the
NOT_RECORDS sentinel (stays silent) or speaks a grounded answer directly and
writes it into the conversation context as an assistant turn. The pipeline
model keeps talking in parallel; it is prompted to defer records questions.

Sits between the user aggregator and the LLM: forwards every frame untouched,
treats LLMContextFrame as the new-turn trigger, and cancels the in-flight
lookup when the user speaks again.
"""

from __future__ import annotations

import asyncio
import datetime

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMContextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

import rag_agent
from prompts import records_analyst
from tools import BUSY_MESSAGE_TYPE, IDLE_MESSAGE_TYPE, RAG_MESSAGE_TYPE


class RecordsObserver(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        db_path,
        embedder,
        answer_fn=None,
        today: datetime.date | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._db_path = db_path
        self._embedder = embedder
        self._answer_fn = answer_fn or rag_agent.answer_question
        self._today = today
        self._task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            self._observe()
        elif isinstance(frame, (UserStartedSpeakingFrame, EndFrame)):
            await self._cancel()

        await self.push_frame(frame, direction)

    def _latest_user_text(self) -> str:
        for message in reversed(self._context.get_messages()):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                return ""
        return ""

    def _observe(self) -> None:
        question = self._latest_user_text()
        if not question:
            return
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = self.create_task(self._answer(question))

    async def _signal(self, message_type: str) -> None:
        await self.push_frame(RTVIServerMessageFrame(data={"type": message_type}))

    async def _cancel(self) -> None:
        if self._task is not None and not self._task.done():
            await self.cancel_task(self._task)
            await self._signal(IDLE_MESSAGE_TYPE)
        self._task = None

    async def _answer(self, question: str) -> None:
        await self._signal(BUSY_MESSAGE_TYPE)
        result = await self._answer_fn(
            question,
            db_path=self._db_path,
            embedder=self._embedder,
            today=self._today,
            system_prompt=records_analyst.OBSERVER_SYSTEM_PROMPT,
        )
        answer = result.answer.strip()
        if records_analyst.NOT_RECORDS in answer or result.error:
            if result.error:
                logger.warning(f"records observer: {result.error}")
            await self._signal(IDLE_MESSAGE_TYPE)
            return
        for trace in result.traces:
            await self.push_frame(
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
        await self.push_frame(TTSSpeakFrame(answer))
        self._context.add_message({"role": "assistant", "content": answer})
