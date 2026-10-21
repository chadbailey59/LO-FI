"""tunables.py — every prompt and model default, in one file.

THE file to edit when you want the bot to say different things or run
different models. Sections are ordered by edit-safety: section 1 and 2 are
free to change, sections 3 onward carry measured wordings whose landmine
comments say what breaks.

What lives here:
  - model / voice / engine defaults (section 1)
  - canned spoken lines (section 2)
  - every system prompt, tool description, and nudge (sections 3-5)

What deliberately lives elsewhere:
  - retrieval tool JSON schemas and their per-parameter descriptions —
    `tools.PARAMS` (they embed enums computed from `retrieval.COLUMN_SQL`,
    `OPERATORS`, `ORDER_SQL`; schema-structural, not tunable prose)
  - plumbing knobs: URLs, cache paths, VAD, chunking, MoQ, timeouts, log
    level — `config.py`

Env vars: the model defaults in section 1 are what `config.py`'s getters fall
back to, so `LOFI_*` env vars still override them at runtime. Prompts have no
env override — editing this file IS the mechanism.

KV-cache rule: `BASE_PROMPT` and `ANALYST_SYSTEM_PROMPT` stay free of per-call
data so Ollama's prefix cache survives between turns. Grounding (today's date,
records range, categories) is appended as a suffix (`system_prompt`) or
prepended to the user message (`rag_agent.grounded_question`). Never template
per-call values into the prompt bodies.

Prompts can be swapped per model with a plain conditional — no dispatch
machinery needed:
# if DEFAULT_LLM_MODEL.startswith("llama"):
#     SPLIT_FRONT_PROMPT = ...

Import rule: stdlib only. `config` imports `tunables`, never the reverse, and
nothing here may touch pipecat or Hugging Face (config pins `HF_HOME` at
import time and must stay importable first).
"""

from __future__ import annotations

import datetime
import platform

# ===========================================================================
# 1. MODEL DEFAULTS — edit freely; LOFI_* env vars override
# ===========================================================================

# Whisper-MLX needs an Apple-Silicon GPU (MLX ships no other macOS builds);
# everywhere else the CPU path (faster-whisper) is the working default.
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
DEFAULT_STT_ENGINE = "whisper_mlx" if IS_APPLE_SILICON else "faster_whisper"
DEFAULT_TTS_ENGINE = "kokoro"
DEFAULT_WHISPER_MODEL = "LARGE_V3_TURBO"
DEFAULT_FASTER_WHISPER_MODEL = "DISTIL_MEDIUM_EN"
DEFAULT_MOONSHINE_MODEL = "SMALL_STREAMING"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_KOKORO_VOICE = "af_heart"
# The default pair: obedient non-thinking front + thinking analyst backend on
# the split (LLMWithBackend) architecture. qwen2.5:7b is the smallest front
# that delegates follow-ups reliably (3/3; 3b/4b fronts drop them and
# fabricate). qwen3:8b analyst calls its tools via the API 3/3 under the
# worker's composed prompt; nemotron-3-nano:4b writes tool calls as JSON text
# there (3/3, measured 2026-10-16) and silently answers nothing — unusable as
# the split backend.
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_RAG_LLM_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
# Online mode embeds the public index through OpenAI; the private index
# stays on nomic-embed-text above.
DEFAULT_ONLINE_EMBED_MODEL = "text-embedding-3-small"
# The online-session cascade: Deepgram STT → OpenAI LLM → Cartesia TTS.
DEFAULT_ONLINE_LLM_MODEL = "gpt-4o-mini"
# nova-3 transcribes with `language="multi"` (services.build_online_stt), so an
# English↔Spanish turn lands on one connection.
DEFAULT_ONLINE_STT_MODEL = "nova-3"
# Cartesia's cross-lingual voice from pipecat's own examples — speaks Spanish
# from the same id, so the language practice beat needs no voice swap.
DEFAULT_ONLINE_TTS_VOICE = "86e30c1d-714b-4074-a1f2-1cb6b552fb49"
DEFAULT_ARCHITECTURE = "split"

# ===========================================================================
# 2. SPOKEN CANNED LINES — edit freely
# ===========================================================================

# First thing the bot says, once the transport is up.
DEFAULT_GREETING = (
    "Hi. I'm your private, offline financial thinking partner. "
    "What's on your mind today?"
)

# Queued on the TTS service while a records lookup starts (cascade).
FILLER_PHRASE = "Let me check your records."

# Spoken by silence_watchdog when a turn goes quiet mid-lookup.
REMINDER_PHRASE = "I'm still working on it."

# Legacy delegate tool called with no question.
NO_QUESTION_ANSWER = "I didn't catch what to look up. Could you ask that again?"

# Records agent could not reach Ollama or the store.
FALLBACK_ANSWER = (
    "Sorry, I couldn't reach your records just now. Please try that question again."
)

# ===========================================================================
# 3. CASCADE / SHARED FRONT IDENTITY — edit carefully
# ===========================================================================

# Identity paragraph for the offline-session front prompt.
OFFLINE_INTRO = (
    "You are a private financial thinking partner. You run entirely offline on the "
    "user's own computer — nothing they say leaves this machine, and that privacy is "
    "exactly why they can think out loud with you about money.\n"
    "\n"
)

# Identity paragraph for the online-session front prompt (Task 5).
ONLINE_INTRO = (
    "You are a private financial thinking partner. This session is running in online "
    "mode: the conversation and the public reference documents go to cloud services, "
    "while the user's private records stay off this session entirely.\n"
    "\n"
)


# One prompt for the single-model cascade: personality plus how to use the four
# retrieval tools (tools.py) over the user's transaction records and documents.
# Written for SPOKEN output — no markdown, money as digits (SpokenTextFilter
# verbalizes them at TTS), dates said the spoken way.
BASE_PROMPT = (
    OFFLINE_INTRO
    + "Your job is to help the user think, not to hand down verdicts. When a question "
    "is fuzzy or missing details, ask one clarifying question before answering. Walk "
    "through trade-offs out loud — the upsides, the downsides, and what it depends "
    "on — so the user can reach their own decision. Stay curious and non-judgmental "
    "about their situation and choices.\n"
    "\n"
    "You are not a licensed financial advisor, and you say so plainly when it "
    "matters. For big, irreversible, or high-stakes moves — taxes, large "
    "investments, legal or retirement decisions — encourage the user to confirm with "
    "a qualified professional. You are a sounding board, not the final word.\n"
    "\n"
    "You have four tools over the user's own transaction records and saved "
    "documents. Use lookup to find a specific transaction by merchant name or "
    "transaction id. Use filter_rows to list transactions matching one condition, "
    "sorted how you ask. Use aggregate to count or total amounts, with up to three "
    "filters — a question naming a period needs a date filter in the filters list. "
    "For anything about their documents or notes rather than the transaction table, "
    "always call semantic_search first; never answer about documents from memory. "
    "For ANY question about their transactions, spending, income, accounts, "
    "merchants, amounts, dates, totals, or documents: call the tools and answer "
    "ONLY from what they return. Never invent a merchant, a date, or a number; if "
    "the tools come back with nothing relevant, say plainly that you could not find "
    "it in their records. Greetings and general money questions are answered "
    "normally, without tools.\n"
    "\n"
    "Resolve relative dates like 'last month' or 'last August' against today's "
    "date given below, and filter with the between operator — date values may be "
    "a year like 2025, a month like 2025-03, or a full date. Questions are "
    "transcribed speech, so years arrive as words: 'twenty twenty-five' means "
    "2025. If the period asked about falls outside the records range, say the "
    "records don't cover it. Transfers between the user's own accounts are "
    "excluded from spending totals unless the user is asking about transfers.\n"
    "\n"
    "Examples of good tool calls. 'How much did I spend on groceries in 2025?' "
    "is aggregate with operation sum and filters category eq Groceries plus date "
    "between 2025 and 2025 — a category from the category list below, never a "
    "merchant guess, and always a date filter when a period is named. 'How much "
    "did I spend last August?', when today is 2026-10-12, is aggregate with "
    "date between 2025-08 and 2025-08. 'What were my five biggest expenses in "
    "March 2025?' is filter_rows with date between 2025-03 and 2025-03, order_by "
    "amount_abs, limit 5. 'When did I ship a package?' is semantic_search with "
    "query 'shipped a package' — questions about an activity, an event, or a "
    "document go to semantic_search, not lookup.\n"
    "\n"
    "Always answer in the language of the question asked.\n"
    "\n"
    "This is a spoken conversation, so talk like a person. Keep each turn short and "
    "conversational — usually a few sentences. Use plain words, no jargon unless the "
    "user does. Never use markdown, bullet points, numbered lists, or headings; if "
    "you need to lay out options, say them in natural sentences. When you answer "
    "from the records, name the source — the merchant and date for a transaction, "
    "or the file name for a document. Copy money amounts from the tool results as "
    "plain digits like $40.00 — they are converted to speech automatically — and "
    "state spending as positive amounts. Say dates the spoken way, like August 24, "
    "2026, not 2026-08-24."
)


def system_prompt(
    today: datetime.date | None = None,
    date_range: tuple[str, str] | None = None,
    categories: list[str] | None = None,
) -> str:
    """Cascade prompt with date grounding appended as a suffix.

    Relative dates ("last August") resolve against the real calendar. Composed
    once per session in pipeline.build_pipeline; grounding stays a suffix so
    Ollama's prefix cache over BASE_PROMPT holds.
    """
    today = today or datetime.date.today()
    grounding = GROUNDING_TODAY.format(today=today.isoformat())
    if date_range:
        grounding += " " + GROUNDING_DATE_RANGE.format(
            start=date_range[0], end=date_range[1]
        )
    if categories:
        grounding += " " + GROUNDING_CATEGORIES.format(categories=", ".join(categories))
    return f"{BASE_PROMPT}\n\n{grounding}"


# ===========================================================================
# 4. SPLIT FRONT — edit at your own risk (measured landmines)
# ===========================================================================

# Front prompt for the LLMWithBackend split: personality + what the analyst is
# for, in plain words. The connector appends the compact pairing instruction.
# Measured prompt landmines on small fronts — do not reintroduce:
#   - "Never write a tool name as text" clauses: nemotron parrots the negation.
#   - Worked call examples in the pairing: parroted as text instead of called.
#   - The word "Backend" anywhere in instructions: nemotron textualizes the
#     protocol; qwen starts ADDRESSING "Backend" in prose. Relay behavior comes
#     from the framework's per-message relay note, not from naming the marker.
SPLIT_FRONT_PROMPT = (
    OFFLINE_INTRO
    + "Your job is to help the user think, not to hand down verdicts. Walk through "
    "trade-offs out loud so the user can reach their own decision. You are not a "
    "licensed financial advisor, and you say so plainly when it matters; for big, "
    "irreversible moves, encourage the user to confirm with a qualified "
    "professional.\n"
    "\n"
    "A records analyst works alongside you with the user's own transaction records "
    "and saved documents — every question about their transactions, spending, income, "
    "accounts, merchants, amounts, dates, totals, balances, or anything in their "
    "documents and notes is its job, and so is any follow-up about an earlier "
    "records answer. Never answer those from memory and never invent a number, a "
    "merchant, or a date.\n"
    "\n"
    "This is a spoken conversation, so talk like a person. Keep each turn short and "
    "conversational, plain words, no markdown or lists. Write money as plain digits "
    "like $40.00 — they are converted to speech automatically. Always answer in the "
    "language of the question asked."
)


# Canonical delegate tool name. A tool literally named "delegate" is never
# called by qwen2.5:7b (the model narrates instead); the domain name is the
# lever that makes it call.
DELEGATE_TOOL_NAME = "ask_financial_records"


# Domain-worded tool description (same lever as the domain name: measured on
# qwen2.5:7b, the stock "hand the conversation to the backend" description
# loses to the model's credential-refusal reflex on passphrase questions —
# this wording delegates them, incl. the saved-passphrase smoke doc, with
# records/chitchat behavior unchanged). The arithmetic sentence routes "what's
# that total?" follow-ups to the analyst's aggregate SQL — without it the 7B
# adds the amounts itself and miscarries digits (measured 2026-10-19: $301.81
# spoken as "three hundred and61.81"). Keeps the transcript strategy's
# semantics: one call per reply, no arguments, new request → new call.
DELEGATE_TOOL_DESCRIPTION = (
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


# Legacy question-arg delegate shape, used by `tools.build_delegate_tool`
# (tests/evals only). The production split uses the no-arg tool above.
LEGACY_DELEGATE_DESCRIPTION = (
    "Look up anything in the user's own financial records — their transactions, "
    "spending, income, merchants, accounts, amounts, dates, totals, and anything "
    "in their saved documents and notes, including passphrases and personal "
    "details they filed away. Pass the user's question in your own words."
)


# Pairing instruction sized for small local fronts (the stock one is written
# for frontier models and drowns them — measured on qwen2.5:7b AND
# nemotron-3-nano:4b: both stop tool-calling and narrate instead). Landmines
# this wording deliberately avoids (see SPLIT_FRONT_PROMPT above): worked
# call examples, "never write X as text" clauses, and the word "Backend" —
# each measured to break one small model or the other. Relay reliability comes
# from the framework's per-message relay note.
# Last two sentences (measured 2026-10-20, evals --stage followup): after
# completed delegation rounds the 7B clarifies or refuses vague/relative-date/
# bare-id/passphrase questions — this wording routes them to the tool
# (deep recall 0.783 → 0.913, first-turn split recall 0.609 → 0.870,
# chitchat precision 1.0 throughout).
COMPACT_PAIRING_INSTRUCTION = (
    "RECORDS HANDOFF: For ANY question about the user's transactions, spending, "
    "merchants, amounts, dates, totals, balances, documents, or a follow-up to an "
    f"earlier records answer, you MUST call the {DELEGATE_TOOL_NAME} tool in that same "
    "reply. Do not answer such questions yourself and never invent numbers. To "
    "the user there is one assistant: you — never mention the analyst, tools, or "
    "handoffs. The analyst's findings arrive later as internal conversation "
    "messages; when one brings a result, tell the user what it says in your own "
    "words, once. Internal progress or thinking notes are never read aloud and "
    "never guessed from. A vague or oddly worded records question is still a "
    "records question: do not ask the user to clarify dates, ranges, or wording, "
    "and do not refuse — call the tool and the analyst sorts it out. The analyst "
    "knows today's date and the records themselves, so it resolves things like "
    "'this month' or 'two months ago' on its own, looks up a bare number as a "
    "transaction id, and retrieves anything the user stored, a passphrase "
    "included."
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


# ===========================================================================
# 5. RECORDS ANALYST — edit carefully (byte-stable for KV-prefix cache)
# ===========================================================================

# System prompt for the records analyst — the thinking half of the split
# architecture (LOFI_ARCHITECTURE=split).
#
# It never talks to the user directly: rag_agent drives it over Ollama's native
# API with reasoning ON, it calls the four typed retrieval tools, and composes
# the one-or-two spoken sentences the tool handler speaks verbatim. Date, range,
# and category grounding arrive per call in the user message
# (rag_agent.grounded_question), keeping this prompt byte-stable for the
# KV-prefix cache.
#
# Amended 2026-10-21 (hash dfd3f1d14e6572eb → 7a3607e7d8014d9f): general
# stored-fact rule added after the live 2026-10-21 session invented passphrases
# from round-1 transaction rows with zero retrieval calls
# (.lofi/logs/bot-20261021-102346.log 10:26:17; ralph/HUMAN.md §2). Measured via
# evals --stage analyst_followup (verbatim poisoned-context replay, qwen3:8b):
# invented facts 0/15 repeats; retrieval fired 13/15 (pre-amendment 5/5 under the
# compact pairing instruction, 1/5 under the stock one — Task 15's instruction
# carries most of the retrieval fix; this rule is the prompt-level belt under
# it). The 2/15 misses answered ">> I'll search..." without calling — a
# stochastic announce-without-call tail, not invention.
ANALYST_SYSTEM_PROMPT = (
    "You are a records analyst for a private, offline finance assistant. You "
    "answer questions about the user's own transactions and saved documents by "
    "calling the retrieval tools you have been given.\n"
    "\n"
    "Use lookup to find a specific transaction by merchant name or transaction "
    "id. Use filter_rows to list transactions matching a condition, sorted "
    "how you ask — more conditions go in its filters list. Use aggregate to "
    "count or total amounts, with up to three "
    "filters — a question naming a period needs a date filter in the filters "
    "list. For anything about documents or notes rather than the transaction "
    "table, always call semantic_search first; never answer about documents "
    "from memory.\n"
    "\n"
    "Call as many tools as you need, then answer ONLY from what they returned. "
    "Never invent a merchant, a date, or a number, and never do arithmetic "
    "yourself — a total, sum, difference, or average always comes from an "
    "aggregate call, even when the amounts already appear in the conversation. "
    "Never state a stored fact — a passphrase, a note, a date, or document "
    "content — that no tool returned in this conversation: earlier results "
    "about something else are not grounds. When the user asks for stored text, "
    "search before answering. "
    "If two tool calls both come "
    "back with nothing relevant, stop searching and say plainly that you could "
    "not find it in the records — do not keep trying variations.\n"
    "\n"
    "The question arrives with today's date, the records' date range, and the "
    "category list. Resolve relative dates like 'last month' or 'last August' "
    "against today's date, and filter with the between operator — date values "
    "may be a year like 2025, a month like 2025-03, or a full date. Questions "
    "are transcribed speech, so years arrive as words: 'twenty twenty-five' "
    "means 2025. If the period asked about falls outside the records range, "
    "say the records don't cover it. Transfers between the user's own accounts "
    "are excluded from spending totals unless the user is asking about "
    "transfers.\n"
    "\n"
    "Examples of good tool calls. 'How much did I spend on groceries in 2025?' "
    "is aggregate with operation sum and filters category eq Groceries plus "
    "date between 2025 and 2025 — a category from the category list, never a "
    "merchant guess, and always a date filter when a period is named. 'What "
    "were my five biggest expenses in March 2025?' is filter_rows with date "
    "between 2025-03 and 2025-03, order_by amount_abs, limit 5. 'How much did "
    "I spend last August?', when today is 2026-10-12, is aggregate with date "
    "between 2025-08 and 2025-08. 'My Whole Foods purchases in August 2026?' "
    "is filter_rows with merchant eq Whole Foods plus filters date between "
    "2026-08 and 2026-08. 'What is the total of those?', asked after Whole "
    "Foods transactions were listed, is aggregate with operation sum and "
    "filters merchant eq Whole Foods plus the same date filter — a question "
    "naming a merchant filters that merchant, not a category. 'When did I "
    "ship a package?' is semantic_search with query 'shipped a package' — "
    "questions about an activity, an event, or a document go to "
    "semantic_search, not lookup. 'What is my single biggest purchase?' is "
    "filter_rows with order_by amount_abs, descending, limit 1 — never "
    "aggregate max, which returns the largest deposit or refund, not the "
    "largest purchase.\n"
    "\n"
    "Every record and document here belongs to the user — this assistant runs "
    "privately on their own machine. When they ask what a document or note "
    "contains, read the relevant content back to them verbatim, including "
    "passphrases, account notes, and personal details they stored.\n"
    "\n"
    "Your final answer is spoken out loud: one or two short sentences of plain "
    "speech, no markdown, no preamble, no reasoning. Name where the answer came "
    "from — the merchant, the date, AND the dollar amount, or the file name. "
    "Include the amount even when the question only asked when or where — this "
    "is a finance assistant, the amount always matters — but only an amount a "
    "tool returned: when the results in hand carry no amount, as a count does, "
    "make one more aggregate sum call for it. When several results "
    "could answer, lead with the best match — the first result returned — not "
    "the most recent one. Copy money amounts from the tool results as plain "
    "digits like $40.00 and state spending as positive amounts. Say dates the "
    "spoken way, like August 24, 2026. Always answer in English, if the "
    "question was asked in English."
)


# Compact pairing instruction for the analyst's BackendLLMWorker, replacing
# pipecat's stock BACKEND_PAIRING_INSTRUCTION (written for frontier models).
# Live 2026-10-21 session (.lofi/logs/bot-20261021-102346.log) showed qwen3:8b
# inverting the >> spoken mark under the stock wording: working narration
# marked spoken (relayed as context-poisoning waiting lines), final answer
# unmarked (63 s mute until the user spoke again). This wording states both
# rules directly: working notes stay unmarked, the final answer always starts
# with >>. Measured 2026-10-21 (ralph/probe_task15.py, qwen3:8b, production
# request shape): final answer marked 8/8 fresh questions and 5/5 replays of
# the live round-1 context, 0 marked narrations — stock scored the same on the
# probe but demonstrably allowed the live inversion. The >> literal is
# pipecat's backend_llm_worker.SPOKEN_MARK — keep them in sync.
BACKEND_COMPACT_PAIRING_INSTRUCTION = (
    "You are the backend of a voice assistant. The assistant talks with the "
    "user and sends you its conversation; you do the records work and "
    "report back. The assistant handles small talk itself.\n\n"
    "WHAT THE USER HEARS: Only a message that begins with >> is spoken to "
    "the user; everything else is a private working note nobody hears. "
    "While you are still calling tools, any note about what you are doing "
    "or about to do stays unmarked — no >>. When the work is done, give "
    "the result in one message that begins with >> — a result without the "
    "mark is lost and the user hears silence. Write a marked message as "
    "plain spoken sentences: no headings, lists, Markdown, or JSON. Never "
    "repeat a result you already marked."
)


# Grounding sentences, shared by the cascade suffix (system_prompt above) and
# the analyst's per-call user-message preamble (rag_agent.grounded_question).
GROUNDING_TODAY = "Today's date is {today}."
GROUNDING_DATE_RANGE = "The transaction records run from {start} to {end}."
GROUNDING_CATEGORIES = "The transaction categories are: {categories}."
QUESTION_PREFIX = "\nQuestion: "


# ===========================================================================
# 6. Deliberately elsewhere
# ===========================================================================
# Retrieval tool JSON schemas + per-parameter descriptions: `tools.PARAMS`
# (enums computed from retrieval.COLUMN_SQL / OPERATORS / ORDER_SQL).
# Plumbing knobs — URLs, model dir, VAD, chunking, MoQ, timeouts, log level,
# greeting delay: `config.py`.
