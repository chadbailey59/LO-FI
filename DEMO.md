# Workshop demo script

A beat-by-beat walkthrough for demoing lofi live. Every expected answer below
comes from `data/` as shipped; if you've swapped in your own transactions, the
amounts change but the beats don't.

## Pre-flight (before the audience arrives)

```bash
./lofi.sh status          # both model tags present, index built
uv run pytest -q          # green
```

- Index missing or stale? `./lofi.sh index-rag`.
- Close GPU-hungry apps — backend thinking time is GPU-bound and contention
  shows up as dead air.
- Warm Ollama: ask one throwaway question before going live so the first
  on-stage answer isn't paying model-load time.

## Start

```bash
./lofi.sh start
```

Open the printed URL. The browser connects over **WebSocket** by default —
the reliable path. (`?transport=webrtc` and `?transport=moq` exist as URL
params; MoQ garbles audio and is not demo-safe.)

**8 GB attendee machines:** split needs two resident models. The fallback is

```bash
LOFI_ARCHITECTURE=cascade ./lofi.sh start
```

— one model talks and retrieves. Lower answer accuracy, same tools, same
panels, stable.

## Beat 1 — passphrase smoke (is it really reading my files?)

Warm up with any throwaway exchange first ("hey, can you hear me?").
A cold-start *first* utterance asking for a "secret passphrase" trips the
front model's credential reflex and it refuses; with any prior turn in
context it delegates fine.

> **Say:** "What's the secret passphrase for the tax vault?"
>
> **Expect:** *BLUE MANATEE 436921* — straight out of
> `data/passphrase-smoke.txt`. The lofi-rag panel prints the backend's
> `semantic_search` call: tool, args, rows, timing.

Point at the panel: the model didn't know this; retrieval did.

## Beat 2 — Whole Foods chain (delegation + SQL arithmetic)

Three turns, one merchant. This is the core demo: the front model never
touches the data; the analyst backend answers everything through typed SQL
tools, including the arithmetic.

> **Say:** "How many times did I shop at Whole Foods in August 2026?"
>
> **Expect:** a count, answered via `aggregate` — watch the panel.

> **Say:** "List them."
>
> **Expect:** the rows via `filter_rows` (merchant + date-range filters in
> one call).

> **Say:** "What's the total?"
>
> **Expect:** **$301.81** — computed by SQL `aggregate`, not by the model
> adding in its head. Call this out: small local models reliably miscarry
> digits; lofi measures that and routes all arithmetic to the database.

## Beat 3 — tax documents (unstructured retrieval)

`data/` ships IRS Pub 17 and Pub 334 alongside the transactions.

> **Say:** "Roughly, what's the standard deduction for a single filer?"
>
> **Expect:** an answer grounded in Pub 17; the panel shows
> `semantic_search` hitting `irs-pub-17.pdf` chunks.

One index, typed rows and prose documents side by side.

## Beat 4 — language-lesson flourish (optional)

Restart with:

```bash
LOFI_WAIT_LESSON_LANGUAGE=Spanish ./lofi.sh start
```

Ask any records question that makes the backend think. At ~12 s of wait the
front fills the silence by teaching one Spanish phrase *tied to your
question* ("cuántas veces…" for a how-many question), then the real answer
arrives.

**Teach-only — do not demo two-way.** STT is English-configured; spoken
Spanish or German comes back as gibberish. Let the bot teach; don't answer
it in Spanish.

(German needs Piper TTS — `LOFI_TTS_ENGINE=piper`, a `de_DE` voice; Kokoro
has no German.)

## Beat 5 — fast ⚡ / accurate 🧠 toggle

**TODO — punch list item 2, not yet wired.** Once live: fast = backend
thinking off, accurate = thinking on (optionally 8b vs 27b). The
latency-vs-accuracy moment of the workshop.

## Recovery moves

- **A lookup is taking forever.** qwen3:8b thinks 29–70 s on identical
  requests — swings are normal. The watchdog speaks a reminder on a 60 s
  grid, so silence ≤ 1 min is expected behavior, not a hang. Narrate it:
  "the analyst is a reasoning model; it's thinking."
- **Do not re-ask during a lookup.** Re-asking mid-churn is the measured
  fabrication trigger — noise turns prime the front to invent results. Wait
  for the answer or barge in cleanly (next bullet) and move to a different
  beat.
- **Barge-in.** Speaking over the bot cuts TTS and it listens — safe on
  WebSocket (this race is why MoQ isn't the default). Use it to cut off a
  rambling answer, not to retry a pending lookup.
- **Latency insurance on a big machine (32 GB+):**
  `LOFI_RAG_LLM_MODEL=qwen3.8:27b ./lofi.sh start` — 25–38 s consistent,
  ~1.6× faster end-to-end than 8b's swings.
- **Wedged session.** Ctrl-C, `./lofi.sh start` again. Cold restart is
  seconds; a confused context is forever.
- **Logs.** `.lofi/logs/bot-*.log` holds every LLM context (grep
  "Generating LLM response"); `./lofi.sh get-debug-log` bundles the latest
  session for later.
