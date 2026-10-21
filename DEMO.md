# Workshop demo script

A beat-by-beat walkthrough for demoing lofi live. Every expected answer below
comes from `data/` as shipped (`data/private/` transactions and notes,
`data/public/` IRS publications); if you've swapped in your own transactions,
the amounts change but the beats don't.

## Pre-flight (before the audience arrives)

```bash
./lofi.sh status          # both model tags present, both indexes built
uv run pytest -q          # green
```

- Index missing or stale? `./lofi.sh index-rag` — it builds the offline db and,
  with `OPENAI_API_KEY` in `.env`, the public one Beat 5 needs.
- Doing Beat 5? Check all three keys are in `.env` before the room fills:

  ```bash
  cut -d= -f1 .env
  ```
- Close GPU-hungry apps — backend thinking time is GPU-bound and contention
  shows up as dead air.
- Warm Ollama: ask one throwaway question before going live so the first
  on-stage answer isn't paying model-load time.

## Start

```bash
./lofi.sh start
```

Open the printed URL. The browser connects over **SmallWebRTC** by default and
falls back to WebSocket automatically if that connect fails. (`?transport=`
pins a starting point; MoQ garbles audio and is not demo-safe.) **On WSL2,
pin WebSocket** — UDP never crosses the boundary, so the default costs you
15–30 s of failing first:

```bash
echo 'http://localhost:7860/?transport=websocket'
```

The session-mode control sits next to Connect: `🔌 offline` is the default and
Beats 1–4 all run there. Beat 5 flips it.

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
> `data/private/passphrase-smoke.txt`. The lofi-rag panel prints the backend's
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

`data/public/` ships IRS Pub 17 and Pub 334 alongside the transactions in
`data/private/`. Offline sees both; Beat 5's online session sees only the public two.

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
it in Spanish. Two-way is Beat 5's payoff — cloud STT code-switches where
the local one can't.

(German needs Piper TTS — `LOFI_TTS_ENGINE=piper`, a `de_DE` voice; Kokoro
has no German.)

## Beat 5 — the online flip (the privacy boundary, out loud)

The contrast beat. Same page, same bot, one control: flip `🔌 offline` to
`☁️ online` and reconnect. The `⚠️ ONLINE MODE` banner goes red, and the
What's Running panel repaints with Deepgram, OpenAI and Cartesia — read the
service names out to the room before you say a word to the bot.

> **Say:** "Roughly, what's the standard deduction for a single filer?"
>
> **Expect:** the same Pub 17 answer as Beat 3, in **about two seconds**.
> This is the honest reason online mode exists: `gpt-4o-mini` over OpenAI
> embeddings is an order of magnitude faster than a 7B on a laptop.

> **Say:** "What's the secret passphrase for the tax vault?"
>
> **Expect:** a refusal that *names the boundary* — this session is running
> online, the private records aren't available here, an offline session can
> answer it. **No passphrase.** This is the beat to linger on: the private db
> isn't filtered out by a prompt, it is never opened. Online code is wired to
> `config.public_rag_db_path()`, full stop.

Then flip back to `🔌 offline`, reconnect, and ask for the passphrase again —
*BLUE MANATEE 436921*, from the same machine, thirty seconds later. The data
didn't move; the session did.

### Beat 5b — two-way Spanish (optional, needs the restart from Beat 4)

With `LOFI_WAIT_LESSON_LANGUAGE=Spanish` set, an online session doesn't teach
during a wait — it just talks to you.

> **Say (in Spanish):** "¿Qué es la deducción estándar?"
>
> **Expect:** an answer **in Spanish**, in a Cartesia voice that pronounces
> it. Deepgram's `nova-3` runs with `language=multi`, so it code-switches
> mid-conversation; switch back to English and so does the bot.

Say the trade-off plainly: this is the one thing the local stack can't do, and
it costs you the privacy guarantee for the whole session.

**No keys in `.env`?** Online connect fails fast and names every missing key.
Skip to Recovery, or demo it on purpose — it's a good 10-second explanation of
where the keys live.

## Beat 6 — fast ⚡ / accurate 🧠 toggle

**TODO — punch list item 2, not yet wired.** The button is hidden unless
`architecture === "observer"`, and observer is retired. Once live: fast =
backend thinking off, accurate = thinking on (optionally 8b vs 27b). The
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
- **Online connect fails.** The error names the missing key. If it's the
  network and not the key, flip back to `🔌 offline` — every beat but 5 runs
  there. A laptop with no local models at all can still serve Beat 5:

  ```bash
  ./lofi.sh start --online-only
  ```
- **Wedged session.** Ctrl-C, `./lofi.sh start` again. Cold restart is
  seconds; a confused context is forever.
- **Logs.** `.lofi/logs/bot-*.log` holds every LLM context (grep
  "Generating LLM response"); `./lofi.sh get-debug-log` bundles the latest
  session for later.
