# Online/Offline Session Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (This plan is also written to be converted into a ralph loop via create-ralph-loop.)

**Goal:** Every session is offline (today's local split stack, all data) or online (Deepgram + OpenAI + Cartesia cascade over the public index only), chosen on the page before Connect.

**Architecture:** Mode rides the connect request's query string into `build_pipeline(transport, mode)`. Privacy is file-level: two SQLite indexes, and online code never constructs the private db path. The legacy `LOFI_LLM_ENGINE=hosted` hybrid is deleted. Config splits into `.lofirc` (settings) and `.env` (three keys only).

**Tech Stack:** pipecat 1.13 (`DeepgramSTTService`, `OpenAILLMService`, `CartesiaTTSService`), OpenAI embeddings API, vite/TS client, bash launcher.

**Spec:** `docs/superpowers/specs/2026-10-21-online-offline-toggle-design.md`

## Global Constraints

- Workshop 2026-10-27: each task leaves the repo demoable; offline must never regress.
- DO NOT START while `ralph/` tunables loop is active; it owns `tunables.py`/prompts until its promise prints.
- Pipecat pinned `>=1.13,<1.14`. Look up every pipecat constructor via the pipecat-context-hub MCP before use (repo rule).
- Baseline `uv run pytest -q`: 264 passed. Stays green every task.
- Keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY` in `.env` only. Never log, never send to client, never in `.lofirc` or examples with values.
- Online code must never construct `config.rag_db_path()`; only `config.public_rag_db_path()`.
- Prompts are measured, not vibed: never touch `COMPACT_PAIRING_INSTRUCTION`; new online prompt text is additive and gets the chitchat/records/passphrase trio run once against the hosted model (Task 10's gate).
- No commits by interactive sessions without the user's ask; ralph iterations commit locally per its own convention.
- One README, top level only. Client changes require `cd client && npm run build`.
- **Naming duality: `offline`/`online`, never `hosted`.** "Hosted" implied the whole bot is hosted; it isn't. `HOSTED_INTRO` → `ONLINE_INTRO` (kept, re-worded, paired with `OFFLINE_INTRO`); no new `hosted*` identifiers or UI copy anywhere (banner says ONLINE).

## Review Focus

1. Legacy `.env` holding `LOFI_*` settings (every existing user): both files are dotenv loads so values still apply, and startup prints a one-line rename pointer. → test in Task 1.
2. Two of three keys present: online connect fail-fast lists ALL missing keys in one message, not the first. → test in Task 6.
3. `?mode=garbage` or absent: resolves to `offline`, never an exception. → test in Task 5.
4. Public index build failing mid-run (network/key revoked): `lofi.db` untouched, clear error, nonzero exit from `index-rag`, stale `lofi-public.db` left as-was. → test in Task 3.
5. Online session with Ollama down: fully functional — no code path touches `config.ollama_base_url()` in online mode. → test in Task 5.

---

### Task 1: Config file split — `.lofirc` + keys-only `.env`

**Files:**
- Modify: `config.py` (~line 61–77 load block; line 139–160 rename-warning machinery), `lofi.sh` (`load_env_defaults` callers, ~line 28–45), `configure.sh` (line 28 `ENV_FILE="${LOFI_ENV_FILE:-.env}"` and the `.env` write/backup block ~line 1023–1031 and help text), `.gitignore` (add `.lofirc`, `.lofirc.bak`)
- Create: `lofirc.example` (current `env.example` content, retitled; its sections mirror `tunables.py`'s groupings — models / behavior — and its header points at `tunables.py` as where defaults and prompts live), new 4-line `env.example` (three key names, empty values, one comment line)
- Test: `tests/test_config_files.py` (new), update `tests/test_lofi_sh.py`, `tests/test_configure_cascades.py`

**Interfaces:**
- Produces: `config.py` loads, in order: `.lofirc` (path overridable via `LOFI_ENV_FILE`), then `.env` (fixed name, repo root). Shell env wins over both (dotenv default no-override semantics, unchanged).
- Produces: `config.legacy_settings_in_env() -> list[str]` — `LOFI_*` names found in `./.env`; `lofi.sh start` and bot startup print one pointer line when non-empty.

- [ ] **Step 1: Write failing tests** — `test_lofirc_loaded_then_env(tmp_path, monkeypatch)` (a `LOFI_GREETING` in `.lofirc` applies; an `OPENAI_API_KEY` in `.env` applies; same var in shell env beats both), `test_legacy_settings_in_env_detected` (`.env` containing `LOFI_LLM_MODEL=x` → `["LOFI_LLM_MODEL"]`), `test_env_example_lists_only_keys` (parse `env.example`: exactly the three key names).
- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/test_config_files.py -q`
- [ ] **Step 3: Implement** config.py load-order change + `legacy_settings_in_env()`; move `env.example` → `lofirc.example` (`git mv`), write new `env.example`; update `lofi.sh` `load_env_defaults` to read `.lofirc` then `.env`; repoint `configure.sh` `ENV_FILE` default to `.lofirc`, backups to `.lofirc.bak`, and its help/prose strings.
- [ ] **Step 4: Run full suite green** — `uv run pytest -q` (fix `test_lofi_sh.py` / `test_configure_cascades.py` expectations as part of this step).
- [ ] **Step 5: Commit** — `config: split .lofirc (settings) from .env (hosted keys)`

### Task 2: Delete the `LOFI_LLM_ENGINE=hosted` hybrid

**Files:**
- Modify: `config.py` (delete `llm_engine()`, `hosted_base_url()`, `hosted_api_key()`, `hosted_key_in_dotenv_only()`, lines ~321–356), `services.py` (`build_llm` hosted branch), `tunables.py` (delete `hosted_variant()`; rename `HOSTED_INTRO` → `ONLINE_INTRO`, re-worded for session-mode framing — it becomes Task 5's online prompt intro, the duality partner of `OFFLINE_INTRO`), `pipeline.py` (drop `hosted_variant` import + call, drop `llm_engine`/`hosted_host` from use), `bot.py` (delete `_preflight_hosted_llm`, its call), `bot_browser.py` (`_lofi_config_message`: drop `llm_engine`/`hosted_host` fields), `client/src/panels.ts` (banner: temporary no-op pending Task 8; drop `llm_engine` branch in config lines), `lofirc.example`/README (remove `LOFI_HOSTED_*`, `LOFI_LLM_ENGINE`)
- Delete: `tests/test_services_llm_engine.py`
- Test: existing suite

**Interfaces:**
- Produces: `services.build_llm(temperature=None)` is Ollama-only. `lofi-config` no longer carries `llm_engine`/`hosted_host` (Task 7 adds `mode`).

- [ ] **Step 1: Delete code paths listed above**, following compile/test errors to every reference (`grep -rn "llm_engine\|hosted_" --include="*.py" --include="*.ts" --include="*.sh"` must come back empty outside git history and this plan).
- [ ] **Step 2: Full suite green** — `uv run pytest -q`; `cd client && npm run build` passes.
- [ ] **Step 3: Commit** — `refactor: delete hosted-hybrid engine; two stories only`

### Task 3: Data split + dual indexes

**Files:**
- Modify: `ingest.py` (`_cli_index` + `index` plumbing; new `OpenAIEmbedder`), `lofi.sh` (`index-rag` runs both builds; `status` shows both), `config.py` (add `public_rag_db_path()`, `online_embed_model()`), `store.py` only if `has_index` needs a param (it takes a path — verify, likely untouched)
- Move: `git mv data/Transactions_2026-09-08T18-09-28.csv data/passphrase-smoke.txt data/private/` and `git mv data/irs-pub-17.pdf data/irs-pub-334.pdf data/public/`
- Test: `tests/test_ingest.py` additions

**Interfaces:**
- Produces: `ingest.OpenAIEmbedder(model: str, api_key: str)` with `.embed(texts: list[str]) -> np.ndarray` (POST `https://api.openai.com/v1/embeddings`); CLI `uv run python ingest.py index [--data-dir D] [--db PATH] [--embedder ollama|openai]` (defaults preserve today's behavior).
- Produces: `config.public_rag_db_path() -> str` (`LOFI_PUBLIC_RAG_DB`, default `<rag_index_dir>/lofi-public.db`); `config.online_embed_model() -> str` (default `text-embedding-3-small`).

- [ ] **Step 1: Write failing tests** — `test_index_params_build_separate_dbs` (FakeEmbedder, two data dirs → two dbs, public db has only public chunks), `test_openai_embedder_payload` (mocked HTTP: model + input echoed, float32 array back), `test_public_build_failure_leaves_private_db` (embedder raises mid-build → `lofi.db` intact, nonzero CLI exit).
- [ ] **Step 2: Verify FAIL** — `uv run pytest tests/test_ingest.py -q`
- [ ] **Step 3: Implement** embedder + CLI params; `git mv` the data files; `lofi.sh index-rag` = offline build, then public build only when `OPENAI_API_KEY` present (else one-line skip notice, exit 0; failure during an attempted public build exits nonzero).
- [ ] **Step 4: Full suite green; rebuild local indexes** — `uv run pytest -q`; `./lofi.sh index-rag` (public build skips or runs depending on key).
- [ ] **Step 5: Commit** — `feat: data/private + data/public, dual RAG indexes`

### Task 4: Online service builders

**Files:**
- Modify: `services.py`, `pyproject.toml` (pipecat extras `deepgram`, `cartesia`; `openai` already present via hosted path — verify, add if not), `config.py` (`online_llm_model()`, `online_stt_model()`, `online_tts_voice()`, `missing_online_keys() -> list[str]`), `tunables.py` (DEFAULT_ONLINE_LLM_MODEL = "gpt-4o-mini", DEFAULT_ONLINE_STT_MODEL = "nova-3", DEFAULT_ONLINE_TTS_VOICE — builder picks a Spanish-capable Cartesia voice id)
- Test: `tests/test_services_online.py` (new)

**Interfaces:**
- Produces: `services.build_online_stt()`, `services.build_online_llm(temperature: float | None = None)`, `services.build_online_tts()` — constructor shapes verified via pipecat-context-hub before writing; keys read from `os.environ` inside the builders; each raises `SystemExit` with the key name when its key is missing (builders assume preflight normally catches this first).

- [ ] **Step 1: Context-hub lookups** — `search_api` for `DeepgramSTTService` Settings/LiveOptions, `CartesiaTTSService.Settings`, `OpenAILLMService.Settings` on 1.13; note multilingual config for Deepgram (`language` multi) in a comment.
- [ ] **Step 2: Write failing tests** — `test_online_builders_construct_without_network` (keys monkeypatched, services instantiate; no Ollama URL referenced), `test_online_builder_missing_key_exits` (clear message naming the key), `test_missing_online_keys_lists_all` (two absent → both names).
- [ ] **Step 3: Verify FAIL, implement, suite green** — `uv run pytest -q` (new extras: `uv sync` first).
- [ ] **Step 4: Commit** — `feat: online service builders (deepgram/openai/cartesia)`

### Task 5: Mode-aware pipeline

**Files:**
- Modify: `pipeline.py` (`build_pipeline(transport, mode="offline")`), `tools.py` (`build_public_tools(db_path, embedder) -> list[FunctionSchema]` — `semantic_search` only, same `lofi-rag` RTVI messages as `build_tools`), `tunables.py` (`ONLINE_SYSTEM_PROMPT` built on `ONLINE_INTRO` from Task 2 — the `OFFLINE_INTRO`/`ONLINE_INTRO` duality — with a `{language_line}` slot; `ONLINE_LANGUAGE_LINE` inviting two-way practice when `config.wait_lesson_language()` is set)
- Test: `tests/test_pipeline_modes.py` (new)

**Interfaces:**
- Consumes: Task 4 builders, Task 3 `public_rag_db_path()`.
- Produces: `build_pipeline(transport, mode: Literal["offline","online"] = "offline") -> BuiltBot`. Online: Silero VAD (unchanged) → Deepgram STT → cascade LLM with `build_public_tools` over the public db + `OpenAIEmbedder` → Cartesia TTS. Missing public db → no tools, panel notice string `"no public index — run ./lofi.sh index-rag with OPENAI_API_KEY set"` carried in `lofi-config`'s `rag` field. No watchdog lesson online. Offline: byte-for-byte today's behavior.

- [ ] **Step 1: Write failing tests** — `test_online_pipeline_uses_public_db_only` (spy on db paths opened; `rag_db_path` never called in online mode), `test_online_tools_semantic_only` (one FunctionSchema, name `semantic_search`), `test_mode_default_and_garbage_offline` (`build_pipeline(t)` and mode="garbage" → offline stack), `test_online_no_ollama_touch` (Ollama base URL helper not called online), `test_online_missing_public_db_degrades` (no tools, notice in rag string).
- [ ] **Step 2: Verify FAIL, implement, suite green.**
- [ ] **Step 3: Commit** — `feat: build_pipeline(mode) — online cascade over public index`

### Task 6: Mode into the server + `--online-only`

**Files:**
- Modify: `bot_browser.py` (`session_mode(runner_args) -> str` reading the connect request/websocket query params — probe the 1.13 runner-args objects for where the starlette request/websocket surfaces; MoQ always offline; call `build_pipeline(transport, mode)`; online fail-fast at connect when `config.missing_online_keys()` non-empty, error listing ALL missing keys reaches the client), `bot.py` preflights (mode-aware: skip `_preflight_llm/_stt/_rag` under online-only), `lofi.sh` (`start --online-only` → env `LOFI_ONLINE_ONLY=1`, skip Ollama wait/pull), `config.py` (`online_only() -> bool`)
- Test: `tests/test_session_mode.py` (new), `tests/test_llm_preflight.py` + `tests/test_stt_preflight.py` updates

**Interfaces:**
- Produces: `session_mode(runner_args)` → `"online"`/`"offline"` (default offline). Server under `--online-only` rejects offline connects with a clear message; checks keys at startup (warn-only for missing public db).

- [ ] **Step 1: Write failing tests** — `test_session_mode_from_query` (fake runner-args with `?mode=online` → online; absent/garbage → offline; MoQ args → offline), `test_online_connect_missing_keys_lists_all`, `test_online_only_skips_local_preflights` (monkeypatched preflights not called), `test_online_only_rejects_offline_connect`.
- [ ] **Step 2: Verify FAIL, implement (probe runner args first, adapt extraction), suite green.**
- [ ] **Step 3: Live smoke (no voice)** — `LOFI_ONLINE_ONLY=1 ./lofi.sh start` on a machine with keys: server starts with Ollama stopped; `curl localhost:7860` 200. Then stop.
- [ ] **Step 4: Commit** — `feat: session mode plumbing + ./lofi.sh start --online-only`

### Task 7: `lofi-config` + What's Running for online

**Files:**
- Modify: `bot_browser.py` (`_lofi_config_message(mode)`: add `"mode"`; online model entries — STT `nova-3 (Deepgram cloud)`, LLM `<model> (OpenAI cloud)`, TTS `<voice> (Cartesia cloud)`, EMBED `<model> (OpenAI cloud)`, paths `"—"`; `rag` shows public db stats/notice), `scripts/print_models.py` (`model_entries(mode="offline")`)
- Test: `tests/test_lofi_config.py` additions

**Interfaces:**
- Produces: `lofi-config` carries `mode: "online"|"offline"`; `model_entries("online")` returns the four hosted entries (role strings unchanged so panel colors apply).

- [ ] **Step 1: Failing tests** — `test_config_message_carries_mode`, `test_online_model_entries_roles` (roles STT/LLM/TTS/EMBED, no local paths, no key material in any field).
- [ ] **Step 2: Verify FAIL, implement, suite green.**
- [ ] **Step 3: Commit** — `feat: lofi-config mode + hosted What's Running entries`

### Task 8: Client — toggle, banner, mode in URL

**Files:**
- Modify: `client/index.html` (two-state control `#mode-select` — `🔌 offline` / `☁️ online` — next to Connect, default offline, disabled while connected), `client/src/main.ts` (init from `?mode=` URL param, write back on change, pass to connect), `client/src/connection.ts` (append `mode` to the connect endpoint query for websocket + webrtc), `client/src/panels.ts` (banner keyed on `config.mode === "online"`; headline `⚠️ ONLINE MODE — …`, one text: conversation AND retrieved public documents go to OpenAI/Deepgram/Cartesia; private records db not mounted; replace `HOSTED_DETAIL` machinery; no "hosted" wording anywhere in UI copy)
- Test: `cd client && npm run build` (type-level); behavior verified in Task 10 ears-on

**Interfaces:**
- Consumes: Task 7's `mode` field.

- [ ] **Step 1: Implement the four files**; keep `#mode-toggle` (⚡/🧠) untouched — separate feature.
- [ ] **Step 2: Build** — `cd client && npm run build` clean.
- [ ] **Step 3: Static check in browser** — serve, confirm toggle renders, URL param round-trips, banner hidden offline.
- [ ] **Step 4: Commit** — `feat: client online/offline session toggle + loud online banner`

### Task 9: SmallWebRTC default + websocket fallback

**Files:**
- Modify: `client/src/connection.ts` (`pickTransport()` default `webrtc`; on webrtc connect failure, automatic one-shot retry on `websocket`, updating `#transport-name`), `pyproject.toml` (move `pipecat-ai[webrtc]` into base deps IF `aiortc` wheels exist for mac-arm64/linux-x86_64/win-amd64 — verify on PyPI first; otherwise keep extra and add a `lofi.sh start` warning when absent), README (default-transport prose)
- Test: `tests/test_bot_transport_dispatch.py` still green; manual fallback check

**Interfaces:**
- Produces: default chain webrtc → websocket; `?transport=` still overrides (moq opt-in unchanged).

- [ ] **Step 1: Verify aiortc wheel coverage** (PyPI file list) — record the verdict in the commit message.
- [ ] **Step 2: Implement client fallback + dependency change; `uv sync`; `npm run build`.**
- [ ] **Step 3: Suite green; manual check** — with webrtc available, label shows `webrtc`; with aiortc uninstalled server-side, client lands on `websocket` without user action.
- [ ] **Step 4: Commit** — `feat: smallwebrtc default transport, websocket auto-fallback`

### Task 10: Docs + ears-on verification (human gate)

**Files:**
- Modify: `README.md` (config-files section + migration note, including the "where a setting comes from" ladder — shell env → `.lofirc` → `tunables.py` default — with `.env` as secrets-only and prompts living exclusively in `tunables.py`; `tunables.py` module docstring gains one line pointing at `.lofirc` as the no-code-edit override path; online mode + `--online-only`; transport story; model-size budget table: Q4 ≈ params/2 GB weights + 1–3 GB KV/overhead, measured rows 7B→~4.5 GB / 14B→~9 GB / 27B→17.7 GB, STT/TTS ~2 GB, leave a third of unified RAM free), `DEMO.md` (Beat 5 becomes the online flip: tax question ~2 s, passphrase "cannot know — and says so", flip back offline; Spanish two-way beat online; cache-bust/no-cache note already in place)
- Test: human

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Ears-on (HUMAN):** offline trio passes by voice (passphrase / records / chitchat); online session: tax question fast, passphrase refused with the boundary stated, short Spanish exchange; banner + panel loud and correct both modes.
- [ ] **Step 3: Prompt-trio check (HUMAN or scripted):** chitchat/records/passphrase against the hosted model once, temp as configured — no regressions attributable to `ONLINE_SYSTEM_PROMPT`.
- [ ] **Step 4: Commit** — `docs: online/offline demo beats, config migration, model-size budget`
