#!/usr/bin/env bash
#
# lofi.sh — the one lifecycle command for the fully-offline voice bot.
#
# Ownership rule: lofi only ever stops PIDs it recorded in $LOFI_STATE_DIR
# (default .lofi/). An Ollama you started yourself is never touched.
#
# Usage:
#   ./lofi.sh start [-t moq|headphones]   # bring up Ollama if needed, then the bot (default: moq)
#   ./lofi.sh stop                        # stop only what lofi started (recorded PIDs)
#   ./lofi.sh status [-v]                 # machine, ownership, bot, models (configured + downloaded), rag; -v adds full catalogs
#   ./lofi.sh index-rag                   # build/rebuild the document index
#   ./lofi.sh consolidate [-n]            # adopt models from default HF/Ollama dirs (symlinks)
#   ./lofi.sh configure [args]            # delegates to ./configure.sh
#   ./lofi.sh get-debug-log [n]           # bundle the latest (or nth-latest) session log for debugging
#   ./lofi.sh <command> help              # help for one command
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
# .env supplies DEFAULTS only — values already in the environment (e.g. set on
# the command line) win, matching config.py's load_dotenv behavior.
load_env_defaults() {
  local file="$1" line key value
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*(#|$) ]]; then continue; fi
    key="${line%%=*}"; value="${line#*=}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then continue; fi
    if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
  done < "$file"
}
ENV_FILE="${LOFI_ENV_FILE:-.env}"
load_env_defaults "$ENV_FILE"
LOFI_REPO_ROOT="$REPO"
# shellcheck source=scripts/model_dir.sh
. "$REPO/scripts/model_dir.sh"

STATE_DIR="${LOFI_STATE_DIR:-.lofi}"
case "$STATE_DIR" in /*) ;; *) STATE_DIR="$REPO/$STATE_DIR" ;; esac

OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE="http://${OLLAMA_HOST}"
# must match tunables.DEFAULT_LLM_MODEL (shell can't import Python)
LOFI_LLM_MODEL="${LOFI_LLM_MODEL:-qwen2.5:7b}"
LOFI_WEB_PORT="${LOFI_WEB_PORT:-7860}"
WEB_URL="http://localhost:${LOFI_WEB_PORT}"

usage() { grep '^#   ' "$0" | sed 's/^#   //'; }

ollama_up()   { curl -sf --max-time 2 "$OLLAMA_BASE/api/tags" -o /dev/null 2>/dev/null; }
tag_ready()   { curl -sf --max-time 3 "$OLLAMA_BASE/api/tags" 2>/dev/null | grep -q "\"$1"; }
model_ready() { tag_ready "$LOFI_LLM_MODEL"; }
missing_models() {
  local missing=""
  tag_ready "$LOFI_LLM_MODEL" || missing="$LOFI_LLM_MODEL"
  echo "$missing"
}

pid_alive()    { [[ -n "${1:-}" ]] && ps -p "$1" >/dev/null 2>&1; }
recorded_pid() { cat "$STATE_DIR/$1.pid" 2>/dev/null || true; }

with_descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do with_descendants "$child"; done
  echo "$1"
}

stop_tree() {
  local pids
  pids="$(with_descendants "$1")"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 20); do pid_alive "$1" || break; sleep 0.25; done
  # shellcheck disable=SC2086
  pid_alive "$1" && kill -KILL $pids 2>/dev/null || true
}

model_lines() {
  uv run python scripts/print_models.py "$@" 2>/dev/null \
    || echo "         (could not resolve models — run 'uv sync' and retry)"
}

rag_line() {
  uv run python ingest.py stats --bare 2>/dev/null \
    || echo "no index (run ./lofi.sh index-rag)"
}

cmd_start() {
  local transport="moq"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--transport) transport="${2:-}"; shift 2 || shift ;;
      -h|--help)      usage; exit 0 ;;
      *) echo "lofi: unknown start option '$1' (try ./lofi.sh -h)" >&2; exit 1 ;;
    esac
  done
  case "$transport" in
    moq|headphones) ;;
    *) echo "lofi: unknown transport '$transport' — valid: moq (default), headphones" >&2; exit 1 ;;
  esac

  mkdir -p "$STATE_DIR"

  local bot_pid
  bot_pid="$(recorded_pid bot)"
  if pid_alive "$bot_pid"; then
    echo "start: bot already running (pid $bot_pid) — run ./lofi.sh stop first." >&2
    exit 1
  fi

  if ! ollama_up; then
    echo "start: bringing up Ollama (store → ${OLLAMA_MODELS}; log → ollama.log)"
    nohup ./scripts/run_ollama.sh >"$REPO/ollama.log" 2>&1 &
    echo $! >"$STATE_DIR/ollama.pid"
  elif ! model_ready; then
    echo "start: Ollama already running — pulling $(missing_models) into it..."
    ./scripts/run_ollama.sh
  fi
  if ! model_ready; then
    echo "start: waiting for $(missing_models) (first pull can take a few minutes)..."
    for _ in $(seq 1 900); do model_ready && break; sleep 1; done
    model_ready || { echo "start: $(missing_models) not ready — see ollama.log" >&2; exit 1; }
  fi

  model_lines

  mkdir -p "$STATE_DIR/logs"
  if [[ -f "$STATE_DIR/bot.log" && ! -L "$STATE_DIR/bot.log" ]]; then
    mv "$STATE_DIR/bot.log" "$STATE_DIR/logs/bot-00000000-pre-upgrade.log"
  fi
  local bot_log="$STATE_DIR/logs/bot-$(date +%Y%m%d-%H%M%S).log"
  ln -sf "$bot_log" "$STATE_DIR/bot.log"
  session_logs | tail -n +11 | xargs rm -f
  case "$transport" in
    moq)
      # No -t pin: the runner then serves every installed browser transport
      # (MoQ always; SmallWebRTC with --extra webrtc) and the client picks per
      # session via /start — the WebRTC fallback path needs this.
      nohup uv run python bot_browser.py --host localhost --port "$LOFI_WEB_PORT" >"$bot_log" 2>&1 &
      ;;
    headphones)
      echo "start: launching the local-audio bot — use headphones 🎧"
      nohup uv run python bot.py >"$bot_log" 2>&1 &
      ;;
  esac
  echo $! >"$STATE_DIR/bot.pid"
  echo "$transport" >"$STATE_DIR/bot.transport"
  echo "start: bot starting (pid $(recorded_pid bot), log → ${bot_log#"$REPO"/})"
  echo "start: stop everything lofi started with ./lofi.sh stop"

  if [[ "$transport" == "moq" ]]; then
    for _ in $(seq 1 120); do
      curl -sf --max-time 2 "$WEB_URL/" -o /dev/null 2>/dev/null && break
      pid_alive "$(recorded_pid bot)" || { echo "start: bot exited early — see ${bot_log#"$REPO"/}" >&2; exit 1; }
      sleep 1
    done
    echo "start: ▶ $WEB_URL — allow the mic and Connect."
    # Auto-open the page. LOFI_OPEN_BROWSER=0 disables. Skipped on WSL: the
    # right browser lives on the Windows side, which this shell can't open —
    # xdg-open would launch a WSLg browser inside the VM instead.
    if [[ "${LOFI_OPEN_BROWSER:-1}" == "0" ]]; then
      :
    elif grep -qi microsoft /proc/version 2>/dev/null; then
      echo "start: (WSL detected — open the URL above in a WINDOWS browser; ?transport=webrtc for audio)"
    elif command -v open >/dev/null 2>&1; then
      open "$WEB_URL"
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$WEB_URL"
    fi
  fi
}

cmd_stop() {
  local stopped=0 pidfile name pid cmd
  for pidfile in "$STATE_DIR"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if ! pid_alive "$pid"; then
      echo "stop: $name pidfile is stale (pid ${pid:-?} not running) — removing it."
      rm -f "$pidfile"
    else
      cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
      if ! echo "$cmd" | grep -qE 'run_ollama|ollama|bot|python|uv|sleep'; then
        echo "stop: pid $pid in $name.pid now runs '$cmd' — not lofi's, removing the pidfile only."
        rm -f "$pidfile"
      else
        echo "stop: stopping $name (pid $pid)."
        stop_tree "$pid"
        rm -f "$pidfile"
        stopped=1
      fi
    fi
    [[ "$name" == "bot" ]] && rm -f "$STATE_DIR/bot.transport"
  done
  if ollama_up && [[ ! -f "$STATE_DIR/ollama.pid" ]]; then
    echo "ollama: running, NOT started by lofi — leaving it alone."
  fi
  [[ "$stopped" -eq 0 ]] && echo "stop: nothing lofi started is running."
  return 0
}

session_logs() {  # newest first; names are timestamped so lexical order is chronological
  { ls -1 "$STATE_DIR/logs"/bot-*.log 2>/dev/null || true; } | sort -r
}

cmd_get_debug_log() {
  local back="${1:-1}" session bundle
  if ! [[ "$back" =~ ^[1-9][0-9]*$ ]]; then
    echo "get-debug-log: expected how many sessions back (a number ≥1), got '$back'" >&2
    exit 1
  fi
  session="$(session_logs | sed -n "${back}p")"
  if [[ -z "$session" ]]; then
    echo "get-debug-log: no session logs in $STATE_DIR/logs — start the bot at least once (./lofi.sh start)" >&2
    exit 1
  fi
  bundle="/tmp/lofi-debug-$(date +%Y%m%d-%H%M%S).log"
  {
    echo "=== lofi debug bundle · $(date) ==="
    echo "=== session log: $session ==="
    echo
    echo "--- lofi status at collection time ---"
    cmd_status
    echo
    echo "--- session log ---"
    cat "$session"
    echo
    echo "--- ollama.log (last 200 lines) ---"
    tail -n 200 "$REPO/ollama.log" 2>/dev/null || echo "(no ollama.log)"
  } >"$bundle"
  echo "session: $session"
  echo "debug bundle: $bundle"
}

cmd_help() {
  case "$1" in
    start)
      echo "usage: ./lofi.sh start [-t moq|headphones]"
      echo "  -t, --transport   moq (browser bot, the default — serves MoQ, plus WebRTC"
      echo "                    when installed via 'uv sync --extra webrtc')"
      echo "                    headphones (local mic/speakers via PyAudio — wear headphones)"
      echo "  brings up Ollama first if nothing answers on \$OLLAMA_HOST (and records its pid)" ;;
    stop)
      echo "usage: ./lofi.sh stop"
      echo "  stops ONLY processes lofi started (pids recorded in $STATE_DIR)"
      echo "  an Ollama you started yourself is reported and left alone" ;;
    status)
      echo "usage: ./lofi.sh status [-v]"
      echo "  machine hardware, ollama (ownership + store), bot, configured models (full paths),"
      echo "  rag index, and an inventory of every model on disk with sizes"
      echo "  -v, --verbose   also print the full STT/LLM/TTS catalogs (every model, even ones"
      echo "                  too big for this machine)" ;;
    index-rag)
      echo "usage: ./lofi.sh index-rag"
      echo "  builds the document index: \$LOFI_RAG_DATA_DIR (default ./data) → \$LOFI_RAG_INDEX_DIR"
      echo "  rebuilds from scratch; run it again after adding/editing/removing documents" ;;
    get-debug-log)
      echo "usage: ./lofi.sh get-debug-log [n]"
      echo "  bundles status + the latest session log + ollama.log into one file under /tmp"
      echo "  n = how many sessions back (default 1 = the most recent); logs live in $STATE_DIR/logs" ;;
    configure)   exec ./configure.sh -h ;;
    consolidate) exec uv run python scripts/consolidate.py -h ;;
  esac
}

cmd_status() {
  local verbose=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -v|--verbose) verbose=1 ;;
      *) echo "lofi: unknown status option '$1' (try ./lofi.sh status help)" >&2; exit 1 ;;
    esac
    shift
  done

  echo "machine"
  ./configure.sh --hardware

  local pid transport
  local store_desc="${OLLAMA_MODELS}"
  if [[ -L "${LOFI_MODEL_DIR}/ollama" ]]; then
    store_desc="borrowed -> $(readlink "${LOFI_MODEL_DIR}/ollama")"
  fi
  pid="$(recorded_pid ollama)"
  if ollama_up; then
    if pid_alive "$pid"; then
      echo "ollama   running (pid $pid), started by lofi, store: ${store_desc}"
    else
      echo "ollama   running, NOT started by lofi (your own instance; ./lofi.sh stop leaves it alone)"
    fi
  else
    echo "ollama   not running (./lofi.sh start brings it up), store: ${store_desc}"
  fi

  pid="$(recorded_pid bot)"
  if pid_alive "$pid"; then
    transport="$(cat "$STATE_DIR/bot.transport" 2>/dev/null || echo "?")"
    if [[ "$transport" == "moq" ]]; then
      echo "bot      running (pid $pid), transport: moq, $WEB_URL"
    else
      echo "bot      running (pid $pid), transport: $transport"
    fi
  else
    echo "bot      not running (./lofi.sh start)"
  fi

  model_lines --bare
  echo "rag      $(rag_line)"

  echo
  echo "downloaded"
  uv run python scripts/print_models.py -d 2>/dev/null | sed 's/^/  /' \
    || echo "  (could not resolve models — run 'uv sync' and retry)"

  if (( verbose )); then
    echo
    ./configure.sh --catalogs
  fi
}

CMD="${1:-}"
[[ $# -gt 0 ]] && shift
case "${1:-}" in
  help|-h|--help)
    case "$CMD" in
      start|stop|status|index-rag|get-debug-log|configure|consolidate)
        cmd_help "$CMD"; exit 0 ;;
    esac ;;
esac
case "$CMD" in
  start)      cmd_start "$@" ;;
  stop)       cmd_stop ;;
  status)     cmd_status "$@" ;;
  index-rag)  exec uv run python ingest.py index ;;
  consolidate) exec uv run python scripts/consolidate.py "$@" ;;
  configure)  exec ./configure.sh "$@" ;;
  get-debug-log) cmd_get_debug_log "$@" ;;
  -h|--help|help|"") usage ;;
  *) echo "lofi: unknown command '$CMD' (try ./lofi.sh -h)" >&2; exit 1 ;;
esac
