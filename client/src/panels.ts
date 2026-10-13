import { RTVIEvent, type PipecatClient } from "@pipecat-ai/client-js";

export const CONFIG_MESSAGE_TYPE = "lofi-config";
export const RAG_MESSAGE_TYPE = "lofi-rag";
const BUSY_MESSAGE_TYPE = "lofi-busy";
const IDLE_MESSAGE_TYPE = "lofi-idle";

interface ModelEntry {
  role: string;
  model: string;
  path: string;
}

interface LofiConfig {
  type: string;
  models: ModelEntry[];
  ollama_host: string;
  ollama_started_by_lofi: boolean;
  rag: string;
  architecture?: string;
}

const OWNERSHIP = {
  lofi: "started by lofi (./lofi.sh stop stops it)",
  foreign: "NOT started by lofi (your own instance; ./lofi.sh stop leaves it alone)",
};

interface RagChunk {
  source: string;
  page: number | null;
  text: string;
}

interface RagRoute {
  tool: string;
  args: Record<string, unknown>;
  count: number;
  elapsed_ms: number;
}

interface LofiRag {
  type: string;
  tool: string;
  args: Record<string, unknown>;
  count: number;
  elapsed_ms: number;
  chunks: RagChunk[];
}

function chunkLocation(chunk: RagChunk): string {
  return chunk.page == null ? chunk.source : `${chunk.source} p.${chunk.page}`;
}

function routeLine(route: RagRoute): string {
  const args = JSON.stringify(route.args ?? {});
  return `${route.tool} ${args} \u00b7 ${route.count} rows \u00b7 ${Math.round(route.elapsed_ms)}ms`;
}

export function attachRagPanel(client: PipecatClient, panel: HTMLElement): void {
  client.on(RTVIEvent.ServerMessage, (data: unknown) => {
    const message = data as Partial<LofiRag> | null;
    if (message?.type !== RAG_MESSAGE_TYPE) return;
    const turn = document.createElement("div");
    turn.className = "rag-turn";
    const line = document.createElement("p");
    line.className = "rag-route";
    line.textContent = routeLine(message as unknown as RagRoute);
    turn.append(line);
    for (const chunk of message.chunks ?? []) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = chunkLocation(chunk);
      const text = document.createElement("p");
      text.className = "rag-text";
      text.textContent = chunk.text;
      details.append(summary, text);
      turn.append(details);
    }
    if (panel.querySelector(".rag-turn")) {
      panel.prepend(turn);
    } else {
      panel.replaceChildren(turn);
    }
  });
}

export function attachConfigPanel(client: PipecatClient, panel: HTMLElement): void {
  client.on(RTVIEvent.ServerMessage, (data: unknown) => {
    const message = data as Partial<LofiConfig> | null;
    if (message?.type !== CONFIG_MESSAGE_TYPE) return;
    const lines = (message.models ?? []).flatMap((entry) => [
      `${entry.role.padEnd(8)}${entry.model}`,
      `       ${entry.path}`,
    ]);
    lines.push(
      `OLLAMA ${message.ollama_host}`,
      `       ${message.ollama_started_by_lofi ? OWNERSHIP.lofi : OWNERSHIP.foreign}`,
      `ARCH   ${message.architecture ?? "cascade"}`,
      `RAG    ${message.rag}`,
    );
    panel.textContent = lines.join("\n");
  });
}

const MONEY_RAIN_MS = 30_000;
const MONEY_DROPS = 18;
const MONEY_EMOJI = ["\u{1F4B8}", "\u{1FA99}", "\u{1F4B5}", "\u{1F4B0}"];

export function attachMoneyRain(client: PipecatClient, host: HTMLElement): void {
  let rain: HTMLElement | null = null;
  let timeout: number | undefined;

  function stop(): void {
    window.clearTimeout(timeout);
    rain?.remove();
    rain = null;
  }

  function start(): void {
    stop();
    rain = document.createElement("div");
    rain.className = "money-rain";
    rain.setAttribute("aria-hidden", "true");
    for (let i = 0; i < MONEY_DROPS; i++) {
      const drop = document.createElement("span");
      drop.className = "money-drop";
      drop.textContent = MONEY_EMOJI[i % MONEY_EMOJI.length];
      drop.style.left = `${Math.random() * 92}%`;
      drop.style.animationDelay = `${Math.random() * 2.4}s`;
      drop.style.animationDuration = `${2 + Math.random() * 2}s`;
      drop.style.fontSize = `${0.9 + Math.random() * 0.9}rem`;
      rain.append(drop);
    }
    host.append(rain);
    timeout = window.setTimeout(stop, MONEY_RAIN_MS);
  }

  client.on(RTVIEvent.ServerMessage, (data: unknown) => {
    const type = (data as { type?: string } | null)?.type;
    if (type === BUSY_MESSAGE_TYPE) start();
    // Route lines arrive when the backend finishes; BotOutput can't be the stop
    // signal because the filler phrase itself is bot output.
    if (type === RAG_MESSAGE_TYPE || type === IDLE_MESSAGE_TYPE) stop();
  });
}
