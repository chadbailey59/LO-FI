import { PipecatClient, type TransportState } from "@pipecat-ai/client-js";
import { MoqTransport } from "@pipecat-ai/moq-transport";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

const RELAY_URL_COMES_FROM_START_RESPONSE = "";

export type TransportName = "moq" | "webrtc" | "websocket";

const TRANSPORT_NAMES: TransportName[] = ["moq", "webrtc", "websocket"];

// ?transport=moq|webrtc|websocket wins; otherwise websocket — pure TCP, no
// FFI, works on every platform including Windows/WSL2 (where neither QUIC nor
// WebRTC's UDP media cross the boundary). MoQ and WebRTC stay reachable via
// the URL param; the auto-fallback chain still walks moq → webrtc → websocket
// when a param-selected transport fails.
export function pickTransport(): TransportName {
  const param = new URLSearchParams(window.location.search).get("transport");
  if (TRANSPORT_NAMES.includes(param as TransportName)) return param as TransportName;
  return "websocket";
}

function buildTransport(transportName: TransportName) {
  if (transportName === "webrtc") return new SmallWebRTCTransport();
  if (transportName === "websocket") return new WebSocketTransport();
  return new MoqTransport({ relayUrl: RELAY_URL_COMES_FROM_START_RESPONSE });
}

export function createClient(
  transportName: TransportName,
  onState: (state: TransportState) => void,
): PipecatClient {
  return new PipecatClient({
    transport: buildTransport(transportName),
    enableMic: true,
    callbacks: {
      onTransportStateChanged: onState,
    },
  });
}

export async function startBot(transportName: TransportName): Promise<unknown> {
  // WebRTC sessions start when the offer hits /api/offer — no /start round
  // trip; hand the transport its offer endpoint directly.
  if (transportName === "webrtc") {
    return { webrtcUrl: new URL("/api/offer", window.location.origin).toString() };
  }
  const response = await fetch("/start", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ transport: transportName }),
  });
  if (!response.ok) {
    throw new Error(`/start failed (${response.status}): ${await response.text()}`);
  }
  const data = (await response.json()) as { wsUrl?: string; token?: string | null };
  // WebSocketTransport rejects unknown keys (the /start response carries
  // sessionId and a null token) — hand it exactly what it validates.
  if (transportName === "websocket") {
    const params: Record<string, string> = { wsUrl: data.wsUrl ?? "" };
    if (data.token) params.token = data.token;
    return params;
  }
  return data;
}
