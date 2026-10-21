import { PipecatClient, type TransportState } from "@pipecat-ai/client-js";
import { MoqTransport } from "@pipecat-ai/moq-transport";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

const RELAY_URL_COMES_FROM_START_RESPONSE = "";

export type TransportName = "moq" | "webrtc" | "websocket";
export type SessionMode = "offline" | "online";

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

// Websocket sessions carry mode as `?mode=` on the connect URL itself (the
// bot reads it off the live `/ws-client` query params — there's no /start
// round trip to hand it a body); WebRTC carries it as `requestData.mode` in
// the /api/offer POST (query params on /api/offer never reach the bot).
function withMode(url: string, mode: SessionMode): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}mode=${mode}`;
}

export async function startBot(transportName: TransportName, mode: SessionMode): Promise<unknown> {
  // WebRTC sessions start when the offer hits /api/offer — no /start round
  // trip; hand the transport its offer endpoint directly.
  if (transportName === "webrtc") {
    return {
      webrtcRequestParams: {
        endpoint: new URL("/api/offer", window.location.origin).toString(),
        requestData: { mode },
      },
    };
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
    const params: Record<string, string> = { wsUrl: withMode(data.wsUrl ?? "", mode) };
    if (data.token) params.token = data.token;
    return params;
  }
  return data;
}
