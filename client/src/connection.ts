import { PipecatClient, type TransportState } from "@pipecat-ai/client-js";
import { MoqTransport } from "@pipecat-ai/moq-transport";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

const RELAY_URL_COMES_FROM_START_RESPONSE = "";

export type TransportName = "moq" | "webrtc";

// ?transport=moq|webrtc wins; otherwise MoQ where WebTransport exists and
// WebRTC everywhere else (Safari/Firefox — and the Windows/WSL2 fallback,
// where QUIC/UDP can't cross the Windows↔WSL boundary).
export function pickTransport(): TransportName {
  const param = new URLSearchParams(window.location.search).get("transport");
  if (param === "moq" || param === "webrtc") return param;
  return typeof WebTransport === "undefined" ? "webrtc" : "moq";
}

export function createClient(
  transportName: TransportName,
  onState: (state: TransportState) => void,
): PipecatClient {
  const transport =
    transportName === "webrtc"
      ? new SmallWebRTCTransport()
      : new MoqTransport({ relayUrl: RELAY_URL_COMES_FROM_START_RESPONSE });
  return new PipecatClient({
    transport,
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
  return response.json();
}
