import "./style.css";
import type { TransportState } from "@pipecat-ai/client-js";
import { createClient, pickTransport, startBot } from "./connection";
import {
  attachConfigPanel,
  attachHostedBanner,
  attachModeToggle,
  attachMoneyRain,
  attachRagPanel,
  attachStopwatch,
} from "./panels";
import { attachTranscript } from "./transcript";

const connectButton = document.querySelector<HTMLButtonElement>("#connect")!;
const muteButton = document.querySelector<HTMLButtonElement>("#mute")!;
const webtransportBanner = document.querySelector<HTMLDivElement>("#webtransport-banner")!;
const connectionState = document.querySelector<HTMLSpanElement>("#connection-state")!;
const transcriptPane = document.querySelector<HTMLElement>("#transcript")!;
const composer = document.querySelector<HTMLFormElement>("#composer")!;
const configLines = document.querySelector<HTMLPreElement>("#config-lines")!;
const ragTurns = document.querySelector<HTMLElement>("#rag-turns")!;
const textInput = document.querySelector<HTMLInputElement>("#text-input")!;
const sendButton = document.querySelector<HTMLButtonElement>("#send")!;

const IDLE_STATES: TransportState[] = [
  "disconnected",
  "initializing",
  "initialized",
  "error",
];

function render(state: TransportState): void {
  connectionState.textContent = state;
  const idle = IDLE_STATES.includes(state);
  connectButton.textContent = idle ? "Connect" : "Disconnect";
  muteButton.disabled = idle;
  textInput.disabled = idle;
  sendButton.disabled = idle;
}

const transportName = pickTransport();
document.querySelector<HTMLSpanElement>("#transport-name")!.textContent = transportName;

// MoQ connect failed (e.g. Windows/WSL2, where QUIC/UDP can't cross the
// boundary): reload once onto WebRTC and auto-connect. Storage guards the loop;
// no storage → no auto-retry.
function fallBackToWebRTC(): boolean {
  if (transportName !== "moq") return false;
  try {
    if (sessionStorage.getItem("lofi-moq-failed")) return false;
    sessionStorage.setItem("lofi-moq-failed", "1");
  } catch {
    return false;
  }
  connectionState.textContent = "MoQ failed — retrying with WebRTC…";
  const url = new URL(window.location.href);
  url.searchParams.set("transport", "webrtc");
  url.searchParams.set("autoconnect", "1");
  window.location.assign(url);
  return true;
}

const client = createClient(transportName, render);
const transcript = attachTranscript(client, transcriptPane);
attachConfigPanel(client, configLines);
attachRagPanel(client, ragTurns);
attachMoneyRain(client, document.querySelector<HTMLElement>("#rag-panel")!);
attachModeToggle(client, document.querySelector<HTMLButtonElement>("#mode-toggle")!);
attachHostedBanner(client, document.querySelector<HTMLElement>("#hosted-banner")!);
attachStopwatch(client, document.querySelector<HTMLElement>("#rag-stopwatch")!);

connectButton.addEventListener("click", async () => {
  connectButton.disabled = true;
  try {
    if (IDLE_STATES.includes(client.state)) {
      await client.connect(await startBot(transportName));
    } else {
      await client.disconnect();
    }
  } catch (error) {
    if (fallBackToWebRTC()) return;
    connectionState.textContent = `error: ${error instanceof Error ? error.message : error}`;
  } finally {
    connectButton.disabled = false;
  }
});

muteButton.addEventListener("click", () => {
  client.enableMic(!client.isMicEnabled);
  muteButton.textContent = client.isMicEnabled ? "Mute" : "Unmute";
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  transcript.addUserText(text);
  try {
    await client.sendText(text);
  } catch (error) {
    connectionState.textContent = `error: ${error instanceof Error ? error.message : error}`;
  }
});

// No WebTransport (Safari/Firefox) is no longer fatal — pickTransport already
// selected WebRTC; the banner just says so.
if (typeof WebTransport === "undefined") {
  webtransportBanner.textContent =
    "This browser has no WebTransport, so lofi is using the WebRTC transport instead.";
  webtransportBanner.hidden = false;
}

render(client.state);

// Arriving via the automatic MoQ→WebRTC fallback: connect without a click
// (mic permission was already granted to this origin on the failed attempt).
if (new URLSearchParams(window.location.search).get("autoconnect") === "1") {
  connectButton.click();
}
