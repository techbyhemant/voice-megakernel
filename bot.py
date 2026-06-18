"""
bot.py — freight-carrier-negotiation voice agent:
    mic -> Whisper (STT) -> LLM brain -> Qwen3-TTS (5090) -> speaker.

The negotiation brain is a SWAPPABLE LLM. In production you'd use a frontier
model for negotiation quality; this defaults to Claude when ANTHROPIC_API_KEY is
set, and falls back to a local Ollama model (keyless, lower quality) otherwise.
The engineering contribution is the real-time TTS (megakernel talker on the 5090).

Run it with:
    ./run.sh        (or: ./.venv/bin/python bot.py)
"""

import asyncio
import os
import sys
import warnings

# --- Quiet terminal: keep ONLY our clean conversation view --------------------
# Pipecat logs everything at DEBUG via loguru (frame links, per-turn context
# dumps, mute toggles...) and the SDK emits DeprecationWarnings. We silence both
# so the terminal shows just the colorized transcript + live metrics. Configure
# this BEFORE importing pipecat so its startup banner is suppressed too. Real
# problems still surface: loguru WARNING+ stays on.
warnings.filterwarnings("ignore", category=DeprecationWarning)
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY (brain), etc.

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from remote_tts_service import RemoteQwenTTSService
from debug_taps import DebugTap

# Brain runs on the 5090 (Ollama), reached over the SSH tunnel — keyless and on
# the reimbursed GPU. Tunnel maps local 11435 -> box 11434 (avoids clashing with
# any Ollama on the Mac). Claude is OPTIONAL (set ANTHROPIC_API_KEY; billed to you).
LLM_MODEL = "qwen2.5:7b-instruct"            # served by Ollama on the 5090
REMOTE_LLM_BASE = "http://localhost:11435/v1"
CLAUDE_MODEL = "claude-sonnet-4-6"           # only used if ANTHROPIC_API_KEY is set

# Barge-in: True = mic stays live so you can interrupt the bot (needs headphones,
# else it hears its own voice); False = mute the mic while it speaks (speaker-safe,
# strict turn-taking). False is the demo default.
ALLOW_INTERRUPTIONS = False

# Carrier-negotiation persona (e3's domain). Warm + brief: on a real phone call
# a good broker is personable, and short replies = far less latency/gaps.
SYSTEM_PROMPT = (
    "You are Marcus, a freight broker on the phone with a truck driver or dispatcher. "
    "Your job: find out where they're headed, then negotiate a rate for a load on that lane.\n\n"
    "FIRST — you do NOT know their route yet. If they haven't told you a pickup and "
    "destination, ASK where they're headed. Never mention a specific load, shipment, "
    "or rate until they have named a lane. Do not invent a shipment.\n\n"
    "HOW THE RATE WORKS — get this right:\n"
    "- You PAY the carrier per mile. They push for a HIGHER rate; you want to keep it LOW.\n"
    "- Your FIRST offer is your anchor: low end, around a dollar fifty a mile.\n"
    "- Every later offer must move UP toward their number, in small five-to-ten-cent steps. "
    "NEVER offer a number lower than one you already said. NEVER jump straight to their ask.\n"
    "- Your CEILING is two dollars a mile — never exceed it.\n"
    "- If they ask for MORE than your last offer, counter with a number BETWEEN your last "
    "offer and their ask (you said a dollar sixty-five, they want a dollar eighty-five, so you "
    "say a dollar seventy-two), or accept if their ask is at or under your ceiling and you're close.\n"
    "- Justify briefly (lane, miles, market) and stay warm and confident.\n\n"
    "MARKET: dry-van spot rates run about a dollar forty to two dollars ten a mile depending on lane.\n\n"
    "SAYING THE RATE — you're on a phone and every word is read ALOUD by a voice:\n"
    "- Always say a rate as spoken dollars and cents: 'a dollar fifty-five a mile' or "
    "'a buck sixty a mile'. NEVER say a bare 'one fifty-five', and NEVER use digits or a "
    "dollar sign (1.55 or $1.55) — they get mispronounced out loud.\n\n"
    "OUTPUT — this is a live phone call, so be warm but TIGHT:\n"
    "- Reply with exactly ONE sentence, then STOP. This includes hellos, confirmations, and "
    "goodbyes — even closing the deal is ONE sentence. If a second sentence is forming, cut it. "
    "Never tack a follow-up question on the end.\n"
    "- Sound personable and human, not robotic or curt — one warm, natural sentence.\n\n"
    "EXAMPLES (ONE sentence each — note rates are spoken in dollars and only move UP):\n"
    "  Them: 'Hey, how's it going?'  You: 'Doing great, where are you headed today?'\n"
    "  Them: 'Dallas to Houston.'  You: 'Nice lane, I can start you at a dollar fifty a mile, that work?'\n"
    "  Them: 'I need one eighty.'  You: 'I hear you, I can come up to a dollar sixty-two on that run.'\n"
    "  Them: 'Closer to one eighty.'  You: 'Best I can do is a dollar seventy, you in?'\n"
    "  Them: 'Deal.'  You: 'Awesome, I'll send the rate confirmation right over.'"
)


def _build_brain():
    """LLM brain on the 5090 via Ollama (over the tunnel) by default — keyless,
    on the reimbursed GPU. Claude only if a real ANTHROPIC_API_KEY is set
    (API cost is on you, not reimbursed)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key.startswith("sk-ant-") and "REPLACE" not in key and "oat01" not in key:
        print(f"Brain: Claude ({CLAUDE_MODEL}) — billed to your account")
        return AnthropicLLMService(api_key=key, model=CLAUDE_MODEL)
    print(f"Brain: Ollama ({LLM_MODEL}) on the 5090 via {REMOTE_LLM_BASE}")
    # max_tokens is a RUNAWAY BACKSTOP, not the length target — one natural sentence
    # is ~15-30 tokens, so 64 never clips a real reply but kills a multi-sentence
    # monologue. Brevity comes from the prompt ("one sentence"); a modest
    # temperature keeps it on-rule while still sounding natural.
    return OLLamaLLMService(
        base_url=REMOTE_LLM_BASE,
        settings=OLLamaLLMService.Settings(
            model=LLM_MODEL,
            max_tokens=64,
            temperature=0.6,
        ),
    )


def _preload_brain():
    """Preload the LLM before the first turn: send the (long, detailed) system
    prompt once so Ollama loads the model AND caches its prefix KV. The system
    prompt is a CONSTANT prefix on every turn, so Ollama reuses that cached KV —
    the detailed prompt is prefilled once here, not re-processed each turn. Makes
    turn 1 warm instead of cold (no ~1s first-token penalty mid-demo)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key.startswith("sk-ant-") and "REPLACE" not in key and "oat01" not in key:
        return  # Claude path: provider handles caching; nothing to preload locally
    import json
    import urllib.request
    try:
        req = urllib.request.Request(
            REMOTE_LLM_BASE.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 1,
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30)
        print("Brain preloaded — system prompt cached on the GPU (turn 1 is warm)")
    except Exception as e:
        print(f"(brain preload skipped: {e})")


async def main():
    # --- The microphone + speaker (the "transport") -----------------------
    # audio_in  = your mic  (16 kHz is what Whisper wants)
    # audio_out = your speaker (24 kHz to match our TTS output)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        )
    )

    # --- The four stations ------------------------------------------------
    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO)     # ears (runs on M4 GPU via MLX)
    llm = _build_brain()                                         # brain (Claude or local)
    _preload_brain()                                             # warm + cache the system prompt on the GPU
    tts = RemoteQwenTTSService()                                 # mouth (streams from 5090 over SSH tunnel)

    # The LLM needs "memory" of the conversation. The context holds the running
    # message list; the aggregator pair feeds user turns in and records the
    # assistant's replies back out. The VAD (voice-activity detector) lives on
    # the user side — it decides when YOU have finished speaking.
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    # When interruptions are OFF, mute the mic while the bot speaks so it can't
    # hear its own voice through the speakers. When ON, leave the mic live so you
    # can barge in (use headphones to avoid the bot hearing itself).
    mute_strategies = [] if ALLOW_INTERRUPTIONS else [AlwaysUserMuteStrategy()]

    # Tuned to avoid FALSE triggers (ambient noise / mic hiss interrupting the
    # bot). Stricter than defaults: needs higher speech confidence, more volume,
    # and ~0.4s of sustained speech before it believes a turn started.
    # Snappier turn detection (headphones avoid the bot's voice triggering it).
    vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.7,
        min_volume=0.6,
        # With barge-in on, require a bit more sustained speech (0.35s) so a cough
        # or stray "uh" doesn't cancel the bot mid-reply; still feels responsive.
        start_secs=0.35 if ALLOW_INTERRUPTIONS else 0.2,
        stop_secs=0.2,     # matches Pipecat's benchmark default (silences the warning)
    ))
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_mute_strategies=mute_strategies,
        ),
    )

    # --- The conveyor belt: order matters, data flows top -> bottom -------
    # The two DebugTaps are pass-through probes that print the live transcript:
    # the STT tap shows what you said, the LLM tap shows Marcus's reply + timing.
    # They forward every frame unchanged (no behavior change).
    pipeline = Pipeline([
        transport.input(),   # mic audio in
        stt,                 # audio -> text
        DebugTap("STT"),     # 👂 print the user's transcription
        user_agg,            # add your words to the conversation
        llm,                 # conversation -> reply text
        DebugTap("LLM"),     # 🧠 print Marcus's reply + LLM timing
        tts,                 # reply text -> audio frames (streamed)
        transport.output(),  # audio -> speaker
        assistant_agg,       # remember the bot's reply
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    print("\n🎙️  Voice agent running. Speak, then pause. Ctrl+C to quit.\n")
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
