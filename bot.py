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

# Brain runs on the 5090 (Ollama), reached over the SSH tunnel — keyless and on
# the reimbursed GPU. Tunnel maps local 11435 -> box 11434 (avoids clashing with
# any Ollama on the Mac). Claude is OPTIONAL (set ANTHROPIC_API_KEY; billed to you).
LLM_MODEL = "qwen2.5:7b-instruct"            # served by Ollama on the 5090
REMOTE_LLM_BASE = "http://localhost:11435/v1"
CLAUDE_MODEL = "claude-sonnet-4-6"           # only used if ANTHROPIC_API_KEY is set

# Barge-in control:
#   True  = mic stays live while the bot speaks, so you can interrupt it (real
#           phone-call feel). REQUIRES HEADPHONES — on speakers the bot hears its
#           own voice and interrupts itself. The server handles a mid-reply cancel
#           cleanly (lock release + watchdog), so this is now safe end-to-end.
#   False = mute the mic while the bot speaks (speaker-safe, strict turn-taking).
# Use headphones with True; switch to False if you must demo on speakers.
ALLOW_INTERRUPTIONS = True

# Carrier-negotiation persona (e3's domain). Warm + brief: on a real phone call
# a good broker is personable, and short replies = far less latency/gaps.
SYSTEM_PROMPT = (
    "You are Marcus, a sharp, friendly freight broker working the load board, live "
    "on the phone with a truck carrier (driver or dispatcher). Your job: book this "
    "load at a rate that protects your margin but gets the truck committed — close "
    "the deal.\n\n"
    "HOW YOU NEGOTIATE:\n"
    "- Anchor a little low but realistic for the lane, then concede in small steps "
    "toward a fair middle; don't cave all at once.\n"
    "- Justify with the lane, miles, equipment, or market ('short haul', 'backhaul's "
    "tight', 'fuel's up').\n"
    "- Know your ceiling and don't blow past it; if they're reasonable, lock it in "
    "fast and confirm the booking.\n"
    "- Build rapport: warm, confident, easy to deal with.\n\n"
    "REALISM: rates are dollars per mile (dry-van spot is ~$1.40–$2.10/mi depending "
    "on lane); reference pickup, destination, miles, and equipment naturally.\n\n"
    "OUTPUT RULES — this is a fast phone call, so:\n"
    "- Reply with EXACTLY ONE sentence, 12 words or fewer. Never two sentences.\n"
    "- Say ONE thing — an offer, a counter, or a short question — then STOP and let "
    "them talk. No monologues, no lists, no thinking out loud.\n"
    "GOOD EXAMPLES:\n"
    "  'I can do one forty-five a mile, that work for you?'\n"
    "  'Best I can stretch is one sixty, you in?'\n"
    "  'Where's it delivering out of Dallas?'\n"
    "  'Done — I'll send the rate con over now.'"
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
    return OLLamaLLMService(model=LLM_MODEL, base_url=REMOTE_LLM_BASE)


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
    pipeline = Pipeline([
        transport.input(),   # mic audio in
        stt,                 # audio -> text
        user_agg,            # add your words to the conversation
        llm,                 # conversation -> reply text
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
