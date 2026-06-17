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

# Barge-in control: False mutes the mic while the bot speaks (speaker-safe, no
# interruptions). True allows barge-in but needs headphones / echo cancellation.
ALLOW_INTERRUPTIONS = False

# Carrier-negotiation persona (e3's domain: voice freight brokerage).
SYSTEM_PROMPT = (
    "You are a freight brokerage voice agent negotiating a load with a truck "
    "carrier over the phone. Be concise, natural, and professional — one or two "
    "sentences per turn, since you are speaking aloud. Acknowledge the carrier, "
    "make and counter offers, and work toward a fair agreed rate for the load."
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
    vad = SileroVADAnalyzer(params=VADParams(
        confidence=0.85,   # default 0.7 — require stronger "this is speech"
        min_volume=0.7,    # default 0.6 — require louder input
        start_secs=0.4,    # default 0.2 — ignore short blips
        stop_secs=0.6,     # default 0.2 — wait longer before ending your turn
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
