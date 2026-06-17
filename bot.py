"""
bot.py — local voice agent: mic -> Whisper -> local LLM -> Qwen3-TTS -> speaker.

Fully local, no API keys. The brain runs on Ollama; the voice is Qwen3-TTS.
The whole pipeline is self-contained (same shape we'll deploy on the 5090,
where the Qwen3-TTS talker decode gets swapped for the CUDA megakernel).

Run it with:
    ./run.sh        (or: ./.venv/bin/python bot.py)

Then just talk. Pause, and the bot replies out loud.
"""

import asyncio

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
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from remote_tts_service import RemoteQwenTTSService
# from qwen_tts_service import Qwen3TTSService  # local in-process baseline (slow, Mac CPU)

# The local chat brain. Any model you've pulled with `ollama pull <name>` works.
# qwen2.5vl:3b is already on this machine; llama3.2:3b is a leaner text-only swap.
LLM_MODEL = "qwen2.5vl:3b"

# Barge-in control:
#   False -> mic is muted while the bot speaks. No interruptions, but safe on
#            laptop SPEAKERS (bot can't hear itself). Good for quick testing.
#   True  -> you can interrupt the bot mid-sentence (real-time feel). REQUIRES
#            HEADPHONES (or echo cancellation), else the bot hears its own voice.
ALLOW_INTERRUPTIONS = False

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Keep replies short and conversational "
    "— one or two sentences — since they will be spoken aloud."
)


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
    llm = OLLamaLLMService(model=LLM_MODEL)                       # brain (local, no key)
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
