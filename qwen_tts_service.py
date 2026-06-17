"""
qwen_tts_service.py — Pipecat TTS service backed by the REAL Qwen3-TTS model.

This is the v1 (whole-utterance) version: it calls the model's high-level
`generate_custom_voice()`, gets the full waveform, then chunks it out as frames.
Same shape as say_tts.py, but with the real model instead of macOS `say`.

LIMITATIONS (intentional, fixed in v2):
  - NOT true streaming: it computes the whole clip first, THEN emits chunks.
    The brief wants chunk-as-decoded; that needs us to get inside the talker's
    decode loop (v2).
  - On Mac CPU this runs at RTF ~1.9, so a ~5s reply takes ~10s before audio
    starts. Fine for proving integration; the 5090 + megakernel fix the speed.

This file is also where the megakernel will eventually slot in (v2): the talker
decode becomes a swappable backend, while the codec stays in PyTorch.
"""

import asyncio
from typing import AsyncGenerator

import numpy as np
import torch

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from qwen_tts import Qwen3TTSModel

# Qwen3-TTS codec outputs 24 kHz mono. We match the rest of the pipeline to it.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2

CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * BYTES_PER_SAMPLE * NUM_CHANNELS

# Valid CustomVoice speakers (from the model's talker_config.spk_id):
#   serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan
DEFAULT_SPEAKER = "ryan"


class Qwen3TTSService(TTSService):
    """Real Qwen3-TTS as a Pipecat TTS service (whole-utterance v1)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = DEFAULT_SPEAKER,
        language: str = "English",
        device: str = "cpu",  # CPU beats MPS for this model on Apple Silicon
        **kwargs,
    ):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=TTSSettings(model=model_name, voice=speaker, language=None),
            **kwargs,
        )
        self._speaker = speaker
        self._language = language
        print(f"Loading Qwen3-TTS ({model_name}) on {device} — first load downloads weights...")
        self._model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        print("Qwen3-TTS loaded.")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return

        yield TTSStartedFrame()

        # Generation is blocking + CPU-heavy; run it off the event loop so the
        # rest of the pipeline (mic, etc.) isn't frozen while we synthesize.
        wavs, _sr = await asyncio.to_thread(
            self._model.generate_custom_voice,
            text=text,
            language=self._language,
            speaker=self._speaker,
        )

        pcm = self._to_pcm16(wavs[0])
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield TTSAudioRawFrame(
                audio=pcm[offset:offset + CHUNK_BYTES],
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        yield TTSStoppedFrame()

    @staticmethod
    def _to_pcm16(wav) -> bytes:
        """Convert a float waveform (numpy or torch, range ~[-1,1]) to 16-bit PCM bytes."""
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        a = np.asarray(wav, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        return (a * 32767.0).astype("<i2").tobytes()
