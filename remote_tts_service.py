"""
remote_tts_service.py — Pipecat TTS client for the remote 5090 streaming server.

This replaces the in-process Qwen3TTSService. Instead of running the model
locally, it POSTs text to the streaming TTS server on the 5090 (reached over an
SSH tunnel) and yields audio frames as the PCM streams back — true frame-by-frame
streaming into the Pipecat pipeline.

Tunnel (run in a separate terminal, keep open):
    ssh -i ~/.ssh/vast_ai -p <PORT> -L 8000:localhost:8000 root@ssh5.vast.ai

Then the server is reachable at http://localhost:8000 from this machine.
"""

import time
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

SAMPLE_RATE = 24000  # server streams 16-bit LE PCM @ 24 kHz mono
NUM_CHANNELS = 1


class RemoteQwenTTSService(TTSService):
    """Streams audio from the remote faster-qwen3-tts server on the 5090."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        speaker: str = "uncle_fu",
        language: str = "English",
        chunk_size: int = 2,  # 2 frames/chunk: ~64 ms TTFC, RTF 0.24 (4=84 ms; 1=55 ms)
        **kwargs,
    ):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=TTSSettings(model=None, voice=speaker, language=None),
            **kwargs,
        )
        self._url = base_url.rstrip("/") + "/tts"
        self._metrics_url = base_url.rstrip("/") + "/metrics"
        self._speaker = speaker
        self._language = language
        self._chunk_size = chunk_size

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return

        payload = {
            "text": text,
            "speaker": self._speaker,
            "language": self._language,
            "chunk_size": self._chunk_size,
        }

        yield TTSStartedFrame()
        # Stream the PCM response and emit a frame per network chunk as it arrives.
        # Observability: per-turn TTFC (time to first audio) + RTF (compute/audio).
        t0 = time.perf_counter()
        ttfc_ms = None
        n_bytes = 0
        timeout = aiohttp.ClientTimeout(total=120, sock_read=60)
        # Fresh session per turn: if a turn is interrupted, the abandoned
        # connection dies with it and can't poison the next request.
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._url, json=payload) as resp:
                resp.raise_for_status()
                async for data in resp.content.iter_chunked(4096):
                    if not data:
                        continue
                    # Guard against an odd byte (keep 16-bit sample alignment).
                    if len(data) % 2:
                        data = data[:-1]
                    if ttfc_ms is None:
                        ttfc_ms = (time.perf_counter() - t0) * 1000.0
                    n_bytes += len(data)
                    yield TTSAudioRawFrame(
                        audio=data,
                        sample_rate=SAMPLE_RATE,
                        num_channels=NUM_CHANNELS,
                    )
            total_s = time.perf_counter() - t0
            audio_s = n_bytes / (SAMPLE_RATE * 2)  # 16-bit mono
            rtf = (total_s / audio_s) if audio_s else float("nan")
            # Server-side COMPUTE metrics (no network) for the same utterance.
            gen = {}
            try:
                async with session.get(self._metrics_url) as r:
                    gen = await r.json()
            except Exception:
                pass
        print(
            f"📊 [TTS] compute(GPU): TTFC={gen.get('gen_ttfc_ms')}ms RTF={gen.get('gen_rtf')}  |  "
            f"end-to-end(+net): TTFC={ttfc_ms:.0f}ms RTF={rtf:.2f}  audio={audio_s:.2f}s  "
            f"«{text[:40]}»",
            flush=True,
        )
        yield TTSStoppedFrame()
