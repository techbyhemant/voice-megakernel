"""
tts_server.py — streaming Qwen3-TTS inference server (runs on the RTX 5090).

This is the brief's "Step 2: inference server — prompt in -> stream out".
It loads faster-qwen3-tts (CUDA-graph optimized Qwen3-TTS), captures the CUDA
graphs once at startup, then exposes a single streaming endpoint:

    POST /tts   {"text": "...", "speaker": "ryan", "chunk_size": 4}
       -> streams raw 16-bit little-endian PCM @ 24 kHz mono, chunk-by-chunk,
          as each audio chunk is decoded (NOT buffered).

The Pipecat client on the Mac connects to this over an SSH tunnel.

Later, the megakernel swaps into faster-qwen3-tts's talker decode step
(talker_graph._decode_step) — this server doesn't change when that happens.

Run:  /venv/main/bin/python tts_server.py --port 8000
"""

import argparse
import asyncio
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from faster_qwen3_tts import FasterQwen3TTS

app = FastAPI(title="qwen3-tts streaming server")

MODEL = None
SR = 24000
_gpu_lock = threading.Lock()  # serialize GPU inference (one utterance at a time)
_last_metrics = {}            # compute-only metrics of the most recent utterance


class TTSRequest(BaseModel):
    text: str
    speaker: str = "uncle_fu"
    language: str = "English"
    chunk_size: int = 2   # ~333 ms/chunk; smaller = lower time-to-first-audio
    temperature: float = 0.4  # talker sampling: lower = more consistent prosody, less expressive (0.9 default)
    top_k: int = 50
    speed: float = 1.15   # pitch-preserving playback speed (1.0 = natural); model has no rate knob


def _to_pcm16(audio) -> bytes:
    """float32 [-1,1] -> 16-bit little-endian PCM bytes."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a * 32768.0, -32768, 32767).astype("<i2").tobytes()


def _stretch(audio, speed: float):
    """Pitch-preserving time-stretch (Qwen3-TTS has no rate knob). Applied
    per-chunk to keep streaming; speed>1 = faster. Phase-vocoder, so a chunk
    boundary can warble slightly — fine at modest speeds (~1.15)."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    # Skip tiny chunks (e.g. the partial final chunk): too short for the n_fft=2048
    # phase vocoder, and ~80 ms at natural speed is imperceptible.
    if abs(speed - 1.0) < 1e-3 or a.size < 2048:
        return a
    import librosa
    return librosa.effects.time_stretch(a, rate=speed)


@app.get("/health")
def health():
    return JSONResponse({"ok": MODEL is not None, "sample_rate": SR})


@app.get("/metrics")
def metrics():
    """Compute-only metrics (no network) of the most recent /tts request."""
    return JSONResponse(_last_metrics)


_SENTINEL = object()  # distinct from a yielded (None, ...) chunk


@app.post("/tts")
async def tts(req: TTSRequest, request: Request):
    """Stream TTS with INTERACTIVE barge-in support.

    Earlier this was a sync generator. On a client disconnect (barge-in),
    Starlette runs sync generators in a threadpool and does NOT reliably throw
    GeneratorExit into them — so it parked at `yield` holding the GPU lock, and
    the next turn blocked 30 s on acquire(). That made interruptions unusable.

    Now it's async: we drive the blocking model generator one chunk at a time in
    the threadpool and check `request.is_disconnected()` between chunks. On
    barge-in we stop within ~one chunk (~150 ms), close the generator, and
    release the lock IMMEDIATELY in `finally` — so the next turn never waits.
    """
    loop = asyncio.get_running_loop()

    async def generate():
        global _last_metrics
        # Acquire off the event loop so a busy GPU can't stall the whole server.
        # timeout=30 fast-fails (returns empty) instead of hanging forever if a
        # prior request is genuinely wedged on the GPU.
        got = await loop.run_in_executor(None, lambda: _gpu_lock.acquire(timeout=30))
        if not got:
            return
        gen = None
        try:
            t0 = time.perf_counter()
            gen_ttfc_ms = None
            nat_samples = 0      # model output, pre-stretch — the honest RTF basis
            stretch_s = 0.0      # post-processing time, excluded from the model RTF
            gen = MODEL.generate_custom_voice_streaming(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                chunk_size=req.chunk_size,
                temperature=req.temperature,
                top_k=req.top_k,
            )
            while True:
                # Barge-in check BEFORE spending a chunk of GPU time.
                if await request.is_disconnected():
                    break
                # One model chunk in the threadpool (CUDA releases the GIL).
                item = await loop.run_in_executor(None, lambda: next(gen, _SENTINEL))
                if item is _SENTINEL:
                    break
                audio, _sr, _timing = item
                if audio is None or len(audio) == 0:
                    continue
                nat_samples += len(audio)
                ts = time.perf_counter()
                audio = _stretch(audio, req.speed)   # playback-speed post-proc (not model compute)
                stretch_s += time.perf_counter() - ts
                if gen_ttfc_ms is None:
                    gen_ttfc_ms = (time.perf_counter() - t0) * 1000.0
                yield _to_pcm16(audio)
            total_s = time.perf_counter() - t0
            nat_audio_s = nat_samples / SR if SR else 0.0
            gen_s = total_s - stretch_s   # megakernel compute only; speed-stretch excluded
            # RTF reflects the MODEL: compute vs the audio it generated (pre-stretch),
            # so the playback-speed knob doesn't distort the megakernel metric.
            _last_metrics = {
                "gen_ttfc_ms": round(gen_ttfc_ms, 1) if gen_ttfc_ms else None,
                "gen_rtf": round(gen_s / nat_audio_s, 3) if nat_audio_s else None,
                "audio_s": round(nat_audio_s, 2),
            }
        finally:
            # Stop the underlying model generator (frees its frame loop) and
            # release the lock immediately — even on barge-in or client drop.
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            try:
                _gpu_lock.release()
            except RuntimeError:
                pass

    return StreamingResponse(generate(), media_type="application/octet-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup-speaker", default="ryan")
    ap.add_argument("--engine", default="megakernel", choices=["cudagraph", "megakernel"],
                    help="talker backbone engine")
    ap.add_argument("--predictor", default="megakernel", choices=["cudagraph", "megakernel"],
                    help="code-predictor engine (megakernel = 2.1x faster than the CUDA graph)")
    args = ap.parse_args()

    print(f"Loading {args.model} (engine={args.engine}) ...", flush=True)
    MODEL = FasterQwen3TTS.from_pretrained(args.model)
    SR = getattr(MODEL, "sample_rate", 24000) or 24000

    import sys
    sys.path.insert(0, "/workspace")
    if args.engine == "megakernel":
        from megakernel_talker import MegakernelTalkerGraph
        base = MODEL.talker_graph.model
        MODEL.talker_graph = MegakernelTalkerGraph(base, base.config)
        print("Talker engine -> MEGAKERNEL", flush=True)
    if args.predictor == "megakernel":
        from megakernel_predictor import MegakernelPredictorGraph
        MODEL.predictor_graph = MegakernelPredictorGraph(MODEL.predictor_graph)
        print("Predictor engine -> MEGAKERNEL (2.1x)", flush=True)

    # Capture CUDA graphs up front so the first real request is already fast.
    print("Warming up (capturing CUDA graphs)...", flush=True)
    for _ in MODEL.generate_custom_voice_streaming(
        text="Warm up.", speaker=args.warmup_speaker, language="English", chunk_size=4
    ):
        pass

    # Warm librosa/numba JIT now so the first time-stretch doesn't compile
    # (10-30s) mid-request and stall the first real reply.
    _stretch(np.zeros(24000, dtype=np.float32), 1.15)

    print(f"READY — streaming TTS on :{args.port} (sr={SR})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
