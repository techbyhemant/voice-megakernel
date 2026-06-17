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
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from faster_qwen3_tts import FasterQwen3TTS

app = FastAPI(title="qwen3-tts streaming server")

MODEL = None
SR = 24000
_gpu_lock = threading.Lock()  # serialize GPU inference (one utterance at a time)
_last_metrics = {}            # compute-only metrics of the most recent utterance
_lock_held_since = None       # monotonic time the lock was acquired (None = free)
WEDGE_TIMEOUT = 20.0          # > any real reply; longer => assume parked/wedged


def _lock_watchdog():
    """A disconnected streaming response leaves its sync generator parked at
    `yield` STILL holding the GPU lock (Starlette doesn't close it), which would
    wedge every later request. If the lock is held longer than any real
    utterance, force-release it. threading.Lock isn't thread-owned, so another
    thread may release it."""
    global _lock_held_since
    while True:
        time.sleep(2.0)
        held = _lock_held_since
        if held is not None and (time.monotonic() - held) > WEDGE_TIMEOUT:
            _lock_held_since = None
            try:
                _gpu_lock.release()
                print("WATCHDOG: force-released a wedged GPU lock", flush=True)
            except RuntimeError:
                pass


class TTSRequest(BaseModel):
    text: str
    speaker: str = "uncle_fu"
    language: str = "English"
    chunk_size: int = 4   # ~333 ms/chunk; smaller = lower time-to-first-audio
    temperature: float = 0.4  # talker sampling: lower = more consistent prosody, less expressive (0.9 default)
    top_k: int = 50


def _to_pcm16(audio) -> bytes:
    """float32 [-1,1] -> 16-bit little-endian PCM bytes."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a * 32768.0, -32768, 32767).astype("<i2").tobytes()


@app.get("/health")
def health():
    return JSONResponse({"ok": MODEL is not None, "sample_rate": SR})


@app.get("/metrics")
def metrics():
    """Compute-only metrics (no network) of the most recent /tts request."""
    return JSONResponse(_last_metrics)


@app.post("/tts")
def tts(req: TTSRequest):
    def generate():
        global _last_metrics
        # Serialize GPU inference (one utterance at a time) so concurrent
        # requests don't corrupt shared state. CRITICAL: acquire with a timeout
        # and release in `finally`, so an interrupted turn (client disconnect ->
        # GeneratorExit) can't leave the lock held and wedge every later request.
        global _lock_held_since
        if not _gpu_lock.acquire(timeout=30):
            return  # a prior request is wedged; fail fast instead of hanging
        _lock_held_since = time.monotonic()
        try:
            t0 = time.perf_counter()
            gen_ttfc_ms = None
            n_samples = 0
            for audio, _sr, _timing in MODEL.generate_custom_voice_streaming(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                chunk_size=req.chunk_size,
                temperature=req.temperature,
                top_k=req.top_k,
            ):
                if audio is None or len(audio) == 0:
                    continue
                if gen_ttfc_ms is None:
                    gen_ttfc_ms = (time.perf_counter() - t0) * 1000.0
                n_samples += len(audio)
                yield _to_pcm16(audio)
            total_s = time.perf_counter() - t0
            audio_s = n_samples / SR if SR else 0.0
            _last_metrics = {
                "gen_ttfc_ms": round(gen_ttfc_ms, 1) if gen_ttfc_ms else None,
                "gen_rtf": round(total_s / audio_s, 3) if audio_s else None,
                "audio_s": round(audio_s, 2),
            }
        finally:
            # Watchdog may have already force-released it; guard the release.
            _lock_held_since = None
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

    threading.Thread(target=_lock_watchdog, daemon=True).start()
    print(f"READY — streaming TTS on :{args.port} (sr={SR})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
