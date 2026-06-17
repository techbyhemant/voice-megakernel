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


class TTSRequest(BaseModel):
    text: str
    speaker: str = "ryan"
    language: str = "English"
    chunk_size: int = 4  # ~333 ms/chunk; smaller = lower time-to-first-audio


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
        # Hold the GPU lock for the whole utterance so concurrent requests
        # don't corrupt the shared CUDA graphs.
        with _gpu_lock:
            t0 = time.perf_counter()
            gen_ttfc_ms = None
            n_samples = 0
            for audio, _sr, _timing in MODEL.generate_custom_voice_streaming(
                text=req.text,
                speaker=req.speaker,
                language=req.language,
                chunk_size=req.chunk_size,
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

    return StreamingResponse(generate(), media_type="application/octet-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup-speaker", default="ryan")
    ap.add_argument("--engine", default="megakernel", choices=["cudagraph", "megakernel"])
    args = ap.parse_args()

    print(f"Loading {args.model} (engine={args.engine}) ...", flush=True)
    MODEL = FasterQwen3TTS.from_pretrained(args.model)
    SR = getattr(MODEL, "sample_rate", 24000) or 24000

    if args.engine == "megakernel":
        import sys
        sys.path.insert(0, "/workspace")
        from megakernel_talker import MegakernelTalkerGraph
        base = MODEL.talker_graph.model
        MODEL.talker_graph = MegakernelTalkerGraph(base, base.config)
        print("Talker engine -> MEGAKERNEL", flush=True)

    # Capture CUDA graphs up front so the first real request is already fast.
    print("Warming up (capturing CUDA graphs)...", flush=True)
    for _ in MODEL.generate_custom_voice_streaming(
        text="Warm up.", speaker=args.warmup_speaker, language="English", chunk_size=4
    ):
        pass

    print(f"READY — streaming TTS on :{args.port} (sr={SR})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
