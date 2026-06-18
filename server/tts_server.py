"""
tts_server.py — streaming Qwen3-TTS inference server (runs on the RTX 5090).

This is the brief's "Step 2: inference server — prompt in -> stream out".
It loads faster-qwen3-tts (CUDA-graph optimized Qwen3-TTS), captures the CUDA
graphs once at startup, then exposes a single streaming endpoint:

    POST /tts   {"text": "...", "speaker": "uncle_fu", "chunk_size": 2}
       -> streams raw 16-bit little-endian PCM @ 24 kHz mono, chunk-by-chunk,
          as each audio chunk is decoded (NOT buffered).

The Pipecat client on the Mac connects to this over an SSH tunnel.

The talker and code-predictor decode run on the megakernel by default (select per
stage with --engine/--predictor cudagraph|megakernel); only the codec stays in
PyTorch (CUDA-graphed). See megakernel_talker.py / megakernel_predictor.py.

Run:  /venv/main/bin/python tts_server.py --port 8000
"""

import argparse
import asyncio
import threading
import time

import numpy as np
import torch
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
    temperature: float = 0.3  # talker sampling: lower = more consistent pace/energy, fewer whisper/slow takes (0.9 default)
    top_k: int = 20           # tighter sampling -> less prosody drift between utterances
    speed: float = 1.15   # pitch-preserving playback speed (1.0 = natural); model has no rate knob
    # Runaway backstop. The talker intermittently fails to emit EOS (Qwen3-TTS
    # issue #118 — happens on stock PyTorch too, ~0.5%; the megakernel amplifies
    # it) and runs to the KV ceiling (~2046 frames ≈ 162s) instead of stopping.
    # 0 = AUTO: size the cap to THIS reply's text (see _dynamic_cap) so a short
    # reply gets a tight cap (a runaway is cut in ~5-6s, not 24s) while long text
    # is never clipped. A positive value forces an explicit fixed cap instead.
    # The multiple-EOS streaming patch is the primary fix; this is the catch-all.
    max_new_tokens: int = 0


# Frame rate of the 12 Hz codec, measured: 2020 frames -> 161.76 s.
FRAMES_PER_SEC = 12.5
_MAX_FRAMES = 2046  # talker KV ceiling (MAX_SEQ_LEN - 2); the loop can't exceed it


def _dynamic_cap(text: str) -> int:
    """Size the talker frame budget to the reply text so a runaway is bounded to
    a few seconds beyond the longest plausible delivery of THIS text — without
    ever clipping a genuine reply.

    Expected delivery ≈ max(5 frames/word, 0.85 frames/char) (≈150 wpm / ~15
    chars-per-second at 12.5 fps). We allow 3× that plus a 50-frame margin, with a
    ~5 s floor. 3× ≈ a 50 wpm floor on speech rate — slower than any natural
    speech — so a real reply never hits it, but a 2-word runaway ("Safe trip!")
    is cut at ~6 s instead of 24 s.
    """
    t = (text or "").strip()
    n_words = len(t.split())
    n_chars = len(t)
    est_frames = max(n_words * 5, int(n_chars * 0.85))
    cap = est_frames * 3 + 50
    return max(60, min(_MAX_FRAMES, cap))


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


class _GraphedCodecDecoder:
    """Per-input-shape CUDA-graph cache around the codec decoder's forward.

    The codec is launch-overhead-bound (~16 ms eager / ~3 ms graphed, lossless) and
    runs once per streamed chunk, so it's the dominant RTF term at small chunk_size.
    Input shapes vary, so we cache one graph per frame-count (capture is near-instant).
    Drop-in for decoder.forward.
    """

    def __init__(self, orig_forward):
        self._orig = orig_forward      # ORIGINAL forward — capture/replay target (no recursion)
        self._cache = {}               # frames -> (static_in, static_out, graph)

    def __call__(self, codes):
        n = int(codes.shape[-1])
        entry = self._cache.get(n)
        if entry is None:
            static_in = codes.clone()
            # Warm on a side stream before capture (required by CUDA graphs).
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._orig(static_in)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = self._orig(static_in)
            entry = (static_in, static_out, g)
            self._cache[n] = entry
        static_in, static_out, g = entry
        static_in.copy_(codes)
        g.replay()
        # Clone: the caller slices/keeps this, and static_out is overwritten next replay.
        return static_out.clone()


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
    """Stream TTS with barge-in support: drive the blocking model generator one
    chunk at a time in the threadpool, check `request.is_disconnected()` between
    chunks, and release the GPU lock in `finally` so a barge-in frees it within
    ~one chunk (~150 ms) and the next turn never waits. (Async, not a sync
    generator: Starlette won't reliably throw GeneratorExit into sync ones.)
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
        cap = req.max_new_tokens if req.max_new_tokens and req.max_new_tokens > 0 \
            else _dynamic_cap(req.text)
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
                max_new_tokens=cap,   # dynamic runaway backstop sized to the text (see _dynamic_cap)
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
    # Codec decode is launch-overhead-bound (~16ms eager, ~3ms graphed — see
    # benchmarks/bench_codec_graph.py), the dominant RTF term at small chunk_size.
    # We wrap it in a per-shape CUDA-graph cache (_GraphedCodecDecoder): LOSSLESS and
    # near-instant to capture. With it, chunk_size=2 measures TTFC ~43ms AND RTF ~0.091
    # — both strict targets at one operating point. Default on; --compile-codec off
    # falls back to the eager codec for comparison.
    ap.add_argument("--compile-codec", default="on", choices=["on", "off"],
                    help="CUDA-graph the codec decoder (lossless; ~16ms->~3ms/call, lowers RTF)")
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
        # temperature=0.3 on the predictor too: keeps timbre/detail consistent across
        # utterances (matches the talker temperature), reducing voice drift.
        MODEL.predictor_graph = MegakernelPredictorGraph(MODEL.predictor_graph, temperature=0.3)
        print("Predictor engine -> MEGAKERNEL (2.1x, temp=0.3)", flush=True)

    if args.compile_codec == "on":
        # The codec (speech-tokenizer decoder) is launch-overhead-bound: ~16 ms/call
        # eager, ~3 ms graphed (measured, benchmarks/bench_codec_graph.py). It runs once
        # per streamed chunk, so that fixed cost is the dominant RTF term at small
        # chunk_size. We wrap its forward in a per-shape CUDA-graph cache: LOSSLESS (same
        # kernels/weights), and capture is near-instant (unlike torch.compile, which took
        # ~5 min). The warmup below captures one graph per chunk window shape.
        try:
            dec = MODEL.speech_tokenizer.model.decoder
            dec.forward = _GraphedCodecDecoder(dec.forward)
            print("Codec decoder -> CUDA-graph cache (lossless)", flush=True)
        except Exception as e:
            print(f"(codec graph cache skipped: {e})", flush=True)

    # Capture CUDA graphs (talker/predictor) + codec graphs up front. Warm across the
    # chunk sizes we serve (1, 2, 4) so every per-shape codec graph is captured before
    # real traffic — no first-request capture stall.
    print("Warming up (CUDA graphs)...", flush=True)
    for cs in (1, 2, 4):
        for _ in MODEL.generate_custom_voice_streaming(
            text="Warm up the decoder now please.",
            speaker=args.warmup_speaker, language="English", chunk_size=cs
        ):
            pass

    # Warm librosa/numba JIT now so the first time-stretch doesn't compile
    # (10-30s) mid-request and stall the first real reply.
    _stretch(np.zeros(24000, dtype=np.float32), 1.15)

    print(f"READY — streaming TTS on :{args.port} (sr={SR})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
