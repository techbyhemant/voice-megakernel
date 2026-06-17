"""
bench_codec.py — measure the codec (speech tokenizer V2 decoder) cost vs window size.

Question this answers: at small chunk_size the streaming path re-decodes a 25-frame
left context every chunk (chunked_decode / context_frames=25). Is that recompute the
real cost (→ a stateful streaming decoder that caches transformer-KV + conv state
would win big), or is the codec dominated by fixed per-call overhead (→ recompute is
cheap and the rewrite isn't worth it)?

We time speech_tokenizer.decode() for a range of frame counts and report ms and
ms-per-frame. If ms scales ~linearly with frames, the per-frame work dominates and
caching the context (decode only NEW frames) removes most of the small-chunk RTF.
If ms is ~flat, fixed overhead dominates and streaming buys little.

Run on the 5090 with an EXCLUSIVE GPU (stop the TTS server first):
    /venv/main/bin/python benchmarks/bench_codec.py
"""
import time

import torch

from faster_qwen3_tts import FasterQwen3TTS

NUM_CODEBOOKS = 16          # talker cb0 + predictor cb1..15
WINDOWS = [1, 2, 4, 8, 16, 25, 26, 33, 50, 100]  # frame counts (12 Hz: 1 frame ~83 ms audio)
RUNS = 20
WARMUP = 5


def _time(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0  # ms


def main():
    print("Loading Qwen3-TTS (codec only needed) ...", flush=True)
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    tok = model.speech_tokenizer
    sr = getattr(model, "sample_rate", 24000) or 24000
    print(f"codec class: {type(tok).__name__}  sample_rate={sr}", flush=True)

    # Small, safely-valid codebook indices (real values are 0..codebook_size-1; any
    # small non-negative int indexes valid embeddings). Shape: [B, frames, codebooks].
    def make(frames):
        return torch.randint(0, 64, (1, frames, NUM_CODEBOOKS), device="cuda", dtype=torch.long)

    print(f"\n{'frames':>7} {'audio_ms':>9} {'decode_ms':>10} {'ms/frame':>9}")
    base_per_frame = None
    for n in WINDOWS:
        codes = make(n)
        try:
            ms = _time(lambda: tok.decode({"audio_codes": codes}))
        except Exception as e:
            print(f"{n:>7}  decode failed: {e}")
            continue
        audio_ms = n / 12.0 * 1000.0
        per = ms / n
        if base_per_frame is None and n >= 25:
            base_per_frame = per
        print(f"{n:>7} {audio_ms:>9.0f} {ms:>10.3f} {per:>9.3f}")

    print(
        "\nReading the result:\n"
        "  • If ms grows ~linearly with frames (ms/frame roughly flat) => per-frame work\n"
        "    dominates; a stateful streaming decoder (cache KV+conv, decode only NEW\n"
        "    frames) removes the 25-frame recompute => big small-chunk RTF win.\n"
        "  • If ms is ~flat across frame counts (ms/frame falls sharply with n) => fixed\n"
        "    per-call overhead dominates; recompute is cheap and the rewrite isn't worth it.\n"
        "  • Streaming win factor at chunk_size=k ≈ (k + 25) / k  IF per-frame-dominated."
    )


if __name__ == "__main__":
    main()
