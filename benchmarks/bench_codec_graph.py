"""
bench_codec_graph.py — is the codec's fixed ~16ms/call launch overhead or real compute?

Codec decode is ~16ms regardless of window size (fixed per-CALL cost dominates).
If that 16ms is PyTorch eager launch overhead (many small unfused
kernels), a CUDA graph / torch.compile replay should slash it — which would lower RTF
at every chunk_size and potentially hit both strict targets. If it's real GPU compute,
the graph won't help and we're capped.

We time the decoder forward three ways at a fixed window: eager, CUDA-graph replay,
torch.compile. Run with an EXCLUSIVE GPU (stop the TTS server first):
    /venv/main/bin/python benchmarks/bench_codec_graph.py
"""
import time

import torch

from faster_qwen3_tts import FasterQwen3TTS

WINDOW = 27           # representative cs2 window (25 ctx + 2 new)
NUM_CODEBOOKS = 16
RUNS = 30
WARMUP = 8


def _time(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def main():
    print("Loading Qwen3-TTS ...", flush=True)
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    tok = model.speech_tokenizer
    # tok is the Qwen3TTSTokenizer inference wrapper; the decoder is nested at
    # tok.model.decoder (Qwen3TTSTokenizerV2Decoder, whose chunked_decode calls self()).
    decoder = None
    for path in ("model.decoder", "decoder", "model.model.decoder"):
        obj = tok
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            decoder = obj
            print(f"resolved decoder at tok.{path}: {type(decoder).__name__}", flush=True)
            break
        except AttributeError:
            continue
    if decoder is None:
        print("could not find decoder; tok attrs:", [a for a in dir(tok) if not a.startswith("__")])
        return

    # decoder forward expects [B, num_quantizers, frames] (decode() transposes (1,2)).
    codes = torch.randint(0, 64, (1, NUM_CODEBOOKS, WINDOW), device="cuda", dtype=torch.long)

    @torch.inference_mode()
    def eager():
        return decoder(codes)

    eager_ms = _time(eager)
    print(f"\n[eager]        decoder fwd @ {WINDOW} frames: {eager_ms:.3f} ms")

    # --- CUDA graph capture/replay ---
    graph_ms = None
    try:
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.inference_mode():
                    _ = decoder(codes)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(g):
                static_out = decoder(codes)  # noqa: F841

        def replay():
            g.replay()

        graph_ms = _time(replay)
        print(f"[cuda-graph]   replay:                    {graph_ms:.3f} ms  "
              f"({eager_ms/graph_ms:.1f}x vs eager)")
    except Exception as e:
        print(f"[cuda-graph]   capture FAILED: {type(e).__name__}: {str(e)[:160]}")

    # --- torch.compile (reduce-overhead = cudagraphs under the hood) ---
    try:
        compiled = torch.compile(decoder, mode="reduce-overhead", fullgraph=False)
        with torch.inference_mode():
            comp_ms = _time(lambda: compiled(codes))
        print(f"[compile]      reduce-overhead:           {comp_ms:.3f} ms  "
              f"({eager_ms/comp_ms:.1f}x vs eager)")
    except Exception as e:
        print(f"[compile]      FAILED: {type(e).__name__}: {str(e)[:160]}")

    print(
        "\nReading it:\n"
        f"  • eager ~16ms is the per-call floor. If cuda-graph/compile is a few ms,\n"
        f"    the 16ms was launch overhead => CUDA-graphing the codec in the server cuts\n"
        f"    RTF at every chunk_size. Projected cs1 codec RTF = 12 * graph_ms/1000.\n"
        f"  • If graph_ms ~= eager_ms, it's real compute and we're capped at this codec."
    )


if __name__ == "__main__":
    main()
