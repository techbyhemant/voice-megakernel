"""
bench_tts.py — rigorous TTS benchmark harness (runs on the RTX 5090).

Measures, over multiple runs and text lengths, in the BRIEF's conventions:
  - TTFC  : time to first audio chunk (ms)            [target < 50-90 ms]
  - RTF   : compute_time / audio_duration (lower=better) [target < 0.1-0.3]
  - ms/step and steps/s : talker decode throughput (12 Hz frames)

It drives faster-qwen3-tts's streaming generator, so the SAME harness measures
the megakernel-backed talker once that's integrated — just point --model / the
backend at it. CUDA-synced timing; warmup runs excluded.

Run:  /venv/main/bin/python bench_tts.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --runs 7
"""

import argparse
import statistics
import time

import torch

from faster_qwen3_tts import FasterQwen3TTS

TEXTS = {
    "short": "Hello there, how are you today?",
    "medium": "Hello, this is the Qwen three T T S model running on an RTX 5090 graphics card.",
    "long": (
        "Real time speech synthesis needs the model to generate audio faster than "
        "it is played back. On this machine, the decode loop runs on the GPU while "
        "the codec turns the predicted tokens into a continuous waveform that streams "
        "out chunk by chunk, with no buffering of the whole utterance before sending."
    ),
}


def bench_once(model, text, speaker, language, chunk_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttfc = None
    samples = 0
    sr = 24000
    for audio, s, _timing in model.generate_custom_voice_streaming(
        text=text, speaker=speaker, language=language, chunk_size=chunk_size
    ):
        if audio is None or len(audio) == 0:
            continue
        if ttfc is None:
            torch.cuda.synchronize()
            ttfc = (time.perf_counter() - t0) * 1000.0  # ms
        samples += len(audio)
        sr = s
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    audio_s = samples / sr if sr else 0.0
    frames = audio_s * 12.0  # talker runs at 12 Hz
    return {
        "ttfc_ms": ttfc if ttfc is not None else float("nan"),
        "rtf": total / audio_s if audio_s else float("nan"),
        "ms_per_step": (total * 1000.0 / frames) if frames else float("nan"),
        "steps_per_s": (frames / total) if total else float("nan"),
        "audio_s": audio_s,
        "total_s": total,
    }


def agg(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return "n/a"
    s = sorted(vals)
    p90 = s[min(len(s) - 1, max(0, round(0.9 * len(s)) - 1))]
    return f"median={statistics.median(vals):.3f} p90={p90:.3f} min={min(vals):.3f} max={max(vals):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--engine", default="cudagraph", choices=["cudagraph", "megakernel"])
    args = ap.parse_args()

    print(f"Loading {args.model} (engine={args.engine}) ...", flush=True)
    model = FasterQwen3TTS.from_pretrained(args.model)

    if args.engine == "megakernel":
        import sys
        sys.path.insert(0, "/workspace")
        from megakernel_talker import MegakernelTalkerGraph
        base = model.talker_graph.model
        model.talker_graph = MegakernelTalkerGraph(base, base.config)
        print("Swapped talker to MEGAKERNEL", flush=True)

    print("Warmup (capture CUDA graphs, 2 runs)...", flush=True)
    for _ in range(2):
        bench_once(model, TEXTS["medium"], args.speaker, args.language, args.chunk_size)

    print(f"\n=== model={args.model} speaker={args.speaker} chunk_size={args.chunk_size} runs={args.runs} ===")
    overall = {"ttfc_ms": [], "rtf": [], "ms_per_step": []}
    for name in args.lengths.split(","):
        name = name.strip()
        if name not in TEXTS:
            continue
        rows = [bench_once(model, TEXTS[name], args.speaker, args.language, args.chunk_size)
                for _ in range(args.runs)]
        a = rows[0]["audio_s"]
        print(f"\n[{name}] audio≈{a:.2f}s")
        for k in ("ttfc_ms", "rtf", "ms_per_step"):
            vals = [r[k] for r in rows]
            overall[k].extend(vals)
            print(f"  {k:12s}: {agg(vals)}")
    print("\n=== OVERALL ===")
    for k in ("ttfc_ms", "rtf", "ms_per_step"):
        print(f"  {k:12s}: {agg(overall[k])}")
    print("\n(RTF = compute/audio, lower is better. Targets: RTF<0.1-0.3, TTFC<50-90ms.)")


if __name__ == "__main__":
    main()
