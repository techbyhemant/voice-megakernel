"""
repro_codec_audio.py — does the graph-cache codec corrupt the REAL streaming audio?

Generates the same sentence twice with a fixed seed (so the talker/predictor produce
identical codes) — once with the eager codec, once with the graph cache wired in
exactly as the server does it — and compares the full waveforms. Tells us if the
graph output is silent (absmax~0), wrong-length, or numerically divergent, and on
which path, so we fix the real bug rather than guess.

Run with exclusive GPU (stop the server first):
    /venv/main/bin/python benchmarks/repro_codec_audio.py
"""
import sys
sys.path.insert(0, "/workspace")
import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS
from megakernel_talker import MegakernelTalkerGraph
from megakernel_predictor import MegakernelPredictorGraph
from tts_server import _GraphedCodecDecoder

TEXT = "I can do one fifty a mile to Chicago, does that work for you?"

m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
base = m.talker_graph.model
m.talker_graph = MegakernelTalkerGraph(base, base.config)
m.predictor_graph = MegakernelPredictorGraph(m.predictor_graph)


def gen(seed):
    torch.manual_seed(seed)
    out = []
    for audio, sr, _ in m.generate_custom_voice_streaming(
        text=TEXT, speaker="uncle_fu", language="English",
        chunk_size=2, temperature=0.4, top_k=50,
    ):
        if audio is not None and len(audio):
            out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


# Warm, then eager reference.
for _ in m.generate_custom_voice_streaming(text="warm up", speaker="uncle_fu", language="English", chunk_size=2):
    pass
a_eager = gen(1234)
print(f"eager:  len={len(a_eager):>7}  absmax={np.abs(a_eager).max():.4f}")

# Wire the graph cache exactly like the server, warm to capture per-shape graphs.
dec = m.speech_tokenizer.model.decoder
dec.forward = _GraphedCodecDecoder(dec.forward)
for cs in (1, 2, 4):
    for _ in m.generate_custom_voice_streaming(text="warm up the decoder now please",
                                               speaker="uncle_fu", language="English", chunk_size=cs):
        pass
a_graph = gen(1234)
print(f"graph:  len={len(a_graph):>7}  absmax={np.abs(a_graph).max():.4f}")

if len(a_eager) == len(a_graph):
    d = float(np.abs(a_eager - a_graph).max())
    print(f"max|eager-graph| = {d:.6f}")
    print("VERDICT:", "MATCH (graph audio is correct)" if d < 1e-2
          else "DIVERGENT (graph corrupts audio)")
else:
    print(f"VERDICT: LENGTH MISMATCH ({len(a_eager)} vs {len(a_graph)}) — graph produces wrong-length audio")
