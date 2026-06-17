"""
verify_codec_graph.py — is the CUDA-graph codec output CORRECT (not just fast)?

The graph cache is only "lossless" if replaying it on a NEW input reproduces what
eager would compute. We capture a graph on codesA, then replay with codesB and
compare to eager(codesB). maxdiff ~0 => correct (lossless). Large diff or ~0 absmax
=> the graph isn't tracking the input (silent/garbage audio) and must be reverted.
"""
import torch
from faster_qwen3_tts import FasterQwen3TTS

m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
dec = m.speech_tokenizer.model.decoder
orig = dec.forward

def capture(codes):
    static_in = codes.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            orig(static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = orig(static_in)
    return static_in, static_out, g

n = 27
codesA = torch.randint(0, 64, (1, 16, n), device="cuda")
codesB = torch.randint(0, 64, (1, 16, n), device="cuda")

eagerB = orig(codesB).float().clone()
si, so, g = capture(codesA)          # graph captured on A
si.copy_(codesB); g.replay()         # replay with B
graphedB = so.float().clone()

diff = (eagerB - graphedB).abs().max().item()
print(f"eagerB absmax   = {eagerB.abs().max().item():.5f}")
print(f"graphedB absmax = {graphedB.abs().max().item():.5f}")
print(f"max|eagerB - graphedB| = {diff:.6f}")
print("VERDICT:", "CORRECT (lossless)" if diff < 1e-2 and graphedB.abs().max().item() > 1e-3
      else "BROKEN — graph not tracking input (silent/garbage audio)")
