"""
megakernel_predictor.py — run Qwen3-TTS code predictor 15-step loop on the megakernel.

The predictor backbone is architecturally identical to Qwen3-0.6B (hidden 1024,
16/8 heads, head_dim 128, intermediate 3072, rope 1e6) but only 5 layers. The
megakernel takes num_layers as a runtime arg, so the SAME compiled kernel runs the
predictor backbone. We drive its 15-step inner loop eagerly (backbone on the
megakernel; per-codebook lm_heads + sampling + codec-embedding feedback in PyTorch).

Drop-in for faster_qwen3_tts PredictorGraph: same .run(pred_input)->[15] interface.
Measured 4.0 ms/frame vs 8.55 ms for the CUDA-graph predictor (2.1x). Teacher-forced
per-step hidden cosine vs PyTorch predictor: min 0.99926, mean 0.99977.
"""
import math, struct, sys
sys.path.insert(0, "/workspace/qwen_megakernel")
import torch
from qwen_megakernel.build import get_extension
from qwen_megakernel.model import (HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_SIZE,
    MAX_SEQ_LEN, NUM_KV_HEADS, Q_SIZE)
from faster_qwen3_tts.sampling import sample_logits

ROPE_THETA = 1_000_000.0

def _pack(lw, n):
    buf = bytearray(n*11*8)
    for i in range(n):
        for j in range(11):
            struct.pack_into("Q", buf, (i*11+j)*8, lw[i*11+j].data_ptr())
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()

def _build_weights(base, n):
    inv = 1.0/(ROPE_THETA**(torch.arange(0,HEAD_DIM,2,dtype=torch.float32)/HEAD_DIM))
    fr = torch.outer(torch.arange(MAX_SEQ_LEN,dtype=torch.float32), inv)
    cos = torch.cos(fr).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    sin = torch.sin(fr).repeat(1,2).to(torch.bfloat16).cuda().contiguous()
    lw = []
    for i in range(n):
        L = base.layers[i]; a = L.self_attn
        lw.extend([L.input_layernorm.weight.contiguous(), a.q_proj.weight.contiguous(),
            a.k_proj.weight.contiguous(), a.v_proj.weight.contiguous(), a.q_norm.weight.contiguous(),
            a.k_norm.weight.contiguous(), a.o_proj.weight.contiguous(),
            L.post_attention_layernorm.weight.contiguous(), L.mlp.gate_proj.weight.contiguous(),
            L.mlp.up_proj.weight.contiguous(), L.mlp.down_proj.weight.contiguous()])
    return lw, base.norm.weight.contiguous(), cos, sin

class _PredKernel:
    def __init__(self, base, n):
        get_extension(); self.n = n
        lw, fn, cos, sin = _build_weights(base, n)
        self.p = _pack(lw, n); self.fn = fn; self.cos = cos; self.sin = sin
        b = dict(dtype=torch.bfloat16, device="cuda"); f = dict(dtype=torch.float32, device="cuda")
        self.kc = torch.zeros(n, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, **b); self.vc = torch.zeros_like(self.kc)
        self.h = torch.empty(HIDDEN_SIZE, **b); self.a = torch.empty(HIDDEN_SIZE, **f); self.r = torch.empty(HIDDEN_SIZE, **f)
        self.q = torch.empty(Q_SIZE, **f); self.k = torch.empty(KV_SIZE, **f); self.v = torch.empty(KV_SIZE, **f)
        self.ao = torch.empty(Q_SIZE, **f); self.m = torch.empty(INTERMEDIATE_SIZE, **f); self.no = torch.empty(HIDDEN_SIZE, **f)
        self.sc = 1.0/math.sqrt(HEAD_DIM)
    def step(self, h, pos):
        torch.ops.qwen_megakernel_C.decode_from_hidden(self.no, h.reshape(-1).contiguous(), self.p,
            self.fn, self.cos, self.sin, self.kc, self.vc, self.h, self.a, self.r, self.q, self.k,
            self.v, self.ao, self.m, self.n, pos, MAX_SEQ_LEN, self.sc)
        return self.no

class MegakernelPredictorGraph:
    """Drop-in for faster_qwen3_tts PredictorGraph using the megakernel backbone."""
    def __init__(self, base_predictor_graph):
        pg = base_predictor_graph
        self.sm = pg.small_to_mtp; self.heads = pg.lm_heads; self.embeds = pg.codec_embeds
        self.ncb = pg.num_codebooks
        self.temperature = pg.temperature; self.top_k = pg.top_k; self.top_p = pg.top_p; self.do_sample = pg.do_sample
        self.pk = _PredKernel(pg.pred_model, pg.num_layers)
        self.out = torch.zeros(self.ncb, dtype=torch.long, device="cuda")
        self.captured = True
    def capture(self, num_warmup=3):
        pass
    @torch.inference_mode()
    def run(self, pred_input):
        h2 = self.sm(pred_input)                     # [1,2,H]
        self.pk.step(h2[0,0], 0)                      # prefill token 0
        hid = self.pk.step(h2[0,1], 1)                # prefill token 1 -> hidden for cb0
        logits = self.heads[0](hid.to(torch.bfloat16).view(1,1,HIDDEN_SIZE))
        tok = sample_logits(logits[:,0,:], temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, do_sample=self.do_sample)
        self.out[0] = tok[0]
        for cb in range(1, self.ncb):
            emb = self.sm(self.embeds[cb-1](tok.unsqueeze(0)))   # [1,1,H]
            hid = self.pk.step(emb[0,0], 1+cb)
            logits = self.heads[cb](hid.to(torch.bfloat16).view(1,1,HIDDEN_SIZE))
            tok = sample_logits(logits[:,0,:], temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, do_sample=self.do_sample)
            self.out[cb] = tok[0]
        return self.out.clone()


def _checks():
    """Teacher-forced per-step fidelity vs the PyTorch predictor + timing vs the
    CUDA-graph predictor. Run on the 5090 with an exclusive GPU:
        /venv/main/bin/python megakernel_predictor.py
    """
    import time
    import torch.nn.functional as Fn
    from transformers import DynamicCache
    from faster_qwen3_tts import FasterQwen3TTS

    m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    pg = m.predictor_graph
    pm, sm, heads, embeds = pg.pred_model, pg.small_to_mtp, pg.lm_heads, pg.codec_embeds
    mk = MegakernelPredictorGraph(pg)

    torch.manual_seed(0)
    pred_input = torch.randn(1, 2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Reference: eager PyTorch predictor, capture per-step hidden + argmax tokens.
    cache = DynamicCache(); h2 = sm(pred_input)
    o = pm(inputs_embeds=h2, use_cache=True, past_key_values=cache, cache_position=torch.arange(2, device="cuda"))
    ref_h = [o.last_hidden_state[0, 1].float().clone()]
    toks = [heads[0](o.last_hidden_state[:, -1:, :]).argmax(-1).view(())]
    for cb in range(1, 15):
        emb = sm(embeds[cb - 1](toks[-1].view(1, 1)))
        o = pm(inputs_embeds=emb, use_cache=True, past_key_values=cache, cache_position=torch.tensor([1 + cb], device="cuda"))
        ref_h.append(o.last_hidden_state[0, 0].float().clone())
        toks.append(heads[cb](o.last_hidden_state[:, -1:, :]).argmax(-1).view(()))

    # Megakernel backbone, teacher-forced with the same tokens.
    mk.pk.step(h2[0, 0], 0)
    mk_h = [mk.pk.step(h2[0, 1], 1).float().clone()]
    for cb in range(1, 15):
        emb = sm(embeds[cb - 1](toks[cb - 1].view(1, 1)))
        mk_h.append(mk.pk.step(emb[0, 0], 1 + cb).float().clone())
    coss = [Fn.cosine_similarity(a, b, dim=0).item() for a, b in zip(mk_h, ref_h)]
    print(f"FIDELITY teacher-forced per-step cosine: min={min(coss):.5f} mean={sum(coss)/len(coss):.5f}")
    print("FIDELITY:", "PASS" if min(coss) > 0.99 else "FAIL")

    # Timing vs the CUDA-graph predictor.
    pg.capture(num_warmup=3)
    gi = torch.randn(1, 2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    def t(fn, N=100, w=10):
        for _ in range(w): fn()
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(N): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - s) / N * 1000
    cg = t(lambda: pg.run(gi)); mkk = t(lambda: mk.run(gi))
    print(f"TIMING predictor CUDA-graph={cg:.3f} ms/frame  megakernel={mkk:.3f} ms/frame  speedup={cg/mkk:.2f}x")


if __name__ == "__main__":
    _checks()
