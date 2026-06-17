"""
megakernel_talker.py — run Qwen3-TTS's talker backbone on the CUDA megakernel.

The talker base (Qwen3TTSTalkerModel) is architecturally identical to Qwen3-0.6B
(28 layers, hidden 1024, 16 q / 8 kv heads, head_dim 128, rope_theta 1e6), so the
megakernel runs it unchanged except: weights come from the talker, RoPE uses
theta=1e6, and we use the `decode_from_hidden` op (hidden in -> final-norm hidden
out, skipping embedding + LM head).

Runs on the 5090. `python megakernel_talker.py` runs a single-step correctness
check against the PyTorch talker.
"""

import math
import sys

sys.path.insert(0, "/workspace/qwen_megakernel")

import torch

from qwen_megakernel.build import get_extension
from qwen_megakernel.model import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    MAX_SEQ_LEN,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_SIZE,
    _pack_layer_weights,
)

ROPE_THETA = 1_000_000.0  # talker uses 1e6 (vs 1e4 in the chat model)


def build_talker_weights(base):
    """Pack Qwen3TTSTalkerModel weights into the megakernel's format."""
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        L = base.layers[i]
        a = L.self_attn
        layer_weights.extend([
            L.input_layernorm.weight.contiguous(),
            a.q_proj.weight.contiguous(),
            a.k_proj.weight.contiguous(),
            a.v_proj.weight.contiguous(),
            a.q_norm.weight.contiguous(),
            a.k_norm.weight.contiguous(),
            a.o_proj.weight.contiguous(),
            L.post_attention_layernorm.weight.contiguous(),
            L.mlp.gate_proj.weight.contiguous(),
            L.mlp.up_proj.weight.contiguous(),
            L.mlp.down_proj.weight.contiguous(),
        ])
    return {
        "layer_weights": layer_weights,
        "final_norm": base.norm.weight.contiguous(),
        "cos": cos,
        "sin": sin,
    }


class TalkerKernel:
    """Megakernel-backed talker: hidden state in -> final-norm hidden state out."""

    def __init__(self, weights):
        get_extension()  # registers torch.ops.qwen_megakernel_C.*
        self._packed = _pack_layer_weights(weights["layer_weights"])
        self._final_norm = weights["final_norm"]
        self._cos = weights["cos"]
        self._sin = weights["sin"]
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, **bf16)
        self._v_cache = torch.zeros_like(self._k_cache)
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._scale = 1.0 / math.sqrt(HEAD_DIM)

    def reset(self):
        self._k_cache.zero_()
        self._v_cache.zero_()

    def step(self, input_hidden, position):
        """input_hidden: bf16 [HIDDEN_SIZE] on cuda. Returns final-norm hidden (f32 [HIDDEN_SIZE])."""
        torch.ops.qwen_megakernel_C.decode_from_hidden(
            self._norm_out, input_hidden.reshape(-1).contiguous(),
            self._packed, self._final_norm, self._cos, self._sin,
            self._k_cache, self._v_cache, self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out, self._mlp,
            NUM_LAYERS, position, MAX_SEQ_LEN, self._scale,
        )
        return self._norm_out.clone()


def _correctness_check():
    from faster_qwen3_tts import FasterQwen3TTS

    print("loading talker...")
    m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    base = m.talker_graph.model  # Qwen3TTSTalkerModel
    tk = TalkerKernel(build_talker_weights(base))
    tk.reset()

    torch.manual_seed(0)
    hidden = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    out_k = tk.step(hidden, position=0).to(torch.bfloat16)

    # PyTorch reference: one token, position 0, fresh cache
    from transformers import DynamicCache
    with torch.no_grad():
        o = base(
            inputs_embeds=hidden.view(1, 1, HIDDEN_SIZE),
            use_cache=True,
            past_key_values=DynamicCache(),
            cache_position=torch.tensor([0], device="cuda"),
        )
    out_ref = o.last_hidden_state[0, 0].to(torch.bfloat16)

    d = (out_k.float() - out_ref.float())
    cos = torch.nn.functional.cosine_similarity(out_k.float(), out_ref.float(), dim=0).item()
    print(f"max_abs_diff={d.abs().max().item():.4f}  mean_abs_diff={d.abs().mean().item():.4f}  cosine_sim={cos:.5f}")
    print("CORRECTNESS:", "PASS" if cos > 0.99 else "FAIL (investigate)")


if __name__ == "__main__":
    _correctness_check()
