# Patches

Two patches turn stock components into this project's TTS backend — one modifies
the CUDA megakernel, the other is a pure-Python fix to the streaming loop.

## 1. `qwen_megakernel_talker.patch` — CUDA kernel (onto `qwen_megakernel` @ `5030e15`)

Lets AlpinDale's megakernel serve as Qwen3-TTS's **talker** (and, reused with a
runtime `num_layers`, the **code predictor**). Changes `csrc/kernel.cu`,
`csrc/torch_bindings.cpp`:

- **`input_hidden` path** — when provided, layer 0 reads an injected hidden state
  instead of an embedding lookup (the talker is fed `inputs_embeds`, not token ids).
- **`skip_lm_head` flag** — the talker wants the final-norm hidden state (for the
  code predictor), not an argmax token, so the LM head is skipped and the hidden
  lands in the `normalized` buffer.
- New op **`decode_from_hidden(...)`** wrapping both; the original `decode` op is
  unchanged (passes `nullptr, false`).
- **Barrier re-arm race fix** — block 0 re-armed the atomic grid barrier
  (`*barrier_counter = 0`) *in-kernel* with no grid-wide ordering vs. the other
  blocks' `atomicAdd(barrier_counter, 1)`, so an increment landing before the reset
  was wiped → the counter never reached `num_blocks` → the barrier spun forever
  (GPU 100%, no Xid). Fixed by zeroing the barrier/flag buffers **host-side**
  (`cudaMemsetAsync`, stream-ordered before launch) and deleting the racy in-kernel
  reset — fidelity intact (cosine 0.99926).

The talker backbone is architecturally identical to Qwen3-0.6B (28 layers, hidden
1024, 16 q / 8 kv heads, head_dim 128), so **no dimension changes** were needed —
only RoPE theta=1e6 (set in `megakernel_talker.py`).

```bash
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && git checkout 5030e154d39ecd054df03eb4dd9c8aa8185414d1
git apply /path/to/patches/qwen_megakernel_talker.patch
```
Verified: megakernel talker matches PyTorch at cosine **0.99978** (single-step; see
`megakernel_talker.py`).

## 2. `faster_qwen3_tts_eos.patch` — streaming loop (pure Python, no CUDA)

Fixes the talker **runaway / missing-EOS failure** in `faster_qwen3_tts/streaming.py`:
the talker intermittently fails to emit its stop token and runs to the cap. It's an
inherent Qwen3-TTS bug ([QwenLM/Qwen3-TTS #118](https://github.com/QwenLM/Qwen3-TTS/issues/118),
~0.5% on stock PyTorch), amplified by the megakernel predictor's feedback drift. Two
changes (full reasoning in the main README's *Talker runaway & EOS robustness*):

- **Multiple-EOS detection** — stop on `codec_eos_token_id` (2150) **or**
  `codec_think_eos_id` (2157, which the suppress-mask previously blocked), not just one.
- **Filler repeat-stop** (`MK_REPEAT_STOP`=40) — break if the primary codec token
  repeats ≥40 in a row; 40 sits in the content (≤34) / filler (50–1800) run-length gap,
  so it trims the runaway and the multi-second drag without clipping real speech.

This — not the kernel barrier above — was the dominant intermittent "hang" we chased
in live use: a reply ballooning to ~160 s of audio with the mic muted, which *felt*
frozen even though compute finished in seconds. Pairs with the server-side dynamic cap
(`tts_server._dynamic_cap`) as the final catch-all.

```bash
SITE=$(python -c "import faster_qwen3_tts,os;print(os.path.dirname(os.path.dirname(faster_qwen3_tts.__file__)))")
patch -p1 -d "$SITE" < /path/to/patches/faster_qwen3_tts_eos.patch
```
