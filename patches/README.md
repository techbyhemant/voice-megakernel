# Kernel modifications

`qwen_megakernel_talker.patch` modifies AlpinDale's `qwen_megakernel` so the
megakernel can serve as Qwen3-TTS's **talker** decode backend.

**What it changes** (`csrc/kernel.cu`, `csrc/torch_bindings.cpp`):
- Adds an `input_hidden` path: when provided, layer 0 reads an injected hidden
  state instead of doing an embedding lookup (the talker is fed `inputs_embeds`,
  not token ids).
- Adds a `skip_lm_head` flag: the talker wants the final-norm hidden state (for
  the code predictor), not an argmax token, so the LM head is skipped and the
  hidden lands in the `normalized` buffer.
- Exposes a new op `decode_from_hidden(...)` wrapping both.
- The original `decode` op is unchanged (passes `nullptr, false`).
- **Barrier re-arm race fix:** the persistent kernel's atomic grid barrier was
  re-armed *in-kernel* by block 0 (`*barrier_counter = 0`) with no grid-wide
  ordering vs. the other blocks' `atomicAdd(barrier_counter, 1)`. If a block
  incremented before block 0's reset landed, that increment was wiped, the
  counter never reached `num_blocks`, and the barrier spun forever (intermittent
  hang: GPU 100%, no Xid). Fixed by zeroing the barrier/flag buffers **host-side**
  (`cudaMemsetAsync`, stream-ordered before the launch) and deleting the racy
  in-kernel reset — preserves fidelity (cosine 0.99926). A residual hang remains
  in the talker's longer 28-layer barrier chain (~84 barriers/launch) — see the
  main README's known-issues.

The talker backbone is architecturally identical to Qwen3-0.6B (28 layers,
hidden 1024, 16 q / 8 kv heads, head_dim 128), so no dimension changes are
needed — only RoPE theta=1e6 (set in `megakernel_talker.py`).

**Apply:**
```bash
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && git checkout 5030e154d39ecd054df03eb4dd9c8aa8185414d1
git apply /path/to/patches/qwen_megakernel_talker.patch
```

**Verified:** the megakernel's talker output matches the PyTorch talker at
cosine similarity 0.99978 (single-step, see `megakernel_talker.py`).
