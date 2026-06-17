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
