# Voice Megakernel — RTX 5090 Decode Megakernel → Qwen3-TTS on Pipecat

Run [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
(a hand-written CUDA megakernel for Qwen3-0.6B) as the decode backend for
**Qwen3-TTS**, streaming real-time speech into a **Pipecat** voice agent.

The voice loop runs **mic → Whisper (STT) → local LLM → Qwen3-TTS → speaker**.
Only the TTS is custom: **both** of its autoregressive stages — the talker
(28-layer backbone) *and* the code predictor (5-layer, 15-step inner loop) — are
driven by the same megakernel; only the codec (tokens → waveform) stays in
PyTorch. Driving the predictor on the kernel too is the project's biggest win and
the task's stated bonus ("improve the megakernel's performance during integration").

## Headline results

Qwen3-TTS `12Hz-0.6B-CustomVoice`, streaming, single RTX 5090, `chunk_size=4`,
`ryan` voice. RTF = compute / audio (lower is better; 1.0 = real-time).

| Engine | ms/step | RTF | TTFC |
|---|---|---|---|
| Naive eager PyTorch (baseline) | — | **1.48** | n/a (buffered) |
| CUDA-graph (faster-qwen3-tts) | 21.05 | 0.253 | 107 ms |
| Megakernel talker only | 15.07 | 0.181 | 83.7 ms |
| **Megakernel talker + predictor (this project)** | **9.51** | **0.114** | **63 ms** |

- The full megakernel (talker + predictor) is **~55% faster on RTF / per step**
  and **41% lower TTFC** than the CUDA-graph baseline.
- **RTF 0.114 beats the `< 0.15` target**; at `chunk_size=2`, **TTFC 52.8 ms beats
  the `< 60` target**. (See the chunk-size sweep below; one non-kernel lever — the
  glue — reaches the strict `< 0.1`.)
- **Two kernel contributions:** the talker is 28% faster than the CUDA-graph
  talker; adding the predictor on the same kernel wins a **further 37% per step**
  (predictor stage: 8.55 → 4.03 ms/frame, **2.1×**).
- Unmodified megakernel decode (Qwen3-0.6B chat, sanity check): **1050 tok/s**.
- Both stages match PyTorch closely: talker **cosine 0.9997**, predictor
  **teacher-forced cosine 0.9993** (min) / 0.9998 (mean). See `megakernel_talker.py`
  and `megakernel_predictor.py`.

## Why this matters (freight voice negotiation)

e3 negotiates loads with carriers **over the phone**. On a live call, the agent's
responsiveness *is* the product: a human expects a reply to begin within a couple
hundred milliseconds, or the bot feels robotic, kills rapport, and loses the
negotiation. So **time-to-first-audio (TTFC) is the metric that maps to call
quality** — not raw throughput.

That's why this kernel work matters here: it cuts TTFC from 107 ms to **63 ms**
(and to **52.8 ms** at `chunk_size=2`), well under the bar where turn-taking feels
natural — and cuts RTF from 0.253 to **0.114**, so the GPU spends far less of each
second of audio on compute (more headroom for concurrent calls).

Product judgment also shaped what we *didn't* build: **no GUI** (the interface is
the voice), and STT/LLM are **swappable off-the-shelf parts** — the engineering
investment went into the one component that actually moves the customer-facing
latency.

## Where the time goes (bottleneck analysis)

Measured per-frame split (megakernel talker only, before the predictor work),
profiled live with CUDA-synced timers on the deployed engine:

| Component | per frame | share |
|---|---|---|
| Talker backbone (28L, megakernel) | 0.97 ms | 6.7% |
| **Code predictor (5L, 15-step loop, CUDA graph)** | **8.6 ms** | **59%** |
| Rest (codec embeds, talker head, sampling, glue) | ~5 ms | 34% |

Key insight: once the megakernel solves the talker (~1 ms), the **code predictor
becomes the bottleneck (59%)**, not the backbone. The predictor's per-layer
architecture is *identical* to Qwen3-0.6B (hidden 1024, 16/8 heads, head_dim 128,
intermediate 3072, rope 1e6) — only the layer count differs (5 vs 28). The
megakernel takes layer-count as a *runtime* argument, so the **same compiled
kernel** drives the predictor backbone (`num_layers=5`). Even run eagerly (16
backbone calls + per-codebook heads/sampling/feedback in PyTorch), it beats the
fused CUDA-graph predictor **2.1×** (8.55 → 4.03 ms/frame) — because the
megakernel's per-step backbone is ~7× faster than the CUDA-graph one, which
swamps the eager-loop overhead. So the win is: **megakernel BOTH autoregressive
stages.** After that, the remaining ~4.5 ms/frame "rest" (a 15-way embedding loop
+ glue, all eager PyTorch) is what stands between RTF 0.114 and the strict `< 0.1`.

## Architecture

```
   YOUR MAC (mic + speaker)            RENTED RTX 5090 (all compute)
┌────────────────────────────┐     ┌──────────────────────────────────────────┐
│ mic → Whisper (MLX STT)     │     │  Ollama LLM (qwen2.5:7b-instruct)          │
│           │ text            │ ──► │        │  negotiation reply                │
│           ▼                 │ ◄── │        ▼                                   │
│    RemoteQwenTTS ───────────────► │  Qwen3-TTS streaming:                      │
│ speaker ◄── audio chunks ◄────────│   talker step    ─► MEGAKERNEL             │
└────────────────────────────┘ PCM │   code predictor ─► MEGAKERNEL · codec=PT  │
   SSH tunnel: 8000 (TTS),          └──────────────────────────────────────────┘
               11435→11434 (LLM)
```

- The **Mac** is the thin audio client: microphone, Whisper STT, speaker. (Audio
  I/O must live where the human is; the GPU box is headless.)
- **All compute runs on the 5090** — the LLM (Ollama) *and* the TTS — reached
  over an SSH tunnel. This is the brief's Step 2 (inference server) + Step 3
  (Pipecat integration).
- Inside the TTS: **both** the talker's per-step decode and the code predictor's
  15-step loop run on the **megakernel**; only the **codec** stays in PyTorch.
- The **LLM brain is swappable** — Ollama on the GPU by default (free, on the
  reimbursed GPU); set `ANTHROPIC_API_KEY` to use Claude instead (better
  negotiation quality, but billed to you — API cost isn't reimbursed).

## Kernel modifications

See `patches/qwen_megakernel_talker.patch` (applies onto `qwen_megakernel`
@`5030e15`) and `patches/README.md`. The talker backbone is architecturally
identical to Qwen3-0.6B (28 layers, hidden 1024, 16 q / 8 kv heads, head_dim
128), so **no dimension changes were needed** — only:

1. **`input_hidden` path** — the talker is fed `inputs_embeds`, not token ids, so
   layer 0 reads an injected hidden state instead of doing an embedding lookup.
2. **`skip_lm_head`** — the talker wants the final-norm hidden state (for the code
   predictor), not an argmax token, so the LM head is skipped.
3. New op `decode_from_hidden(...)`; the original `decode` op is unchanged.
4. RoPE theta = 1e6 (talker value; set in `megakernel_talker.py`), and
   `rope_deltas = 0` holds for custom-voice/voice-clone so RoPE position = cache
   position with no offset.

**Code predictor on the same kernel** (`megakernel_predictor.py`) — *no new CUDA*.
The predictor backbone is the same architecture as the talker/0.6B but 5 layers
instead of 28. Since the kernel's `decode_from_hidden` op takes `num_layers` as a
**runtime argument** (only the Python-side weight pack and KV cache are sized per
model), the same compiled kernel runs the predictor backbone by packing 5 layers
of predictor weights and calling with `num_layers=5`. `MegakernelPredictorGraph`
is a drop-in for faster-qwen3-tts's `PredictorGraph` (same `run(pred_input)→[15]`
interface): it runs the 15-step inner loop eagerly — backbone on the kernel,
per-codebook `lm_head` + top-k/p sampling + codec-embedding feedback in PyTorch.
This was the surprise: even with eager per-step glue it beats the fused CUDA-graph
predictor 2.1×, so no persistent/fused predictor kernel was needed.

KV-cache prefill is copied from the PyTorch prefill into the kernel's cache
(`TalkerKernel.prefill_from_cache`); convention verified by the multi-step check.

## Repo layout

```
bot.py                 Pipecat voice agent (Mac client)
remote_tts_service.py  Pipecat TTS client → streams from the 5090 server
run.sh                 launch the agent
server/tts_server.py   streaming TTS server (5090); --engine + --predictor cudagraph|megakernel
megakernel_talker.py   TalkerKernel + MegakernelTalkerGraph + correctness/e2e checks
megakernel_predictor.py  MegakernelPredictorGraph (predictor 15-step loop on the kernel)
benchmarks/bench_tts.py  TTFC/RTF/ms-step harness; --engine + --predictor cudagraph|megakernel
patches/               kernel modification patch + notes
.env.example           HF_TOKEN (server) — copy to .env
```

## How to run

### Server (RTX 5090, sm_120 / Blackwell, CUDA ≥ 12.8)

```bash
# 1. Base image with CUDA 12.8 + cu128 PyTorch (e.g. vastai/pytorch:cuda-12.8.1
#    or pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel). Verify:
nvcc --version            # >= 12.8
python -c "import torch; print(torch.cuda.get_device_capability())"   # (12, 0)

# 2. Build the modified megakernel
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && git checkout 5030e154d39ecd054df03eb4dd9c8aa8185414d1
git apply /path/to/patches/qwen_megakernel_talker.patch && cd ..

# 3. Deps
pip install qwen-tts faster-qwen3-tts fastapi "uvicorn[standard]"
export HF_TOKEN=hf_...     # free read-only token (avoids HF rate-limiting)

# 4. LLM on the GPU (Ollama) — keyless, on the reimbursed GPU
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                       # run in tmux/background
ollama pull qwen2.5:7b-instruct

# 5. Streaming TTS server. Both stages on the megakernel (default); pass
#    --engine/--predictor cudagraph to fall back per stage.
python server/tts_server.py --port 8000 --engine megakernel --predictor megakernel
```

### Client (Mac, Apple Silicon)

```bash
brew install portaudio uv ffmpeg
uv venv --python 3.12 && uv pip install "pipecat-ai[mlx-whisper,local]" python-dotenv
# No local LLM — the brain runs on the GPU (Ollama on the 5090). The Mac only
# does mic capture, Whisper STT, and speaker playback.

# Tunnel both GPU services (TTS 8000, LLM 11435->11434), then run the agent:
ssh -i ~/.ssh/vast_ai -p <PORT> -L 8000:localhost:8000 -L 11435:localhost:11434 root@<host>
./run.sh
```
(Non-Mac clients: swap `WhisperSTTServiceMLX` → `WhisperSTTService` (faster-whisper)
in `bot.py`; STT is the only Mac-specific piece.)

Talk, pause, and the agent replies in the Qwen3-TTS voice.

## Benchmarking

```bash
# on the 5090 (exclusive GPU — stop the server first; the megakernel needs all SMs)
python benchmarks/bench_tts.py --engine cudagraph  --predictor cudagraph  --runs 7  # baseline
python benchmarks/bench_tts.py --engine megakernel --predictor cudagraph  --runs 7  # talker only
python benchmarks/bench_tts.py --engine megakernel --predictor megakernel --runs 7  # both (headline)
```
Multi-run, CUDA-synced, reports median/p90/min/max for TTFC, RTF, ms/step across
short/medium/long texts. `--engine` selects the talker engine, `--predictor` the
code-predictor engine, so each kernel's contribution is measurable in isolation.

## Observability

A phone agent fails *quietly* — audio just gets laggy or choppy — so you
instrument the signals that map to call quality:

- **Live per-turn metrics, split by layer:** every reply prints **compute(GPU)**
  TTFC/RTF (measured server-side, no network — `tts_server.py /metrics`) *and*
  **end-to-end** TTFC/RTF incl. network (`remote_tts_service.py`). Seeing both
  side by side attributes latency to the right layer (measured demo run, both
  megakernels, `chunk_size=2`: **~55 ms compute vs ~590 ms end-to-end ⇒ ~535 ms is
  network/geography over the SSH tunnel, not the model**).
- **What you'd monitor in production:**
  - **TTFC p50/p99** — the turn-taking latency a caller feels (alert if p99 climbs).
  - **RTF** — must stay < 1.0 or audio stutters (alert as it approaches 1.0).
  - **STT / LLM / TTS latency breakdown** — pinpoint which stage caused a slow turn.
  - **Dropped/late audio frames, GPU utilization, error rate** — health + capacity.
- **Offline rigor:** `benchmarks/bench_tts.py` reports median/p90/min/max over
  multiple runs and text lengths — the same discipline applied offline that the
  live metrics apply online.

Note: live end-to-end TTFC includes network round-trip (hundreds of ms over the
SSH tunnel); the **on-GPU compute figure** (63 ms at `chunk_size=4`, ~53 ms at the
`chunk_size=2` demo default, both megakernels) is what a co-located deployment
would see.

## What works, what's rough (honest)

**Works:** end-to-end real-time voice agent; the megakernel verifiably drives
**both** the talker (cosine 0.9997) and the code predictor (teacher-forced cosine
0.9993) — together 55% faster than the CUDA-graph baseline (RTF 0.255 → 0.114);
streaming is true frame-by-frame (not buffered); reproducible kernel patch + a
no-recompile predictor reuse.

**Rough / known issues:**
- **Megakernel startup contention:** the megakernel launches 128 *persistent
  grid-synced* blocks that must all be co-resident on the SMs at once. If another
  process is actively using the GPU *at launch* (e.g. Ollama loading weights, or
  a server restart mid-flight), the blocks can't all schedule and the grid-sync
  spins (GPU pegged, no progress). The dual-megakernel warmup launches the kernel
  many more times, so it hits this more often — in practice **one server restart**
  lands it on a settled GPU and it then runs reliably (verified live across the
  streaming server + repeated requests). Benchmarks need the GPU **exclusive**
  (stop the server first). `--engine/--predictor cudagraph` remain available as
  per-stage fallbacks. The robust fix is to lower `LDG_NUM_BLOCKS` so the kernel
  co-resides even under contention.
- **TTFC over the network:** box-local TTFC is 53–63 ms, but end-to-end over the
  SSH tunnel is ~590 ms — dominated by network round-trip + per-request HTTP
  setup, not compute. A persistent connection / co-locating the client would
  remove most of it. Reported separately from the on-GPU numbers.
- **vs targets:** with both megakernels, `chunk_size` trades TTFC vs RTF. Measured
  (CUDA-synced, both megakernels):

  | `chunk_size` | TTFC | RTF | audio/chunk |
  |---|---|---|---|
  | 4 | 63 ms | **0.114** | 333 ms |
  | **2** (demo default) | **52.8 ms** | 0.162 | 167 ms |

  At `chunk_size=4` **RTF 0.114 beats the `< 0.15` target**; at `chunk_size=2`
  **TTFC 52.8 ms beats the `< 60` target**. The brief's strictest numbers
  (TTFC `< 50` *and* RTF `< 0.1` jointly) remain just out of reach, and honestly
  so: per-frame is now 9.51 ms = talker 0.97 + predictor 4.03 + **~4.5 ms eager
  "rest"** (a 15-way codec-embedding loop + talker head + sampling, all uncaptured
  PyTorch in faster-qwen3-tts's streaming loop). Vectorizing that glue (not kernel
  work) is the last lever to RTF `< 0.1` — documented but not done, since it's
  upstream-library surgery with correctness risk and little kernel relevance.

## Credits

- Megakernel: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
- CUDA-graph streaming base: [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS](https://huggingface.co/Qwen) · [Pipecat](https://docs.pipecat.ai)
