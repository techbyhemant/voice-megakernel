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

Qwen3-TTS `12Hz-0.6B-CustomVoice`, single RTX 5090 (sm_120, torch 2.11+cu128),
streaming, median of 7 runs, **`chunk_size=2`** (the demo setting), on-GPU. RTF =
compute/audio (lower is better; 1.0 = real-time). **Each row adds one optimization,
so the bottom row *is* the shipped demo config:**

| talker / predictor / codec | ms/step | RTF | TTFC |
|---|---|---|---|
| cudagraph / cudagraph / eager (faster-qwen3-tts baseline) | 28.7 | 0.344 | 82 ms |
| megakernel / cudagraph / eager (talker only) | 21.5 | 0.258 | 67 ms |
| megakernel / megakernel / eager (both stages) | 14.3 | 0.171 | 54 ms |
| **megakernel / megakernel / graph (the demo)** | 14.3 | **0.093** | **43 ms** |

The demo (bottom row) clears the brief's **Step-4 strict** targets — **TTFC < 50 ms
AND RTF < 0.1 at once** — plus every looser tier, streaming frame-by-frame (not
buffered).

- **Demo vs the CUDA-graph baseline:** TTFC **82 → 43 ms** (47% lower), RTF
  **0.344 → 0.093** (73% lower).
- **Where the wins come from:** (1) talker on the megakernel — ~25% faster/step
  (28.7 → 21.5 ms); (2) predictor on the *same* kernel — a further **2.95×** on the
  predictor in isolation (10.7 → 3.6 ms/frame), taking the pipeline 21.5 → 14.3
  ms/step; (3) the codec CUDA-graphed (lossless) — drops the last step (54 → 43 ms,
  0.171 → 0.093) without touching talker ms/step.
- Both stages match PyTorch: talker **cosine 0.99978**, predictor teacher-forced
  **0.99926 (min) / 0.99977 (mean)** — re-verified this run (`megakernel_talker.py`,
  `megakernel_predictor.py`).
- At `chunk_size=4`: full megakernel = **9.9 ms/step, RTF 0.119, TTFC 64 ms** (larger
  chunks lower RTF but raise TTFC — the usual tradeoff).

## Why this matters (freight voice negotiation)

e3 negotiates loads with carriers **over the phone**. On a live call, the agent's
responsiveness *is* the product: a human expects a reply to begin within a couple
hundred milliseconds, or the bot feels robotic, kills rapport, and loses the
negotiation. So **time-to-first-audio (TTFC) is the metric that maps to call
quality** — not raw throughput.

That's why this kernel work matters here: at the demo's `chunk_size=2` it cuts TTFC
from **82 ms** (CUDA-graph baseline) to **43 ms** (full megakernel + codec graph) —
well under the bar where turn-taking feels natural — and RTF from **0.344 to 0.093**,
so the GPU spends far less of each second of audio on compute (more headroom for
concurrent calls).

Product judgment also shaped what we *didn't* build: **no GUI** (the interface is
the voice), and STT/LLM are **swappable off-the-shelf parts** — the engineering
investment went into the one component that actually moves the customer-facing
latency.

## Where the time goes (bottleneck analysis)

Measured per-frame split (megakernel talker only, before the predictor work),
profiled live with CUDA-synced timers on the deployed engine:

| Component | per frame | share |
|---|---|---|
| Talker backbone (28L, megakernel) | ~1 ms | ~6% |
| **Code predictor (5L, 15-step loop, CUDA graph)** | **~10.7 ms** | **~64%** |
| Rest (codec embeds, talker head, sampling, glue) | ~5 ms | ~30% |

Key insight: once the megakernel solves the talker (~1 ms), the **code predictor
becomes the bottleneck (59%)**, not the backbone. The predictor's per-layer
architecture is *identical* to Qwen3-0.6B (hidden 1024, 16/8 heads, head_dim 128,
intermediate 3072, rope 1e6) — only the layer count differs (5 vs 28). The
megakernel takes layer-count as a *runtime* argument, so the **same compiled
kernel** drives the predictor backbone (`num_layers=5`). Even run eagerly (16
backbone calls + per-codebook heads/sampling/feedback in PyTorch), it beats the
fused CUDA-graph predictor **2.95×** (10.7 → 3.6 ms/frame) — because the
megakernel's per-step backbone is ~7× faster than the CUDA-graph one, which
swamps the eager-loop overhead. So the win is: **megakernel BOTH autoregressive
stages.** The remaining ~4.5 ms/frame "rest" (a 15-way embedding loop + glue, all
eager PyTorch) is the next target — though CUDA-graphing the codec already takes
RTF under 0.1 at `chunk_size=2`.

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
predictor 2.95×, so no persistent/fused predictor kernel was needed.

KV-cache prefill is copied from the PyTorch prefill into the kernel's cache
(`TalkerKernel.prefill_from_cache`); convention verified by the multi-step check.

## Talker runaway & EOS robustness (`patches/faster_qwen3_tts_eos.patch`)

The integration's hardest problem wasn't the kernel — it was an **inherent
Qwen3-TTS failure mode** the megakernel amplifies. Intermittently the talker
**fails to emit its stop token** and runs to the generation cap, producing minutes
of audio (or a garbled burst) for a one-line reply. It's a known upstream bug:
[QwenLM/Qwen3-TTS #118](https://github.com/QwenLM/Qwen3-TTS/issues/118) reports
~0.5% of inferences fail to emit EOS **on stock PyTorch** (closed "as not
planned"); the megakernel predictor's ~0.02%/frame fidelity drift, fed back
through the talker each frame, amplifies the rate.

We diagnosed it by dumping the talker's primary-codec-token stream — the runaway
is a **filler-token collapse, not noise.** Measured on the exact (cudagraph)
predictor, real speech **content never repeats the primary token past ~34 frames**;
only filler/silence tokens stretch (e.g. 706/668 to ~50), and the collapse
attractor (token 1318) runs 700–1800. Content and runaways are therefore cleanly
separable by run length. Three layered fixes (in the patch + server, all
engine-agnostic — they don't touch the CUDA kernel):

1. **Multiple-EOS detection** — the stock loop checks one stop id
   (`codec_eos_token_id`=2150); the talker also terminates via the "think" EOS
   (`codec_think_eos_id`=2157), which the suppress-mask was *blocking*. We
   un-suppress it and stop on either (`vocab_size`=3072, so text-space EOS ids are
   unreachable and omitted).
2. **Filler repeat-stop (`MK_REPEAT_STOP`=40)** — if the primary token repeats ≥40
   in a row, stop. 40 sits in the content(≤34)/filler(50–1800) gap, so it cuts the
   collapse *and* the multi-second drag without clipping real speech.
3. **Dynamic length cap** (`tts_server._dynamic_cap`) — size `max_new_tokens` to
   the reply text (~5 frames/word ×3 + margin) as a final catch-all so no reply can
   balloon, even if 1+2 miss.

Result: **no freezes, no 160-second runaways**; replies recover their natural
length. **Residual (honest):** on the full-megakernel config the predictor still
injects a *sub-threshold* filler burst on ~1-in-4 replies, which can sound garbled
— the drift overlaps real-content run-lengths, so no run-length threshold removes
it without clipping speech. The **cudagraph predictor is clean** (0 runaways in
50-utterance sweeps) at ~2–3× the RTF (still < 0.3). The project ships
full-megakernel for the strict RTF; `--predictor cudagraph` is the clean-audio
fallback — one flag.

## Repo layout

```
bot.py                   Pipecat voice agent (Mac client) — "Marcus", the freight broker
debug_taps.py            clean colorized terminal transcript + per-turn live metrics
remote_tts_service.py    Pipecat TTS client → streams from the 5090 server
run.sh                   launch the agent (colorized live view + ANSI-stripped log file)
setup.sh                 one-shot box provisioner (clone+patch kernel, deps, EOS patch)
server/tts_server.py     streaming TTS server (5090); --engine/--predictor cudagraph|megakernel; dynamic runaway cap
megakernel_talker.py     TalkerKernel + MegakernelTalkerGraph + correctness/e2e checks
megakernel_predictor.py  MegakernelPredictorGraph (predictor 15-step loop on the kernel)
benchmarks/bench_tts.py  TTFC/RTF/ms-step harness; --engine + --predictor cudagraph|megakernel
benchmarks/bench_codec_graph.py  proves the codec's ~16ms/call is launch overhead (graphed -> ~3ms)
patches/qwen_megakernel_talker.patch  kernel mods (input_hidden, skip_lm_head, barrier fix)
patches/faster_qwen3_tts_eos.patch    streaming EOS/runaway fix (multi-EOS + filler repeat-stop)
patches/README.md        how to apply both patches + what each changes
.env.example             HF_TOKEN (server) + optional ANTHROPIC_API_KEY (client) — copy to .env
```

## How to run

### Server (RTX 5090, sm_120 / Blackwell, CUDA ≥ 12.8)

Base image: CUDA 12.8 + cu128 PyTorch (e.g. `vastai/pytorch:cuda-12.8.1`). Verify
with `nvcc --version` (≥12.8) and `python -c "import torch;print(torch.cuda.get_device_capability())"` → `(12, 0)`.

**Shortcut:** copy this repo to the box and run **`./setup.sh`** — it does steps 1–2
below (clone+patch kernel, deps, EOS patch) idempotently. Or step through manually:

```bash
# 1. Build the patched megakernel
git clone https://github.com/AlpinDale/qwen_megakernel.git
cd qwen_megakernel && git checkout 5030e154d39ecd054df03eb4dd9c8aa8185414d1
git apply /path/to/patches/qwen_megakernel_talker.patch && cd ..

# 2. Install deps + patch the EOS/runaway fix into the installed faster-qwen3-tts
pip install qwen-tts faster-qwen3-tts fastapi "uvicorn[standard]"
export HF_TOKEN=hf_...     # free read-only token (avoids HF rate-limiting)
SITE=$(python -c "import faster_qwen3_tts,os;print(os.path.dirname(os.path.dirname(faster_qwen3_tts.__file__)))")
patch -p1 -d "$SITE" < /path/to/patches/faster_qwen3_tts_eos.patch

# 3. Brain: Ollama on the GPU (keyless)
curl -fsSL https://ollama.com/install.sh | sh && ollama serve &
ollama pull qwen2.5:7b-instruct

# 4. Start the streaming TTS server (both stages on the megakernel;
#    --predictor cudagraph = clean audio, --engine cudagraph = the baseline)
python server/tts_server.py --port 8000 --engine megakernel --predictor megakernel
```

### Client (Mac, Apple Silicon)

```bash
brew install portaudio uv ffmpeg
uv venv --python 3.12 && uv pip install "pipecat-ai[mlx-whisper,local]" python-dotenv
# No local LLM — the brain runs on the GPU (Ollama on the 5090). The Mac only
# does mic capture, Whisper STT, and speaker playback.

# Tunnel both GPU services (local 8009 -> box TTS 8000; local 11435 -> box Ollama
# 11434), then run the agent. 8009 avoids clashing with anything on local 8000;
# remote_tts_service.py connects to localhost:8009.
ssh -i ~/.ssh/vast_ai -p <PORT> -L 8009:localhost:8000 -L 11435:localhost:11434 root@<host>
./run.sh        # colorized transcript + live metrics; full session saved to logs/
```
(Non-Mac clients: swap `WhisperSTTServiceMLX` → `WhisperSTTService` (faster-whisper)
in `bot.py`; STT is the only Mac-specific piece.)

Talk, pause, and the agent (Marcus, a freight broker) replies in the Qwen3-TTS
voice. Headphones recommended. The terminal shows just the conversation:

```
────────────────────────────────────────────────────────────────
👤 You      I'm headed Phoenix to Denver.
🤖 Marcus   Nice lane, I can start you at a dollar fifty a mile, that work?
   ⚡ brain: 906 ms  (first token 905 ms)
   🔊 speech: GPU ttfc 46 ms · rtf 0.09    heard: ttfc 595 ms · rtf 0.23 · 5.6s audio
```

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
instrument the signals that map to call quality. The terminal is pruned to a
**clean colorized transcript** (`debug_taps.py`: 👤 you / 🤖 Marcus + per-turn
metrics; pipecat's DEBUG firehose is silenced); `run.sh` mirrors it to a
timestamped, ANSI-stripped log under `logs/`.

- **Live per-turn metrics, split by layer:** every reply prints **compute(GPU)**
  TTFC/RTF (measured server-side, no network — `tts_server.py /metrics`) *and*
  **end-to-end** TTFC/RTF incl. network (`remote_tts_service.py`). Seeing both
  side by side attributes latency to the right layer (measured demo run, both
  megakernels + codec graph, `chunk_size=2`: **~43 ms compute vs ~590 ms end-to-end
  ⇒ ~547 ms is network/geography over the SSH tunnel, not the model**).
- **What you'd monitor in production:**
  - **TTFC p50/p99** — the turn-taking latency a caller feels (alert if p99 climbs).
  - **RTF** — must stay < 1.0 or audio stutters (alert as it approaches 1.0).
  - **STT / LLM / TTS latency breakdown** — pinpoint which stage caused a slow turn.
  - **Dropped/late audio frames, GPU utilization, error rate** — health + capacity.
- **Offline rigor:** `benchmarks/bench_tts.py` reports median/p90/min/max over
  multiple runs and text lengths — the same discipline applied offline that the
  live metrics apply online.

Note: live end-to-end TTFC includes network round-trip (hundreds of ms over the
SSH tunnel); the **on-GPU compute figure** (64 ms at `chunk_size=4`, ~43 ms at the
`chunk_size=2` demo default with the codec graph) is what a co-located deployment
would see.

## What works, what's rough (honest)

**Works:** end-to-end real-time voice agent; the megakernel verifiably drives
**both** the talker (cosine 0.99978) and the code predictor (teacher-forced cosine
0.99926 min) — together ~60% faster than the CUDA-graph baseline (RTF 0.293 → 0.119);
streaming is true frame-by-frame (not buffered); reproducible kernel patch + a
no-recompile predictor reuse.

**Rough / known issues:**
- **Predictor-megakernel audio quality — the main rough edge (and the speed/quality
  knob).** On the full-megakernel config, ~1-in-4 replies have a sub-threshold
  filler burst that can sound garbled or 2–3× too long. This is the residual of the
  talker-runaway fix (see *Talker runaway & EOS robustness* above): the megakernel
  predictor's feedback drift over-sustains filler tokens, and that overlaps
  real-speech run-lengths, so the repeat-stop can't trim it further without clipping
  speech. The fixes (multi-EOS + repeat-stop + dynamic cap) guarantee **no freezes
  and no minutes-long runaways**, but they bound the symptom, not the cause. The
  **clean fallback is one flag** — `--predictor cudagraph` (talker still on the
  megakernel) is 0 runaways at RTF ~0.2–0.3 (still `<0.3`). So it's a deliberate
  trade: ship full-megakernel for the strict RTF and accept occasional garble, or
  cudagraph-predictor for clean audio. We ship full-megakernel; the flag is there.
  *(Earlier this looked like an intermittent "hang" — it was usually this runaway:
  the reply ballooned to ~160 s of audio with the mic muted, so it felt frozen even
  though compute finished in ~18 s. The dynamic cap + repeat-stop remove that.)*
- **Barrier re-arm race (fixed — stability hygiene).** The kernel's atomic grid
  barrier could spin forever: block 0 reset the counter *in-kernel* with no
  grid-wide ordering, so a competing `atomicAdd` got wiped and the counter never
  reached `num_blocks` (GPU 100%, no Xid — a live spin, not a fault). **Fix (in
  the kernel patch):** zero the barrier/flag buffers **host-side**
  (`cudaMemsetAsync`, before launch) and drop the racy in-kernel reset — fidelity
  intact (cosine 0.99926). Run benchmarks with the GPU exclusive (the megakernel
  wants all SMs).
- **Barge-in — implemented, off by default on one GPU.** The `/tts` endpoint
  cancels an in-flight reply within ~150 ms of a client disconnect, so it works.
  But the LLM (Ollama) and the megakernel share one GPU, so a barge-in collides a
  fresh LLM decode with the next kernel launch and the kernel can't claim all its
  SMs. `bot.py` ships `ALLOW_INTERRUPTIONS = False` (strict turn-taking). It's a
  single-GPU deployment artifact, not an architecture limit: point the brain at an
  API (`ANTHROPIC_API_KEY`) and the GPU is the kernel's alone — barge-in is then
  contention-free. Headline TTFC/RTF are measured during generation, identical
  either way.
- **TTFC over the network.** On-GPU TTFC is ~43–64 ms; end-to-end over the SSH
  tunnel is ~590 ms — network round-trip + per-request HTTP, not compute.
  Co-locating the client removes most of it; reported separately from the on-GPU
  figures.

**Next lever.** With the codec graphed, the remaining TTFC floor is the ~20 ms
PyTorch prefill (the talker's variable-length text forward, still eager) —
kernelizing it would push TTFC lower still. The chunk-size / codec tradeoff and all
the operating points are in *Headline results*.

## Credits

- Megakernel: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
- CUDA-graph streaming base: [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS](https://huggingface.co/Qwen) · [Pipecat](https://docs.pipecat.ai)
