# Voice Megakernel — RTX 5090 Decode Megakernel → Qwen3-TTS on Pipecat

Run [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
(a hand-written CUDA megakernel for Qwen3-0.6B) as the **talker decode backend**
for **Qwen3-TTS**, streaming real-time speech into a **Pipecat** voice agent.

The voice loop runs **mic → Whisper (STT) → local LLM → Qwen3-TTS → speaker**.
Only the TTS is custom: its talker (the 28-layer autoregressive backbone) is
driven by the megakernel; the rest of Qwen3-TTS (code predictor + codec) stays
in PyTorch.

## Headline results

Qwen3-TTS `12Hz-0.6B-CustomVoice`, streaming, single RTX 5090, `chunk_size=4`,
`ryan` voice. RTF = compute / audio (lower is better; 1.0 = real-time).

| Talker engine | ms/step | RTF | TTFC |
|---|---|---|---|
| Naive eager PyTorch (baseline) | — | **1.48** | n/a (buffered) |
| CUDA-graph (faster-qwen3-tts) | 21.05 | 0.253 | 107 ms |
| **Megakernel (this project)** | **15.07** | **0.181** | **83.7 ms** |

- The megakernel is **~28% faster per step / on RTF** and **22% lower TTFC**
  than an already-CUDA-graph-optimized talker — and **beats the < 90 ms TTFC
  target** box-local.
- Unmodified megakernel decode (Qwen3-0.6B chat, sanity check): **1050 tok/s**.
- Talker output matches the PyTorch talker at **cosine 0.9997** (single- and
  multi-step, see `megakernel_talker.py`).

## Where the time goes (bottleneck analysis)

Profiling the naive PyTorch path (per decode step) showed the surprising split:

| Component | Share of decode time |
|---|---|
| Codec (tokens → waveform) | **0.3%** (free) |
| **Talker backbone (28L)** — *megakernel target* | 22% |
| **Code predictor (5L, 15-step inner loop/frame)** | **78%** |

Key insight: the megakernel only accelerates the **backbone (22%)**. Hitting
real-time requires the **code predictor** to also be fast — which CUDA graphs
(from faster-qwen3-tts) provide. The megakernel then replaces the CUDA-graph
backbone and wins a further 28%. So the win is: **CUDA-graph the predictor +
megakernel the talker.**

## Architecture

```
   YOUR MAC (mic + speaker)                 RENTED RTX 5090 (TTS server)
┌──────────────────────────────┐        ┌────────────────────────────────────┐
│ mic → Whisper(MLX) → LLM      │  text  │  POST /tts → streaming PCM           │
│        (Ollama)               │ ─────► │   faster-qwen3-tts generate loop:    │
│            │                  │        │     prefill (PyTorch)                │
│            ▼  RemoteQwenTTS    │        │     loop: talker step ──► MEGAKERNEL │
│ speaker ◄── audio chunks ◄─────────────│           code predictor (CUDA graph)│
└──────────────────────────────┘  PCM   │           codec → audio chunk        │
        (over SSH tunnel)                └────────────────────────────────────┘
```

- **STT / LLM** run on the Mac (off-the-shelf). The LLM is any local Ollama model.
- **TTS** is a streaming server on the 5090 (brief's Step 2); the Mac connects
  over an SSH tunnel (brief's Step 3).
- Inside the TTS, the **talker's per-step decode** is the megakernel; the **code
  predictor** is a captured CUDA graph; the **codec** is PyTorch.

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

KV-cache prefill is copied from the PyTorch prefill into the kernel's cache
(`TalkerKernel.prefill_from_cache`); convention verified by the multi-step check.

## Repo layout

```
bot.py                 Pipecat voice agent (Mac client)
remote_tts_service.py  Pipecat TTS client → streams from the 5090 server
run.sh                 launch the agent
server/tts_server.py   streaming TTS server (5090); --engine cudagraph|megakernel
megakernel_talker.py   TalkerKernel + MegakernelTalkerGraph + correctness/e2e checks
benchmarks/bench_tts.py  TTFC/RTF/ms-step harness; --engine cudagraph|megakernel
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

# 4. Start the streaming TTS server (cudagraph = reliable; megakernel = faster)
python server/tts_server.py --port 8000 --engine cudagraph
```

### Client (Mac, Apple Silicon)

```bash
brew install portaudio uv ffmpeg
uv venv --python 3.12 && uv pip install "pipecat-ai[mlx-whisper,local]" python-dotenv
# Ollama for the local LLM brain:
ollama pull llama3.2:3b     # or any small model; set LLM_MODEL in bot.py

# Tunnel the server port, then run the agent:
ssh -i ~/.ssh/vast_ai -p <PORT> -L 8000:localhost:8000 root@<host>   # keep open
./run.sh
```

Talk, pause, and the agent replies in the Qwen3-TTS voice.

## Benchmarking

```bash
# on the 5090 (exclusive GPU — stop the server first)
python benchmarks/bench_tts.py --engine cudagraph --runs 7
python benchmarks/bench_tts.py --engine megakernel --runs 7
```
Multi-run, CUDA-synced, reports median/p90/min/max for TTFC, RTF, ms/step across
short/medium/long texts. The `chunk_size` knob trades TTFC vs RTF (chunk 4:
TTFC 107 ms / RTF 0.25; chunk 12: TTFC 232 ms / RTF 0.22 on the CUDA-graph path).

## What works, what's rough (honest)

**Works:** end-to-end real-time voice agent; megakernel verifiably drives the
talker (cosine 0.9997) and is 28% faster than the CUDA-graph talker; streaming
is true frame-by-frame (not buffered); reproducible kernel patch.

**Rough / known issues:**
- **Megakernel ⇄ predictor-CUDA-graph coexistence:** the megakernel
  occasionally grid-sync **deadlocks** (GPU pegged, no progress) when co-running
  with the predictor's CUDA graph in the streaming-server warmup. It runs fine
  standalone (benchmark + e2e), so the **live server defaults to `--engine
  cudagraph`** for reliability; the megakernel win is established via the
  benchmark and standalone e2e. Fix (future work): give the megakernel a
  dedicated CUDA stream, or replace the predictor CUDA graph.
- **TTFC over the network:** box-local TTFC is 84–107 ms, but end-to-end over the
  SSH tunnel is ~700 ms — dominated by network round-trip + per-request HTTP
  setup, not compute. A persistent connection / co-locating the client would
  remove most of it. Reported separately from the on-GPU numbers.
- **TTFC vs target:** 83.7 ms (megakernel, box-local) is under the 90 ms goal but
  above the 50 ms stretch goal; smaller chunks lower it further at some RTF cost.

## Credits

- Megakernel: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
- CUDA-graph streaming base: [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS](https://huggingface.co/Qwen) · [Pipecat](https://docs.pipecat.ai)
