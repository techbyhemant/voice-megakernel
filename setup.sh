#!/usr/bin/env bash
# setup.sh — provision a fresh RTX 5090 (sm_120 / Blackwell, CUDA ≥ 12.8) as the
# Qwen3-TTS megakernel server. Run from the repo root ON THE GPU BOX.
#
# Idempotent: safe to re-run — it skips a patch that's already applied. Mirrors the
# exact sequence used to provision the benchmark box. After it finishes, export
# HF_TOKEN and launch the server (printed at the end). The live agent additionally
# needs Ollama (commented at the bottom); benchmarking does not.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
KREV=5030e154d39ecd054df03eb4dd9c8aa8185414d1
PY=${PY:-/venv/main/bin/python}
PIP=${PIP:-/venv/main/bin/pip}

echo "[1/3] megakernel: clone @ ${KREV:0:10} + apply talker patch"
[ -d qwen_megakernel ] || git clone https://github.com/AlpinDale/qwen_megakernel.git
( cd qwen_megakernel
  git checkout -q "$KREV"
  if git apply --check "$REPO/patches/qwen_megakernel_talker.patch" 2>/dev/null; then
    git apply "$REPO/patches/qwen_megakernel_talker.patch"; echo "    kernel patch applied"
  else
    echo "    kernel patch already applied — skipping"
  fi )

echo "[2/3] install deps + apply the EOS/runaway patch to faster_qwen3_tts"
$PIP install -q qwen-tts faster-qwen3-tts fastapi "uvicorn[standard]" librosa
SITE=$($PY -c "import faster_qwen3_tts,os;print(os.path.dirname(os.path.dirname(faster_qwen3_tts.__file__)))")
if grep -q "eos_ids" "$SITE/faster_qwen3_tts/streaming.py"; then
  echo "    EOS patch already applied — skipping"
else
  patch -p1 -d "$SITE" < "$REPO/patches/faster_qwen3_tts_eos.patch"; echo "    EOS patch applied"
fi

echo "[3/3] done."
echo "Set HF_TOKEN, then start the streaming TTS server:"
echo "  export HF_TOKEN=hf_...   # free read-only token"
echo "  $PY $REPO/server/tts_server.py --port 8000 --engine megakernel --predictor megakernel --compile-codec on"
echo
echo "Live agent only (not needed for benchmarks) — Ollama brain on the GPU:"
echo "  curl -fsSL https://ollama.com/install.sh | sh && ollama serve & ollama pull qwen2.5:7b-instruct"
