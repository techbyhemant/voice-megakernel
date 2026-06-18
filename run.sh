#!/usr/bin/env bash
# Launch the freight-negotiation voice agent. Usage:  ./run.sh
# Requires the SSH tunnel open first (TTS 8000, LLM 11435->11434).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1   # models are cached; skip per-turn HF cache checks (quiet logs)

# Capture the FULL session (Whisper STT taps, Ollama LLM taps, TTS metrics,
# pipecat/loguru, any errors) to a timestamped log so a failed live call can be
# analyzed afterward. PYTHONUNBUFFERED so no lines are lost if it crashes.
mkdir -p logs
LOG="logs/bot_$(date +%Y%m%d_%H%M%S).log"
echo "📒 Logging this session to $LOG"
# FORCE_COLOR keeps the colored transcript even though we pipe through tee.
# tee shows the colored output live; the log file gets the ANSI codes stripped
# (via perl) so pasted logs stay clean and readable.
FORCE_COLOR=1 PYTHONUNBUFFERED=1 ./.venv/bin/python bot.py 2>&1 \
  | tee >(perl -pe 's/\e\[[0-9;]*m//g' > "$LOG")
