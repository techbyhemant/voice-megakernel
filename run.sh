#!/usr/bin/env bash
# Launch the freight-negotiation voice agent. Usage:  ./run.sh
# Requires the SSH tunnel open first (TTS 8000, LLM 11435->11434).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1   # models are cached; skip per-turn HF cache checks (quiet logs)
exec ./.venv/bin/python bot.py
