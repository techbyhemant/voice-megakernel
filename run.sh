#!/usr/bin/env bash
# Launch the Phase 1 voice agent. Usage:  ./run.sh
cd "$(dirname "$0")"
exec ./.venv/bin/python bot.py
