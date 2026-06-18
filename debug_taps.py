"""
debug_taps.py — clean, colorized conversation view for the voice agent.

Pipecat's own DEBUG logging is silenced in bot.py; these pass-through taps print
the only thing worth watching live: what the user said, what Marcus replied, and
the per-turn latency/quality metrics — color-coded and easy to scan. A DebugTap
forwards every frame unchanged, so it never alters pipeline behavior.

Place one after STT (label="STT") for the user transcript, and one after the LLM
(label="LLM") for Marcus's reply + LLM timing. The TTS service prints its own
metrics line (it imports the color helpers from here).
"""
import os
import sys
import time

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

# ANSI colors — on for a real terminal, or when FORCE_COLOR=1 (run.sh sets this
# because it pipes through `tee`, which would otherwise hide the color).
_TTY = sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def USER(s):  return _c("96;1", s)   # bright cyan, bold  — the carrier/driver
def BOT(s):   return _c("92;1", s)   # bright green, bold — Marcus
def DIM(s):   return _c("2", s)      # dim — metrics
def ERR(s):   return _c("91;1", s)   # bright red, bold  — errors
def RULE(s):  return _c("90", s)     # grey — turn separator


class DebugTap(FrameProcessor):
    """Logs the user transcript / Marcus reply / LLM timing, then forwards."""

    def __init__(self, label: str, **kwargs):
        super().__init__(name=f"DebugTap[{label}]", **kwargs)
        self._label = label
        self._llm_start = None
        self._llm_first = None
        self._buf = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        try:
            self._show(frame)
        except Exception:
            pass  # never let the view break the pipeline
        await self.push_frame(frame, direction)

    def _show(self, frame):
        # --- user said (after STT) ---
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                print(f"\n{RULE('─' * 64)}")
                print(f"{USER('👤 You    ')}  {text}", flush=True)

        # --- Marcus replied (after LLM) ---
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_start = time.perf_counter()
            self._llm_first = None
            self._buf = []
        elif isinstance(frame, LLMTextFrame):
            if self._llm_first is None:
                self._llm_first = time.perf_counter()
            self._buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            reply = "".join(self._buf).strip()
            if reply:
                ttft = (self._llm_first - self._llm_start) * 1000 if (self._llm_first and self._llm_start) else 0.0
                total = (time.perf_counter() - self._llm_start) * 1000 if self._llm_start else 0.0
                print(f"{BOT('🤖 Marcus ')}  {reply}", flush=True)
                print(DIM(f"   ⚡ brain: {total:.0f} ms  (first token {ttft:.0f} ms)"), flush=True)

        # --- errors (rare, but make them loud) ---
        elif isinstance(frame, ErrorFrame):
            print(ERR(f"   ❌ {getattr(frame, 'error', frame)}"), flush=True)
