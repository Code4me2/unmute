"""Programmatic audio cue generation for tool call events.

Generates short synthesized pings at 24kHz float32 to inject directly into the
audio output queue, bypassing TTS.
"""

import numpy as np

from unmute.kyutai_constants import SAMPLE_RATE


def _envelope(
    t: np.ndarray, attack: float = 0.01, decay_rate: float = 12.0
) -> np.ndarray:
    """Smooth attack ramp + exponential decay."""
    attack_samples = max(1, int(attack * SAMPLE_RATE))
    env = np.exp(-t * decay_rate)
    ramp = np.linspace(0.0, 1.0, min(attack_samples, len(env)), dtype=np.float32)
    env[: len(ramp)] *= ramp
    return env


def generate_tool_call_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Rising two-tone chirp (~150ms). 'Something is happening.'"""
    dur = 0.15
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)

    tone1 = np.sin(2 * np.pi * 660 * t) * _envelope(t, decay_rate=15)

    mid = n // 2
    tone2 = np.zeros(n, dtype=np.float32)
    t2 = np.linspace(0, (n - mid) / sr, n - mid, dtype=np.float32)
    tone2[mid:] = np.sin(2 * np.pi * 880 * t2) * _envelope(t2, decay_rate=15)

    return (0.4 * (tone1 + tone2)).astype(np.float32)


def generate_agent_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ascending C-major arpeggio (~250ms). 'Spawning a sub-agent.'"""
    dur = 0.25
    freqs = [523.25, 659.25, 783.99]  # C5, E5, G5
    total = int(sr * dur)
    step = total // len(freqs)
    samples = np.zeros(total, dtype=np.float32)

    for i, freq in enumerate(freqs):
        start = i * step
        end = min(start + step, total)
        seg_len = end - start
        seg_t = np.linspace(0, seg_len / sr, seg_len, dtype=np.float32)
        samples[start:end] = np.sin(2 * np.pi * freq * seg_t) * _envelope(
            seg_t, decay_rate=10
        )

    return (0.4 * samples).astype(np.float32)


def generate_error_ping(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Descending tone (~150ms). 'Something went wrong.'"""
    dur = 0.15
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    freq = 440 * np.exp(-t * 3)  # descending frequency sweep
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return (0.35 * np.sin(phase) * _envelope(t)).astype(np.float32)


# Module-level singletons — generated once at import time
PING_TOOL_CALL: np.ndarray = generate_tool_call_ping()
PING_AGENT: np.ndarray = generate_agent_ping()
PING_ERROR: np.ndarray = generate_error_ping()
