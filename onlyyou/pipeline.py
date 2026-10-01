"""Apply a frame-level gate to audio, and the single-mic gate used for live / recorded demos.

A laptop or earbud recording has one mic and no body-vibration sensor, so the demo gate
uses only Hush + voice activity + voiceprint. The pendant gate (gate.py) adds the
vibration and two-mic cues on top of these.
"""

import numpy as np

from . import vad, voiceprint
from .audio import HOP, SR
from .enhance import hush


def apply_gate(x, decisions, hangover=3, ramp_ms=20):
    """Zero audio outside accepted frames, keeping `hangover` frames after each one."""
    d = np.asarray(decisions, bool).copy()
    for k in range(1, hangover + 1):
        d[k:] |= decisions[:-k]
    mask = np.repeat(d.astype(np.float32), HOP)
    mask = np.pad(mask, (0, max(0, len(x) - len(mask))))[: len(x)]
    r = int(ramp_ms * SR / 1000)
    if r > 1:
        mask = np.convolve(mask, np.ones(r) / r, mode="same")
    return (x * mask).astype(np.float32)


def calibrate_threshold(enroll_audio, profile):
    """Threshold from the wearer's own enrollment scores (scores of other people sit far lower)."""
    s = voiceprint.score_track(enroll_audio, profile)
    speech = vad.speech_prob(enroll_audio)[: len(s)] > 0.5
    own = s[speech] if speech.any() else s
    return float(max(0.2, np.percentile(own, 10) * 0.6))


def single_mic(x, profile, threshold):
    """Returns dict with raw / Hush / Only You audio and per-frame cues."""
    enhanced = hush(x)
    v = vad.speech_prob(enhanced)
    vp = voiceprint.score_track(enhanced, profile)
    n = min(len(v), len(vp))
    decisions = (v[:n] > 0.5) & (vp[:n] >= threshold)
    return {"raw": x, "hush": enhanced, "only_you": apply_gate(enhanced, decisions),
            "vad": v[:n], "voiceprint": vp[:n], "decisions": decisions}
