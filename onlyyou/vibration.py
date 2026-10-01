"""Body-vibration channel: IMU emulation and per-frame features.

A pendant IMU sits on the sternum. When the wearer talks, chest vibration reaches it;
other people's voices mostly do not. We emulate the IMU from Vibravox body-conduction
recordings by reducing the sample rate, optionally without an anti-aliasing filter
(many MEMS IMUs let you disable the digital low-pass filter, which folds voice energy
into the low band instead of removing it), adding walking motion and a noise floor.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from .audio import HOP, SR


def add_gait(x: np.ndarray, rng: np.random.Generator, cadence_hz=None, rel_db=10.0) -> np.ndarray:
    """Walking motion: a ~2 Hz step fundamental with harmonics, louder than speech vibration."""
    cadence = cadence_hz or rng.uniform(1.6, 2.2)
    t = np.arange(len(x)) / SR
    gait = sum((1 / k) * np.sin(2 * np.pi * k * cadence * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 8))
    speech_rms = np.sqrt(np.mean(x**2)) + 1e-9
    gait *= speech_rms * 10 ** (rel_db / 20) / (np.sqrt(np.mean(gait**2)) + 1e-12)
    return (x + gait).astype(np.float32)


def emulate_imu(x: np.ndarray, rate: int, aliased: bool, rng: np.random.Generator,
                noise_db=-50.0) -> np.ndarray:
    """Return the body signal as an IMU sampled at `rate` would see it, at `rate` Hz."""
    if rate >= SR:
        y = x.copy()
    elif aliased:
        y = x[:: SR // rate].copy()
    else:
        from .audio import resample

        y = resample(x, SR, rate)
    floor = np.sqrt(np.mean(y**2)) * 10 ** (noise_db / 20)
    return (y + rng.normal(0, floor, len(y))).astype(np.float32)


def features(imu: np.ndarray, rate: int, mic_db: np.ndarray) -> np.ndarray:
    """Per-HOP-frame features from an IMU stream sampled at `rate`.

    - band energy above gait frequencies, in dB above its own running noise floor
    - body energy minus mic energy (high only when the sound originates in the body)
    - short-window correlation between body and mic energy envelopes
    """
    n = len(mic_db)
    cutoff = min(25.0, 0.4 * rate / 2)
    sos = butter(4, cutoff, btype="highpass", fs=rate, output="sos")
    hp = sosfilt(sos, imu)
    # energy per HOP-length frame at the IMU rate (at least one sample per frame)
    t_edges = np.arange(n + 1) * HOP / SR
    idx = np.minimum((t_edges * rate).astype(int), len(hp))
    e = np.array([np.mean(hp[a:max(b, a + 1)] ** 2) if a < len(hp) else 0.0
                  for a, b in zip(idx[:-1], idx[1:])])
    body_db = 10 * np.log10(e + 1e-12)
    floor = _running_percentile(body_db, 10, win=int(4 * SR / HOP))
    rel = body_db - floor
    diff = body_db - mic_db
    diff = diff - np.median(diff)
    corr = _running_corr(body_db, mic_db, win=7)
    return np.stack([rel, diff, corr], axis=1).astype(np.float32)


def _running_percentile(x, q, win):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.percentile(x[max(0, i - win) : i + 1], q)
    return out


def _running_corr(a, b, win):
    out = np.zeros(len(a), np.float32)
    for i in range(len(a)):
        sa, sb = a[max(0, i - win + 1) : i + 1], b[max(0, i - win + 1) : i + 1]
        if len(sa) > 2 and sa.std() > 1e-6 and sb.std() > 1e-6:
            out[i] = np.corrcoef(sa, sb)[0, 1]
    return out
