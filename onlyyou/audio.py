from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
HOP = 512  # 32 ms frames; matches Silero VAD's window at 16 kHz
FPS = SR / HOP


def load(path, sr=SR):
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample(x, file_sr, sr)


def save(path, x, sr=SR):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def n_frames(n_samples):
    return n_samples // HOP


def frame_view(x, hop=HOP):
    n = len(x) // hop
    return x[: n * hop].reshape(n, hop)


def frame_db(x, hop=HOP, eps=1e-10):
    return 10 * np.log10(np.mean(frame_view(x, hop) ** 2, axis=1) + eps)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
