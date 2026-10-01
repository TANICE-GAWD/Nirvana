from functools import lru_cache

import numpy as np
import torch

from .audio import HOP, SR


@lru_cache(maxsize=1)
def _model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


@torch.no_grad()
def speech_prob(x: np.ndarray) -> np.ndarray:
    """Per-frame (HOP=512 samples) speech probability from Silero VAD."""
    model = _model()
    model.reset_states()
    n = len(x) // HOP
    frames = torch.from_numpy(np.asarray(x[: n * HOP], np.float32)).reshape(n, HOP)
    return np.array([float(model(f, SR)) for f in frames], dtype=np.float32)
