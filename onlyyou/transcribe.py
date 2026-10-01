from functools import lru_cache

import numpy as np

from .audio import SR


@lru_cache(maxsize=2)
def _model(size):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def _regions(x, min_gap_s=0.4, min_len_s=0.3, pad_s=0.1):
    """Non-silent regions of gated audio (gating leaves exact zeros between kept parts)."""
    active = np.abs(x) > 1e-6
    if active.all():
        return [(0, len(x))]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    runs, gap, pad = [], int(min_gap_s * SR), int(pad_s * SR)
    for s, e in zip(edges[::2], edges[1::2]):
        if runs and s - runs[-1][1] < gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return [(max(0, s - pad), min(len(x), e + pad)) for s, e in runs if e - s >= min_len_s * SR]


def transcribe(x: np.ndarray, language=None, size="small"):
    """[(start_s, end_s, text)] for 16 kHz mono audio, CPU int8.

    Gated audio is transcribed region by region: Whisper hallucinates on long stretches
    of digital silence, and a pendant would only upload the kept regions anyway.
    """
    x = np.asarray(x, np.float32)
    if len(x) == 0 or np.abs(x).max() < 1e-4:
        return []
    out = []
    for a, b in _regions(x):
        segments, _ = _model(size).transcribe(
            x[a:b], language=language, beam_size=1, vad_filter=False, condition_on_previous_text=False,
        )
        out += [(a / SR + s.start, a / SR + s.end, s.text.strip()) for s in segments
                if s.text.strip().strip(".… ")]
    return out
