from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _model(size):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(x: np.ndarray, language=None, size="small"):
    """[(start_s, end_s, text)] for 16 kHz mono audio. CPU int8."""
    if len(x) == 0 or np.abs(x).max() < 1e-4:
        return []
    segments, _ = _model(size).transcribe(
        np.asarray(x, np.float32), language=language, beam_size=1,
        vad_filter=False, condition_on_previous_text=False,
    )
    return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
