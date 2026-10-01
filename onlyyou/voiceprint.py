"""Wearer voiceprint with SpeechBrain's pretrained ECAPA-TDNN (CPU inference only)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .audio import HOP, SR

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _model():
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "models" / "ecapa"),
        run_opts={"device": "cpu"},
    )


@torch.no_grad()
def embed(chunks: list[np.ndarray]) -> np.ndarray:
    """L2-normalised embeddings for equal-length chunks, shape [N, 192]."""
    if not chunks:
        return np.zeros((0, 192), np.float32)
    out = []
    for i in range(0, len(chunks), 32):
        batch = torch.from_numpy(np.stack(chunks[i : i + 32]).astype(np.float32))
        e = _model().encode_batch(batch).squeeze(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out)


def enroll(waves: list[np.ndarray], win_s=3.0) -> np.ndarray:
    """Average embedding over fixed windows cut from the wearer's enrollment audio."""
    win = int(win_s * SR)
    chunks = []
    for w in waves:
        if len(w) < win:
            w = np.pad(w, (0, win - len(w)))
        chunks += [w[s : s + win] for s in range(0, len(w) - win + 1, win // 2)]
    e = embed(chunks).mean(0)
    return e / np.linalg.norm(e)


def score_track(x: np.ndarray, profile: np.ndarray, win_s=1.5, hop_s=0.5) -> np.ndarray:
    """Cosine similarity to the profile, one value per HOP frame.

    Each frame takes the score of the most recent window that ends at or after it,
    so the score is available within win_s of real time.
    """
    win, hop = int(win_s * SR), int(hop_s * SR)
    n_frames = len(x) // HOP
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    starts = list(range(0, len(x) - win + 1, hop))
    scores = embed([x[s : s + win] for s in starts]) @ profile
    centers = np.array([(s + win / 2) / HOP for s in starts])
    frames = np.arange(n_frames)
    return np.interp(frames, centers, scores).astype(np.float32)
