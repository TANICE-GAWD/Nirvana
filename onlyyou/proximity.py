"""Two-mic proximity cues for a pendant whose mics are a couple of cm apart.

The wearer's mouth is ~15 cm above the pendant, so their voice arrives louder at the
top mic, from a fixed direction, and with mostly direct (coherent) sound. Other talkers
are metres away: nearly equal level at both mics, varying direction, more reverberant.
"""

import numpy as np

from .audio import HOP, SR, frame_view

NFFT = 1024
BAND = (300, 3500)


def features(mics: np.ndarray, expected_lag: float, upsample=16, smooth=0.7) -> np.ndarray:
    """Per-frame [top-mic level dB, inter-mic level difference dB,
    |TDOA - wearer TDOA| (samples), GCC-PHAT peak height, band coherence]."""
    top, bot = mics[0], mics[1]
    n = len(top) // HOP
    win = np.hanning(NFFT)
    pad = NFFT - HOP
    t_pad = np.concatenate([np.zeros(pad, np.float32), top])
    b_pad = np.concatenate([np.zeros(pad, np.float32), bot])
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    level = 10 * np.log10(np.mean(frame_view(top) ** 2, axis=1) + 1e-10)
    level_b = 10 * np.log10(np.mean(frame_view(bot) ** 2, axis=1) + 1e-10)
    s11 = s22 = s12 = None
    out = np.zeros((n, 5), np.float32)
    max_lag = 4 * upsample
    for i in range(n):
        a = np.fft.rfft(t_pad[i * HOP : i * HOP + NFFT] * win)
        b = np.fft.rfft(b_pad[i * HOP : i * HOP + NFFT] * win)
        p11, p22, p12 = np.abs(a) ** 2, np.abs(b) ** 2, a * np.conj(b)
        if s11 is None:
            s11, s22, s12 = p11, p22, p12
        else:
            s11 = smooth * s11 + (1 - smooth) * p11
            s22 = smooth * s22 + (1 - smooth) * p22
            s12 = smooth * s12 + (1 - smooth) * p12
        coh = np.abs(s12[band]) ** 2 / (s11[band] * s22[band] + 1e-12)
        phat = np.where(band, s12 / (np.abs(s12) + 1e-12), 0)
        cc = np.fft.irfft(phat, NFFT * upsample)
        cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
        k = int(np.argmax(cc))
        lag = (k - max_lag) / upsample
        out[i] = [level[i], level[i] - level_b[i], abs(lag - expected_lag), cc[k] * upsample, coh.mean()]
    return out


def wearer_lag(spacing_m: float, mouth_dist_m: float = 0.15, c=343.0) -> float:
    """TDOA (samples, top minus bottom) for a mouth directly above a vertical mic pair."""
    return -(spacing_m / c) * SR * (mouth_dist_m / np.hypot(mouth_dist_m, 0.05))
