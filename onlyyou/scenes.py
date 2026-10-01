"""Synthetic 'pendant days': wearer + loud friend (+ TV) + traffic/babble in a room.

Sound is rendered with pyroomacoustics at two pendant mics on the wearer's chest.
Body-vibration tracks come from the wearer's own Vibravox throat / forehead recordings,
plus leakage of external sound scaled from each sensor's measured signal-to-external-
noise ratio (Vibravox paper, Table 1). Throat (25.6 dB) is a well-coupled best case,
forehead accelerometer (-3.5 dB) a poorly-coupled worst case; a chest IMU is unknown
and assumed to sit somewhere between.
"""

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, sosfilt

from .audio import SR, frame_db, rms
from .vibration import add_gait

SENSOR_SXNR_DB = {"throat": 25.6, "forehead": -3.5}


@dataclass
class SceneConfig:
    max_s: float = 28.0
    mic_spacing: float = 0.025
    chest_height: float = 1.3
    mouth_offset: tuple = (0.05, 0.0, 0.15)
    friend_sir_db: tuple = (-6.0, 12.0)  # wearer vs friend at the top mic; <0: friend louder
    tv_prob: float = 0.5
    tv_sir_db: tuple = (0.0, 15.0)
    snr_db: tuple = (0.0, 15.0)
    walking_prob: float = 0.5


def _timeline(utts, rng, max_len, gap=(0.3, 2.0), start=0.0):
    track = np.zeros(max_len, np.float32)
    placed, t = [], int(start * SR)
    for u in utts:
        if t + len(u) > max_len:
            break
        track[t : t + len(u)] += u
        placed.append((t, t + len(u)))
        t += len(u) + int(rng.uniform(*gap) * SR)
    return track, placed


def _traffic(n, rng):
    """Brown-ish rumble plus a few car pass-bys (amplitude-modulated low-passed noise)."""
    base = np.cumsum(rng.normal(size=n)).astype(np.float32)
    base = sosfilt(butter(2, 40, "highpass", fs=SR, output="sos"), base)
    cars = sosfilt(butter(4, 1200, "lowpass", fs=SR, output="sos"), rng.normal(size=n))
    env = np.zeros(n)
    t = np.arange(n) / SR
    for _ in range(max(1, int(n / SR / 6))):
        c, w = rng.uniform(0, n / SR), rng.uniform(1.0, 3.0)
        env += np.exp(-0.5 * ((t - c) / w) ** 2)
    x = base / (rms(base) + 1e-9) + 2.0 * env * cars / (rms(cars) + 1e-9)
    return x.astype(np.float32)


def _babble(n, rng, pool):
    x = np.zeros(n, np.float32)
    for _ in range(6):
        u = pool[rng.integers(len(pool))]
        reps = np.tile(u, int(np.ceil(n / len(u))) + 1)
        off = rng.integers(len(u))
        x += reps[off : off + n] / (rms(u) + 1e-9)
    return x


def _scale_to(sig_at_mic, ref_at_mic, target_db, mask_sig, mask_ref):
    """Gain so that ref/sig level (over active regions) at the mic equals target_db."""
    p_ref = np.mean(ref_at_mic[mask_ref] ** 2) if mask_ref.any() else np.mean(ref_at_mic**2)
    p_sig = np.mean(sig_at_mic[mask_sig] ** 2) if mask_sig.any() else np.mean(sig_at_mic**2)
    return np.sqrt(p_ref / (p_sig + 1e-12) / 10 ** (target_db / 10))


def _active(track):
    db = frame_db(track)
    return db > max(db.max() - 35.0, -70.0)


def build(wearer, friend_utts, tv_utts, babble_pool, rng, cfg=SceneConfig()):
    """wearer: list of dicts with 'headset', 'throat', 'forehead' arrays (16 kHz).
    friend_utts / tv_utts: lists of headset arrays from other speakers."""
    max_len = int(cfg.max_s * SR)
    w_track, w_placed = _timeline([u["headset"] for u in wearer], rng, max_len)
    f_track, _ = _timeline(friend_utts, rng, max_len, start=rng.uniform(0.5, 3.0))
    n = max(p[1] for p in w_placed) + int(0.5 * SR)
    n = min(max(n, int(10 * SR)), max_len)
    w_track, f_track = w_track[:n], f_track[:n]
    body = {}
    for s in SENSOR_SXNR_DB:
        b = np.zeros(n, np.float32)
        for (a, e), u in zip(w_placed, wearer):
            seg = u[s][: min(len(u[s]), n - a)]
            b[a : a + len(seg)] += seg
        body[s] = b
    tv_on = rng.random() < cfg.tv_prob and len(tv_utts) > 0
    t_track = _timeline(tv_utts, rng, n, gap=(0.1, 0.4))[0] if tv_on else np.zeros(n, np.float32)

    room_dim = np.array([rng.uniform(4, 9), rng.uniform(3.5, 7), rng.uniform(2.6, 3.4)])
    rt60 = rng.uniform(0.2, 0.7)
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(absorption), max_order=min(max_order, 12))
    chest = np.array([rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), cfg.chest_height])
    half = np.array([0, 0, cfg.mic_spacing / 2])
    room.add_microphone_array(np.stack([chest + half, chest - half], axis=1))
    mouth = chest + np.array(cfg.mouth_offset)

    def place(dist):
        for _ in range(100):
            az = rng.uniform(-np.pi * 0.6, np.pi * 0.6)
            p = chest + np.array([dist * np.cos(az), dist * np.sin(az), 0.0])
            p[2] = rng.uniform(1.1, 1.7)
            if np.all(p[:2] > 0.3) and np.all(p[:2] < room_dim[:2] - 0.3):
                return p
        return np.clip(chest + np.array([0.8, 0.0, 0.2]), 0.3, room_dim - 0.3)

    friend_dist = rng.uniform(0.8, 3.0)
    sources = {"wearer": (mouth, w_track), "friend": (place(friend_dist), f_track)}
    if tv_on:
        sources["tv"] = (place(rng.uniform(2.0, 3.5)), t_track)
    noise_kind = "traffic" if rng.random() < 0.5 else "babble"
    for k in range(3):
        nz = _traffic(n, rng) if noise_kind == "traffic" else _babble(n, rng, babble_pool)
        corner = np.array([rng.choice([0.3, room_dim[0] - 0.3]), rng.choice([0.3, room_dim[1] - 0.3]), rng.uniform(0.5, 2.5)])
        sources[f"noise{k}"] = (corner, nz)

    at_mic = {}
    for name, (pos, sig) in sources.items():
        r = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(absorption), max_order=min(max_order, 12))
        r.add_microphone_array(room.mic_array.R)
        r.add_source(pos, signal=sig)
        r.simulate()
        at_mic[name] = r.mic_array.signals[:, :n].astype(np.float32)

    w_act, f_act, t_act = _active(w_track), _active(f_track), _active(t_track) if tv_on else None
    from .audio import HOP

    def mask(act):
        return np.repeat(act, HOP)[:n] if act is not None else np.zeros(n, bool)

    w_m = mask(w_act)
    sir_f = rng.uniform(*cfg.friend_sir_db)
    at_mic["friend"] *= _scale_to(at_mic["friend"][0], at_mic["wearer"][0], sir_f, mask(f_act), w_m)
    if tv_on:
        sir_t = rng.uniform(*cfg.tv_sir_db)
        at_mic["tv"] *= _scale_to(at_mic["tv"][0], at_mic["wearer"][0], sir_t, mask(t_act), w_m)
    noise = sum(v for k, v in at_mic.items() if k.startswith("noise"))
    snr = rng.uniform(*cfg.snr_db)
    noise *= _scale_to(noise[0], at_mic["wearer"][0], snr, np.ones(n, bool), w_m)
    external = at_mic["friend"] + noise + (at_mic["tv"] if tv_on else 0)
    mics = at_mic["wearer"] + external
    peak = np.abs(mics).max() + 1e-9
    mics, external, wearer_at_mic = mics / peak * 0.9, external / peak * 0.9, at_mic["wearer"] / peak * 0.9

    walking = rng.random() < cfg.walking_prob
    for s, sxnr in SENSOR_SXNR_DB.items():
        g = rms(body[s][w_m]) / (rms(wearer_at_mic[0][w_m]) + 1e-9) * 10 ** (-sxnr / 20)
        b = body[s] + g * external[0]
        if walking:
            b = add_gait(b, rng)
        body[s] = b.astype(np.float32)

    nf = len(w_act)
    others = f_act[:nf] | (t_act[:nf] if tv_on else False)
    return {
        "mics": mics.astype(np.float32),
        "wearer_ref": wearer_at_mic[0].astype(np.float32),
        "throat": body["throat"], "forehead": body["forehead"],
        "wearer_active": w_act[:nf], "others_active": others,
        "meta": {"friend_sir_db": float(sir_f), "tv": bool(tv_on), "snr_db": float(snr),
                 "noise": noise_kind, "walking": bool(walking), "rt60": float(rt60),
                 "friend_dist_m": float(friend_dist)},
    }
