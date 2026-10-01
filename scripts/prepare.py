"""Build synthetic pendant scenes from Vibravox and cache per-frame base features.

Train and test scenes use disjoint speakers (Vibravox's own speaker-disjoint splits).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onlyyou import proximity, scenes, vad, voiceprint  # noqa: E402
from onlyyou.audio import frame_db, load  # noqa: E402
from onlyyou.enhance import hush  # noqa: E402

DATA = ROOT / "data"
VIB = DATA / "vibravox"


def load_index():
    by = defaultdict(lambda: defaultdict(list))
    for line in (VIB / "index.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[r["split"]][r["speaker"]].append(r)
    return by


def utt(r, channels=("headset", "throat", "forehead")):
    return {c: load(VIB / f"{r['path']}_{c}.flac") for c in channels}


def make_scenes(split, n_scenes, seed):
    rng = np.random.default_rng(seed)
    speakers = load_index()[split]
    wearers = [s for s, rs in speakers.items() if len(rs) >= 4]
    all_rows = [r for rs in speakers.values() for r in rs]
    out = DATA / "scenes" / split
    out.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n_scenes), desc=f"scenes/{split}"):
        path = out / f"{i:03d}.npz"
        if path.exists():
            continue
        w = wearers[i % len(wearers)]
        rows = list(speakers[w])
        rng.shuffle(rows)
        enroll_rows, scene_rows = rows[:2], rows[2:8]
        others = [s for s in speakers if s != w]
        f_spk, t_spk = rng.choice(others, 2, replace=False)
        f_rows = list(rng.permutation(speakers[f_spk]))[:6]
        t_rows = list(rng.permutation(speakers[t_spk]))[:8]
        pool_rows = [all_rows[j] for j in rng.choice(len(all_rows), 12, replace=False) if all_rows[j]["speaker"] != w]
        sc = scenes.build(
            [utt(r) for r in scene_rows],
            [utt(r, ("headset",))["headset"] for r in f_rows],
            [utt(r, ("headset",))["headset"] for r in t_rows],
            [utt(r, ("headset",))["headset"] for r in pool_rows],
            rng,
            enroll_utts=[utt(r, ("headset",))["headset"] for r in enroll_rows],
        )
        m = sc["meta"]
        m.update(wearer=w, friend=str(f_spk), tv_speaker=str(t_spk),
                 wearer_texts=[r["text"] for r in scene_rows[: m["n_wearer"]]],
                 other_texts=[r["text"] for r in f_rows[: m["n_friend"]]] + [r["text"] for r in t_rows[: m["n_tv"]]])
        np.savez_compressed(path, **{k: v for k, v in sc.items() if k != "meta"}, meta=json.dumps(m))


def base_features(split):
    src, out = DATA / "scenes" / split, DATA / "features" / split
    out.mkdir(parents=True, exist_ok=True)
    lag = proximity.wearer_lag(scenes.SceneConfig().mic_spacing)
    for p in tqdm(sorted(src.glob("*.npz")), desc=f"features/{split}"):
        dst = out / p.name
        if dst.exists():
            continue
        s = np.load(p)
        top = s["mics"][0]
        enhanced = hush(top)
        profile = voiceprint.enroll([s["enroll"]])
        n = len(s["wearer_active"])
        feats = {
            "mic_db": frame_db(top)[:n],
            "vad_raw": vad.speech_prob(top)[:n],
            "vad_hush": vad.speech_prob(enhanced)[:n],
            "vp_raw": voiceprint.score_track(top, profile)[:n],
            "vp_hush": voiceprint.score_track(enhanced, profile)[:n],
            "prox": proximity.features(s["mics"], lag)[:n],
        }
        np.savez_compressed(dst, **feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--test", type=int, default=40)
    args = ap.parse_args()
    make_scenes("train", args.train, seed=1)
    make_scenes("test", args.test, seed=2)
    base_features("train")
    base_features("test")


if __name__ == "__main__":
    main()
