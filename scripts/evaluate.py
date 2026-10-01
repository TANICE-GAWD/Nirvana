"""Leakage table and IMU sample-rate sweep.

Every gate is calibrated on training scenes to keep 90% of the wearer's speech, then
measured on test scenes with different speakers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onlyyou import vibration  # noqa: E402
from onlyyou.gate import SYSTEMS, Gate, metrics  # noqa: E402

DATA, RESULTS = ROOT / "data", ROOT / "results"
RATES = [16000, 4000, 2000, 1000, 500, 250, 100]


def load(split):
    out = []
    for p in sorted((DATA / "features" / split).glob("*.npz")):
        s, f = np.load(DATA / "scenes" / split / p.name), np.load(p)
        feats = {k: f[k] for k in f.files}
        feats["prox"] = feats["prox"].copy()
        feats["prox"][:, 0] -= np.median(feats["prox"][:, 0])  # level relative to the scene
        out.append({
            "name": p.stem, "feats": feats, "meta": json.loads(str(s["meta"])),
            "wearer": s["wearer_active"], "others": s["others_active"],
            "body": {"throat": s["throat"], "forehead": s["forehead"]},
        })
    return out


def set_vibration(scenes, sensor, rate, aliased):
    for i, sc in enumerate(scenes):
        rng = np.random.default_rng(1000 + i)
        imu = vibration.emulate_imu(sc["body"][sensor], rate, aliased, rng)
        sc["feats"]["vib"] = vibration.features(imu, rate, sc["feats"]["mic_db"])


def run(system, train, test):
    cues = SYSTEMS[system]
    gate = Gate(cues).fit([s["feats"] for s in train], [s["wearer"] for s in train])
    dec = gate.decide([s["feats"] for s in test])
    res = metrics(dec, [s["wearer"] for s in test], [s["others"] for s in test])
    # 95% interval for leakage by resampling whole test scenes
    passed = np.array([(d & s["others"] & ~s["wearer"]).sum() for d, s in zip(dec, test)])
    total = np.array([(s["others"] & ~s["wearer"]).sum() for s in test])
    rng = np.random.default_rng(0)
    boots = [passed[i].sum() / max(total[i].sum(), 1) for i in rng.integers(0, len(test), (1000, len(test)))]
    res["leakage_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    buckets = {"friend louder than wearer": lambda m: m["friend_sir_db"] < 0,
               "friend 0-6 dB quieter": lambda m: 0 <= m["friend_sir_db"] < 6,
               "friend 6+ dB quieter": lambda m: m["friend_sir_db"] >= 6}
    res["leakage_by_condition"] = {}
    for name, cond in buckets.items():
        idx = [i for i, s in enumerate(test) if cond(s["meta"])]
        if idx:
            sub = metrics([dec[i] for i in idx], [test[i]["wearer"] for i in idx], [test[i]["others"] for i in idx])
            res["leakage_by_condition"][name] = sub["leakage"]
    return res, gate, dec


def sweep(train, test):
    rows = []
    for sensor in ("throat", "forehead"):
        for aliased in (False, True):
            for rate in tqdm(RATES, desc=f"{sensor} {'aliased' if aliased else 'filtered'}"):
                if rate == 16000 and aliased:
                    continue
                set_vibration(train, sensor, rate, aliased)
                set_vibration(test, sensor, rate, aliased)
                for system in ("Vibration only", "Only You (vibration + two mics + voiceprint)"):
                    r, _, _ = run(system, train, test)
                    rows.append({"sensor": sensor, "aliased": aliased, "rate": rate, "system": system,
                                 "leakage": r["leakage"], "wearer_recall": r["wearer_recall"]})
    return rows


def plot_sweep(rows, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {("throat", False): ("tab:blue", "-"), ("throat", True): ("tab:blue", "--"),
              ("forehead", False): ("tab:orange", "-"), ("forehead", True): ("tab:orange", "--")}
    for (sensor, aliased), (color, ls) in styles.items():
        pts = sorted((r["rate"], 100 * r["leakage"]) for r in rows if r["sensor"] == sensor and r["aliased"] == aliased
                     and r["system"].startswith("Only You"))
        if pts:
            label = f"Only You, {sensor}-like coupling, {'no anti-alias filter' if aliased else 'standard IMU filter'}"
            ax.plot(*zip(*pts), color=color, ls=ls, marker="o", label=label)
    for name, color in (("Hush alone", "black"), ("Voiceprint only", "gray"),
                        ("Voiceprint + two mics (no vibration)", "tab:green")):
        ax.axhline(baselines[name], color=color, ls=":", label=name)
    ax.set_xscale("log")
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{r // 1000}k" if r >= 1000 else str(r) for r in RATES])
    ax.invert_xaxis()
    ax.set_xlabel("IMU sample rate (Hz)")
    ax.set_ylabel("Leakage: share of others' speech passed (%)")
    ax.set_title("How slow can the pendant's IMU be? (wearer recall fixed at 90%)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=1000, help="IMU rate for the main table")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    train, test = load("train"), load("test")
    print(f"{len(train)} train scenes, {len(test)} test scenes")

    table = {}
    for sensor in ("throat", "forehead"):
        set_vibration(train, sensor, args.rate, aliased=False)
        set_vibration(test, sensor, args.rate, aliased=False)
        for system in SYSTEMS:
            uses_vib = "vibration" in SYSTEMS[system]
            if not uses_vib and system in table:
                continue
            key = f"{system} [{sensor}-like]" if uses_vib else system
            table[key] = run(system, train, test)[0]

    lines = ["| System | Wearer speech kept | Others' speech leaked (95% CI) | Leaked when friend is louder | False triggers on noise |",
             "|---|---|---|---|---|"]
    for name, r in table.items():
        louder = r["leakage_by_condition"].get("friend louder than wearer", float("nan"))
        lo, hi = r["leakage_ci95"]
        lines.append(f"| {name} | {pct(r['wearer_recall'])} | **{pct(r['leakage'])}** ({100 * lo:.0f}-{100 * hi:.0f}%) "
                     f"| {pct(louder)} | {pct(r['false_triggers'])} |")
    md = "\n".join(lines)
    print(md)
    (RESULTS / "leakage_table.md").write_text(
        f"IMU at {args.rate} Hz with a standard anti-alias filter. {len(train)} train / {len(test)} test scenes, "
        "speaker-disjoint. Thresholds set on train for 90% wearer recall.\n\n" + md + "\n")
    (RESULTS / "leakage_table.json").write_text(json.dumps(table, indent=2))

    if not args.skip_sweep:
        rows = sweep(train, test)
        (RESULTS / "imu_rate_sweep.json").write_text(json.dumps(rows, indent=2))
        baselines = {k: 100 * table[k]["leakage"]
                     for k in ("Hush alone", "Voiceprint only", "Voiceprint + two mics (no vibration)")}
        plot_sweep(rows, baselines, RESULTS / "imu_rate_sweep.png")
        print("saved", RESULTS / "imu_rate_sweep.png")


if __name__ == "__main__":
    main()
