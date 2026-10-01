"""Timeline figure + audio for one hard test scene (friend louder than the wearer)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onlyyou.audio import FPS, save  # noqa: E402
from onlyyou.enhance import hush  # noqa: E402
from onlyyou.gate import SYSTEMS, Gate  # noqa: E402
from onlyyou.pipeline import apply_gate  # noqa: E402
from scripts.evaluate import DATA, RESULTS, load, set_vibration  # noqa: E402

SHOW = {"Hush alone": "Hush alone", "Voiceprint only": "Voiceprint only",
        "Voiceprint + two mics (no vibration)": "Voiceprint + two mics",
        "Only You (vibration + two mics + voiceprint)": "Only You (+ vibration)"}


def main(sensor="throat", rate=1000):
    train, test = load("train"), load("test")
    set_vibration(train, sensor, rate, False)
    set_vibration(test, sensor, rate, False)
    candidates = [s for s in test if s["meta"]["friend_sir_db"] < 0 and (s["others"] & ~s["wearer"]).mean() > 0.2]
    sc = min(candidates or test, key=lambda s: s["meta"]["friend_sir_db"])
    gates = {k: Gate(SYSTEMS[k]).fit([s["feats"] for s in train], [s["wearer"] for s in train]) for k in SHOW}
    dec = {k: g.decide([sc["feats"]])[0] for k, g in gates.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("Wearer talking (truth)", sc["wearer"], "tab:blue"), ("Others talking (truth)", sc["others"], "tab:red")]
    rows += [(f"{SHOW[k]} keeps", d, "tab:green" if k.startswith("Only") else "gray") for k, d in dec.items()]
    t = np.arange(len(sc["wearer"])) / FPS
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 0.55 * len(rows) + 1.2), sharex=True)
    for ax, (label, d, color) in zip(axes, rows):
        ax.fill_between(t, 0, d.astype(float), step="post", color=color, alpha=0.8, lw=0)
        leak = d & sc["others"] & ~sc["wearer"]
        if label.endswith("keeps"):
            ax.fill_between(t, 0, leak.astype(float), step="post", color="tab:red", alpha=0.9, lw=0)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("seconds")
    m = sc["meta"]
    fig.suptitle(f"Friend {-m['friend_sir_db']:.0f} dB louder than the wearer at the pendant"
                 f"{', TV on' if m['tv'] else ''}, {m['noise']} noise. Red = other people's speech let through.",
                 fontsize=10)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "demo_timeline.png", dpi=150)

    top = np.load(DATA / "scenes" / "test" / f"{sc['name']}.npz")["mics"][0]
    enh = hush(top)
    out = RESULTS / "demo"
    save(out / "1_pendant_mic_raw.wav", top)
    save(out / "2_hush_alone.wav", enh)
    save(out / "3_voiceprint_only.wav", apply_gate(enh, dec["Voiceprint only"]))
    save(out / "4_only_you.wav", apply_gate(enh, dec["Only You (vibration + two mics + voiceprint)"]))
    (out / "scene.json").write_text(json.dumps({"scene": sc["name"], **m}, indent=2, ensure_ascii=False))
    print("scene", sc["name"], m["friend_sir_db"])


if __name__ == "__main__":
    main()
