"""Word-level check: how many of other people's words end up in the transcript?

Each system's output audio is transcribed (Whisper small, French, CPU). A word counts as
leaked if it appears in what the friend / TV said but not in what the wearer said.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onlyyou.enhance import hush  # noqa: E402
from onlyyou.gate import SYSTEMS, Gate  # noqa: E402
from onlyyou.memory import STOP  # noqa: E402
from onlyyou.pipeline import apply_gate  # noqa: E402
from onlyyou.transcribe import transcribe  # noqa: E402
from scripts.evaluate import DATA, RESULTS, load, set_vibration  # noqa: E402

GATED = ["Voiceprint only", "Only You (vibration + two mics + voiceprint)"]


def words(text):
    return [w for w in re.findall(r"[\w']+", text.lower()) if len(w) > 3 and w not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--rate", type=int, default=1000)
    ap.add_argument("--sensor", default="forehead", help="forehead = pessimistic coupling")
    ap.add_argument("--whisper", default="small")
    args = ap.parse_args()
    train, test = load("train"), load("test")
    set_vibration(train, args.sensor, args.rate, False)
    set_vibration(test, args.sensor, args.rate, False)
    gates = {g: Gate(SYSTEMS[g]).fit([s["feats"] for s in train], [s["wearer"] for s in train]) for g in GATED}
    names = ["No filter", "Hush alone"] + GATED
    totals = {k: {"leaked": 0, "wearer_hit": 0} for k in names}
    wearer_total = 0
    examples = []
    for sc in tqdm(test[: args.scenes], desc="transcribing"):
        top = np.load(DATA / "scenes" / "test" / f"{sc['name']}.npz")["mics"][0]
        enhanced = hush(top)
        outs = {"No filter": top, "Hush alone": enhanced}
        for g, gate in gates.items():
            outs[g] = apply_gate(enhanced, gate.decide([sc["feats"]])[0])
        w_set = set(words(" ".join(sc["meta"]["wearer_texts"])))
        o_set = set(words(" ".join(sc["meta"]["other_texts"]))) - w_set
        wearer_total += len(w_set)
        row = {"scene": sc["name"], "friend_sir_db": round(sc["meta"]["friend_sir_db"], 1)}
        for k, audio in outs.items():
            hyp = " ".join(t for _, _, t in transcribe(audio, language="fr", size=args.whisper))
            h = words(hyp)
            leaked = [w for w in h if w in o_set]
            totals[k]["leaked"] += len(leaked)
            totals[k]["wearer_hit"] += len(w_set & set(h))
            row[k] = {"transcript": hyp, "leaked_words": leaked}
        examples.append(row)

    n = len(examples)
    lines = [f"Whisper-{args.whisper} transcripts of {n} test scenes. IMU {args.rate} Hz, {args.sensor}-like coupling.",
             "", "| System | Other people's words in transcript (per scene) | Wearer's words recovered |", "|---|---|---|"]
    for k in names:
        t = totals[k]
        lines.append(f"| {k} | {t['leaked'] / n:.1f} | {100 * t['wearer_hit'] / max(wearer_total, 1):.0f}% |")
    md = "\n".join(lines)
    print(md)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "word_leakage.md").write_text(md + "\n")
    (RESULTS / "word_leakage_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
