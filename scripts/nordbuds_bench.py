"""Real-world comparison against the OnePlus Nord Buds 3's built-in call noise cancellation.

You read a script while the laptop speaker plays another person talking (LibriSpeech,
known transcripts). The earbud mic (already processed by the buds) and the laptop mic are
recorded at the same time. Scoring counts how many of the other person's words reach each
transcript, and how much of your script survives.

  python scripts/nordbuds_bench.py record --buds "Nord Buds 3-95" --laptop "Digital Microphone" --speaker "Speaker"
  python scripts/nordbuds_bench.py score
"""

import argparse
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audio_io import MultiRecorder, find  # noqa: E402
from onlyyou.audio import SR, resample, save, load  # noqa: E402
from onlyyou.memory import STOP  # noqa: E402

REC = ROOT / "recordings"
SCRIPT = [
    "This morning I finally fixed the bicycle chain that kept slipping.",
    "My sister wants to visit in November, so I need to clean the spare room.",
    "The meeting about the budget went better than I expected.",
    "I should drink more water and sleep before midnight this week.",
    "Tomorrow I will try the new noodle place near the station.",
    "Honestly I feel calmer when I write things down in the evening.",
    "Remember to send the invoice and book the dentist appointment.",
    "The weather was perfect for a picnic, warm but not too sunny.",
]


def librispeech():
    """73 short English utterances with transcripts (hf-internal-testing/librispeech_asr_dummy)."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=str(ROOT / "data" / "librispeech_dummy"))
    t = pq.read_table(p).to_pydict()
    out = []
    for a, text in zip(t["audio"], t["text"]):
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        out.append((resample(x, sr, SR), text.lower()))
    return out


def cmd_record(args):
    import sounddevice as sd

    REC.mkdir(exist_ok=True)
    utts = librispeech()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(utts))
    spk = find(args.speaker, "output")
    out_rate = int(sd.query_devices(spk, "output")["default_samplerate"]) if spk is not None else 48000
    gap = np.zeros(int(0.4 * SR), np.float32)
    played, chunks, total = [], [], 0
    for i in order:
        x, text = utts[i]
        chunks += [x, gap]
        played.append(text)
        total += len(x) + len(gap)
        if total > args.seconds * SR:
            break
    friend = np.concatenate(chunks) * args.volume
    friend_out = resample(friend, SR, out_rate)

    print("You will read 8 sentences while 'a friend' talks from the laptop speaker.")
    print("Wear the earbuds. Sit at arm's length from the laptop.\n")
    input("Press Enter to start...")
    with MultiRecorder({"buds": find(args.buds), "laptop": find(args.laptop)}) as rec:
        time.sleep(1.0)
        player = threading.Thread(target=lambda: sd.play(friend_out, out_rate, device=spk, blocking=True), daemon=True)
        player.start()
        for i, line in enumerate(SCRIPT, 1):
            input(f"\n[{i}/8] Read aloud, then press Enter:\n    {line}\n")
        sd.stop()
        time.sleep(0.5)
    res = rec.result()
    save(REC / "bench_buds.wav", res["buds"])
    save(REC / "bench_laptop.wav", res["laptop"])
    (REC / "bench_meta.json").write_text(json.dumps({"script": SCRIPT, "friend_texts": played,
                                                     "volume": args.volume}, indent=2))
    print("saved recordings/bench_buds.wav, bench_laptop.wav, bench_meta.json")


def words(text):
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP]


def cmd_score(args):
    from live_demo import load_profile
    from onlyyou.pipeline import single_mic
    from onlyyou.transcribe import transcribe

    meta = json.loads((REC / "bench_meta.json").read_text())
    script = set(words(" ".join(meta["script"])))
    friend = set(words(" ".join(meta["friend_texts"]))) - script
    systems = {}
    for dev in ("buds", "laptop"):
        x = load(REC / f"bench_{dev}.wav")
        prof, th = load_profile(dev if (REC / f"profile_{dev}.npy").exists() else "laptop")
        out = single_mic(x, prof, th)
        label = "Nord Buds 3 (built-in noise cancellation)" if dev == "buds" else "Laptop mic"
        systems[f"{label}, raw"] = out["raw"]
        systems[f"{label} + Hush"] = out["hush"]
        systems[f"{label} + Only You (voiceprint + Hush)"] = out["only_you"]
    rows = ["| Recording | Friend's words in transcript | Your script words recovered |", "|---|---|---|"]
    detail = {}
    for name, audio in systems.items():
        hyp = " ".join(t for _, _, t in transcribe(audio, language="en"))
        h = words(hyp)
        leaked = [w for w in h if w in friend]
        kept = len(script & set(h)) / max(len(script), 1)
        rows.append(f"| {name} | {len(leaked)} | {100 * kept:.0f}% |")
        detail[name] = {"transcript": hyp, "leaked": leaked}
    md = "\n".join(rows)
    print(md)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nordbuds_benchmark.md").write_text(
        "Real recording: wearer reads a script while a second voice plays from a laptop speaker.\n"
        "Single mic, no vibration sensor, so Only You here = voiceprint + Hush.\n\n" + md + "\n")
    (ROOT / "results" / "nordbuds_benchmark.json").write_text(json.dumps(detail, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--buds", default="Nord Buds 3-95")
    r.add_argument("--laptop", default="Digital Microphone")
    r.add_argument("--speaker", default="Speaker")
    r.add_argument("--seconds", type=int, default=120)
    r.add_argument("--volume", type=float, default=0.8)
    sub.add_parser("score")
    args = ap.parse_args()
    {"record": cmd_record, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
