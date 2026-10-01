"""Live and offline demos on a laptop or earbuds (one mic, no vibration sensor).

  python scripts/live_demo.py devices
  python scripts/live_demo.py enroll --mic "Digital Microphone"     # read aloud for ~25 s
  python scripts/live_demo.py live   --mic "Digital Microphone"     # Ctrl+C to stop
  python scripts/live_demo.py process recordings/street.wav --profile laptop
"""

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audio_io import MultiRecorder, countdown, find, list_devices  # noqa: E402
from onlyyou import memory, vad, voiceprint  # noqa: E402
from onlyyou.audio import HOP, SR, load, save  # noqa: E402
from onlyyou.enhance import hush  # noqa: E402
from onlyyou.pipeline import calibrate_threshold, single_mic  # noqa: E402
from onlyyou.transcribe import transcribe  # noqa: E402

REC = ROOT / "recordings"

ENROLL_TEXT = """Read this aloud in your normal voice:

  The best part of my week was the long walk by the river on Sunday.
  I keep meaning to call my parents more often, and I should do it tonight.
  Work has been busy, but the new project is finally starting to make sense.
  Coffee in the morning, a short run, and then I feel ready for anything.
  Sometimes I talk too fast when I am excited about an idea.
  Remind me to buy groceries, water the plants, and reply to my friend.
"""


def profile_paths(tag):
    return REC / f"profile_{tag}.npy", REC / f"enroll_{tag}.wav", REC / f"threshold_{tag}.json"


def save_profile(tag, audio):
    p, w, t = profile_paths(tag)
    save(w, audio)
    prof = voiceprint.enroll([audio])
    np.save(p, prof)
    th = calibrate_threshold(audio, prof)
    t.write_text(json.dumps({"threshold": th}))
    print(f"saved voiceprint '{tag}' (threshold {th:.2f})")


def load_profile(tag):
    p, _, t = profile_paths(tag)
    if not p.exists():
        raise SystemExit(f"No voiceprint '{tag}'. Run `enroll` first.")
    return np.load(p), json.loads(t.read_text())["threshold"]


def cmd_enroll(args):
    REC.mkdir(exist_ok=True)
    devices = {"laptop": find(args.mic)}
    if args.buds:
        devices["buds"] = find(args.buds)
    print(ENROLL_TEXT)
    input("Press Enter, then start reading...")
    with MultiRecorder(devices) as rec:
        _, t = countdown(args.seconds, "recording")
        t.join()
    for tag, audio in rec.result().items():
        save_profile(tag, audio)


def summarize(only_you_audio, raw_audio, out_prefix, language):
    print("\ntranscribing (CPU)...")
    raw_t = transcribe(raw_audio, language=language)
    ours = transcribe(only_you_audio, language=language)
    moments = memory.build_moments(ours)
    text = memory.journal(moments)
    check = memory.check_grounding(text, moments)
    md = ["# Only You: session journal", "", "## Everything the mic heard (raw)", "",
          " ".join(t for _, _, t in raw_t) or "(nothing)", "", "## What Only You kept (wearer only)", "",
          " ".join(t for _, _, t in ours) or "(nothing)", "", "## Moments", ""]
    md += [f"- [{m.id}] {m.start:.1f}-{m.end:.1f}s: {m.text}" for m in moments]
    md += ["", "## Journal", "", text or "(no wearer speech)", "",
           f"Grounding check: {100 * check['grounded_share']:.0f}% of lines cite moments that support them."]
    Path(f"{out_prefix}_journal.md").write_text("\n".join(md) + "\n")
    Path(f"{out_prefix}_moments.json").write_text(json.dumps(memory.to_json(moments), indent=2))
    print("\n".join(md))
    print(f"\nsaved {out_prefix}_journal.md")


def cmd_live(args):
    profile, threshold = load_profile(args.profile)
    threshold = args.threshold or threshold
    dev = find(args.mic)
    import sounddevice as sd

    rate = int(sd.query_devices(dev, "input")["default_samplerate"])
    q = queue.Queue()
    stream = sd.InputStream(device=dev, channels=1, samplerate=rate, dtype="float32",
                            callback=lambda d, f, t, s: q.put(d[:, 0].copy()))
    from onlyyou.audio import resample

    buf = np.zeros(0, np.float32)
    raw_all, ours_all = [], []
    step, ctx = int(0.5 * SR), int(2.0 * SR)
    print(f"Listening on '{sd.query_devices(dev)['name']}'. Talk, and have someone else talk too. Ctrl+C to stop.\n")
    stream.start()
    try:
        pending = np.zeros(0, np.float32)
        while True:
            pending = np.concatenate([pending, resample(q.get(), rate, SR)])
            if len(pending) < step:
                continue
            chunk, pending = pending[:step], pending[step:]
            buf = np.concatenate([buf, chunk])[-ctx:]
            raw_all.append(chunk)
            if len(buf) < int(1.5 * SR):
                ours_all.append(np.zeros_like(chunk))
                continue
            enh = hush(buf)
            v = vad.speech_prob(enh[-step:]).mean()
            s = float(voiceprint.score_track(enh[-int(1.5 * SR):], profile, win_s=1.5, hop_s=1.5)[-1])
            keep = v > 0.5 and s >= threshold
            ours_all.append(enh[-step:] if keep else np.zeros(step, np.float32))
            bar = "#" * int(max(0, s) * 30)
            label = "YOU    " if keep else ("someone" if v > 0.5 else "quiet  ")
            print(f"\r{label}  speech {v:.2f}  voiceprint {s:+.2f} {bar:<30}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw, ours = np.concatenate(raw_all), np.concatenate(ours_all)
    save(REC / f"live_{stamp}_raw.wav", raw)
    save(REC / f"live_{stamp}_only_you.wav", ours)
    summarize(ours, raw, REC / f"live_{stamp}", args.language)


def cmd_process(args):
    profile, threshold = load_profile(args.profile)
    x = load(args.input)
    out = single_mic(x, profile, args.threshold or threshold)
    prefix = REC / Path(args.input).stem
    for k in ("raw", "hush", "only_you"):
        save(f"{prefix}_{k}.wav", out[k])
    kept = out["decisions"].mean()
    print(f"kept {100 * kept:.0f}% of frames as wearer speech; wrote {prefix}_{{raw,hush,only_you}}.wav")
    summarize(out["only_you"], x, prefix, args.language)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    e = sub.add_parser("enroll")
    e.add_argument("--mic", default=None, help="laptop mic name (substring) or index")
    e.add_argument("--buds", default=None, help="also enroll the earbud mic at the same time")
    e.add_argument("--seconds", type=int, default=25)
    lv = sub.add_parser("live")
    lv.add_argument("--mic", default=None)
    lv.add_argument("--profile", default="laptop")
    lv.add_argument("--threshold", type=float, default=None)
    lv.add_argument("--language", default="en")
    pr = sub.add_parser("process")
    pr.add_argument("input")
    pr.add_argument("--profile", default="laptop")
    pr.add_argument("--threshold", type=float, default=None)
    pr.add_argument("--language", default="en")
    args = ap.parse_args()
    if args.cmd == "devices":
        list_devices()
    else:
        {"enroll": cmd_enroll, "live": cmd_live, "process": cmd_process}[args.cmd](args)


if __name__ == "__main__":
    main()
