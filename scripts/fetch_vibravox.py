"""Download a few Vibravox shards and keep only the channels we use, at 16 kHz.

Each shard is ~0.5 GB with ~100 utterances from ~20 speakers. We keep the
near-mouth headset mic (clean reference for the wearer's voice), the throat
microphone and the forehead accelerometer (the body-vibration sensors), then
delete the shard.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onlyyou.audio import SR, resample, save  # noqa: E402

CHANNELS = {
    "headset": "audio.headset_microphone",
    "throat": "audio.throat_microphone",
    "forehead": "audio.forehead_accelerometer",
}


def extract(shard_path, out_dir, split):
    pf = pq.ParquetFile(shard_path)
    meta_cols = ["speaker_id", "sentence_id", "gender", "normalized_text"]
    rows = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=list(CHANNELS.values()) + meta_cols).to_pydict()
        for i in range(len(t["speaker_id"])):
            spk, sid = t["speaker_id"][i], t["sentence_id"][i]
            base = out_dir / split / spk / str(sid)
            for name, col in CHANNELS.items():
                x, sr = sf.read(io.BytesIO(t[col][i]["bytes"]), dtype="float32")
                save(f"{base}_{name}.flac", resample(x, sr, SR))
            rows.append({
                "split": split, "speaker": spk, "sentence_id": sid,
                "gender": t["gender"][i], "text": t["normalized_text"][i],
                "path": str(base.relative_to(out_dir)),
            })
        del t
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vibravox")
    ap.add_argument("--train-shards", type=int, default=4)
    ap.add_argument("--test-shards", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(l)["shard"] for l in index.read_text().splitlines() if l}

    plan = [("test", f"speech_clean/test-{i:05d}-of-00030.parquet") for i in range(args.test_shards)]
    plan += [("train", f"speech_clean/train-{i:05d}-of-00201.parquet") for i in range(args.train_shards)]
    for split, name in plan:
        if name in done:
            continue
        print(f"downloading {name}", flush=True)
        p = Path(hf_hub_download("Cnam-LMSSC/vibravox", name, repo_type="dataset", local_dir="data/raw"))
        rows = extract(p, out, split)
        with index.open("a") as f:
            for r in rows:
                f.write(json.dumps({**r, "shard": name}) + "\n")
        p.unlink()
        print(f"  {len(rows)} utterances from {len({r['speaker'] for r in rows})} speakers", flush=True)


if __name__ == "__main__":
    main()
