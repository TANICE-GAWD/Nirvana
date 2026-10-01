# Only You

**A pendant should hear only its wearer.** Only You is a CPU-only pipeline that decides, every 32 ms,
whether the person wearing an AI pendant is the one speaking, drops everyone else before transcription,
and turns what is left into a journal where every line cites the moment it came from.

Built as a working answer to two problems an always-listening AI necklace (like [Nirva](https://www.nirva.life/))
has to solve:

1. **Listen only to the wearer.** Speech-enhancement models such as Hush, Krisp or ai-coustics keep the
   *loudest* voice. On a necklace the loudest voice is often a friend across the table or the TV.
2. **Grounded memory.** Journals and insights must come from what the wearer actually said.

<!-- RESULTS -->

## The idea: feel the voice, don't just hear it

[antimattr](https://www.antimattr.one/) (YC F26) isolates the wearer in its earbuds with **bone conduction**:
the wearer's voice arrives as vibration through the body, which nobody else's voice can produce.
A pendant resting on the sternum already carries a sensor that can feel that vibration: its **IMU**.

Only You fuses three cues a pendant already has:

| Cue | Why it separates the wearer | Source in this repo |
|---|---|---|
| **Chest vibration (IMU)** | Only the wearer's own speech shakes the pendant | Vibravox throat-mic and forehead-accelerometer recordings, re-sampled to IMU rates |
| **Two-mic proximity** | Mouth is ~15 cm away: louder at the top mic, fixed arrival-time difference, mostly direct sound | Two mics 2.5 cm apart, rendered with `pyroomacoustics` |
| **Voiceprint** | Who is speaking | SpeechBrain ECAPA-TDNN (pretrained, CPU) |

then cleans what passes with **Hush** (weya-ai's open-source speech enhancer) and hands only the
wearer's speech to transcription and memory.

```
2 mics ──► Hush ──► voice activity ─┐
   │                voiceprint ─────┤
   └──► level / arrival time / ─────┼──► gate (gradient-boosted trees, causal) ──► wearer-only audio
        coherence                   │                                                 │
IMU ──► vibration energy / ─────────┘                                     Whisper (CPU) ▼
        body-vs-air ratio                                       moments ──► journal with [m3] citations
                                                                                 └─► grounding check
```

## How it is tested

No hardware is needed. Everything is simulated from public recordings, with speakers that never
appear in training:

- **Scenes**: 60 training and 40 test "pendant days", each 10-28 s. A wearer talks; a friend talks at
  0.8-3 m (from 6 dB *louder* than the wearer to 12 dB quieter at the pendant); half the scenes add a TV;
  traffic or crowd babble at 0-15 dB SNR; rooms with 0.2-0.7 s reverberation.
- **Realism knobs**: the pendant's position relative to the mouth varies per scene, the two mics have
  up to ±2 dB gain mismatch and their own noise floor, and half the scenes include walking motion on the IMU.
- **Vibration leakage**: outside sound also reaches body sensors. Leakage is scaled from each sensor's
  measured signal-to-external-noise ratio in the Vibravox paper: throat mic 25.6 dB (well-coupled,
  best case) and forehead accelerometer -3.5 dB (poorly coupled, worst case). A chest IMU should sit
  between the two, so results are reported for both.
- **Metric**: every system is calibrated on training scenes to keep **90% of the wearer's speech**, then
  measured on test scenes. *Leakage* is the share of frames where only other people are talking that
  still get through.

## Run it

```bash
./scripts/setup.sh                      # CPU-only: Python 3.11 venv, Hush source, deps
source .venv/bin/activate
python scripts/fetch_vibravox.py        # ~4 GB download, keeps ~150 MB at 16 kHz
python scripts/prepare.py               # build scenes + cache features (~1 h on a laptop)
python scripts/evaluate.py              # leakage table + IMU sample-rate sweep
python scripts/eval_words.py            # word-level check with Whisper
python -m pytest
```

### Live demo (laptop mic or earbuds)

A laptop has one mic and no vibration sensor, so the live gate uses voiceprint + Hush only.

```bash
python scripts/live_demo.py devices
python scripts/live_demo.py enroll --mic "Digital Microphone"   # read the prompt for 25 s
python scripts/live_demo.py live   --mic "Digital Microphone"   # Ctrl+C: saves audio + journal
python scripts/live_demo.py process street.wav                  # any recording
```

### Real-world earbud comparison

Compare against the OnePlus Nord Buds 3's built-in call noise cancellation: you read a script while
the laptop speaker plays another voice with known transcripts.

```bash
python scripts/live_demo.py enroll --mic "Digital Microphone" --buds "Nord Buds 3-95"
python scripts/nordbuds_bench.py record
python scripts/nordbuds_bench.py score   # -> results/nordbuds_benchmark.md
```

## Limits

- Vibravox is French speech with sensors on the throat and forehead, not the chest. It stands in for a
  pendant IMU; detecting vibration does not depend on language, but absolute numbers will move on real
  hardware. That is why both a best-case and a worst-case coupling are reported.
- The two-mic cues are simulated. Real pendants add clothing rustle and body reflections.
- IMU emulation (sample rate, optional missing anti-alias filter, noise floor, walking) is a model,
  not a measurement of any specific part.
- The journal layer is deliberately thin: it shows the contract (wearer-only input, every line cited,
  automatic grounding check), not a finished product.

## Layout

```
onlyyou/      audio, enhance (Hush), vad, voiceprint, vibration, proximity, scenes, gate, pipeline,
              transcribe, memory
scripts/      fetch_vibravox, prepare, evaluate, eval_words, live_demo, nordbuds_bench
results/      tables, plots, transcripts
tests/        unit tests
```

Credits: [Hush](https://github.com/pulp-vision/Hush) (Apache 2.0), [Vibravox](https://huggingface.co/datasets/Cnam-LMSSC/vibravox)
(CC BY 4.0), [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
[Silero VAD](https://github.com/snakers4/silero-vad), [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics).
