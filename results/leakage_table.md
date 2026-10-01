IMU at 1000 Hz with a standard anti-alias filter. 60 train / 40 test scenes, speaker-disjoint. Thresholds set on train for 90% wearer recall.

| System | Wearer speech kept | Others' speech leaked (95% CI) | Leaked when friend is louder | False triggers on noise |
|---|---|---|---|---|
| No filter (any speech) | 93.9% | **69.9%** (63-77%) | 78.9% | 18.1% |
| Hush alone | 91.5% | **54.8%** (48-63%) | 66.4% | 13.4% |
| Voiceprint only | 90.2% | **38.0%** (33-44%) | 43.8% | 15.9% |
| Two mics only | 92.5% | **23.8%** (19-29%) | 27.8% | 12.0% |
| Vibration only [throat-like] | 89.5% | **8.3%** (7-10%) | 12.8% | 4.5% |
| Voiceprint + two mics (no vibration) | 91.9% | **19.5%** (16-24%) | 21.3% | 9.3% |
| Only You (vibration + two mics + voiceprint) [throat-like] | 90.7% | **5.9%** (5-8%) | 7.2% | 3.3% |
| Vibration only [forehead-like] | 91.2% | **44.6%** (40-50%) | 54.1% | 13.0% |
| Only You (vibration + two mics + voiceprint) [forehead-like] | 91.1% | **16.0%** (13-19%) | 16.0% | 8.0% |
