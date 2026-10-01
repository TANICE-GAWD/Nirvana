import numpy as np

from onlyyou import memory
from onlyyou.gate import metrics
from onlyyou.pipeline import apply_gate
from onlyyou.audio import HOP


def test_metrics_counts_leakage_only_on_others_only_frames():
    wearer = [np.array([1, 1, 0, 0, 0], bool)]
    others = [np.array([0, 1, 1, 1, 0], bool)]
    passed = [np.array([1, 0, 1, 0, 1], bool)]
    m = metrics(passed, wearer, others)
    assert m["wearer_recall"] == 0.5
    assert m["leakage"] == 0.5  # frames 2,3 are others-only; frame 2 passed
    assert m["false_triggers"] == 1.0  # frame 4 is silence and passed


def test_moments_merge_close_segments():
    segs = [(0.0, 2.0, "I fixed the bike"), (2.5, 4.0, "then went running"), (10.0, 12.0, "called mum")]
    ms = memory.build_moments(segs)
    assert [m.id for m in ms] == ["m1", "m2"]
    assert "running" in ms[0].text


def test_grounding_flags_unsupported_and_uncited_lines():
    ms = memory.build_moments([(0.0, 3.0, "I finally fixed the bicycle chain"), (9.0, 12.0, "dentist appointment tomorrow")])
    text = "\n".join([
        "You fixed your bicycle chain. [m1]",
        "You booked a flight to Paris. [m2]",
        "You remembered the dentist appointment.",
        "You mentioned the dentist appointment. [m9]",
    ])
    lines = memory.check_grounding(text, ms)["lines"]
    assert [l["grounded"] for l in lines] == [True, False, False, False]


def test_extractive_journal_is_fully_grounded(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ms = memory.build_moments([(0.0, 3.0, "the meeting about the budget went well"), (8.0, 9.0, "need more sleep")])
    assert memory.check_grounding(memory.journal(ms), ms)["grounded_share"] == 1.0


def test_apply_gate_keeps_only_accepted_frames():
    x = np.ones(HOP * 10, np.float32)
    d = np.zeros(10, bool)
    d[2] = True
    y = apply_gate(x, d, hangover=0, ramp_ms=0)
    assert y[2 * HOP : 3 * HOP].min() == 1.0
    assert y[: 2 * HOP].max() == 0.0 and y[4 * HOP :].max() == 0.0
