"""Fuse per-frame cues into a 'wearer is speaking' gate, and score it.

Frames are classed by ground truth:
  W = wearer speaking (others may overlap)
  O = only other people speaking (friend / TV)
  S = no speech (noise only)
Wearer recall = pass rate on W. Leakage = pass rate on O. False triggers = pass rate on S.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

CUES = {
    "vad": ["vad_raw"],
    "hush": ["vad_hush"],
    "voiceprint": ["vp_raw", "vp_hush"],
    "mics": ["prox"],
    "vibration": ["vib"],
}

SYSTEMS = {
    "No filter (any speech)": ["vad"],
    "Hush alone": ["hush"],
    "Voiceprint only": ["vad", "hush", "voiceprint"],
    "Two mics only": ["vad", "mics"],
    "Vibration only": ["vad", "vibration"],
    "Only You (vibration + two mics + voiceprint)": ["vad", "hush", "voiceprint", "mics", "vibration"],
}


def _causal_mean(x, k):
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def matrix(feats: dict, cues: list[str]) -> np.ndarray:
    cols = []
    for cue in cues:
        for key in CUES[cue]:
            v = feats[key]
            cols.append(v[:, None] if v.ndim == 1 else v)
    x = np.hstack(cols).astype(np.float32)
    return np.hstack([x, _causal_mean(x, 4), _causal_mean(x, 12)])


class Gate:
    def __init__(self, cues, target_recall=0.9):
        self.cues, self.target_recall = cues, target_recall
        self.raw = len(cues) == 1 and cues[0] in ("vad", "hush")
        self.model, self.threshold = None, 0.5

    def _new(self):
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)

    def score(self, feats_list):
        if self.raw:
            return [f[CUES[self.cues[0]][0]] for f in feats_list]
        return [self.model.predict_proba(matrix(f, self.cues))[:, 1] for f in feats_list]

    def fit(self, feats_list, labels_list):
        y = np.concatenate(labels_list)
        if self.raw:
            oof = np.concatenate(self.score(feats_list))
        else:
            xs = [matrix(f, self.cues) for f in feats_list]
            x = np.vstack(xs)
            groups = np.concatenate([np.full(len(a), i) for i, a in enumerate(xs)])
            oof = np.zeros(len(y))
            for tr, va in GroupKFold(n_splits=min(4, len(xs))).split(x, y, groups):
                oof[va] = self._new().fit(x[tr], y[tr]).predict_proba(x[va])[:, 1]
            self.model = self._new().fit(x, y)
        # threshold reaching the target wearer recall on held-out training predictions
        self.threshold = float(np.quantile(oof[y.astype(bool)], 1 - self.target_recall))
        return self

    def decide(self, feats_list):
        return [s >= self.threshold for s in self.score(feats_list)]


def metrics(decisions, wearer, others):
    d, w, o = (np.concatenate(a).astype(bool) for a in (decisions, wearer, others))
    only_o, silent = o & ~w, ~o & ~w
    return {
        "wearer_recall": float(d[w].mean()),
        "leakage": float(d[only_o].mean()) if only_o.any() else float("nan"),
        "false_triggers": float(d[silent].mean()) if silent.any() else float("nan"),
        "frames_wearer": int(w.sum()), "frames_others_only": int(only_o.sum()),
    }
