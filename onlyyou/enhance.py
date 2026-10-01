"""Hush (weya-ai) speech enhancement wrapper."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HUSH_SRC = ROOT / "third_party" / "Hush"
HUSH_CKPT = ROOT / "models" / "hush" / "model_best.ckpt"


@lru_cache(maxsize=1)
def _model():
    if str(HUSH_SRC) not in sys.path:
        sys.path.insert(0, str(HUSH_SRC))
    from model.dfnet_se import DfNetSE, get_config

    if not HUSH_CKPT.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download("weya-ai/hush", "model_best.ckpt", local_dir=str(HUSH_CKPT.parent))
    model = DfNetSE(get_config())
    state = torch.load(str(HUSH_CKPT), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    return model.eval()


@torch.no_grad()
def hush(x: np.ndarray) -> np.ndarray:
    """Enhance 16 kHz mono audio. Output is aligned with the input."""
    model = _model()
    from libdf import DF, erb, erb_norm, unit_norm
    from model.dfnet_se import as_complex, as_real, get_norm_alpha

    cfg = model.config
    df = DF(sr=cfg.sr, fft_size=cfg.fft_size, hop_size=cfg.hop_size,
            nb_bands=cfg.nb_erb, min_nb_erb_freqs=cfg.min_nb_freqs)
    n_fft, hop = df.fft_size(), df.hop_size()
    audio = torch.nn.functional.pad(torch.from_numpy(np.asarray(x, np.float32))[None], (0, n_fft))
    alpha = get_norm_alpha(df.sr(), hop, cfg.norm_tau)
    spec_np = df.analysis(audio.numpy(), reset=True)
    erb_feat = torch.as_tensor(erb_norm(erb(spec_np, df.erb_widths()), alpha)).unsqueeze(1)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec_np[..., : cfg.nb_df], alpha))).unsqueeze(1)
    spec = as_real(torch.as_tensor(spec_np)).unsqueeze(1)
    enh = as_complex(model.model(spec.clone(), erb_feat, spec_feat)[0].squeeze(1))
    out = np.asarray(df.synthesis(enh.numpy(), reset=True), dtype=np.float32)[0]
    delay = n_fft - hop
    return out[delay : delay + len(x)]
