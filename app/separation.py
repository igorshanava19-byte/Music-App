"""Source separation with Demucs: stems + the minus (instrumental) track."""
from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
import torch

from . import config

log = logging.getLogger(__name__)

# Human friendly names for Demucs stems.
STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano / Keys",
    "other": "Other (synths, strings, ...)",
}

_models: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


ROFORMER = "bs_roformer_sw"  # BS-RoFormer-SW by jarredou, 6 stems (pip install bs-roformer-infer, ~700 MB weights)


def _roformer():
    from ml_collections import ConfigDict
    import yaml
    from bs_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config
    from bs_roformer.inference import SafeLoaderWithTuple

    import sys

    sys.stdout.reconfigure(errors="replace")  # its downloader prints emoji, which crash a cp1252 Windows pipe
    ckpt, cfg_path = ensure_model_assets(DEFAULT_MODEL)  # downloads + sha256-checks on first use
    with open(cfg_path) as f:
        cfg = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.config = cfg
    model.samplerate = 44100
    model.sources = list(cfg.training.instruments)
    return model


def get_model(name: str):
    with _lock:
        if name == ROFORMER and name not in _models:
            log.info("loading %s", name)
            _models[name] = _roformer().to(config.device()).eval()
        if name not in _models:
            from demucs.pretrained import get_model as demucs_get_model

            log.info("loading demucs model %s", name)
            model = demucs_get_model(name)
            model.to(config.device())
            model.eval()
            _models[name] = model
        return _models[name]


def run_model(name: str, mix: np.ndarray, sr: int, progress: Callable[[float], None] | None = None) -> dict[str, np.ndarray]:
    """Separate a [2, n] float mix; returns {stem_name: [2, n] array}."""
    model = get_model(name)
    if sr != model.samplerate:
        raise ValueError(f"expected {model.samplerate} Hz audio, got {sr}")
    if name == ROFORMER:
        from bs_roformer.utils import demix_track

        out, _ = demix_track(model.config, model, torch.from_numpy(mix), config.device())
        if config.device() == "cuda":
            torch.cuda.empty_cache()
        return {k: v.astype(np.float32) for k, v in out.items()}

    from demucs.apply import apply_model

    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((wav - mean) / std)[None]
    length = x.shape[-1]

    def callback(d: dict) -> None:
        if progress and d.get("state") == "end":
            models = d.get("models", 1)
            frac = (d.get("model_idx_in_bag", 0) + min(1.0, (d.get("segment_offset", 0) + 1) / length)) / models
            progress(min(1.0, frac))

    device = config.device()
    with torch.no_grad():
        out = apply_model(model, x, shifts=1, split=True, overlap=0.25, progress=False,
                          device=device, num_workers=0, callback=callback)[0]
    out = (out * std + mean).cpu().numpy()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {src: out[i] for i, src in enumerate(model.sources)}


def active_frames(mix: np.ndarray, stem: np.ndarray, sr: int) -> np.ndarray:
    """Per 100 ms frame: is the stem within 30 dB of the mix there (playing, not bleed)?"""
    hop = sr // 10
    n = stem.shape[-1] // hop
    frames = stem.mean(0)[: n * hop].reshape(n, hop)
    mix_frames = mix.mean(0)[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    mix_rms = np.sqrt((mix_frames ** 2).mean(1) + 1e-12)
    return (rms / mix_rms) > 10 ** (-30 / 20)


def stem_presence(mix: np.ndarray, stem: np.ndarray, sr: int) -> dict:
    """How much of the song a stem actually plays in (to skip empty instruments)."""
    mono_mix = mix.mean(0)
    mono = stem.mean(0)
    active = active_frames(mix, stem, sr)
    if len(active) == 0:
        return {"level_db": -120.0, "active": 0.0}
    overall = 20 * np.log10(np.sqrt((mono ** 2).mean() + 1e-12) / (np.sqrt((mono_mix ** 2).mean()) + 1e-12))
    return {"level_db": round(float(overall), 1), "active": round(float(active.mean()), 3)}


def is_present(presence: dict) -> bool:
    return presence["level_db"] > -30 and presence["active"] > 0.08
