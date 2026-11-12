"""Paths and runtime settings. Override any of them with environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MUSICAPP_DATA_DIR", ROOT_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
WEB_DIR = ROOT_DIR / "web"

HOST = os.environ.get("MUSICAPP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MUSICAPP_PORT", "8000"))

# Demucs model that splits the song into 6 stems (vocals, drums, bass, guitar, piano, other).
STEMS_MODEL = os.environ.get("MUSICAPP_STEMS_MODEL", "htdemucs_6s")
# Model used for the vocal estimate the minus track is built from. "auto" picks the
# slower, cleaner fine-tuned model on a GPU and reuses the 6-stem result on CPU.
MINUS_MODEL = os.environ.get("MUSICAPP_MINUS_MODEL", "auto")

# Optional: let yt-dlp read YouTube cookies from a browser (e.g. "chrome", "edge",
# "firefox") if YouTube asks you to sign in.
YTDLP_COOKIES_FROM_BROWSER = os.environ.get("MUSICAPP_COOKIES_FROM_BROWSER", "").strip() or None

MAX_UPLOAD_MB = int(os.environ.get("MUSICAPP_MAX_UPLOAD_MB", "300"))

# Password for the live link. When set, every page and API call needs a login.
PASSWORD = os.environ.get("MUSICAPP_PASSWORD", "")


@lru_cache(maxsize=1)
def device() -> str:
    forced = os.environ.get("MUSICAPP_DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def minus_model() -> str:
    if MINUS_MODEL != "auto":
        return MINUS_MODEL
    return "htdemucs_ft" if device() == "cuda" else STEMS_MODEL
