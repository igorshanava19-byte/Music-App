"""Getting audio in (YouTube / upload) and out (wav, mp3, flac) via ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

SAMPLE_RATE = 44100


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # bundled static ffmpeg, so Windows needs no separate install

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any audio/video file to 44.1 kHz stereo 16-bit WAV."""
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)])


def encode_mp3(src: Path, dst: Path, bitrate: str = "320k") -> None:
    _run_ffmpeg(["-i", str(src), "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)])


def load(path: Path) -> tuple[np.ndarray, int]:
    """Return (samples[channels, n] float32, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T.copy(), sr


def save(path: Path, samples: np.ndarray, sr: int) -> None:
    """Save [channels, n] float audio; format follows the extension (.wav / .flac)."""
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0.999:  # avoid clipping when stems are summed
        samples = samples * (0.999 / peak)
    subtype = "PCM_16" if path.suffix.lower() in (".wav", ".flac") else None
    sf.write(str(path), samples.T, sr, subtype=subtype)


def _deno_path() -> str | None:
    found = shutil.which("deno")
    if found:
        return found
    try:
        import deno  # pip package that ships the Deno binary

        return deno.find_deno_bin()
    except Exception:
        return None


def download_youtube(url: str, out_dir: Path, progress=None) -> tuple[Path, str]:
    """Download the best audio stream of a YouTube (or other yt-dlp supported) URL.

    Returns (downloaded_file, title). For personal use only: downloading from
    YouTube is against YouTube's terms of service.
    """
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "ffmpeg_location": ffmpeg_exe(),
    }
    deno = _deno_path()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
    if config.YTDLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.YTDLP_COOKIES_FROM_BROWSER,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if "entries" in info:  # a search or playlist slipped through
            info = info["entries"][0]
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        candidates = sorted(out_dir.glob("download.*"))
        if not candidates:
            raise RuntimeError("Download finished but no audio file was found")
        path = candidates[0]
    return path, info.get("title") or "Untitled"
