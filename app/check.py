"""Setup self-check: verifies the install, downloads the AI models, and reports the GPU.

Run with:  python -m app.check
"""
from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ok = True

    import torch

    from . import audio_io, config, separation, transcription

    dev = config.device()
    print(f"PyTorch {torch.__version__}, processing device: {dev}")
    if dev == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)")
        try:  # make sure the CUDA build actually has kernels for this GPU
            (torch.ones(8, device="cuda") * 2).sum().item()
        except Exception as exc:
            print(f"  !! The GPU is not usable with this PyTorch build: {exc}")
            ok = False
    elif torch.version.cuda is None and sys.platform == "win32":
        print("  (CPU-only PyTorch: songs will take several minutes each)")

    print(f"ffmpeg: {audio_io.ffmpeg_exe()}")
    print(f"deno (for YouTube): {audio_io._deno_path() or 'NOT FOUND - YouTube links may fail'}")

    for name in dict.fromkeys([config.STEMS_MODEL, config.minus_model()]):
        print(f"Downloading / loading separation model {name} ...")
        try:
            separation.get_model(name)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            ok = False

    print("Loading the note transcription model ...")
    try:
        transcription._basic_pitch_model()
    except Exception as exc:
        print(f"  !! failed: {exc}")
        ok = False

    print("All good!" if ok else "Some checks failed (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
