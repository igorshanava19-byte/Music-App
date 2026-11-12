"""Audio -> notes for each instrument stem.

Pitched instruments use Spotify's Basic Pitch (ONNX model, runs on CPU).
Drums use a simple band-split onset detector (kick / snare / hi-hat).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Note:
    start: float  # seconds
    end: float
    pitch: int  # MIDI number (GM drum number for drums)
    velocity: int = 90


@dataclass
class InstrumentSpec:
    min_hz: float
    max_hz: float
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_ms: float = 70
    monophonic: str | None = None  # "lowest" or "loudest"
    max_polyphony: int = 6


SPECS = {
    "bass": InstrumentSpec(30, 400, 0.7, 0.3, 90, monophonic="lowest"),
    "guitar": InstrumentSpec(78, 900, 0.65, 0.3, 70, max_polyphony=6),
    "piano": InstrumentSpec(27, 4200, 0.65, 0.3, 60, max_polyphony=10),
    "other": InstrumentSpec(60, 3000, 0.65, 0.32, 80, max_polyphony=6),
    "vocals": InstrumentSpec(80, 1100, 0.6, 0.3, 100, monophonic="loudest"),
}

_model = None
_model_lock = threading.RLock()


def _basic_pitch_model():
    global _model
    with _model_lock:
        if _model is None:
            import logging as _logging

            # basic-pitch logs a warning for every backend it can't find; we only need ONNX
            _logging.getLogger().setLevel(_logging.ERROR)
            from basic_pitch.inference import Model
            from basic_pitch import FilenameSuffix, build_icassp_2022_model_path

            _logging.getLogger().setLevel(_logging.INFO)
            _model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
        return _model


def transcribe_pitched(path: Path, instrument: str) -> list[Note]:
    from basic_pitch.inference import predict

    spec = SPECS[instrument]
    with _model_lock:  # the ONNX session is not shared across threads
        _, _, events = predict(
            str(path), _basic_pitch_model(),
            onset_threshold=spec.onset_threshold,
            frame_threshold=spec.frame_threshold,
            minimum_note_length=spec.min_note_ms,
            minimum_frequency=spec.min_hz,
            maximum_frequency=spec.max_hz,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    if not events:
        return []
    amps = np.array([e[3] for e in events])
    # drop the quietest ghost notes (bleed from other instruments, overtones)
    floor = max(0.12, float(np.percentile(amps, 90)) * 0.6)
    notes = [Note(float(s), float(e), int(p), int(np.clip(30 + a * 97, 1, 127)))
             for s, e, p, a, *_ in events if a >= floor]
    notes.sort(key=lambda n: (n.start, n.pitch))
    if spec.monophonic:
        notes = _make_monophonic(notes, spec.monophonic)
    else:
        notes = _limit_polyphony(notes, spec.max_polyphony)
    return notes


def _make_monophonic(notes: list[Note], keep: str) -> list[Note]:
    """Keep one note at a time: the lowest (bass) or loudest (melody)."""
    out: list[Note] = []
    for n in notes:
        if out and n.start < out[-1].end - 0.03:
            prev = out[-1]
            if abs(n.start - prev.start) < 0.06:  # simultaneous onset: choose one
                better = (n.pitch < prev.pitch) if keep == "lowest" else (n.velocity > prev.velocity)
                if better:
                    out[-1] = n
                continue
            prev.end = n.start  # later onset cuts the previous note
            if prev.end - prev.start < 0.05:
                out.pop()
        out.append(n)
    return out


def _limit_polyphony(notes: list[Note], limit: int) -> list[Note]:
    """Drop the quietest notes of any onset cluster bigger than the instrument can play."""
    out: list[Note] = []
    i = 0
    while i < len(notes):
        j = i
        while j < len(notes) and notes[j].start - notes[i].start < 0.04:
            j += 1
        cluster = notes[i:j]
        if len(cluster) > limit:
            cluster = sorted(cluster, key=lambda n: -n.velocity)[:limit]
        out.extend(sorted(cluster, key=lambda n: n.pitch))
        i = j
    return out


# ---------------------------------------------------------------- drums

KICK, SNARE, HIHAT = 36, 38, 42
_BANDS = {KICK: (20, 130), SNARE: (180, 4000), HIHAT: (7000, 16000)}


def transcribe_drums(samples: np.ndarray, sr: int) -> list[Note]:
    """Rough kick / snare / hi-hat transcription from the drum stem."""
    mono = samples.mean(0) if samples.ndim == 2 else samples
    target_sr = 32000
    mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    hop = 256
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=2048)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    notes: list[Note] = []
    envs = {}
    for drum, (lo, hi) in _BANDS.items():
        band = log_S[(freqs >= lo) & (freqs < hi)]
        env = np.maximum(0.0, np.diff(band, axis=1, prepend=band[:, :1])).mean(0)
        env = env / (np.percentile(env, 99.5) + 1e-9)
        envs[drum] = env
    for drum, env in envs.items():
        peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                                       delta=0.12 if drum != HIHAT else 0.08, wait=6)
        for p in peaks:
            if env[p] < {KICK: 0.45, SNARE: 0.3, HIHAT: 0.2}[drum]:
                continue
            if drum == SNARE and envs[KICK][p] > env[p] * 1.3:
                continue  # low-mid energy of a kick, not a snare
            t = librosa.frames_to_time(p, sr=target_sr, hop_length=hop)
            vel = int(np.clip(50 + env[p] * 60, 1, 127))
            notes.append(Note(float(t), float(t) + 0.1, drum, vel))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
