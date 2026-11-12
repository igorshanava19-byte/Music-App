"""Tempo / beat grid, key, and chord recognition from the separated stems."""
from __future__ import annotations

import numpy as np
import librosa

from . import theory

SR = 22050
HOP = 512


def to_mono(samples: np.ndarray, sr: int) -> np.ndarray:
    mono = samples.mean(0) if samples.ndim == 2 else samples
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono.astype(np.float32)


# ---------------------------------------------------------------- beats / tempo

def beat_grid(rhythm_mono: np.ndarray, duration: float) -> tuple[float, np.ndarray]:
    """Return (tempo_bpm, beat_times) covering the whole song."""
    onset = librosa.onset.onset_strength(y=rhythm_mono, sr=SR, hop_length=HOP, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:  # no clear pulse: fall back to a steady grid
        tempo = tempo if tempo > 0 else 120.0
        return tempo, np.arange(0.0, duration + 60 / tempo, 60 / tempo)
    period = float(np.median(np.diff(beats)))
    # extend the grid to the start and end of the song
    head = np.arange(beats[0] - period, -period * 0.5, -period)[::-1]
    tail = np.arange(beats[-1] + period, duration + period, period)
    beats = np.concatenate([head, beats, tail])
    return 60.0 / period, beats


def downbeat_phase(beats: np.ndarray, chord_change_beats: list[int], low_onsets: np.ndarray | None) -> int:
    """Pick which beat (0-3) starts each bar, assuming 4/4.

    Chords tend to change on the downbeat and kick drums land there, so we
    score every phase by how many changes/kicks fall on it.
    """
    scores = np.zeros(4)
    for b in chord_change_beats:
        scores[b % 4] += 1.0
    if low_onsets is not None and len(low_onsets) == len(beats):
        for i, v in enumerate(low_onsets):
            scores[i % 4] += 0.5 * v
    return int(np.argmax(scores))


def beat_strengths(mono: np.ndarray, beats: np.ndarray, fmax: float = 150.0) -> np.ndarray:
    """Low-frequency onset strength at each beat (kick-drum proxy), normalized 0-1."""
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = S[freqs < fmax]
    env = np.maximum(0, np.diff(low, axis=1, prepend=low[:, :1])).sum(0)
    frames = librosa.time_to_frames(np.clip(beats, 0, None), sr=SR, hop_length=HOP)
    frames = np.clip(frames, 0, len(env) - 1)
    vals = np.array([env[max(0, f - 2): f + 3].max() for f in frames])
    return vals / (vals.max() + 1e-9)


# ---------------------------------------------------------------- key

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(chroma: np.ndarray) -> tuple[int, str, float]:
    """Return (tonic_pc, 'major'|'minor', confidence) from a [12, T] chromagram."""
    profile = chroma.mean(1)
    best = (0, "major", -2.0)
    for mode, ref in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(profile, np.roll(ref, tonic))[0, 1])
            if r > best[2]:
                best = (tonic, mode, r)
    return best


# ---------------------------------------------------------------- chords

# Small prior against rarer/extended chords so plain triads win ties.
_QUALITY_BIAS = {"": 0.0, "m": 0.0, "7": -0.045, "maj7": -0.05, "m7": -0.05,
                 "sus4": -0.05, "sus2": -0.06, "dim": -0.06}


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    temps, labels = [], []
    for quality, intervals in theory.QUALITIES.items():
        for root in range(12):
            t = np.zeros(12)
            for k, iv in enumerate(intervals):
                t[(root + iv) % 12] = 1.0 if k == 0 else 0.85 if k < 3 else 0.7
            temps.append(t / np.linalg.norm(t))
            labels.append((root, quality))
    return np.array(temps), labels


def _beat_rms(signal: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """RMS level of `signal` inside each beat."""
    edges = (np.clip(beats, 0, None) * SR).astype(int)
    out = np.zeros(len(beats))
    for i in range(len(beats)):
        seg = signal[edges[i]: edges[i + 1] if i + 1 < len(beats) else len(signal)]
        out[i] = np.sqrt((seg ** 2).mean()) if len(seg) else 0.0
    return out


def recognize_chords(harmonic: np.ndarray, bass: np.ndarray | None, beats: np.ndarray,
                     flats: bool) -> tuple[list[dict], np.ndarray]:
    """Chord label for each beat-length segment.

    Returns (segments, chroma): segments are [{start_beat, end_beat, root,
    quality, bass, name}] with start/end as indices into `beats`; chroma is
    the [12, T] chromagram (reused for key detection).
    """
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP, bins_per_octave=36)
    n = chroma.shape[1]
    beat_frames = librosa.time_to_frames(np.clip(beats, 0, None), sr=SR, hop_length=HOP)
    beat_frames = np.clip(beat_frames, 0, n)
    # aggregate per beat (median is robust to passing tones)
    seg_chroma = np.zeros((12, len(beats)))
    for i in range(len(beats)):
        a = beat_frames[i]
        b = beat_frames[i + 1] if i + 1 < len(beats) else n
        if b > a:
            seg_chroma[:, i] = np.median(chroma[:, a:b], axis=1)
    energy = _beat_rms(harmonic, beats)

    bass_seg = None
    if bass is not None:
        bchroma = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E1"), n_octaves=3)
        bass_seg = np.zeros((12, len(beats)))
        for i in range(len(beats)):
            a = beat_frames[i]
            b = beat_frames[i + 1] if i + 1 < len(beats) else n
            if b > a:
                bass_seg[:, i] = bchroma[:, a:b].mean(1)
        bass_energy = _beat_rms(bass, beats)
        bass_active = bass_energy > (np.percentile(bass_energy, 90) * 0.1)

    temps, labels = _templates()
    norm = seg_chroma / (np.linalg.norm(seg_chroma, axis=0, keepdims=True) + 1e-9)
    scores = temps @ norm  # [n_labels, n_beats]
    scores += np.array([_QUALITY_BIAS[q] for _, q in labels])[:, None]
    if bass_seg is not None:
        bnorm = bass_seg / (bass_seg.max(0, keepdims=True) + 1e-9)
        roots = np.array([r for r, _ in labels])
        scores += 0.12 * bnorm[roots] * bass_active[None, :]

    # "no chord" state for silent beats
    silent = energy < (np.percentile(energy, 95) * 0.03)
    n_states = len(labels) + 1
    nc_score = np.where(silent, 1.0, 0.3)
    all_scores = np.vstack([scores, nc_score[None, :]])
    prob = np.exp(18.0 * (all_scores - all_scores.max(0, keepdims=True)))
    prob /= prob.sum(0, keepdims=True)
    transition = librosa.sequence.transition_loop(n_states, 0.85)
    path = librosa.sequence.viterbi(prob, transition)

    # merge consecutive beats with the same label
    segments = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            state = int(path[start])
            if state < len(labels):
                root, quality = labels[state]
                bass_pc = None
                if bass_seg is not None and bass_active[start:i].mean() > 0.5:
                    bpc = int(np.argmax(bass_seg[:, start:i].sum(1)))
                    tones = [(root + iv) % 12 for iv in theory.QUALITIES[quality]]
                    if bpc != root and bpc in tones[1:]:
                        bass_pc = bpc
                segments.append({"start_beat": start, "end_beat": i, "root": root, "quality": quality,
                                 "bass": bass_pc, "name": theory.chord_name(root, quality, bass_pc, flats)})
            start = i
    # drop one-beat blips between identical neighbours
    cleaned = []
    for seg in segments:
        if cleaned and seg["name"] == cleaned[-1]["name"] and seg["start_beat"] <= cleaned[-1]["end_beat"] + 1:
            cleaned[-1]["end_beat"] = seg["end_beat"]
        else:
            cleaned.append(seg)
    return cleaned, chroma
