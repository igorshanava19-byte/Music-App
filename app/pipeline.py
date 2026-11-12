"""The full processing pipeline for one song."""
from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Callable


from . import analysis, audio_io, config, notation, separation, theory, transcription

log = logging.getLogger(__name__)

# Fraction of the progress bar each step takes.
STEPS = [
    ("fetch", "Getting the audio", 0.06),
    ("separate", "Separating instruments", 0.44),
    ("minus", "Building the minus track", 0.12),
    ("analyze", "Finding tempo, key and chords", 0.12),
    ("transcribe", "Transcribing notes", 0.2),
    ("notate", "Writing sheet music", 0.06),
]

PITCHED_STEMS = ["vocals", "guitar", "piano", "other", "bass"]


class Progress:
    def __init__(self, report: Callable[[str, str, float], None]):
        self.report = report
        self._base = {}
        acc = 0.0
        for key, _, w in STEPS:
            self._base[key] = acc
            acc += w
        self._weights = {k: w for k, _, w in STEPS}
        self._labels = {k: label for k, label, _ in STEPS}

    def __call__(self, step: str, frac: float = 0.0, detail: str | None = None) -> None:
        total = self._base[step] + self._weights[step] * max(0.0, min(1.0, frac))
        self.report(step, detail or self._labels[step], total)


def run(job_dir: Path, source: dict, report: Callable[[str, str, float], None]) -> dict:
    progress = Progress(report)
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(exist_ok=True)

    # 1. get audio -------------------------------------------------------------
    progress("fetch", 0.0)
    title = source.get("title") or "Untitled"
    if source["type"] == "url":
        downloaded, title = audio_io.download_youtube(
            source["url"], job_dir, progress=lambda f: progress("fetch", 0.8 * f, "Downloading from YouTube"))
        src_file = downloaded
    else:
        src_file = job_dir / source["filename"]
    original = job_dir / "original.wav"
    audio_io.decode_to_wav(src_file, original)
    if src_file != original and src_file.exists():
        src_file.unlink()
    mix, sr = audio_io.load(original)
    duration = mix.shape[1] / sr
    if duration < 3:
        raise ValueError("The audio is too short (under 3 seconds).")
    audio_io.encode_mp3(original, job_dir / "original.mp3")
    progress("fetch", 1.0)

    # 2. separate --------------------------------------------------------------
    progress("separate", 0.0, f"Separating instruments ({config.device().upper()})")
    stems = separation.run_model(config.STEMS_MODEL, mix, sr,
                                 progress=lambda f: progress("separate", f, f"Separating instruments ({config.device().upper()})"))
    instruments = []
    for name, audio in stems.items():
        audio_io.save(stems_dir / f"{name}.flac", audio, sr)
        pres = separation.stem_presence(mix, audio, sr)
        instruments.append({"stem": name, "label": separation.STEM_LABELS.get(name, name.title()),
                            "present": separation.is_present(pres), **pres,
                            "file": f"stems/{name}.flac"})

    # 3. minus track -----------------------------------------------------------
    minus_model = config.minus_model()
    if minus_model != config.STEMS_MODEL:
        vocals = separation.run_model(minus_model, mix, sr,
                                      progress=lambda f: progress("minus", f))["vocals"]
    else:
        vocals = stems["vocals"]
    minus = mix - vocals  # keeps every instrument exactly as mixed
    audio_io.save(job_dir / "minus.wav", minus, sr)
    audio_io.encode_mp3(job_dir / "minus.wav", job_dir / "minus.mp3")
    audio_io.save(job_dir / "vocals_only.wav", vocals, sr)
    audio_io.encode_mp3(job_dir / "vocals_only.wav", job_dir / "vocals_only.mp3")
    (job_dir / "vocals_only.wav").unlink()
    progress("minus", 1.0)

    # 4. tempo / key / chords --------------------------------------------------
    progress("analyze", 0.0)
    mono = {k: analysis.to_mono(v, sr) for k, v in stems.items()}
    present = {i["stem"] for i in instruments if i["present"]}
    harmonic = sum(mono[k] for k in ("guitar", "piano", "other") if k in mono)
    if not any(k in present for k in ("guitar", "piano", "other")):
        harmonic = harmonic + mono["vocals"] * 0.5  # a cappella-ish: use the voice
    rhythm = harmonic + mono["bass"] + mono["drums"]
    tempo, beats = analysis.beat_grid(rhythm, duration)
    progress("analyze", 0.3)
    bass_for_chords = mono["bass"] if "bass" in present else None
    # key first (for chord spelling), from the same harmonic signal
    import librosa
    key_chroma = librosa.feature.chroma_cqt(y=harmonic + 0.5 * mono["bass"], sr=analysis.SR, hop_length=analysis.HOP)
    tonic, mode, key_conf = analysis.detect_key(key_chroma)
    fifths = theory.key_fifths(tonic, mode)
    flats = theory.use_flats(fifths)
    chord_segments, _ = analysis.recognize_chords(harmonic, bass_for_chords, beats, flats)
    progress("analyze", 0.8)
    kick = analysis.beat_strengths(mono["drums"], beats) if "drums" in present else None
    phase = analysis.downbeat_phase(beats, [c["start_beat"] for c in chord_segments], kick)
    grid = notation.Grid(beats, phase)
    n_bars = max(1, math.ceil(grid.tick(duration) / notation.BAR_TICKS))
    chart = _chord_chart(chord_segments, grid, n_bars, duration)
    progress("analyze", 1.0)

    # 5. transcribe ------------------------------------------------------------
    parts: dict[str, list[transcription.Note]] = {}
    todo = [s for s in PITCHED_STEMS if s in present] + (["drums"] if "drums" in present else [])
    for i, stem in enumerate(todo):
        label = separation.STEM_LABELS.get(stem, stem)
        progress("transcribe", i / max(1, len(todo)), f"Transcribing {label}")
        if stem == "drums":
            notes = transcription.transcribe_drums(stems["drums"], sr)
        else:
            notes = transcription.transcribe_pitched(stems_dir / f"{stem}.flac", stem)
        if len(notes) >= 8:
            parts[stem] = notes
    progress("transcribe", 1.0)

    # 6. notation --------------------------------------------------------------
    progress("notate", 0.0)
    xml, part_info = notation.build_musicxml(title, parts, chord_segments, grid, tonic, mode, tempo, n_bars)
    (job_dir / "score.musicxml").write_text(xml, encoding="utf-8")
    notation.build_midi(parts, tempo, job_dir / "score.mid")
    notated = {p["stem"] for p in part_info}
    for inst in instruments:
        inst["notated"] = inst["stem"] in notated
    progress("notate", 1.0)

    result = {
        "title": title,
        "duration": round(duration, 2),
        "tempo": round(tempo, 1),
        "key": theory.key_name(tonic, mode),
        "key_tonic": tonic,
        "key_mode": mode,
        "key_confidence": round(key_conf, 2),
        "flats": flats,
        "time_signature": "4/4",
        "device": config.device(),
        "instruments": instruments,
        "parts": part_info,
        "chords": chart,
        "files": {
            "original": "original.mp3",
            "minus_mp3": "minus.mp3",
            "minus_wav": "minus.wav",
            "vocals": "vocals_only.mp3",
            "musicxml": "score.musicxml",
            "midi": "score.mid",
        },
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    original.unlink(missing_ok=True)  # the mp3 + stems are enough from here on
    return result


def _chord_chart(segments: list[dict], grid: notation.Grid, n_bars: int, duration: float) -> dict:
    """Chords laid out bar by bar, as the chord-chart view needs them."""
    bars = [{"bar": b + 1, "time": round(max(0.0, grid.bar_start_time(b)), 3), "chords": []} for b in range(n_bars)]
    for seg in segments:
        tick = grid.beat_to_tick(seg["start_beat"])
        bar = tick // notation.BAR_TICKS
        if 0 <= bar < n_bars:
            bars[bar]["chords"].append({
                "beat": (tick % notation.BAR_TICKS) // notation.SUBDIV,
                "name": seg["name"], "root": seg["root"], "quality": seg["quality"], "bass": seg["bass"],
                "end_tick": grid.beat_to_tick(seg["end_beat"]),
            })
    # carry a still-sounding chord into bars without a change, so every bar shows
    # something (but not across silence, where no chord was detected)
    current = None
    for b, bar in enumerate(bars):
        bar_tick = b * notation.BAR_TICKS
        if current is not None and current["end_tick"] <= bar_tick:
            current = None
        if not (bar["chords"] and bar["chords"][0]["beat"] == 0) and current is not None:
            bar["chords"].insert(0, {**current, "beat": 0, "continued": True})
        if bar["chords"]:
            current = bar["chords"][-1]
    for bar in bars:
        for c in bar["chords"]:
            c.pop("end_tick", None)
    return {"beats_per_bar": notation.BEATS_PER_BAR, "bars": bars,
            "segments": [{"start": round(float(grid.beats[s["start_beat"]]), 3) if s["start_beat"] < len(grid.beats) else duration,
                          "name": s["name"], "root": s["root"], "quality": s["quality"], "bass": s["bass"]}
                         for s in segments]}


def cleanup_job(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)
