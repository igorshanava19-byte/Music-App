"""Small music-theory helpers: note spelling, keys, chord names."""
from __future__ import annotations

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Key signature (number of sharps > 0 / flats < 0) of each major key by tonic pitch class.
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}

# Chord qualities: suffix -> intervals above the root (first = root).
QUALITIES = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "dim": (0, 3, 6),
}

# MusicXML <kind> values for each suffix.
MUSICXML_KIND = {
    "": "major", "m": "minor", "7": "dominant", "maj7": "major-seventh",
    "m7": "minor-seventh", "sus4": "suspended-fourth", "sus2": "suspended-second", "dim": "diminished",
}


def key_fifths(tonic_pc: int, mode: str) -> int:
    major_pc = tonic_pc if mode == "major" else (tonic_pc + 3) % 12
    return MAJOR_FIFTHS[major_pc]


def use_flats(fifths: int) -> bool:
    return fifths < 0


def pc_name(pc: int, flats: bool) -> str:
    return (FLAT_NAMES if flats else SHARP_NAMES)[pc % 12]


def key_name(tonic_pc: int, mode: str) -> str:
    flats = use_flats(key_fifths(tonic_pc, mode))
    name = pc_name(tonic_pc, flats)
    # F# major / D# minor read better than Gb / Eb in their usual sharp spelling
    return f"{name} {mode}"


def chord_name(root_pc: int, quality: str, bass_pc: int | None, flats: bool) -> str:
    name = pc_name(root_pc, flats) + quality
    if bass_pc is not None and bass_pc != root_pc:
        name += "/" + pc_name(bass_pc, flats)
    return name


def step_alter(pc: int, flats: bool) -> tuple[str, int]:
    """Return (step letter, alter) for a pitch class, e.g. 10 -> ('B', -1) with flats."""
    name = pc_name(pc, flats)
    step = name[0]
    alter = 1 if name.endswith("#") else -1 if name.endswith("b") else 0
    return step, alter


def pitch_to_xml(midi: int, flats: bool) -> tuple[str, int, int]:
    """MIDI pitch -> (step, alter, octave) for MusicXML."""
    step, alter = step_alter(midi % 12, flats)
    octave = (midi - alter) // 12 - 1
    return step, alter, octave
