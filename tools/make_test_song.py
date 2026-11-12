"""Generate a synthetic multi-instrument test song with a known answer key.

Renders a General MIDI arrangement (drums, bass, piano, guitar, lead "vocal")
through FluidSynth so the pipeline can be checked against ground truth:
the chords, tempo and key are written next to the audio as JSON.

Usage:
    python tools/make_test_song.py out_dir [--soundfont path.sf2]
Requires the `fluidsynth` command-line program and a GM soundfont.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pretty_midi

BPM = 100
BEAT = 60.0 / BPM
BAR = 4 * BEAT
LEAD_IN_BARS = 1  # one bar of drums only before the chords start

# (chord name, root midi for bass, chord tones for piano/guitar)
CHORDS = {
    "G": (43, [55, 59, 62, 67]),
    "D": (38, [50, 54, 57, 62]),
    "Em": (40, [52, 55, 59, 64]),
    "C": (36, [48, 52, 55, 60]),
    "Am": (45, [57, 60, 64, 69]),
    "D7": (38, [50, 54, 57, 60]),
    "Cmaj7": (36, [48, 52, 55, 59]),
    "Bm": (47, [54, 59, 62, 66]),
}
PROGRESSION = ["G", "D", "Em", "C"] * 2 + ["Am", "C", "G", "D7"] + ["Em", "Bm", "Cmaj7", "D"] + ["G", "D", "Em", "C"]

# Simple melody, one list of (pitch, beats) per bar; None = rest.
MELODY = [
    [(74, 1), (71, 1), (67, 2)], [(69, 1), (66, 1), (62, 2)], [(67, 1.5), (71, 0.5), (74, 2)], [(72, 2), (None, 2)],
] * 5


def build_midi() -> tuple[pretty_midi.PrettyMIDI, list[dict]]:
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    bass = pretty_midi.Instrument(program=33, name="Bass")  # fingered bass
    piano = pretty_midi.Instrument(program=0, name="Piano")
    guitar = pretty_midi.Instrument(program=25, name="Guitar")  # steel acoustic
    lead = pretty_midi.Instrument(program=73, name="Lead")  # flute as a stand-in voice

    n_bars = LEAD_IN_BARS + len(PROGRESSION)
    for bar in range(n_bars):
        t0 = bar * BAR
        for beat in range(4):
            tb = t0 + beat * BEAT
            if beat in (0, 2):
                drums.notes.append(pretty_midi.Note(100, 36, tb, tb + 0.1))
            if beat in (1, 3):
                drums.notes.append(pretty_midi.Note(95, 38, tb, tb + 0.1))
            for half in (0, 0.5):
                drums.notes.append(pretty_midi.Note(70, 42, tb + half * BEAT, tb + half * BEAT + 0.05))

    truth = []
    for i, name in enumerate(PROGRESSION):
        t0 = (LEAD_IN_BARS + i) * BAR
        root, tones = CHORDS[name]
        truth.append({"bar": LEAD_IN_BARS + i, "start": round(t0, 4), "end": round(t0 + BAR, 4), "chord": name})
        # bass: root on 1 and 3, fifth-ish passing on the "and" of 4
        for b, p in ((0, root), (2, root), (3.5, root + 7)):
            bass.notes.append(pretty_midi.Note(100, p, t0 + b * BEAT, t0 + b * BEAT + (0.45 if b == 3.5 else 1.9) * BEAT))
        # piano: block chords on beats 1 and 3
        for b in (0, 2):
            for p in tones:
                piano.notes.append(pretty_midi.Note(80, p, t0 + b * BEAT, t0 + (b + 1.9) * BEAT))
        # guitar: eighth-note arpeggio an octave around the piano voicing
        arp = [tones[0], tones[1], tones[2], tones[3], tones[2], tones[1], tones[2], tones[3]]
        for k, p in enumerate(arp):
            st = t0 + k * 0.5 * BEAT
            guitar.notes.append(pretty_midi.Note(85, p, st, st + 0.48 * BEAT))
        # melody
        t = t0
        for pitch, beats in MELODY[i % len(MELODY)]:
            if pitch is not None:
                lead.notes.append(pretty_midi.Note(95, pitch, t, t + beats * BEAT * 0.95))
            t += beats * BEAT

    pm.instruments += [drums, bass, piano, guitar, lead]
    return pm, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--soundfont", default="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("fluidsynth"):
        raise SystemExit("fluidsynth not found on PATH")

    pm, truth = build_midi()
    midi_path = args.out_dir / "test_song.mid"
    wav_path = args.out_dir / "test_song.wav"
    pm.write(str(midi_path))
    subprocess.run(
        ["fluidsynth", "-ni", "-g", "0.8", "-r", "44100", "-F", str(wav_path), args.soundfont, str(midi_path)],
        check=True, capture_output=True,
    )
    (args.out_dir / "test_song_truth.json").write_text(json.dumps(
        {"bpm": BPM, "key": "G major", "time_signature": "4/4", "chords": truth}, indent=2))
    print(f"wrote {wav_path}")


if __name__ == "__main__":
    main()
