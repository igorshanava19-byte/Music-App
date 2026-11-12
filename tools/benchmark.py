"""Accuracy benchmark: synthetic songs with an exact answer key, run through the full pipeline and scored.

    python tools/benchmark.py              # make the songs (first time), process them, print the scores
    python tools/benchmark.py --rescore    # score the last run again without reprocessing
    python tools/benchmark.py --songs 3    # quicker run on the first 3 songs

Each song is rendered one instrument at a time from a General MIDI soundfont, so the answer key is
exact: key, tempo, chords and every note of every part. There is no singer: a soundfont "voice" is not
separated like a real one, so vocals and the minus track are checked on real songs instead.
Songs and results go to data/bench/ (report.json for the last run, history.jsonl for every run).
Rendering needs `pip install tinysoundfont` or the `fluidsynth` program; the soundfont defaults to the
small GM one that ships with pretty_midi (pass --soundfont for a better one, e.g. FluidR3_GM.sf2).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, notation, theory  # noqa: E402

SR = 44100
BENCH_DIR = config.DATA_DIR / "bench"  # MUSICAPP_DATA_DIR moves it, e.g. to compare models side by side
GENERATOR_VERSION = 2  # bump when the song generator changes, so old songs are regenerated

MAJOR_SCALE, MINOR_SCALE = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
TRIADS = {"major": ["", "m", "m", "", "", "m", "dim"], "minor": ["m", "dim", "", "m", "m", "", ""]}
PROGRESSIONS = {  # scale degrees, 0 = tonic
    "major": [[0, 4, 5, 3], [5, 3, 0, 4], [0, 5, 3, 4], [1, 4, 0, 0], [0, 3, 4, 3], [0, 3, 5, 4], [3, 4, 2, 5]],
    "minor": [[0, 5, 2, 6], [0, 3, 4, 0], [0, 6, 5, 6], [0, 5, 3, 4]],
}
# mir_eval chord labels for the app's chord suffixes
MIREX_QUALITY = {"": "maj", "m": "min", "7": "7", "maj7": "maj7", "m7": "min7", "sus4": "sus4", "sus2": "sus2", "dim": "dim"}
PROGRAMS = {"piano": 0, "guitar": 25, "bass": 33}  # GM, 0-based
GAINS = {"drums": 0.8, "bass": 0.9, "piano": 0.8, "guitar": 0.8}
# answer-key part -> the stem/part the app writes it to
PART_OF = {"guitar": "guitar", "piano": "piano", "bass": "bass"}


# ---------------------------------------------------------------- song generator

def make_song(seed: int) -> dict:
    """A random but musical song: notes per part, chords, key and tempo."""
    rng = random.Random(seed)
    tonic, mode = rng.randrange(12), "major" if rng.random() < 0.7 else "minor"
    bpm = rng.randrange(72, 141)
    beat = 60.0 / bpm
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    has = {"piano": rng.random() < 0.85, "guitar": rng.random() < 0.7, "bass": rng.random() < 0.8,
           "drums": rng.random() < 0.85}
    if not (has["piano"] or has["guitar"]):
        has[rng.choice(["piano", "guitar"])] = True
    beats_per_chord = 2 if rng.random() < 0.3 else 4

    def chord(deg: int) -> tuple[int, str]:
        q = TRIADS[mode][deg]
        if mode == "minor" and deg == 4 and rng.random() < 0.4:
            q = "7"  # harmonic-minor V7
        elif rng.random() < 0.2:
            q = {"": "7" if deg == 4 else "maj7", "m": "m7"}.get(q, q)
        return (tonic + scale[deg]) % 12, q

    loops = [[chord(d) for d in rng.choice(PROGRESSIONS[mode])] for _ in range(2)]
    chords_seq = []  # 16 bars: verse loop x2, chorus loop x2 (a 4-chord loop is 4 bars, or 2 with 2 chords a bar)
    for loop in (loops[0], loops[0], loops[1], loops[1]):
        chords_seq += loop * (1 if beats_per_chord == 4 else 2)
    n_chord_beats = len(chords_seq) * beats_per_chord
    lead_in = 4  # one bar of drums (or silence) first

    parts = {k: [] for k in ("piano", "guitar", "bass", "drums")}
    truth_chords = []
    p_style = rng.choice(["block", "arpeggio", "pad"])
    g_style = rng.choice(["strum", "strum_half", "arpeggio"])
    b_style = rng.choice(["half", "eighths", "root_fifth"])

    def add(part, pitch, start_beat, dur_beats, vel):
        vel = max(30, min(127, vel + rng.randrange(-8, 9)))
        parts[part].append((int(pitch), start_beat * beat, (start_beat + dur_beats) * beat, vel))

    def above(pc, low):  # lowest pitch with pitch class `pc` at or above `low`
        return low + (pc - low) % 12

    for i, (root, q) in enumerate(chords_seq):
        b0 = lead_in + i * beats_per_chord
        span = beats_per_chord
        ivs = theory.QUALITIES[q]
        pcs = [(root + iv) % 12 for iv in ivs]
        truth_chords.append({"start": round(b0 * beat, 4), "end": round((b0 + span) * beat, 4),
                             "label": f"{theory.SHARP_NAMES[root]}:{MIREX_QUALITY[q]}",
                             "name": theory.chord_name(root, q, None, False)})
        if has["piano"]:
            rh = sorted(above(pc, 60) for pc in pcs)
            lh = above(root, 36)
            if p_style == "arpeggio":
                add("piano", lh, b0, span * 0.95, 75)
                seq = rh + rh[-2:0:-1]
                for k in range(span * 2):
                    add("piano", seq[k % len(seq)], b0 + k * 0.5, 0.48, 70)
            else:
                step = span if p_style == "pad" else 2
                for s in range(0, span, step):
                    for p in [lh] + rh:
                        add("piano", p, b0 + s, step * 0.95, 72)
        if has["guitar"]:
            r = above(root, 40)
            v = [r, r + ivs[2], r + 12, r + 12 + ivs[1], r + 12 + ivs[2]]
            if len(ivs) > 3:
                v[2] = r + ivs[3]
            v = sorted(v)
            if g_style == "arpeggio":
                pat = [v[0], v[2], v[3], v[4], v[3], v[2], v[1], v[2]]
                for k in range(span * 2):
                    add("guitar", pat[k % len(pat)], b0 + k * 0.5, 0.48, 80)
            else:
                step = 1 if g_style == "strum" else 2
                for s in range(0, span, step):
                    for j, p in enumerate(v):  # a down-strum: low to high, 12 ms apart
                        add("guitar", p, b0 + s + j * 0.012 / beat, step * 0.9, 82)
        if has["bass"]:
            r = above(root, 28)
            fifth = r + 7
            if b_style == "half":
                for s in range(0, span, 2):
                    add("bass", r, b0 + s, 1.9, 95)
            elif b_style == "eighths":
                for k in range(span * 2):
                    add("bass", r, b0 + k * 0.5, 0.45, 90)
            else:
                for s in range(span):
                    add("bass", r if s % 2 == 0 else fifth, b0 + s, 0.9, 95)

    total_beats = lead_in + n_chord_beats
    if has["drums"]:
        for bt in range(total_beats):
            if bt % 2 == 0:
                add("drums", 36, bt, 0.1, 100)
            else:
                add("drums", 38, bt, 0.1, 95)
            for h in (0, 0.5):
                add("drums", 42, bt + h, 0.05, 65)
    return {
        "seed": seed, "generator": GENERATOR_VERSION, "bpm": bpm, "tonic": tonic, "mode": mode,
        "key": theory.key_name(tonic, mode), "duration": round(total_beats * beat + 1.5, 3),
        "chords": [{"start": 0.0, "end": round(lead_in * beat, 4), "label": "N", "name": "N.C."}] + truth_chords,
        "parts": {k: [list(n) for n in v] for k, v in parts.items() if has[k] and v},
        "styles": {"piano": p_style, "guitar": g_style, "bass": b_style, "beats_per_chord": beats_per_chord},
    }


# ---------------------------------------------------------------- rendering

def _render_tinysoundfont(notes, program, is_drum, sf2, n_samples):
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(str(sf2))
    synth.program_select(0, sfid, 0, 0 if is_drum else program, is_drums=is_drum)
    events = sorted([(s, 1, p, v) for p, s, e, v in notes] + [(e, 0, p, 0) for p, s, e, v in notes])
    out, pos = [], 0
    for t, on, p, v in events:
        target = min(n_samples, int(t * SR))
        if target > pos:
            out.append(np.frombuffer(synth.generate(target - pos), dtype=np.float32).reshape(-1, 2))
            pos = target
        synth.noteon(0, p, v) if on else synth.noteoff(0, p)
    if n_samples > pos:
        out.append(np.frombuffer(synth.generate(n_samples - pos), dtype=np.float32).reshape(-1, 2))
    return np.concatenate(out)[:n_samples]


def _render_fluidsynth(notes, program, is_drum, sf2, n_samples):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    inst.notes = [pretty_midi.Note(v, p, s, e) for p, s, e, v in notes]
    pm.instruments.append(inst)
    with tempfile.TemporaryDirectory() as tmp:
        mid, wav = Path(tmp) / "t.mid", Path(tmp) / "t.wav"
        pm.write(str(mid))
        subprocess.run(["fluidsynth", "-ni", "-g", "0.5", "-r", str(SR), "-F", str(wav), str(sf2), str(mid)],
                       check=True, capture_output=True)
        audio, _ = sf.read(str(wav), dtype="float32", always_2d=True)
    out = np.zeros((n_samples, 2), np.float32)
    out[: min(n_samples, len(audio))] = audio[:n_samples]
    return out


def render_song(song: dict, out_dir: Path, sf2: Path) -> None:
    import importlib.util

    if importlib.util.find_spec("tinysoundfont"):
        render = _render_tinysoundfont
    else:
        if not shutil.which("fluidsynth"):
            raise SystemExit("To render the test songs, install one of: `pip install tinysoundfont` or the fluidsynth program.")
        render = _render_fluidsynth
    n = int(song["duration"] * SR)
    tracks = {}
    for part, notes in song["parts"].items():
        audio = render(notes, PROGRAMS.get(part, 0), part == "drums", sf2, n)
        rms = np.sqrt((audio ** 2).mean()) + 1e-9
        tracks[part] = audio * (GAINS[part] * 0.1 / rms)  # equal-ish loudness, then the mix balance
    mix = sum(tracks.values())
    scale = 0.9 / (np.abs(mix).max() + 1e-9)
    sf.write(out_dir / "mix.wav", mix * scale, SR)
    (out_dir / "truth.json").write_text(json.dumps(song))


# ---------------------------------------------------------------- scoring

def score(song: dict, run_dir: Path) -> dict:
    import mir_eval

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    out = {"key": float(result["key_tonic"] == song["tonic"] and result["key_mode"] == song["mode"])}
    # tempo: within 4%
    out["tempo"] = float(abs(result["tempo"] - song["bpm"]) / song["bpm"] < 0.04)

    # chords, by duration (mir_eval: majmin = root + major/minor right; sevenths count as their triad)
    ref_iv = np.array([[c["start"], c["end"]] for c in song["chords"]])
    ref_lab = [c["label"] for c in song["chords"]]
    segs = result["chords"]["segments"]
    starts = [s["start"] for s in segs]
    est_iv, est_lab = [[0.0, starts[0] if starts else song["duration"]]], ["N"]
    for k, s in enumerate(segs):
        end = starts[k + 1] if k + 1 < len(segs) else result["duration"]
        if end > s["start"]:
            est_iv.append([s["start"], end])
            est_lab.append(f"{theory.SHARP_NAMES[s['root']]}:{MIREX_QUALITY[s['quality']]}")
    est_iv = np.array(est_iv)
    ref_iv, ref_lab = mir_eval.util.adjust_intervals(ref_iv, ref_lab, 0.0, ref_iv[-1, 1], "N", "N")
    est_iv, est_lab = mir_eval.util.adjust_intervals(est_iv, est_lab, 0.0, ref_iv[-1, 1], "N", "N")
    iv, rl, el = mir_eval.util.merge_labeled_intervals(ref_iv, ref_lab, est_iv, est_lab)
    dur = mir_eval.util.intervals_to_durations(iv)
    for name, fn in (("chords", mir_eval.chord.majmin), ("chord_roots", mir_eval.chord.root),
                     ("chords_7ths", mir_eval.chord.sevenths)):
        cmp = fn(rl, el)
        out[name] = float(mir_eval.chord.weighted_accuracy(cmp, dur)) if (cmp >= 0).any() else None

    # notes: onset within 50 ms and the right pitch (mir_eval note F-measure, offsets ignored)
    est = {}
    mid_path = run_dir / "score.mid"
    if mid_path.exists():
        key_of = {s.name: s.key for s in notation.PART_SPECS}
        for inst in pretty_midi.PrettyMIDI(str(mid_path)).instruments:
            est[key_of.get(inst.name, inst.name)] = [(n.pitch, n.start, n.end) for n in inst.notes]

    def f1(ref, est_notes):
        if not ref:
            return None
        if not est_notes:
            return 0.0
        ri = np.array([[s, max(e, s + 0.01)] for _, s, e, *_ in ref])
        ei = np.array([[s, max(e, s + 0.01)] for _, s, e in est_notes])
        rp = pretty_midi.note_number_to_hz(np.array([p for p, *_ in ref]))
        ep = pretty_midi.note_number_to_hz(np.array([p for p, *_ in est_notes]))
        return float(mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=0.05,
                                                                        offset_ratio=None)[2])

    for part, stem in PART_OF.items():
        out[f"notes_{part}"] = f1(song["parts"].get(part), est.get(stem, []))
    # every pitched note, whichever part it was written to: separates "wrong notes" from "right notes, wrong part"
    all_ref = [n for p, v in song["parts"].items() if p != "drums" for n in v]
    all_est = [n for k, v in est.items() if k != "drums" for n in v]
    out["notes_all_parts"] = f1(all_ref, all_est)

    return out


# ---------------------------------------------------------------- main

PCT_METRICS = ["key", "tempo", "chords", "chord_roots", "chords_7ths",
               "notes_guitar", "notes_piano", "notes_bass", "notes_all_parts"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", type=int, default=10, help="how many test songs (default 10)")
    ap.add_argument("--rescore", action="store_true", help="score the last run again, don't reprocess")
    ap.add_argument("--soundfont", type=Path, default=Path(pretty_midi.__file__).parent / "TimGM6mb.sf2")
    ap.add_argument("--note", default="", help="a label for this run in history.jsonl")
    args = ap.parse_args()

    rows = []
    for i in range(args.songs):
        song_dir = BENCH_DIR / f"song{i + 1:02d}"
        song_dir.mkdir(parents=True, exist_ok=True)
        truth_path = song_dir / "truth.json"
        if not truth_path.exists() or json.loads(truth_path.read_text()).get("generator") != GENERATOR_VERSION:
            print(f"rendering song {i + 1} ...", flush=True)
            render_song(make_song(1000 + i), song_dir, args.soundfont)
        song = json.loads(truth_path.read_text())
        run_dir = song_dir / "run"
        secs = None
        if not args.rescore or not (run_dir / "result.json").exists():
            from app import pipeline

            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir()
            shutil.copy(song_dir / "mix.wav", run_dir / "input.wav")
            t0 = time.time()
            pipeline.run(run_dir, {"type": "file", "filename": "input.wav", "title": song_dir.name},
                         lambda *a: None)
            secs = round(time.time() - t0, 1)
        row = {"song": song_dir.name, "truth_key": song["key"], "bpm": song["bpm"],
               "parts": "+".join(song["parts"]), "seconds": secs, **score(song, run_dir)}
        rows.append(row)
        print(_fmt_row(row), flush=True)

    avg = {}
    for m in PCT_METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None]
        avg[m] = round(float(np.mean(vals)), 3) if vals else None
    print("\nAVERAGE over", len(rows), "songs (target 90%):")
    for m in PCT_METRICS:
        if avg[m] is not None:
            print(f"  {m:16s} {avg[m] * 100:5.1f}%  {'PASS' if avg[m] >= 0.9 else 'below target'}")
    report = {"time": time.strftime("%Y-%m-%d %H:%M"), "note": args.note, "average": avg, "songs": rows}
    (BENCH_DIR / "report.json").write_text(json.dumps(report, indent=1))
    with (BENCH_DIR / "history.jsonl").open("a") as h:
        h.write(json.dumps({k: report[k] for k in ("time", "note", "average")}) + "\n")


def _fmt_row(r: dict) -> str:
    cells = [f"{r['song']} {r['truth_key']:9s} {r['bpm']:3d}bpm"]
    for m in PCT_METRICS:
        v = r.get(m)
        cells.append(f"{m.replace('notes_', 'n_')}={'-' if v is None else f'{v * 100:.0f}'}")
    return "  ".join(cells)


if __name__ == "__main__":
    main()
