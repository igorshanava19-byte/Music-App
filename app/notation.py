"""Notes -> sheet music.

Snaps transcribed notes onto the detected beat grid (16th-note resolution, 4/4),
picks guitar/bass fingerings for the tab staff, and writes a MusicXML score
(one part per instrument) plus a MIDI file.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import numpy as np

from . import theory
from .transcription import HIHAT, KICK, SNARE, Note

SUBDIV = 4  # grid steps per beat (16th notes)
BEATS_PER_BAR = 4
BAR_TICKS = SUBDIV * BEATS_PER_BAR

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4, low -> high
BASS_TUNING = [28, 33, 38, 43]  # E1 A1 D2 G2


@dataclass
class PartSpec:
    key: str  # stem name
    name: str
    abbreviation: str
    kind: str  # "treble" | "grand" | "guitar" | "bass" | "drums"
    midi_program: int  # 1-based GM program
    chords: bool = True


PART_SPECS = [
    PartSpec("vocals", "Vocal Melody", "Voc.", "treble", 54),
    PartSpec("guitar", "Guitar", "Gtr.", "guitar", 26),
    PartSpec("piano", "Piano / Keys", "Pno.", "grand", 1),
    PartSpec("other", "Other (Synth / Strings)", "Oth.", "grand", 49),
    PartSpec("bass", "Bass", "Bass", "bass", 34),
    PartSpec("drums", "Drums", "Dr.", "drums", 1, chords=False),
]


# ---------------------------------------------------------------- beat grid

class Grid:
    """Maps seconds <-> 16th-note ticks using the detected (possibly drifting) beats."""

    def __init__(self, beats: np.ndarray, downbeat_phase: int):
        self.beats = np.asarray(beats, dtype=float)
        # bar 0 starts on the first downbeat at or before the first beat
        self.first_bar_beat = downbeat_phase - BEATS_PER_BAR if downbeat_phase > 0 else 0
        idx = np.arange(len(self.beats), dtype=float)
        self._idx = idx
        period = float(np.median(np.diff(self.beats))) if len(self.beats) > 1 else 0.5
        self.period = period

    def beat_pos(self, t: float) -> float:
        b = self.beats
        if t < b[0]:
            return (t - b[0]) / self.period
        if t > b[-1]:
            return len(b) - 1 + (t - b[-1]) / self.period
        return float(np.interp(t, b, self._idx))

    def tick(self, t: float) -> int:
        return int(round((self.beat_pos(t) - self.first_bar_beat) * SUBDIV))

    def beat_to_tick(self, beat_index: int) -> int:
        return (beat_index - self.first_bar_beat) * SUBDIV

    def bar_of_beat(self, beat_index: int) -> int:
        return (beat_index - self.first_bar_beat) // BEATS_PER_BAR

    def bar_start_time(self, bar: int) -> float:
        beat = bar * BEATS_PER_BAR + self.first_bar_beat
        if 0 <= beat < len(self.beats):
            return float(self.beats[beat])
        if beat < 0:
            return float(self.beats[0] + beat * self.period)
        return float(self.beats[-1] + (beat - len(self.beats) + 1) * self.period)


# ---------------------------------------------------------------- events

@dataclass
class Event:
    start: int  # tick
    end: int
    pitches: list[int]
    frets: list[tuple[int, int]] = field(default_factory=list)  # (string 1=highest, fret)


def quantize(notes: list[Note], grid: Grid) -> list[tuple[int, int, int]]:
    out = set()
    for n in notes:
        s = grid.tick(n.start)
        e = max(s + 1, grid.tick(n.end))
        if s < 0:
            continue
        out.add((s, e, n.pitch))
    return sorted(out)


def to_events(qnotes: list[tuple[int, int, int]], max_notes: int | None = None,
              keep: str = "outer") -> list[Event]:
    """Merge notes into a single voice of non-overlapping chords/notes.

    With max_notes=1, `keep` picks the "lowest" (bass) or "highest" (melody) note.
    """
    by_start: dict[int, list[tuple[int, int]]] = {}
    for s, e, p in qnotes:
        by_start.setdefault(s, []).append((e, p))
    starts = sorted(by_start)
    events = []
    for i, s in enumerate(starts):
        group = by_start[s]
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        end = max(e for e, _ in group)
        if nxt is not None:
            end = min(end, nxt)
        pitches = sorted({p for _, p in group})
        if max_notes == 1:
            pitches = [pitches[0] if keep == "lowest" else pitches[-1]]
        elif max_notes and len(pitches) > max_notes:
            # keep the outer voices and the strongest-looking middle notes
            pitches = sorted(set([pitches[0], pitches[-1]] + pitches[1:-1][: max_notes - 2]))
        events.append(Event(s, max(end, s + 1), pitches))
    return events


# ---------------------------------------------------------------- fingering

def assign_frets(events: list[Event], tuning: list[int], max_fret: int) -> None:
    """Choose (string, fret) for each note, keeping the hand in one area."""
    n_strings = len(tuning)
    hand = 3.0
    for ev in events:
        pitches = []
        for p in ev.pitches:
            while p < tuning[0]:
                p += 12
            while p > tuning[-1] + max_fret:
                p -= 12
            pitches.append(p)
        pitches = sorted(set(pitches))[-n_strings:]
        best, best_cost = None, float("inf")
        options = [[(s, p - tuning[s]) for s in range(n_strings) if 0 <= p - tuning[s] <= max_fret] for p in pitches]
        for combo in itertools.product(*options):
            strings = [s for s, _ in combo]
            if len(set(strings)) != len(strings):
                continue
            fretted = [f for _, f in combo if f > 0]
            span = (max(fretted) - min(fretted)) if fretted else 0
            pos = min(fretted) if fretted else hand
            cost = 0.35 * abs(pos - hand) + 0.04 * sum(f for _, f in combo)
            cost += 10 * max(0, span - 4) + 0.5 * span
            if fretted and min(fretted) > 7 and any(f == 0 for _, f in combo):
                cost += 2.0  # open strings mixed with high positions are awkward
            cost -= 0.15 * sum(1 for _, f in combo if f == 0)
            if cost < best_cost:
                best, best_cost = combo, cost
        if best is None:  # unplayable cluster: keep what fits
            best, used = [], set()
            for opts in reversed(options):
                for s, f in opts:
                    if s not in used:
                        best.append((s, f))
                        used.add(s)
                        break
        fretted = [f for _, f in best if f > 0]
        if fretted:
            hand = 0.6 * hand + 0.4 * min(fretted)
        ev.pitches = [tuning[s] + f for s, f in best]
        # MusicXML strings are numbered from the highest string (1)
        ev.frets = [(n_strings - s, f) for s, f in best]


# ---------------------------------------------------------------- durations

_TYPES = {16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1),
          4: ("quarter", 0), 3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0)}


def _allowed(pos: int, d: int) -> bool:
    in_beat = pos % SUBDIV
    if d == 16:
        return pos == 0
    if d == 12:
        return pos in (0, 4)
    if d in (8, 6):
        return pos % 8 == 0
    if d == 4:
        return in_beat == 0
    if d in (3, 2):
        return in_beat + d <= SUBDIV and (d == 3 or pos % 2 == 0)
    return True


def split_duration(pos: int, dur: int) -> list[int]:
    """Break a duration starting at `pos` (ticks into the bar) into notatable pieces."""
    pieces = []
    while dur > 0:
        for d in (16, 12, 8, 6, 4, 3, 2, 1):
            if d <= dur and pos + d <= BAR_TICKS and _allowed(pos, d):
                pieces.append(d)
                pos += d
                dur -= d
                break
    return pieces


# ---------------------------------------------------------------- MusicXML writing

DRUM_DISPLAY = {KICK: ("F", 4, None, "Kick"), SNARE: ("C", 5, None, "Snare"), HIHAT: ("G", 5, "x", "Hi-Hat")}


class MeasureWriter:
    def __init__(self, flats: bool):
        self.flats = flats

    def staff_measures(self, events: list[Event], n_bars: int, staff: int, voice: int,
                       kind: str, part_id: str, tab: bool) -> list[list[str]]:
        """XML note elements for each bar of one staff."""
        bars: list[list[str]] = [[] for _ in range(n_bars)]
        cursor = 0
        # pad events with rests so the sequence is continuous
        timeline: list[tuple[int, int, Event | None]] = []
        for ev in events:
            if ev.start >= n_bars * BAR_TICKS:
                break
            if ev.start > cursor:
                timeline.append((cursor, ev.start, None))
            timeline.append((ev.start, min(ev.end, n_bars * BAR_TICKS), ev))
            cursor = min(ev.end, n_bars * BAR_TICKS)
        if cursor < n_bars * BAR_TICKS:
            timeline.append((cursor, n_bars * BAR_TICKS, None))

        for start, end, ev in timeline:
            t = start
            first = True
            while t < end:
                bar = t // BAR_TICKS
                pos = t % BAR_TICKS
                seg_end = min(end, (bar + 1) * BAR_TICKS)
                if ev is None and pos == 0 and seg_end - t == BAR_TICKS:
                    bars[bar].append(self._rest(BAR_TICKS, staff, voice, whole_bar=True))
                    t = seg_end
                    continue
                pieces = split_duration(pos, seg_end - t)
                for k, d in enumerate(pieces):
                    last = (t + d) >= end
                    if ev is None:
                        bars[bar].append(self._rest(d, staff, voice))
                    else:
                        bars[bar].append(self._chord(ev, d, staff, voice, kind, part_id, tab,
                                                     tie_stop=not first, tie_start=not last))
                    first = False
                    t += d
        return bars

    def _rest(self, d: int, staff: int, voice: int, whole_bar: bool = False) -> str:
        if whole_bar:
            return (f'<note><rest measure="yes"/><duration>{d}</duration><voice>{voice}</voice>'
                    f'<type>whole</type><staff>{staff}</staff></note>')
        typ, dots = _TYPES[d]
        return (f"<note><rest/><duration>{d}</duration><voice>{voice}</voice><type>{typ}</type>"
                + "<dot/>" * dots + f"<staff>{staff}</staff></note>")

    def _chord(self, ev: Event, d: int, staff: int, voice: int, kind: str, part_id: str, tab: bool,
               tie_stop: bool, tie_start: bool) -> str:
        typ, dots = _TYPES[d]
        out = []
        for i, pitch in enumerate(ev.pitches):
            parts = ["<note>"]
            if i > 0:
                parts.append("<chord/>")
            if kind == "drums":
                step, octave, head, _ = DRUM_DISPLAY.get(pitch, ("C", 5, None, ""))
                parts.append(f"<unpitched><display-step>{step}</display-step>"
                             f"<display-octave>{octave}</display-octave></unpitched>")
            else:
                step, alter, octave = theory.pitch_to_xml(pitch, self.flats)
                alter_xml = f"<alter>{alter}</alter>" if alter else ""
                parts.append(f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>")
            parts.append(f"<duration>{d}</duration>")
            if tie_stop:
                parts.append('<tie type="stop"/>')
            if tie_start:
                parts.append('<tie type="start"/>')
            if kind == "drums":
                parts.append(f'<instrument id="{part_id}-I{pitch}"/>')
            parts.append(f"<voice>{voice}</voice><type>{typ}</type>" + "<dot/>" * dots)
            if kind == "drums":
                parts.append("<stem>up</stem>")
                if DRUM_DISPLAY.get(pitch, (None, None, None))[2]:
                    parts.append(f"<notehead>{DRUM_DISPLAY[pitch][2]}</notehead>")
            parts.append(f"<staff>{staff}</staff>")
            notations = []
            if tie_stop:
                notations.append('<tied type="stop"/>')
            if tie_start:
                notations.append('<tied type="start"/>')
            if tab and i < len(ev.frets):
                s, f = ev.frets[i]
                notations.append(f"<technical><string>{s}</string><fret>{f}</fret></technical>")
            if notations:
                parts.append("<notations>" + "".join(notations) + "</notations>")
            parts.append("</note>")
            out.append("".join(parts))
        return "".join(out)


def _harmony_xml(chord: dict, offset: int, flats: bool) -> str:
    step, alter = theory.step_alter(chord["root"], flats)
    root = f"<root><root-step>{step}</root-step>" + (f"<root-alter>{alter}</root-alter>" if alter else "") + "</root>"
    kind_text = chord["quality"]
    kind = f'<kind text="{escape(kind_text)}">{theory.MUSICXML_KIND[chord["quality"]]}</kind>'
    bass = ""
    if chord.get("bass") is not None:
        bstep, balter = theory.step_alter(chord["bass"], flats)
        bass = f"<bass><bass-step>{bstep}</bass-step>" + (f"<bass-alter>{balter}</bass-alter>" if balter else "") + "</bass>"
    off = f"<offset>{offset}</offset>" if offset else ""
    return f"<harmony>{root}{kind}{bass}{off}<staff>1</staff></harmony>"


def _insert_harmonies(bar_notes: list[str], chords_in_bar: list[tuple[int, dict]], flats: bool) -> list[str]:
    """Place chord symbols before the note sounding at their position within the bar."""
    if not chords_in_bar:
        return bar_notes
    # walk the (staff 1) note list, tracking the position, to find insertion points
    out: list[str] = []
    pos = 0
    pending = sorted(chords_in_bar, key=lambda x: x[0])
    for note in bar_notes:
        # each entry is one rhythmic position (a note, a rest, or all notes of a chord)
        dur = _note_duration(note)
        while pending and pending[0][0] < pos + dur:
            at, chord = pending.pop(0)
            out.append(_harmony_xml(chord, max(0, at - pos), flats))
        out.append(note)
        pos += dur
    for at, chord in pending:
        out.append(_harmony_xml(chord, 0, flats))
    return out


def _note_duration(note_xml: str) -> int:
    a = note_xml.index("<duration>") + len("<duration>")
    return int(note_xml[a: note_xml.index("</duration>", a)])


def _part_list_entry(pid: str, spec: PartSpec, used_drums: list[int]) -> str:
    xml = [f'<score-part id="{pid}"><part-name>{escape(spec.name)}</part-name>'
           f"<part-abbreviation>{escape(spec.abbreviation)}</part-abbreviation>"]
    if spec.kind == "drums":
        for d in used_drums:
            xml.append(f'<score-instrument id="{pid}-I{d}"><instrument-name>{DRUM_DISPLAY[d][3]}</instrument-name></score-instrument>')
        for d in used_drums:
            xml.append(f'<midi-instrument id="{pid}-I{d}"><midi-channel>10</midi-channel>'
                       f"<midi-program>1</midi-program><midi-unpitched>{d + 1}</midi-unpitched></midi-instrument>")
    else:
        xml.append(f'<score-instrument id="{pid}-I1"><instrument-name>{escape(spec.name)}</instrument-name></score-instrument>')
        xml.append(f'<midi-instrument id="{pid}-I1"><midi-channel>{1 + PART_SPECS.index(spec)}</midi-channel>'
                   f"<midi-program>{spec.midi_program}</midi-program></midi-instrument>")
    xml.append("</score-part>")
    return "".join(xml)


def _tuning_xml(tuning: list[int]) -> str:
    out = [f"<staff-lines>{len(tuning)}</staff-lines>"]
    for line, midi in enumerate(tuning, start=1):
        step, alter, octave = theory.pitch_to_xml(midi, False)
        alter_xml = f"<tuning-alter>{alter}</tuning-alter>" if alter else ""
        out.append(f'<staff-tuning line="{line}"><tuning-step>{step}</tuning-step>{alter_xml}'
                   f"<tuning-octave>{octave}</tuning-octave></staff-tuning>")
    return "".join(out)


def _attributes(spec: PartSpec, fifths: int, mode: str) -> str:
    key = f"<key><fifths>{fifths}</fifths><mode>{mode}</mode></key>"
    time = f"<time><beats>{BEATS_PER_BAR}</beats><beat-type>4</beat-type></time>"
    if spec.kind == "treble":
        return f"<attributes><divisions>{SUBDIV}</divisions>{key}{time}<clef><sign>G</sign><line>2</line></clef></attributes>"
    if spec.kind == "grand":
        return (f"<attributes><divisions>{SUBDIV}</divisions>{key}{time}<staves>2</staves>"
                '<clef number="1"><sign>G</sign><line>2</line></clef>'
                '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>')
    if spec.kind in ("guitar", "bass"):
        tuning = GUITAR_TUNING if spec.kind == "guitar" else BASS_TUNING
        clef1 = ('<clef number="1"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>'
                 if spec.kind == "guitar" else
                 '<clef number="1"><sign>F</sign><line>4</line><clef-octave-change>-1</clef-octave-change></clef>')
        return (f"<attributes><divisions>{SUBDIV}</divisions>{key}{time}<staves>2</staves>{clef1}"
                '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                f'<staff-details number="2">{_tuning_xml(tuning)}</staff-details></attributes>')
    # drums
    return (f"<attributes><divisions>{SUBDIV}</divisions><key><fifths>0</fifths></key>{time}"
            "<clef><sign>percussion</sign><line>2</line></clef></attributes>")


def build_musicxml(title: str, parts: dict[str, list[Note]], chords: list[dict], grid: Grid,
                   tonic_pc: int, mode: str, tempo: float, n_bars: int) -> tuple[str, list[dict]]:
    """Return (musicxml_string, part_info list)."""
    fifths = theory.key_fifths(tonic_pc, mode)
    flats = theory.use_flats(fifths)
    writer = MeasureWriter(flats)

    chords_by_bar: dict[int, list[tuple[int, dict]]] = {}
    for c in chords:
        tick = grid.beat_to_tick(c["start_beat"])
        if 0 <= tick < n_bars * BAR_TICKS:
            chords_by_bar.setdefault(tick // BAR_TICKS, []).append((tick % BAR_TICKS, c))

    part_list, part_bodies, info = [], [], []
    for spec in PART_SPECS:
        notes = parts.get(spec.key)
        if not notes:
            continue
        pid = f"P{len(part_list) + 1}"
        q = quantize(notes, grid)
        staves: list[tuple[int, int, list[Event], bool]] = []  # (staff, voice, events, tab)
        if spec.kind == "grand":
            upper = [n for n in q if n[2] >= 60]
            lower = [n for n in q if n[2] < 60]
            staves = [(1, 1, to_events(upper, 5), False), (2, 5, to_events(lower, 5), False)]
        elif spec.kind in ("guitar", "bass"):
            evs = to_events(q, 6) if spec.kind == "guitar" else to_events(q, 1, keep="lowest")
            if spec.kind == "guitar":
                assign_frets(evs, GUITAR_TUNING, 19)
            else:
                assign_frets(evs, BASS_TUNING, 20)
            staves = [(1, 1, evs, False), (2, 5, evs, True)]
        elif spec.kind == "drums":
            staves = [(1, 1, to_events(q), False)]
        else:
            staves = [(1, 1, to_events(q, 1, keep="highest"), False)]

        staff_bars = [writer.staff_measures(evs, n_bars, staff, voice, spec.kind, pid, tab)
                      for staff, voice, evs, tab in staves]
        used_drums = sorted({n.pitch for n in notes}) if spec.kind == "drums" else []
        part_list.append(_part_list_entry(pid, spec, used_drums))

        measures = []
        for b in range(n_bars):
            body = []
            if b == 0:
                body.append(_attributes(spec, fifths, mode))
                body.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                            f"<per-minute>{round(tempo)}</per-minute></metronome></direction-type>"
                            f'<sound tempo="{round(tempo)}"/></direction>')
            for k, bars in enumerate(staff_bars):
                notes_xml = bars[b]
                if k == 0 and spec.chords:
                    notes_xml = _insert_harmonies(notes_xml, chords_by_bar.get(b, []), flats)
                if k > 0:
                    body.append(f"<backup><duration>{BAR_TICKS}</duration></backup>")
                body.extend(notes_xml)
            if b == n_bars - 1:
                body.append('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
            measures.append(f'<measure number="{b + 1}">' + "".join(body) + "</measure>")
        part_bodies.append(f'<part id="{pid}">' + "".join(measures) + "</part>")
        info.append({"id": pid, "stem": spec.key, "name": spec.name, "notes": len(notes)})

    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">\n'
           '<score-partwise version="4.0">'
           f"<work><work-title>{escape(title)}</work-title></work>"
           "<identification><creator type=\"arranger\">Music App (automatic transcription)</creator>"
           "<encoding><software>Music App</software></encoding></identification>"
           "<part-list>" + "".join(part_list) + "</part-list>"
           + "".join(part_bodies) + "</score-partwise>\n")
    return xml, info


# ---------------------------------------------------------------- MIDI

def build_midi(parts: dict[str, list[Note]], tempo: float, path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for spec in PART_SPECS:
        notes = parts.get(spec.key)
        if not notes:
            continue
        inst = pretty_midi.Instrument(program=max(0, spec.midi_program - 1),
                                      is_drum=spec.kind == "drums", name=spec.name)
        for n in notes:
            inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, max(n.end, n.start + 0.05)))
        pm.instruments.append(inst)
    pm.write(str(path))
