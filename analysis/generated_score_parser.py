"""
Parser for generated pipeline output (debug JSON and MusicXML).

Extracts note data into a format comparable to the reference score,
enabling measure-by-measure comparison.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GeneratedNote:
    measure: int
    position_in_measure: float
    pitch_midi: int
    pitch_name: str
    duration_quarters: float
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    tie_type: str = ""
    source: str = ""  # "quantized" or "raw"


@dataclass
class GeneratedScore:
    title: str
    key: str
    time_signature: str
    tempo_bpm: float
    source_mode: str
    notes: List[GeneratedNote] = field(default_factory=list)
    raw_notes: List[GeneratedNote] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "key": self.key,
            "time_signature": self.time_signature,
            "tempo_bpm": self.tempo_bpm,
            "source_mode": self.source_mode,
            "metadata": self.metadata,
            "notes": [asdict(n) for n in self.notes],
            "raw_notes": [asdict(n) for n in self.raw_notes],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def pitch_sequence(self, skip_ties=True) -> List[int]:
        """Return MIDI pitch sequence (optionally skipping tie-stop notes)."""
        return [
            n.pitch_midi for n in self.notes
            if not (skip_ties and n.tie_type == "stop")
        ]

    def notes_in_measure(self, measure: int) -> List[GeneratedNote]:
        return [n for n in self.notes if n.measure == measure]

    def measures_with_notes(self) -> List[int]:
        return sorted(set(n.measure for n in self.notes))

    def pitch_range(self) -> Tuple[int, int]:
        pitches = self.pitch_sequence()
        if not pitches:
            return (0, 0)
        return (min(pitches), max(pitches))


MIDI_TO_NAME = {}
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
for _m in range(21, 109):
    _oct = (_m // 12) - 1
    _n = NOTE_NAMES[_m % 12]
    MIDI_TO_NAME[_m] = f"{_n}{_oct}"


def parse_debug_json(path: str) -> GeneratedScore:
    """Parse a pipeline debug JSON file into a GeneratedScore."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    tempo = d.get("tempo_bpm", 120)
    beats_per_second = tempo / 60.0

    score = GeneratedScore(
        title=Path(path).stem,
        key=d.get("detected_key", "unknown"),
        time_signature=d.get("time_signature", "4/4"),
        tempo_bpm=tempo,
        source_mode=d.get("source_mode", "unknown"),
    )

    for qn in d.get("quantized_notes", []):
        if qn.get("is_rest", False):
            continue
        start_ql = qn["start_ql"]
        dur_ql = qn["duration_ql"]
        midi = qn["midi_pitch"]
        measure = int(start_ql // 4) + 1
        pos = start_ql % 4

        score.notes.append(GeneratedNote(
            measure=measure,
            position_in_measure=pos,
            pitch_midi=midi,
            pitch_name=MIDI_TO_NAME.get(midi, f"midi{midi}"),
            duration_quarters=dur_ql,
            tie_type=qn.get("tie_type") or "",
            source="quantized",
        ))

    for rn in d.get("raw_notes", []):
        midi = rn["midi_pitch"]
        start = rn["start_s"]
        end = rn["end_s"]
        beat_pos = start * beats_per_second
        measure = int(beat_pos // 4) + 1
        pos = beat_pos % 4
        dur_s = end - start
        dur_ql = dur_s * beats_per_second

        score.raw_notes.append(GeneratedNote(
            measure=measure,
            position_in_measure=pos,
            pitch_midi=midi,
            pitch_name=MIDI_TO_NAME.get(midi, f"midi{midi}"),
            duration_quarters=dur_ql,
            start_time_s=start,
            end_time_s=end,
            source="raw",
        ))

    # Collect metadata
    score.metadata = {
        "melody_stats": d.get("melody_stats", {}),
        "vocal_abstraction_stats": d.get("vocal_abstraction_stats", {}),
        "pitch_refinement_stats": d.get("pitch_refinement_stats", {}),
        "notation_stats": d.get("notation_stats", {}),
        "stem_metadata": d.get("stem_metadata", {}),
    }

    return score


def parse_musicxml(path: str) -> GeneratedScore:
    """Parse a MusicXML file into a GeneratedScore (basic extraction)."""
    tree = ET.parse(path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    score = GeneratedScore(
        title=Path(path).stem,
        key="unknown",
        time_signature="4/4",
        tempo_bpm=120,
        source_mode="unknown",
    )

    step_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    for part in root.iter(f"{ns}part"):
        for measure_el in part.iter(f"{ns}measure"):
            measure_num = int(measure_el.get("number", 0))
            current_offset = 0.0
            divisions = 1

            for attr in measure_el.iter(f"{ns}divisions"):
                divisions = int(attr.text)

            for el in measure_el:
                if el.tag == f"{ns}attributes":
                    for div_el in el.iter(f"{ns}divisions"):
                        divisions = int(div_el.text)
                    for key_el in el.iter(f"{ns}fifths"):
                        fifths = int(key_el.text)
                        key_names = {
                            -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab",
                            -3: "Eb", -2: "Bb", -1: "F", 0: "C",
                            1: "G", 2: "D", 3: "A", 4: "E",
                            5: "B", 6: "F#", 7: "C#"
                        }
                        mode = "major"
                        for m_el in el.iter(f"{ns}mode"):
                            mode = m_el.text
                        score.key = f"{key_names.get(fifths, '?')} {mode}"
                    for beats_el in el.iter(f"{ns}beats"):
                        beats = beats_el.text
                    for bt_el in el.iter(f"{ns}beat-type"):
                        beat_type = bt_el.text
                        score.time_signature = f"{beats}/{beat_type}"
                elif el.tag == f"{ns}direction":
                    for tempo_el in el.iter(f"{ns}per-minute"):
                        score.tempo_bpm = float(tempo_el.text)
                elif el.tag == f"{ns}note":
                    rest_el = el.find(f"{ns}rest")
                    dur_el = el.find(f"{ns}duration")
                    duration = int(dur_el.text) if dur_el is not None else 0
                    dur_ql = duration / divisions if divisions else 0

                    if rest_el is not None:
                        current_offset += dur_ql
                        continue

                    pitch_el = el.find(f"{ns}pitch")
                    if pitch_el is None:
                        current_offset += dur_ql
                        continue

                    step = pitch_el.find(f"{ns}step").text
                    octave = int(pitch_el.find(f"{ns}octave").text)
                    alter_el = pitch_el.find(f"{ns}alter")
                    alter = int(float(alter_el.text)) if alter_el is not None else 0

                    midi = (octave + 1) * 12 + step_to_pc[step] + alter

                    tie_type = ""
                    for tie_el in el.iter(f"{ns}tie"):
                        t = tie_el.get("type", "")
                        if t == "start":
                            tie_type = "start"
                        elif t == "stop" and tie_type != "start":
                            tie_type = "stop"

                    score.notes.append(GeneratedNote(
                        measure=measure_num,
                        position_in_measure=current_offset,
                        pitch_midi=midi,
                        pitch_name=MIDI_TO_NAME.get(midi, f"midi{midi}"),
                        duration_quarters=dur_ql,
                        tie_type=tie_type,
                        source="musicxml",
                    ))

                    current_offset += dur_ql

    return score


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generated_score_parser.py <debug.json or .musicxml>")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith(".json"):
        score = parse_debug_json(path)
    else:
        score = parse_musicxml(path)

    print(f"Title: {score.title}")
    print(f"Key: {score.key}, Time: {score.time_signature}, Tempo: {score.tempo_bpm}")
    print(f"Notes: {len(score.notes)}, Raw notes: {len(score.raw_notes)}")
    pitches = score.pitch_sequence()
    if pitches:
        print(f"Pitch range: {min(pitches)}-{max(pitches)} "
              f"(median={sorted(pitches)[len(pitches)//2]})")
    score.save(str(Path(path).with_suffix(".parsed.json")))
