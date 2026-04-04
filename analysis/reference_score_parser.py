"""
Reference score parser — extracts structured melody data from reference images.

This module provides a framework for building reference melody representations
from sheet music images. The first implementation is manually extracted from
the MuseScore "Raise y_our glass - Violin Version" reference screenshots.

The parser produces structured ReferenceNote objects with confidence levels,
enabling honest comparison against generated output.

Design: Reusable for additional songs — supply a new JSON reference file.
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class ReferenceNote:
    measure: int
    position_in_measure: float  # 0-based quarter-note offset
    pitch_midi: int
    pitch_name: str
    duration_class: str  # "whole", "half", "quarter", "eighth", "sixteenth", "dotted-quarter", etc.
    duration_quarters: float
    confidence: str = Confidence.MEDIUM
    tie_start: bool = False
    tie_stop: bool = False
    is_rest: bool = False
    notes: str = ""  # free-text annotation


@dataclass
class ReferenceScore:
    title: str
    key: str
    time_signature: str
    tempo_bpm: int
    total_measures: int
    source_description: str
    notes: List[ReferenceNote] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "key": self.key,
            "time_signature": self.time_signature,
            "tempo_bpm": self.tempo_bpm,
            "total_measures": self.total_measures,
            "source_description": self.source_description,
            "metadata": self.metadata,
            "notes": [asdict(n) for n in self.notes],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ReferenceScore":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        notes = [ReferenceNote(**n) for n in d.pop("notes", [])]
        return cls(**d, notes=notes)

    def pitch_sequence(self, min_confidence: str = Confidence.UNCERTAIN) -> List[int]:
        """Return MIDI pitch sequence for non-rest notes above confidence threshold."""
        conf_order = [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW, Confidence.UNCERTAIN]
        threshold_idx = conf_order.index(min_confidence)
        return [
            n.pitch_midi for n in self.notes
            if not n.is_rest and conf_order.index(n.confidence) <= threshold_idx
        ]

    def notes_in_measure(self, measure: int) -> List[ReferenceNote]:
        return [n for n in self.notes if n.measure == measure and not n.is_rest]

    def measures_with_notes(self) -> List[int]:
        return sorted(set(n.measure for n in self.notes if not n.is_rest))

    def pitch_range(self) -> Tuple[int, int]:
        pitches = self.pitch_sequence()
        if not pitches:
            return (0, 0)
        return (min(pitches), max(pitches))


def midi_to_name(midi: int) -> str:
    """Convert MIDI number to note name (enharmonic, preferring flats for Ab major)."""
    note_names = [
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"
    ]
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


def name_to_midi(name: str) -> int:
    """Convert note name to MIDI number. e.g. 'Ab4' -> 68."""
    name_map = {
        "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
        "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
        "C#": 1, "D#": 3, "F#": 6, "G#": 8, "A#": 10,
    }
    for length in (2, 1):
        prefix = name[:length]
        if prefix in name_map:
            pc = name_map[prefix]
            octave = int(name[length:])
            return (octave + 1) * 12 + pc
    raise ValueError(f"Cannot parse note name: {name}")
