"""
Candidate note normalization — Stage B½.

Takes raw Basic Pitch output and normalizes into a clean, validated
CandidateNote list suitable for the melody-path resolver.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("mp3_to_sheet.candidate_notes")


@dataclass
class CandidateNote:
    """
    A single candidate note from pitch detection, enriched with
    derived features used by the melody-path scorer.
    """
    start_s: float
    end_s: float
    pitch_midi: int
    confidence: float = 0.5
    salience: float = 0.0
    source: str = "basic_pitch"
    raw_index: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def pitch_class(self) -> int:
        return self.pitch_midi % 12

    @property
    def octave(self) -> int:
        return (self.pitch_midi // 12) - 1

    def overlaps(self, other: "CandidateNote") -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s

    def overlap_duration(self, other: "CandidateNote") -> float:
        overlap_start = max(self.start_s, other.start_s)
        overlap_end = min(self.end_s, other.end_s)
        return max(0.0, overlap_end - overlap_start)


def normalize_candidates(
    raw_note_events: list,
    min_duration_ms: float = 50.0,
    max_duration_s: float = 30.0,
    min_confidence: float = 0.15,
) -> List[CandidateNote]:
    """
    Convert raw Basic Pitch note events into validated CandidateNotes.

    Parameters
    ----------
    raw_note_events
        List of tuples/lists: (start_s, end_s, pitch_midi, amplitude, ...).
    min_duration_ms
        Discard notes shorter than this (milliseconds).
    max_duration_s
        Discard notes longer than this (likely artifacts).
    min_confidence
        Discard notes with amplitude/confidence below this.

    Returns
    -------
    Sorted list of CandidateNote, deterministically ordered by (start_s, pitch_midi).
    """
    min_dur_s = min_duration_ms / 1000.0
    candidates: List[CandidateNote] = []
    n_invalid = 0
    n_short = 0
    n_long = 0
    n_low_conf = 0

    for idx, ev in enumerate(raw_note_events):
        start_s = float(ev[0])
        end_s = float(ev[1])
        pitch_midi = int(ev[2])
        confidence = float(ev[3]) if len(ev) > 3 else 0.5

        if _is_invalid(start_s, end_s, pitch_midi, confidence):
            n_invalid += 1
            continue

        duration_s = end_s - start_s
        if duration_s < min_dur_s:
            n_short += 1
            continue
        if duration_s > max_duration_s:
            n_long += 1
            continue
        if confidence < min_confidence:
            n_low_conf += 1
            continue

        candidates.append(CandidateNote(
            start_s=start_s,
            end_s=end_s,
            pitch_midi=pitch_midi,
            confidence=confidence,
            salience=confidence,
            source="basic_pitch",
            raw_index=idx,
        ))

    candidates.sort(key=lambda n: (n.start_s, n.pitch_midi))

    logger.info(
        "Normalized %d → %d candidates  "
        "(invalid=%d, short=%d, long=%d, low_conf=%d)",
        len(raw_note_events), len(candidates),
        n_invalid, n_short, n_long, n_low_conf,
    )
    return candidates


def _is_invalid(start_s: float, end_s: float, pitch_midi: int, confidence: float) -> bool:
    if math.isnan(start_s) or math.isnan(end_s) or math.isnan(confidence):
        return True
    if end_s <= start_s:
        return True
    if pitch_midi < 0 or pitch_midi > 127:
        return True
    if math.isinf(start_s) or math.isinf(end_s):
        return True
    return False


def compute_clutter_density(
    candidates: List[CandidateNote],
    window_ms: float = 200.0,
) -> List[int]:
    """
    For each candidate, count how many other candidates overlap within
    a time window centred on the note's midpoint.  High density suggests
    the note is part of accompaniment texture rather than melody.
    """
    window_s = window_ms / 1000.0
    densities: List[int] = []

    for note in candidates:
        mid = (note.start_s + note.end_s) / 2.0
        win_start = mid - window_s / 2.0
        win_end = mid + window_s / 2.0
        count = 0
        for other in candidates:
            if other is note:
                continue
            if other.start_s < win_end and other.end_s > win_start:
                count += 1
        densities.append(count)

    return densities
