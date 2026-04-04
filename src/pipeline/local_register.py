"""
Local register modeling — adaptive pitch context for melody selection.

Instead of using a single global register target for the entire song,
this module computes time-local pitch statistics in sliding windows.
This allows the melody resolver to adapt to register changes between
verse and chorus sections, and to correctly identify the vocal line
even when low accompaniment dominates the global pitch distribution.

The core insight: in a typical pop song, the vocal melody sits in the
upper portion of the local pitch texture at any given moment.  A global
upper-quartile target can be poisoned by the sheer number of low
accompaniment notes.  A local window focuses on what is happening
*right now*, giving the resolver much better register guidance.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from .candidate_notes import CandidateNote

logger = logging.getLogger("mp3_to_sheet.local_register")


@dataclass
class LocalRegisterContext:
    """Pitch statistics for a local time window around a candidate note."""
    local_median: float
    local_upper_quartile: float
    local_min: float
    local_max: float
    relative_position: float   # 0.0 = lowest in window, 1.0 = highest
    notes_in_window: int


def compute_local_register_contexts(
    candidates: List[CandidateNote],
    window_s: float = 2.0,
) -> List[LocalRegisterContext]:
    """
    For each candidate, compute pitch statistics from nearby candidates
    within a sliding time window.

    Parameters
    ----------
    candidates : list[CandidateNote]
        Sorted by start_s.
    window_s : float
        Half-width of the time window (total window = 2 * window_s centred
        on each note's midpoint).

    Returns
    -------
    List of LocalRegisterContext, one per candidate.
    """
    if not candidates:
        return []

    contexts: List[LocalRegisterContext] = []
    n = len(candidates)

    for i in range(n):
        note = candidates[i]
        mid_t = (note.start_s + note.end_s) / 2.0
        win_start = mid_t - window_s
        win_end = mid_t + window_s

        neighbors: List[Tuple[int, float]] = []  # (pitch_midi, duration_s)
        for j in range(n):
            other = candidates[j]
            other_mid = (other.start_s + other.end_s) / 2.0
            if win_start <= other_mid <= win_end:
                neighbors.append((other.pitch_midi, max(other.duration_s, 0.001)))

        ctx = _stats_from_neighbors(note.pitch_midi, neighbors)
        contexts.append(ctx)

    return contexts


def _stats_from_neighbors(
    note_pitch: int,
    neighbors: List[Tuple[int, float]],
) -> LocalRegisterContext:
    """Compute weighted percentile statistics from (pitch, weight) pairs."""
    if not neighbors:
        return LocalRegisterContext(
            local_median=float(note_pitch),
            local_upper_quartile=float(note_pitch),
            local_min=float(note_pitch),
            local_max=float(note_pitch),
            relative_position=0.5,
            notes_in_window=0,
        )

    sorted_by_pitch = sorted(neighbors, key=lambda x: x[0])
    total_weight = sum(w for _, w in sorted_by_pitch)

    if total_weight <= 0:
        pitches_only = [p for p, _ in sorted_by_pitch]
        mid = float(pitches_only[len(pitches_only) // 2])
        return LocalRegisterContext(
            local_median=mid,
            local_upper_quartile=mid,
            local_min=float(pitches_only[0]),
            local_max=float(pitches_only[-1]),
            relative_position=0.5,
            notes_in_window=len(sorted_by_pitch),
        )

    local_median = _weighted_percentile(sorted_by_pitch, total_weight, 0.50)
    local_upper_quartile = _weighted_percentile(sorted_by_pitch, total_weight, 0.75)

    local_min = float(sorted_by_pitch[0][0])
    local_max = float(sorted_by_pitch[-1][0])

    pitch_range = local_max - local_min
    if pitch_range > 0:
        relative_position = max(0.0, min(1.0, (note_pitch - local_min) / pitch_range))
    else:
        relative_position = 0.5

    return LocalRegisterContext(
        local_median=local_median,
        local_upper_quartile=local_upper_quartile,
        local_min=local_min,
        local_max=local_max,
        relative_position=relative_position,
        notes_in_window=len(sorted_by_pitch),
    )


def _weighted_percentile(
    sorted_items: List[Tuple[int, float]],
    total_weight: float,
    percentile: float,
) -> float:
    """Compute a weighted percentile from pre-sorted (value, weight) pairs."""
    target = total_weight * percentile
    cumulative = 0.0
    for pitch, weight in sorted_items:
        cumulative += weight
        if cumulative >= target:
            return float(pitch)
    return float(sorted_items[-1][0])
