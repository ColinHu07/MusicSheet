"""
Rhythm simplification — make notation human-readable.

Converts overly precise quantized durations into the common note
values a human arranger would write: whole, half, dotted-half,
quarter, dotted-quarter, eighth, dotted-eighth, sixteenth.

Also removes tiny inter-note rests that result from timing jitter
rather than genuine musical silence.

Design:
  1. ``simplify_durations`` — snap each note/rest duration to the
     nearest "standard" note value.
  2. ``remove_micro_rests`` — absorb tiny rests between notes by
     extending the previous note.
  3. ``simplify_notation`` — orchestrates both in the right order.
"""

import logging
from typing import List, Set

from .utils import QuantizedNote

logger = logging.getLogger("mp3_to_sheet.rhythm_simplify")

STANDARD_DURATIONS: List[float] = sorted([
    4.0,        # whole
    3.0,        # dotted half
    2.0,        # half
    1.5,        # dotted quarter
    1.0,        # quarter
    0.75,       # dotted eighth
    0.5,        # eighth
    0.25,       # sixteenth
])

_STANDARD_SET: Set[float] = set(STANDARD_DURATIONS)


def simplify_notation(
    notes: List[QuantizedNote],
    micro_rest_threshold_ql: float = 0.25,
) -> List[QuantizedNote]:
    """
    Full rhythm simplification pass.

    1. Remove micro-rests (absorb into previous note).
    2. Snap durations to standard note values.
    3. Re-seal any small gaps introduced by snapping.
    """
    if not notes:
        return notes

    notes = remove_micro_rests(notes, micro_rest_threshold_ql)
    notes = simplify_durations(notes)
    notes = _seal_gaps(notes)
    return notes


def simplify_durations(notes: List[QuantizedNote]) -> List[QuantizedNote]:
    """
    Snap each note/rest duration to the nearest standard note value.
    Tied notes (tie_type is set) are left untouched — they are already
    structurally necessary.
    """
    result: List[QuantizedNote] = []
    for n in notes:
        if n.tie_type is not None:
            result.append(n)
            continue

        best = _nearest_standard(n.duration_ql)
        if best != n.duration_ql:
            n = QuantizedNote(
                start_ql=n.start_ql,
                duration_ql=best,
                midi_pitch=n.midi_pitch,
                is_rest=n.is_rest,
                tie_type=n.tie_type,
                spelled_name=n.spelled_name,
            )
        result.append(n)
    return result


def remove_micro_rests(
    notes: List[QuantizedNote],
    threshold_ql: float = 0.25,
) -> List[QuantizedNote]:
    """
    Absorb tiny rests (≤ threshold_ql) by extending the preceding note.

    A rest is only absorbed if:
      - it is shorter than the threshold
      - it is flanked by notes (not at the start/end)

    This eliminates the "note-rest-note-rest" stutter that timing
    jitter produces.
    """
    if len(notes) < 3:
        return list(notes)

    result: List[QuantizedNote] = [notes[0]]
    i = 1
    while i < len(notes):
        n = notes[i]
        if (n.is_rest
                and n.duration_ql <= threshold_ql
                and result
                and not result[-1].is_rest
                and i + 1 < len(notes)
                and not notes[i + 1].is_rest):
            result[-1] = QuantizedNote(
                start_ql=result[-1].start_ql,
                duration_ql=round(result[-1].duration_ql + n.duration_ql, 6),
                midi_pitch=result[-1].midi_pitch,
                is_rest=result[-1].is_rest,
                tie_type=result[-1].tie_type,
                spelled_name=result[-1].spelled_name,
            )
        else:
            result.append(n)
        i += 1
    return result


def count_unusual_durations(notes: List[QuantizedNote]) -> int:
    """Count notes/rests whose duration is not a standard note value."""
    count = 0
    for n in notes:
        if n.tie_type is not None:
            continue
        if round(n.duration_ql, 4) not in _STANDARD_SET:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _nearest_standard(dur: float) -> float:
    """Return the standard duration closest to *dur*."""
    dur_r = round(dur, 4)
    if dur_r in _STANDARD_SET:
        return dur_r

    best = min(STANDARD_DURATIONS, key=lambda s: abs(s - dur))
    if best < 0.25:
        best = 0.25
    return best


def _seal_gaps(notes: List[QuantizedNote]) -> List[QuantizedNote]:
    """
    After duration snapping, notes may no longer tile perfectly.
    Adjust start times so each note begins where the previous one
    ends (within a tolerance).
    """
    if len(notes) < 2:
        return notes

    result = [notes[0]]
    for i in range(1, len(notes)):
        prev = result[-1]
        prev_end = round(prev.start_ql + prev.duration_ql, 6)
        n = notes[i]

        gap = round(n.start_ql - prev_end, 6)
        if abs(gap) <= 0.25 and gap != 0:
            n = QuantizedNote(
                start_ql=prev_end,
                duration_ql=n.duration_ql,
                midi_pitch=n.midi_pitch,
                is_rest=n.is_rest,
                tie_type=n.tie_type,
                spelled_name=n.spelled_name,
            )
        result.append(n)
    return result
