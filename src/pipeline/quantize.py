"""
Quantize raw note events onto a beat grid.

Produces a list of ``QuantizedNote`` objects with durations expressed in
quarter-note lengths, ready for MusicXML export.  Also fills gaps with
rests and (optionally) splits notes at bar-lines.
"""

import logging
import math
from typing import List, Optional

from .beat_grid import BeatGrid
from .utils import NoteEvent, QuantizedNote

logger = logging.getLogger("mp3_to_sheet.quantize")


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #

def quantize_notes(
    notes: List[NoteEvent],
    grid: BeatGrid,
    simplify: bool = False,
    allow_triplets: bool = False,
) -> List[QuantizedNote]:
    """
    Snap note events to the grid and return notation-ready notes + rests.

    Parameters
    ----------
    notes : list[NoteEvent]
        From melody extraction (times in seconds).
    grid : BeatGrid
        Tempo / time-sig info.
    simplify : bool
        If *True* use 1/8-note grid (lead-sheet style).
    allow_triplets : bool
        If *True* add a 1/3 ql grid for triplets.

    Returns
    -------
    list[QuantizedNote]  — sorted by ``start_ql``.
    """
    if not notes:
        return []

    # grid resolution (in quarter-note lengths)
    if simplify:
        step = 0.5        # eighth notes
    else:
        step = 0.25       # sixteenth notes

    triplet_step: Optional[float] = (1.0 / 3.0) if allow_triplets else None

    logger.info(
        "Quantizing %d notes  (step=%.4f ql, triplets=%s, simplify=%s)",
        len(notes), step, allow_triplets, simplify,
    )

    quantized: List[QuantizedNote] = []
    for n in notes:
        s_ql = grid.time_to_ql(n.start_s)
        e_ql = grid.time_to_ql(n.end_s)

        # discard notes entirely before the score start
        if e_ql <= 0:
            continue
        s_ql = max(s_ql, 0.0)

        # snap
        s_ql = _snap(s_ql, step, triplet_step)
        e_ql = _snap(e_ql, step, triplet_step)
        if e_ql <= s_ql:
            e_ql = s_ql + step

        quantized.append(QuantizedNote(
            start_ql=_rnd(s_ql),
            duration_ql=_rnd(e_ql - s_ql),
            midi_pitch=n.midi_pitch,
        ))

    quantized.sort(key=lambda q: q.start_ql)
    quantized = _remove_overlaps(quantized, step)
    quantized = _fill_rests(quantized, step)

    n_notes = sum(1 for q in quantized if not q.is_rest)
    n_rests = sum(1 for q in quantized if q.is_rest)
    logger.info("Quantisation done: %d notes + %d rests", n_notes, n_rests)

    return quantized


def split_at_barlines(
    notes: List[QuantizedNote],
    grid: BeatGrid,
) -> List[QuantizedNote]:
    """
    Split notes/rests that cross a measure boundary and annotate ties.

    This is also done by music21's ``makeNotation``, but pre-splitting
    makes the debug JSON easier to inspect.
    """
    meas_len = grid.measure_duration_ql
    result: List[QuantizedNote] = []

    for note in notes:
        start = note.start_ql
        end = start + note.duration_ql
        start_m = int(start / meas_len)
        end_m = int((end - 1e-6) / meas_len)

        if start_m == end_m or note.is_rest:
            result.append(note)
            continue

        cur = start
        for m in range(start_m, end_m + 1):
            m_end = (m + 1) * meas_len
            seg_end = min(end, m_end)
            seg_dur = seg_end - cur
            if seg_dur <= 0:
                continue

            seg = QuantizedNote(
                start_ql=_rnd(cur),
                duration_ql=_rnd(seg_dur),
                midi_pitch=note.midi_pitch,
                is_rest=note.is_rest,
            )
            if not note.is_rest:
                if m == start_m:
                    seg.tie_type = "start"
                elif m == end_m:
                    seg.tie_type = "stop"
                else:
                    seg.tie_type = "continue"
            result.append(seg)
            cur = m_end

    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _snap(value: float, step: float, triplet_step: Optional[float]) -> float:
    snapped = round(value / step) * step
    if triplet_step is not None:
        snapped_t = round(value / triplet_step) * triplet_step
        if abs(value - snapped_t) < abs(value - snapped):
            snapped = snapped_t
    return snapped


def _rnd(v: float) -> float:
    return round(v, 6)


def _remove_overlaps(
    notes: List[QuantizedNote], step: float
) -> List[QuantizedNote]:
    if len(notes) <= 1:
        return notes
    result = [notes[0]]
    for note in notes[1:]:
        prev = result[-1]
        prev_end = prev.start_ql + prev.duration_ql
        if note.start_ql < prev_end:
            new_dur = note.start_ql - prev.start_ql
            if new_dur >= step:
                prev.duration_ql = _rnd(new_dur)
                result.append(note)
            else:
                result[-1] = note
        else:
            result.append(note)
    return result


def _fill_rests(
    notes: List[QuantizedNote], step: float
) -> List[QuantizedNote]:
    if not notes:
        return notes

    result: List[QuantizedNote] = []

    # leading rest
    if notes[0].start_ql > step / 2:
        rest_dur = _snap(notes[0].start_ql, step, None)
        if rest_dur > 0:
            result.append(QuantizedNote(
                start_ql=0.0,
                duration_ql=_rnd(rest_dur),
                midi_pitch=0,
                is_rest=True,
            ))

    for i, note in enumerate(notes):
        result.append(note)
        if i < len(notes) - 1:
            note_end = note.start_ql + note.duration_ql
            gap = notes[i + 1].start_ql - note_end
            if gap >= step / 2:
                gap_r = _snap(gap, step, None)
                if gap_r > 0:
                    result.append(QuantizedNote(
                        start_ql=_rnd(note_end),
                        duration_ql=_rnd(gap_r),
                        midi_pitch=0,
                        is_rest=True,
                    ))

    return result
