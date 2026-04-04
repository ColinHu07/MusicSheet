"""
Pitch refinement — improve pitch accuracy after note abstraction.

This module sits AFTER vocal abstraction and BEFORE quantization.
It corrects pitch drift, splits segments that contain multiple real
pitches, and smooths implausible intervals.

Stages:
  1. **Segment splitting** — detect merged segments that span multiple
     distinct pitch regions and split them at change points.
  2. **Pitch re-estimation** — recompute representative pitch for each
     note using clustering-based estimation.
  3. **Interval smoothing** — correct isolated pitch outliers that
     create implausible melodic intervals.
  4. **Octave correction** — fix notes that jump an octave from
     both neighbors.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.pitch_refinement")


@dataclass
class PitchRefinementConfig:
    """Tuneable parameters for pitch refinement."""

    # -- segment splitting ----------------------------------------------------
    split_enabled: bool = True
    split_min_pitch_range: int = 3
    split_min_segment_duration_ms: float = 200.0

    # -- interval smoothing ---------------------------------------------------
    interval_smoothing_enabled: bool = True
    interval_max_jump: int = 7
    interval_large_jump: int = 12
    interval_context_window: int = 3

    # -- octave correction ----------------------------------------------------
    octave_correction_enabled: bool = True

    # -- neighbor consistency -------------------------------------------------
    neighbor_correction_enabled: bool = True
    neighbor_max_correction: int = 2


@dataclass
class PitchRefinementStats:
    """Metrics from the pitch refinement pass."""
    notes_before: int = 0
    notes_after_split: int = 0
    notes_final: int = 0
    segments_split: int = 0
    notes_added_by_split: int = 0
    octave_corrections: int = 0
    interval_corrections: int = 0
    neighbor_corrections: int = 0


def refine_pitches(
    notes: List[NoteEvent],
    original_notes: Optional[List[NoteEvent]] = None,
    config: Optional[PitchRefinementConfig] = None,
) -> Tuple[List[NoteEvent], PitchRefinementStats]:
    """
    Main entry point: refine pitches after vocal abstraction.

    Parameters
    ----------
    notes : list[NoteEvent]
        Notes after vocal abstraction (may have drift/errors).
    original_notes : list[NoteEvent], optional
        The pre-abstraction notes (used for re-estimation via lookup).
    config : PitchRefinementConfig, optional

    Returns (refined_notes, stats).
    """
    if config is None:
        config = PitchRefinementConfig()

    stats = PitchRefinementStats()
    stats.notes_before = len(notes)

    # 1. Segment splitting (may increase note count, so do before short-circuit)
    if config.split_enabled and original_notes and len(notes) >= 1:
        notes, split_count, added = split_multi_pitch_segments(
            notes, original_notes,
            min_range=config.split_min_pitch_range,
            min_dur_ms=config.split_min_segment_duration_ms,
        )
        stats.segments_split = split_count
        stats.notes_added_by_split = added
    stats.notes_after_split = len(notes)

    if len(notes) < 2:
        stats.notes_final = len(notes)
        return list(notes), stats

    # 2. Pitch re-estimation from original notes
    if original_notes:
        notes = reestimate_pitches(notes, original_notes)

    # 3. Octave correction
    if config.octave_correction_enabled:
        notes, oct_count = correct_octave_errors(notes)
        stats.octave_corrections = oct_count

    # 4. Interval smoothing
    if config.interval_smoothing_enabled:
        notes, int_count = smooth_intervals(
            notes,
            max_jump=config.interval_max_jump,
            large_jump=config.interval_large_jump,
            context_window=config.interval_context_window,
        )
        stats.interval_corrections = int_count

    # 5. Neighbor consistency
    if config.neighbor_correction_enabled:
        notes, nbr_count = correct_neighbor_outliers(
            notes,
            max_correction=config.neighbor_max_correction,
        )
        stats.neighbor_corrections = nbr_count

    stats.notes_final = len(notes)
    return notes, stats


# =========================================================================== #
# 1. Segment splitting
# =========================================================================== #

def split_multi_pitch_segments(
    notes: List[NoteEvent],
    original_notes: List[NoteEvent],
    min_range: int = 3,
    min_dur_ms: float = 200.0,
) -> Tuple[List[NoteEvent], int, int]:
    """
    Detect merged segments that span multiple distinct pitch regions
    and split them at pitch change points.

    For each note in ``notes``, look up the original sub-notes that
    were merged into it (by time overlap).  If those sub-notes span
    a wide pitch range AND show a clear pitch transition (not just
    oscillation), split the merged note back into distinct segments.

    Returns (split_notes, segments_split, notes_added).
    """
    min_dur_s = min_dur_ms / 1000.0
    result: List[NoteEvent] = []
    segments_split = 0
    notes_added = 0

    for note in notes:
        sub_notes = _find_overlapping(note, original_notes)

        if len(sub_notes) < 2:
            result.append(note)
            continue

        pitches = [n.midi_pitch for n in sub_notes]
        pitch_range = max(pitches) - min(pitches)

        if pitch_range < min_range:
            result.append(note)
            continue

        clusters = _find_pitch_clusters(sub_notes)
        if len(clusters) < 2:
            result.append(note)
            continue

        split_result = _split_at_cluster_boundaries(
            sub_notes, clusters, min_dur_s)

        if len(split_result) > 1:
            result.extend(split_result)
            segments_split += 1
            notes_added += len(split_result) - 1
        else:
            result.append(note)

    return result, segments_split, notes_added


def _find_overlapping(
    target: NoteEvent, candidates: List[NoteEvent],
) -> List[NoteEvent]:
    """Find original notes that overlap in time with target."""
    result = []
    for c in candidates:
        if c.end_s <= target.start_s:
            continue
        if c.start_s >= target.end_s:
            break
        overlap_start = max(c.start_s, target.start_s)
        overlap_end = min(c.end_s, target.end_s)
        if overlap_end - overlap_start > 0.01:
            result.append(c)
    return result


def _find_pitch_clusters(notes: List[NoteEvent]) -> List[List[NoteEvent]]:
    """
    Group notes into pitch clusters using a simple sequential approach:
    walk through time-ordered notes and start a new cluster whenever
    the pitch changes by more than 1 semitone from the cluster's
    central pitch AND the change is sustained (not a brief blip).
    """
    if not notes:
        return []

    clusters: List[List[NoteEvent]] = [[notes[0]]]

    for note in notes[1:]:
        current_cluster = clusters[-1]
        cluster_pitches = [n.midi_pitch for n in current_cluster]
        cluster_dur = sum(n.duration_s for n in current_cluster)

        pitch_dur: Counter = Counter()
        for n in current_cluster:
            pitch_dur[n.midi_pitch] += n.duration_s
        cluster_center = pitch_dur.most_common(1)[0][0]

        if abs(note.midi_pitch - cluster_center) <= 1:
            current_cluster.append(note)
        elif note.duration_s < 0.05:
            current_cluster.append(note)
        else:
            clusters.append([note])

    significant = [c for c in clusters
                   if sum(n.duration_s for n in c) >= 0.05]
    return significant if len(significant) >= 2 else clusters


def _split_at_cluster_boundaries(
    sub_notes: List[NoteEvent],
    clusters: List[List[NoteEvent]],
    min_dur_s: float,
) -> List[NoteEvent]:
    """
    Create new NoteEvents for each pitch cluster, choosing the
    representative pitch from each cluster.
    """
    result: List[NoteEvent] = []

    for cluster in clusters:
        if not cluster:
            continue

        start = cluster[0].start_s
        end = cluster[-1].end_s
        if end - start < min_dur_s:
            continue

        pitch = _cluster_pitch(cluster)
        confidence = max(n.confidence for n in cluster)

        result.append(NoteEvent(
            start_s=start, end_s=end,
            midi_pitch=pitch, confidence=confidence,
        ))

    return result if result else [NoteEvent(
        sub_notes[0].start_s, sub_notes[-1].end_s,
        _cluster_pitch(sub_notes), max(n.confidence for n in sub_notes),
    )]


def _cluster_pitch(notes: List[NoteEvent]) -> int:
    """Pick representative pitch for a cluster using duration-weighted mode."""
    pitch_dur: Counter = Counter()
    for n in notes:
        pitch_dur[n.midi_pitch] += n.duration_s
    return pitch_dur.most_common(1)[0][0]


# =========================================================================== #
# 2. Pitch re-estimation
# =========================================================================== #

def reestimate_pitches(
    notes: List[NoteEvent],
    original_notes: List[NoteEvent],
) -> List[NoteEvent]:
    """
    For each note, look up the original sub-notes that fall within
    its time span and recompute the representative pitch using a
    stability-weighted approach.

    Pitch stability = the longest continuous run at a given pitch.
    This is more robust than simple duration weighting because it
    favors pitches that the singer actually held, not pitches that
    appear in many short fragments.
    """
    result: List[NoteEvent] = []

    for note in notes:
        sub = _find_overlapping(note, original_notes)

        if len(sub) < 2:
            result.append(note)
            continue

        pitch = _stability_weighted_pitch(sub)
        result.append(NoteEvent(
            start_s=note.start_s,
            end_s=note.end_s,
            midi_pitch=pitch,
            confidence=note.confidence,
        ))

    return result


def _stability_weighted_pitch(notes: List[NoteEvent]) -> int:
    """
    Choose pitch using stability weighting: for each distinct pitch,
    compute the duration of the longest continuous run at that pitch.
    Then use duration-weighted mode among pitches, with a bonus for
    the pitch that has the longest single continuous run.

    This prevents scattered short fragments from overwhelming a
    genuine sustained pitch.
    """
    if len(notes) == 1:
        return notes[0].midi_pitch

    pitch_total_dur: Counter = Counter()
    pitch_longest_run: Counter = Counter()

    current_pitch = notes[0].midi_pitch
    current_run_start = notes[0].start_s

    for i, n in enumerate(notes):
        pitch_total_dur[n.midi_pitch] += n.duration_s

        if n.midi_pitch == current_pitch:
            run_dur = n.end_s - current_run_start
            pitch_longest_run[current_pitch] = max(
                pitch_longest_run[current_pitch], run_dur)
        else:
            current_pitch = n.midi_pitch
            current_run_start = n.start_s
            pitch_longest_run[n.midi_pitch] = max(
                pitch_longest_run[n.midi_pitch], n.duration_s)

    total_dur = sum(pitch_total_dur.values())
    if total_dur <= 0:
        return notes[0].midi_pitch

    best_pitch = notes[0].midi_pitch
    best_score = -1.0

    for pitch in pitch_total_dur:
        dur_score = pitch_total_dur[pitch] / total_dur
        run_score = pitch_longest_run.get(pitch, 0) / total_dur
        score = dur_score * 0.6 + run_score * 0.4
        if score > best_score:
            best_score = score
            best_pitch = pitch

    return best_pitch


# =========================================================================== #
# 3. Octave correction
# =========================================================================== #

def correct_octave_errors(
    notes: List[NoteEvent],
) -> Tuple[List[NoteEvent], int]:
    """
    Fix notes that jump exactly ±12 semitones from both neighbors.
    These are octave-detection artifacts from Basic Pitch.

    Also catches notes that are an octave away from the local median
    when they create an implausible contour.

    Returns (corrected_notes, count).
    """
    if len(notes) < 3:
        return list(notes), 0

    corrections = 0
    result = [NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.confidence)
              for n in notes]

    for i in range(1, len(result) - 1):
        prev_p = result[i - 1].midi_pitch
        curr_p = result[i].midi_pitch
        next_p = result[i + 1].midi_pitch

        dp = curr_p - prev_p
        dn = curr_p - next_p

        if abs(dp) in (12, 24) and abs(dn) in (12, 24):
            if (dp > 0 and dn > 0) or (dp < 0 and dn < 0):
                direction = -1 if dp > 0 else 1
                result[i].midi_pitch += 12 * direction
                corrections += 1
                continue

        if abs(dp) in (12, 24) and abs(next_p - prev_p) <= 4:
            result[i].midi_pitch = prev_p + (next_p - prev_p) // 2
            corrections += 1
        elif abs(dn) in (12, 24) and abs(prev_p - next_p) <= 4:
            result[i].midi_pitch = prev_p + (next_p - prev_p) // 2
            corrections += 1

    return result, corrections


# =========================================================================== #
# 4. Interval smoothing
# =========================================================================== #

def smooth_intervals(
    notes: List[NoteEvent],
    max_jump: int = 7,
    large_jump: int = 12,
    context_window: int = 3,
) -> Tuple[List[NoteEvent], int]:
    """
    Correct isolated pitch outliers that create implausible intervals.

    An outlier is defined as a note that:
    - creates a jump > max_jump semitones from BOTH neighbors
    - is surrounded by neighbors that are close to each other
    - would create a more plausible contour if adjusted

    Large jumps (> large_jump) that are contextually supported (the
    melody stays in the new register) are preserved.

    Returns (smoothed_notes, corrections_count).
    """
    if len(notes) < 3:
        return list(notes), 0

    result = [NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.confidence)
              for n in notes]
    corrections = 0

    for i in range(1, len(result) - 1):
        prev_p = result[i - 1].midi_pitch
        curr_p = result[i].midi_pitch
        next_p = result[i + 1].midi_pitch

        jump_prev = abs(curr_p - prev_p)
        jump_next = abs(curr_p - next_p)

        if jump_prev <= max_jump or jump_next <= max_jump:
            continue

        neighbor_spread = abs(next_p - prev_p)
        if neighbor_spread > max_jump:
            continue

        is_supported = _is_register_shift_supported(
            result, i, context_window)
        if is_supported:
            continue

        midpoint = round((prev_p + next_p) / 2.0)
        result[i].midi_pitch = midpoint
        corrections += 1

    return result, corrections


def _is_register_shift_supported(
    notes: List[NoteEvent], idx: int, window: int,
) -> bool:
    """
    Check whether the pitch at idx is part of a genuine register
    shift (the melody moves to a new register and stays there).

    Look at notes after idx: if they stay near the current pitch,
    it's a real shift, not an outlier.
    """
    curr_p = notes[idx].midi_pitch
    after = notes[idx + 1: idx + 1 + window]

    if not after:
        return False

    close_count = sum(1 for n in after if abs(n.midi_pitch - curr_p) <= 4)
    return close_count >= len(after) * 0.5


# =========================================================================== #
# 5. Neighbor consistency
# =========================================================================== #

def correct_neighbor_outliers(
    notes: List[NoteEvent],
    max_correction: int = 2,
) -> Tuple[List[NoteEvent], int]:
    """
    For each note, check if it deviates from both neighbors by a
    small but suspicious amount (2-4 semitones both up or both down).
    If so, nudge it toward the local trend.

    This catches the case where a merge picked a pitch 1-2 semitones
    off from the true pitch center, creating a subtle but audible error.

    Only corrects when:
    - the note is 2-4 semitones away from BOTH neighbors in the SAME direction
    - neighbors are within 3 semitones of each other
    - correction is at most max_correction semitones

    Returns (corrected_notes, count).
    """
    if len(notes) < 3:
        return list(notes), 0

    result = [NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.confidence)
              for n in notes]
    corrections = 0

    for i in range(1, len(result) - 1):
        prev_p = result[i - 1].midi_pitch
        curr_p = result[i].midi_pitch
        next_p = result[i + 1].midi_pitch

        diff_prev = curr_p - prev_p
        diff_next = curr_p - next_p

        if abs(prev_p - next_p) > 3:
            continue

        both_above = diff_prev >= 2 and diff_next >= 2
        both_below = diff_prev <= -2 and diff_next <= -2

        if not (both_above or both_below):
            continue

        target = round((prev_p + next_p) / 2.0)
        correction_size = abs(curr_p - target)

        if correction_size > max_correction:
            continue

        result[i].midi_pitch = target
        corrections += 1

    return result, corrections
