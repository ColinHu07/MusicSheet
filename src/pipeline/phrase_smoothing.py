"""
Phrase smoothing — sustain modeling and fragmentation reduction.

Converts fragmented note sequences into smooth, sustained musical
phrases.  Works AFTER melody-path selection so it does not interfere
with melody stream identity.

The core fragmentation problem:
  Basic Pitch often splits a single held vocal note into multiple
  short events with slight pitch jitter (±1 semitone) and tiny gaps
  (50–200 ms).  The existing cleanup handles exact-pitch merges, but
  pitch jitter causes fragments to survive.  This module adds:

  1. Pitch-stability grouping — merge ±N semitone jittery fragments,
     but with CONTOUR AWARENESS to avoid flattening real melodic steps
  2. Phrase-segment smoothing — collapse runs of short near-pitch notes,
     with contour-direction detection to preserve melodic turns
  3. Re-attack suppression — final pass to catch remaining same-pitch
     adjacencies after all other cleanup
  4. Post-quantization merge — merge adjacent same-pitch quantized notes
"""

import logging
from collections import Counter
from typing import List

from .utils import NoteEvent, QuantizedNote

logger = logging.getLogger("mp3_to_sheet.phrase_smoothing")


def group_by_pitch_stability(
    notes: List[NoteEvent],
    max_gap_ms: float = 200.0,
    pitch_semitones: int = 1,
) -> List[NoteEvent]:
    """
    Group consecutive notes that represent the same sustained pitch
    (within jitter tolerance) into single longer notes, while preserving
    genuine melodic contour.

    Two consecutive notes are merged when:
      - pitch difference <= pitch_semitones
      - time gap <= max_gap_ms
      - AND the group does NOT show a clear directional pitch trend

    If a group develops a monotonic or near-monotonic pitch trend
    (consistent stepwise rise/fall), it is split so each step is
    preserved as a distinct note.

    Parameters
    ----------
    notes : list[NoteEvent]
        Sorted by start_s.
    max_gap_ms : float
        Maximum gap between notes to consider them part of the same
        sustained pitch (milliseconds).
    pitch_semitones : int
        Maximum pitch difference (inclusive) to treat as jitter rather
        than a real melodic step.
    """
    if len(notes) < 2:
        return list(notes)

    max_gap_s = max_gap_ms / 1000.0
    groups: List[List[NoteEvent]] = [[notes[0]]]

    for n in notes[1:]:
        current_group = groups[-1]
        prev = current_group[-1]
        gap = n.start_s - prev.end_s
        pitch_diff = abs(n.midi_pitch - prev.midi_pitch)

        if pitch_diff <= pitch_semitones and gap <= max_gap_s:
            current_group.append(n)
        else:
            groups.append([n])

    result: List[NoteEvent] = []
    for group in groups:
        result.extend(_contour_aware_merge(group, pitch_semitones))

    return result


def smooth_phrase_segments(
    notes: List[NoteEvent],
    max_short_note_ms: float = 250.0,
    max_pitch_diff: int = 2,
) -> List[NoteEvent]:
    """
    Collapse runs of consecutive short, near-pitch notes into longer
    sustained notes, but respect melodic contour.

    A "short" note is one shorter than max_short_note_ms.  A run of
    2+ consecutive short notes within max_pitch_diff semitones of
    each other gets merged into a single note — UNLESS the run shows
    a clear directional contour.

    Notes that are already long (> max_short_note_ms) are never merged
    by this function — they represent deliberate melodic events.
    """
    if len(notes) < 2:
        return list(notes)

    max_short_s = max_short_note_ms / 1000.0
    result: List[NoteEvent] = []
    i = 0

    while i < len(notes):
        if notes[i].duration_s >= max_short_s:
            result.append(notes[i])
            i += 1
            continue

        run_start = i
        run = [notes[i]]
        j = i + 1
        while j < len(notes):
            if notes[j].duration_s >= max_short_s:
                break
            if abs(notes[j].midi_pitch - run[-1].midi_pitch) > max_pitch_diff:
                break
            gap = notes[j].start_s - run[-1].end_s
            if gap > 0.3:
                break
            run.append(notes[j])
            j += 1

        if len(run) >= 2:
            result.extend(_contour_aware_merge(run, max_pitch_diff))
        else:
            result.append(notes[run_start])
        i = j

    return result


def suppress_reattacks(
    notes: List[NoteEvent],
    max_gap_ms: float = 200.0,
) -> List[NoteEvent]:
    """
    Final-pass merge of adjacent same-pitch notes separated by tiny
    gaps.  This catches re-attacks that survive earlier cleanup stages
    (e.g. after octave smoothing brings two notes to the same pitch).
    """
    if len(notes) < 2:
        return list(notes)

    max_gap_s = max_gap_ms / 1000.0
    result = [notes[0]]

    for n in notes[1:]:
        prev = result[-1]
        gap = n.start_s - prev.end_s
        if prev.midi_pitch == n.midi_pitch and -0.02 <= gap <= max_gap_s:
            prev.end_s = max(prev.end_s, n.end_s)
            prev.confidence = max(prev.confidence, n.confidence)
        else:
            result.append(n)

    return result


def merge_quantized_same_pitch(
    notes: List[QuantizedNote],
    max_merge_ql: float = 0.375,
) -> List[QuantizedNote]:
    """
    After quantization, merge adjacent notes of the same pitch that
    are contiguous (end of one = start of next).  This catches
    fragmentation introduced or exposed by the snap-to-grid process.

    Only merges when both fragments are short (each <= max_merge_ql),
    so that intentional re-articulation splits are preserved.

    Rests and tied notes are not merged.
    """
    if len(notes) < 2:
        return list(notes)

    result = [notes[0]]

    for n in notes[1:]:
        prev = result[-1]
        if (not prev.is_rest and not n.is_rest
                and prev.midi_pitch == n.midi_pitch
                and prev.tie_type is None and n.tie_type is None):
            prev_end = round(prev.start_ql + prev.duration_ql, 6)
            if abs(prev_end - n.start_ql) < 1e-4:
                # Only merge if at least one fragment is very short
                # (artifact of grid snapping), not a real re-articulation
                if prev.duration_ql <= max_merge_ql or n.duration_ql <= max_merge_ql:
                    combined = prev.duration_ql + n.duration_ql
                    if combined <= max_merge_ql * 2.5:
                        prev.duration_ql = round(combined, 6)
                        continue
        result.append(n)

    return result


# --------------------------------------------------------------------------- #
# Contour-aware merging
# --------------------------------------------------------------------------- #

def _contour_aware_merge(
    group: List[NoteEvent],
    pitch_tolerance: int = 1,
) -> List[NoteEvent]:
    """
    Merge a group of NoteEvents, but split it first if it contains
    a meaningful directional contour (stepwise rise or fall).

    A group with <=2 notes is merged unless the excursion exceeds the
    pitch tolerance.  For larger groups, we check whether the pitches
    form a consistent directional trend.  If so, each step is preserved.

    The excursion threshold adapts to pitch_tolerance so that when the
    caller explicitly permits larger jitter (e.g. pitch_semitones=2),
    only excursions LARGER than that tolerance trigger splitting.

    Heuristic:
    - Compute signed pitch deltas across consecutive notes.
    - If most deltas (>=50% of non-zero deltas) point the same direction
      AND total pitch excursion > pitch_tolerance, treat as a real
      melodic run and preserve individual notes.
    - Otherwise merge the whole group into one sustained note.
    """
    excursion_threshold = pitch_tolerance + 1

    if len(group) <= 1:
        return list(group)

    if len(group) == 2:
        if group[0].midi_pitch == group[1].midi_pitch:
            return [_merge_pair(group)]
        total_excursion = abs(group[1].midi_pitch - group[0].midi_pitch)
        if total_excursion >= excursion_threshold:
            return list(group)
        return [_merge_pair(group)]

    pitches = [n.midi_pitch for n in group]
    deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return [_merge_pair(group)]

    ups = sum(1 for d in nonzero if d > 0)
    downs = sum(1 for d in nonzero if d < 0)
    total = len(nonzero)

    dominant_direction_ratio = max(ups, downs) / total
    total_excursion = abs(pitches[-1] - pitches[0])

    if dominant_direction_ratio >= 0.5 and total_excursion >= excursion_threshold:
        return list(group)

    pitch_range = max(pitches) - min(pitches)
    if pitch_range >= excursion_threshold + 1:
        return _split_by_pitch_variance(group)

    return [_merge_pair(group)]


def _split_by_pitch_variance(group: List[NoteEvent]) -> List[NoteEvent]:
    """
    Split a group at points where pitch changes direction, producing
    sub-groups of consistent pitch that can each be merged safely.
    """
    if len(group) <= 2:
        return list(group)

    subgroups: List[List[NoteEvent]] = [[group[0]]]

    for i in range(1, len(group)):
        prev_pitch = group[i-1].midi_pitch
        curr_pitch = group[i].midi_pitch

        if curr_pitch == prev_pitch:
            subgroups[-1].append(group[i])
            continue

        if len(subgroups[-1]) == 1:
            subgroups[-1].append(group[i])
            continue

        last_sub = subgroups[-1]
        if len(last_sub) >= 2:
            prev_dir = last_sub[-1].midi_pitch - last_sub[-2].midi_pitch
            curr_dir = curr_pitch - prev_pitch
            if (prev_dir > 0 and curr_dir > 0) or (prev_dir < 0 and curr_dir < 0):
                last_sub.append(group[i])
                continue

        subgroups.append([group[i]])

    result: List[NoteEvent] = []
    for sub in subgroups:
        if len(sub) == 1:
            result.append(sub[0])
        else:
            pitches = [n.midi_pitch for n in sub]
            if max(pitches) - min(pitches) <= 1:
                result.append(_merge_pair(sub))
            else:
                result.extend(sub)

    return result


def _merge_pair(group: List[NoteEvent]) -> NoteEvent:
    """
    Merge a group of NoteEvents into a single sustained note.

    Pitch is chosen by stability-weighted scoring: 60% total duration
    at each pitch + 40% longest continuous run at that pitch.  This
    prefers pitches that were actually sustained over pitches that
    appear in many scattered fragments.
    """
    if len(group) == 1:
        return group[0]

    pitch_dur: Counter = Counter()
    pitch_longest_run: Counter = Counter()

    current_pitch = group[0].midi_pitch
    current_run_start = group[0].start_s

    for n in group:
        pitch_dur[n.midi_pitch] += n.duration_s
        if n.midi_pitch == current_pitch:
            run_dur = n.end_s - current_run_start
            pitch_longest_run[current_pitch] = max(
                pitch_longest_run[current_pitch], run_dur)
        else:
            current_pitch = n.midi_pitch
            current_run_start = n.start_s
            pitch_longest_run[n.midi_pitch] = max(
                pitch_longest_run[n.midi_pitch], n.duration_s)

    total_dur = sum(pitch_dur.values())
    if total_dur <= 0:
        best_pitch = group[0].midi_pitch
    else:
        best_pitch = group[0].midi_pitch
        best_score = -1.0
        for pitch in pitch_dur:
            dur_score = pitch_dur[pitch] / total_dur
            run_score = pitch_longest_run.get(pitch, 0) / total_dur
            score = dur_score * 0.6 + run_score * 0.4
            if score > best_score:
                best_score = score
                best_pitch = pitch

    return NoteEvent(
        start_s=group[0].start_s,
        end_s=group[-1].end_s,
        midi_pitch=best_pitch,
        confidence=max(n.confidence for n in group),
    )
