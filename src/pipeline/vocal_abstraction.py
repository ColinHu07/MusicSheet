"""
Vocal abstraction — convert dense vocal pitch tracking into musical note events.

When transcribing from an isolated vocal stem, Basic Pitch produces far
more note events than from a full mix.  Vibrato, pitch wobble, breath
noise, and micro-fluctuations all become separate note events that
overwhelm the output.

This module sits AFTER melody-path selection + post-cleanup and BEFORE
quantization.  It abstracts the raw vocal performance into the kind of
note events a human transcriber would write on a lead sheet.

Stages:
  1. **Vibrato suppression** — detect oscillating pitch clusters and
     collapse them into single sustained notes.
  2. **Phrase consolidation** — merge sequences of short notes that
     belong to one sustained vocal event (held syllable, legato passage).
  3. **Representative pitch** — choose pitch for merged groups using
     a robust estimator (duration-weighted trimmed median) that is
     insensitive to vibrato outliers.
  4. **Micro-rest absorption** — remove tiny gaps between phrase notes.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.vocal_abstraction")


@dataclass
class VocalAbstractionConfig:
    """Tuneable parameters for vocal-level note abstraction."""

    # -- vibrato suppression --------------------------------------------------
    vibrato_merge_enabled: bool = True
    vibrato_max_semitone_span: int = 2
    vibrato_window_ms: float = 1200.0
    vibrato_min_notes: int = 3
    vibrato_max_gap_ms: float = 250.0

    # -- sustained-pitch consolidation (same/near pitch with small gaps) ------
    sustain_merge_enabled: bool = True
    sustain_max_gap_ms: float = 350.0
    sustain_pitch_tolerance: int = 1

    # -- phrase consolidation -------------------------------------------------
    phrase_merge_enabled: bool = True
    phrase_max_gap_ms: float = 250.0
    phrase_pitch_tolerance: int = 2
    phrase_min_short_note_ms: float = 0.0
    phrase_max_short_note_ms: float = 350.0

    # -- second-pass phrase consolidation (catches adjacencies from prior) ----
    second_pass_enabled: bool = True

    # -- micro-rest absorption ------------------------------------------------
    micro_rest_max_ms: float = 200.0

    # -- minimum note duration after abstraction ------------------------------
    min_note_ms: float = 120.0


@dataclass
class AbstractionStats:
    """Metrics from the vocal abstraction pass."""
    notes_before: int = 0
    notes_after_vibrato: int = 0
    notes_after_phrase: int = 0
    notes_after_micro_rest: int = 0
    notes_after_min_dur: int = 0
    notes_final: int = 0
    vibrato_groups_merged: int = 0
    sustain_groups_merged: int = 0
    phrase_groups_merged: int = 0
    micro_rests_removed: int = 0
    avg_duration_before: float = 0.0
    avg_duration_after: float = 0.0


def abstract_vocal_notes(
    notes: List[NoteEvent],
    config: Optional[VocalAbstractionConfig] = None,
) -> Tuple[List[NoteEvent], AbstractionStats]:
    """
    Main entry point: abstract a dense vocal note stream into
    musically meaningful note events.

    Returns the simplified note list and statistics about what was done.
    """
    if config is None:
        config = VocalAbstractionConfig()

    stats = AbstractionStats()
    stats.notes_before = len(notes)
    if notes:
        stats.avg_duration_before = (
            sum(n.duration_s for n in notes) / len(notes))

    if len(notes) < 2:
        stats.notes_final = len(notes)
        stats.avg_duration_after = stats.avg_duration_before
        return list(notes), stats

    # 1. Vibrato suppression
    if config.vibrato_merge_enabled:
        notes, vib_count = suppress_vibrato(
            notes,
            max_span=config.vibrato_max_semitone_span,
            window_ms=config.vibrato_window_ms,
            min_notes=config.vibrato_min_notes,
            max_gap_ms=config.vibrato_max_gap_ms,
        )
        stats.vibrato_groups_merged = vib_count
    stats.notes_after_vibrato = len(notes)

    # 2. Sustained-pitch consolidation (same/near pitch with small gaps)
    if config.sustain_merge_enabled:
        notes, sustain_count = consolidate_sustained(
            notes,
            max_gap_ms=config.sustain_max_gap_ms,
            pitch_tolerance=config.sustain_pitch_tolerance,
        )
        stats.sustain_groups_merged = sustain_count

    # 3. Phrase consolidation
    if config.phrase_merge_enabled:
        notes, phrase_count = consolidate_phrases(
            notes,
            max_gap_ms=config.phrase_max_gap_ms,
            pitch_tolerance=config.phrase_pitch_tolerance,
            max_short_note_ms=config.phrase_max_short_note_ms,
        )
        stats.phrase_groups_merged = phrase_count

    # 4. Second-pass phrase consolidation
    if config.second_pass_enabled and config.phrase_merge_enabled:
        notes, phrase_count_2 = consolidate_phrases(
            notes,
            max_gap_ms=config.phrase_max_gap_ms,
            pitch_tolerance=config.phrase_pitch_tolerance,
            max_short_note_ms=config.phrase_max_short_note_ms * 1.5,
        )
        stats.phrase_groups_merged += phrase_count_2

    stats.notes_after_phrase = len(notes)

    # 5. Micro-rest absorption (in NoteEvent domain)
    notes, rest_count = absorb_micro_rests(
        notes, max_gap_ms=config.micro_rest_max_ms)
    stats.micro_rests_removed = rest_count
    stats.notes_after_micro_rest = len(notes)

    # 6. Remove notes that are too short after all merging
    min_dur_s = config.min_note_ms / 1000.0
    notes = [n for n in notes if n.duration_s >= min_dur_s]
    stats.notes_after_min_dur = len(notes)

    stats.notes_final = len(notes)
    if notes:
        stats.avg_duration_after = (
            sum(n.duration_s for n in notes) / len(notes))

    return notes, stats


# =========================================================================== #
# 1. Vibrato suppression
# =========================================================================== #

def suppress_vibrato(
    notes: List[NoteEvent],
    max_span: int = 2,
    window_ms: float = 600.0,
    min_notes: int = 3,
    max_gap_ms: float = 150.0,
) -> Tuple[List[NoteEvent], int]:
    """
    Detect clusters of notes that oscillate around a pitch center
    (vibrato) and collapse them into single sustained notes.

    A vibrato cluster is identified when:
    - ``min_notes`` or more consecutive notes
    - all pitches within ``max_span`` semitones of each other
    - each note starts within ``max_gap_ms`` of the previous note's end
    - total duration does not exceed ``window_ms``
    - the pitch pattern oscillates (changes direction at least once)

    Real melodic movement (consistent ascent/descent) is preserved.

    Returns (merged_notes, groups_merged_count).
    """
    if len(notes) < min_notes:
        return list(notes), 0

    max_gap_s = max_gap_ms / 1000.0
    window_s = window_ms / 1000.0
    result: List[NoteEvent] = []
    groups_merged = 0
    i = 0

    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1

        while j < len(notes):
            prev = cluster[-1]
            curr = notes[j]
            gap = curr.start_s - prev.end_s
            if gap > max_gap_s:
                break

            all_pitches = [n.midi_pitch for n in cluster] + [curr.midi_pitch]
            if max(all_pitches) - min(all_pitches) > max_span:
                break

            span_s = curr.end_s - cluster[0].start_s
            if span_s > window_s:
                break

            cluster.append(curr)
            j += 1

        if len(cluster) >= min_notes and _is_oscillating(cluster):
            result.append(_merge_with_robust_pitch(cluster))
            groups_merged += 1
        else:
            result.append(notes[i])
            j = i + 1

        i = j

    return result, groups_merged


def _is_oscillating(notes: List[NoteEvent]) -> bool:
    """
    Check whether a sequence of notes oscillates around a pitch center
    rather than moving in one direction.

    Requirements for oscillation:
    - at least 2 direction changes (up-down-up or down-up-down)
    - OR all notes at the same pitch (flat line = sustained note)
    - a single direction change (up-down) could be a melodic turn,
      so we require more evidence

    A monotonic sequence (all ascending or all descending) is NOT vibrato.
    A single up-down or down-up (arch/valley) is NOT vibrato either —
    it's more likely a real ornamental turn.
    """
    if len(notes) < 3:
        return False

    pitches = [n.midi_pitch for n in notes]
    deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    nonzero = [d for d in deltas if d != 0]

    if len(nonzero) < 2:
        return True

    direction_changes = 0
    for k in range(1, len(nonzero)):
        if (nonzero[k] > 0) != (nonzero[k-1] > 0):
            direction_changes += 1

    return direction_changes >= 2


# =========================================================================== #
# 2. Sustained-pitch consolidation
# =========================================================================== #

def consolidate_sustained(
    notes: List[NoteEvent],
    max_gap_ms: float = 350.0,
    pitch_tolerance: int = 1,
) -> Tuple[List[NoteEvent], int]:
    """
    Merge consecutive notes at the same or nearly-same pitch that
    are separated by small gaps.  Unlike phrase consolidation, this
    applies to notes of ANY duration — it catches fragmented sustains
    where one held note was split into multiple events by the detector.

    If the group shows a directional contour (stepwise rise/fall),
    each step is preserved.

    Returns (merged_notes, groups_merged_count).
    """
    if len(notes) < 2:
        return list(notes), 0

    max_gap_s = max_gap_ms / 1000.0
    result: List[NoteEvent] = []
    groups_merged = 0
    i = 0

    while i < len(notes):
        group = [notes[i]]
        j = i + 1

        while j < len(notes):
            prev = group[-1]
            curr = notes[j]
            gap = curr.start_s - prev.end_s
            if gap > max_gap_s:
                break
            if abs(curr.midi_pitch - prev.midi_pitch) > pitch_tolerance:
                break
            group.append(curr)
            j += 1

        if len(group) >= 2:
            if _has_directional_contour(group, pitch_tolerance):
                result.extend(group)
            else:
                result.append(_merge_with_robust_pitch(group))
                groups_merged += 1
        else:
            result.append(notes[i])

        i = j

    return result, groups_merged


# =========================================================================== #
# 3. Phrase consolidation
# =========================================================================== #

def consolidate_phrases(
    notes: List[NoteEvent],
    max_gap_ms: float = 200.0,
    pitch_tolerance: int = 2,
    max_short_note_ms: float = 250.0,
) -> Tuple[List[NoteEvent], int]:
    """
    Merge consecutive short notes that belong to the same sustained
    vocal event into single longer notes.

    A phrase group is a run of notes where:
    - each note is shorter than ``max_short_note_ms``
    - consecutive pitches differ by at most ``pitch_tolerance``
    - gaps between notes are at most ``max_gap_ms``

    Long notes (> max_short_note_ms) act as phrase boundaries and are
    never merged.

    Returns (merged_notes, groups_merged_count).
    """
    if len(notes) < 2:
        return list(notes), 0

    max_gap_s = max_gap_ms / 1000.0
    max_short_s = max_short_note_ms / 1000.0
    result: List[NoteEvent] = []
    groups_merged = 0
    i = 0

    while i < len(notes):
        if notes[i].duration_s > max_short_s:
            result.append(notes[i])
            i += 1
            continue

        group = [notes[i]]
        j = i + 1

        while j < len(notes):
            curr = notes[j]
            prev = group[-1]

            if curr.duration_s > max_short_s:
                break
            if abs(curr.midi_pitch - prev.midi_pitch) > pitch_tolerance:
                break
            gap = curr.start_s - prev.end_s
            if gap > max_gap_s:
                break

            group.append(curr)
            j += 1

        if len(group) >= 2:
            if len(group) >= 3 and _has_directional_contour(
                    group, pitch_tolerance):
                result.extend(group)
            else:
                result.append(_merge_with_robust_pitch(group))
                groups_merged += 1
        else:
            result.append(notes[i])
            j = i + 1

        i = j

    return result, groups_merged


def _has_directional_contour(
    group: List[NoteEvent], pitch_tolerance: int = 2,
) -> bool:
    """
    Return True if the group shows a clear ascending or descending
    melodic contour that should be preserved rather than merged.

    A group has directional contour when:
    - total pitch excursion exceeds pitch_tolerance
    - at least 50% of non-zero pitch deltas move in the same direction
    """
    if len(group) < 2:
        return False

    pitches = [n.midi_pitch for n in group]
    total_excursion = abs(pitches[-1] - pitches[0])
    if total_excursion < pitch_tolerance:
        return False

    deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return False

    ups = sum(1 for d in nonzero if d > 0)
    downs = sum(1 for d in nonzero if d < 0)
    dominant = max(ups, downs) / len(nonzero)

    return dominant >= 0.5


# =========================================================================== #
# 4. Representative pitch selection
# =========================================================================== #

def _merge_with_robust_pitch(group: List[NoteEvent]) -> NoteEvent:
    """
    Merge a group of NoteEvents into one, choosing pitch via
    stability-weighted estimation.

    Algorithm:
    1. Build a pitch histogram weighted by duration.
    2. Compute the longest continuous run at each pitch.
    3. Score each pitch: 60% duration weight + 40% run-length weight.
    4. Choose the pitch with the highest combined score.

    This is robust to vibrato outliers and scattered fragments while
    preferring pitches that the singer actually sustained.
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
        majority_pitch = group[0].midi_pitch
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

        majority_pitch = best_pitch

    return NoteEvent(
        start_s=group[0].start_s,
        end_s=group[-1].end_s,
        midi_pitch=majority_pitch,
        confidence=max(n.confidence for n in group),
    )


def _weighted_median(pitch_dur: Dict[int, float]) -> int:
    """
    Compute the weighted median pitch from a {pitch: duration} dict.

    Expands each pitch by its weight (in small increments) and finds
    the value at the 50th percentile.
    """
    if not pitch_dur:
        return 60

    if len(pitch_dur) == 1:
        return next(iter(pitch_dur))

    sorted_pitches = sorted(pitch_dur.keys())
    total = sum(pitch_dur.values())
    cumulative = 0.0

    for p in sorted_pitches:
        cumulative += pitch_dur[p]
        if cumulative >= total / 2.0:
            return p

    return sorted_pitches[-1]


# =========================================================================== #
# 4. Micro-rest absorption (NoteEvent domain)
# =========================================================================== #

def absorb_micro_rests(
    notes: List[NoteEvent],
    max_gap_ms: float = 150.0,
) -> Tuple[List[NoteEvent], int]:
    """
    Remove tiny gaps between consecutive notes by extending the
    preceding note to fill them.

    Only gaps smaller than ``max_gap_ms`` are absorbed, and only when
    the pitch difference is <=2 semitones (indicating phrase continuity
    rather than an actual rest).

    Returns (adjusted_notes, gaps_absorbed_count).
    """
    if len(notes) < 2:
        return list(notes), 0

    max_gap_s = max_gap_ms / 1000.0
    result = [NoteEvent(notes[0].start_s, notes[0].end_s,
                        notes[0].midi_pitch, notes[0].confidence)]
    absorbed = 0

    for i in range(1, len(notes)):
        prev = result[-1]
        curr = notes[i]
        gap = curr.start_s - prev.end_s

        if 0 < gap <= max_gap_s and abs(curr.midi_pitch - prev.midi_pitch) <= 2:
            prev.end_s = curr.start_s
            absorbed += 1

        result.append(NoteEvent(
            curr.start_s, curr.end_s,
            curr.midi_pitch, curr.confidence))

    return result, absorbed
