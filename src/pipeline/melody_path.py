"""
Melody-path resolver — global optimal monophonic melody selection.

This is the algorithmic heart of the transcription system.  Given a set
of overlapping candidate notes (from Basic Pitch), it finds the single
best monophonic melodic line using dynamic programming over a segmented
time lattice.

The approach:
    1. Build a timeline of unique event boundaries.
    2. At each boundary, identify which candidate notes are active.
    3. Construct a lattice where each "state" at a boundary is one of:
       - a specific active candidate note, or
       - a gap/rest (no melody note sounding).
    4. Score each state (intrinsic note quality) and each transition
       (melodic plausibility between consecutive states).
    5. Run Viterbi-style dynamic programming to find the globally optimal
       path through the lattice.
    6. Apply post-selection cleanup (merge micro-gaps, consolidate repeated
       pitches, remove micro-notes, octave smoothing, range enforcement).

The result is a clean monophonic melody that a trained musician would
recognise as the principal tune of the song.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..config.melody_config import MelodyConfig, InstrumentRange
from .candidate_notes import CandidateNote, compute_clutter_density
from .local_register import LocalRegisterContext, compute_local_register_contexts
from .melody_features import (
    compute_median_pitch,
    compute_upper_quartile_pitch,
    score_gap_continuation,
    score_gap_entry,
    score_note_intrinsic,
    score_transition,
)
from .phrase_smoothing import (
    group_by_pitch_stability,
    smooth_phrase_segments,
    suppress_reattacks,
)
from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.melody_path")

GAP_STATE = -1  # sentinel id for the gap/rest state


@dataclass
class LatticeColumn:
    """One column of the DP lattice, corresponding to a time boundary."""
    time_s: float
    candidate_ids: List[int]   # indices into the candidates list
    has_gap: bool = True       # whether gap state is available


@dataclass
class PathResult:
    """Output of the melody-path resolver."""
    selected_notes: List[NoteEvent]
    selected_candidate_ids: List[int]
    stats: Dict[str, object] = field(default_factory=dict)


def resolve_melody(
    candidates: List[CandidateNote],
    config: MelodyConfig,
) -> PathResult:
    """
    Select the best monophonic melody path from overlapping candidates.

    This is the main entry point.  It builds the lattice, runs DP,
    extracts the best path, and applies post-selection cleanup.

    Parameters
    ----------
    candidates : list[CandidateNote]
        Normalised candidate notes (sorted by start time).
    config : MelodyConfig
        All scoring weights and thresholds.

    Returns
    -------
    PathResult with the selected monophonic melody as NoteEvents.
    """
    if not candidates:
        return PathResult(selected_notes=[], selected_candidate_ids=[], stats={})

    inst_range = config.get_instrument_range()

    # -- Pitch floor (explicit or auto-computed for vocals mode) -------------
    pitch_floor_filtered = 0
    auto_floor_applied = False
    effective_floor: Optional[int] = config.pitch_floor_midi

    # Auto-compute floor for vocals mode when none was explicitly provided.
    # Sub-harmonics from Basic Pitch cluster ~1 octave below the real vocal,
    # so we remove candidates below (upper_quartile − offset) to eliminate
    # those artifacts while preserving the real vocal range.
    if (effective_floor is None
            and config.vocals_mode
            and config.vocals_auto_pitch_floor):
        pre_uq = compute_upper_quartile_pitch(candidates)
        effective_floor = round(pre_uq - config.vocals_floor_semitones_below_uq)
        auto_floor_applied = True
        logger.info(
            "Vocals auto-floor: UQ=%.0f  offset=%d  floor=%d",
            pre_uq, config.vocals_floor_semitones_below_uq, effective_floor,
        )

    if effective_floor is not None:
        original_count = len(candidates)
        candidates = [c for c in candidates if c.pitch_midi >= effective_floor]
        pitch_floor_filtered = original_count - len(candidates)
        logger.info(
            "Pitch floor %d: filtered %d/%d candidates",
            effective_floor, pitch_floor_filtered, original_count,
        )

    if not candidates:
        return PathResult(
            selected_notes=[], selected_candidate_ids=[],
            stats={"pitch_floor_filtered": pitch_floor_filtered},
        )

    median_pitch = compute_median_pitch(candidates)
    upper_q_pitch = compute_upper_quartile_pitch(candidates)
    clutter = compute_clutter_density(candidates, config.clutter_window_ms)

    # -- Compute local register contexts ------------------------------------
    local_contexts: Optional[List[LocalRegisterContext]] = None
    if config.local_register_mode != "off":
        local_contexts = compute_local_register_contexts(
            candidates, config.local_register_window_s,
        )

    # -- Pre-compute intrinsic scores for each candidate --------------------
    intrinsic_scores: List[float] = []
    for i, note in enumerate(candidates):
        s = score_note_intrinsic(
            note, config, inst_range,
            clutter_density=clutter[i],
            median_pitch=median_pitch,
            upper_quartile_pitch=upper_q_pitch,
            local_context=local_contexts[i] if local_contexts else None,
        )
        intrinsic_scores.append(s)

    # -- Build the lattice --------------------------------------------------
    lattice = _build_lattice(candidates, config)

    if not lattice:
        return PathResult(selected_notes=[], selected_candidate_ids=[], stats={})

    # -- Run DP/Viterbi -----------------------------------------------------
    best_path = _viterbi_decode(
        lattice, candidates, intrinsic_scores, config, inst_range, median_pitch,
    )

    # -- Extract selected notes from path -----------------------------------
    selected_ids: List[int] = [cid for cid in best_path if cid != GAP_STATE]
    selected_ids_set: Set[int] = set(selected_ids)

    raw_notes: List[NoteEvent] = []
    for cid in selected_ids:
        c = candidates[cid]
        raw_notes.append(NoteEvent(
            start_s=c.start_s,
            end_s=c.end_s,
            midi_pitch=c.pitch_midi,
            confidence=c.confidence,
        ))

    # -- Deduplicate (a candidate can appear in multiple columns) -----------
    raw_notes = _deduplicate_notes(raw_notes)
    notes_before_cleanup = len(raw_notes)
    avg_dur_before = (sum(n.duration_s for n in raw_notes) / len(raw_notes)
                      if raw_notes else 0.0)

    # -- Global register correction (vocals mode) --------------------------
    global_register_corrections = 0
    if (config.vocals_mode
            and config.vocals_global_register_correction
            and raw_notes):
        raw_notes, global_register_corrections = _global_register_correction(
            raw_notes, inst_range,
        )

    # -- Post-selection cleanup ---------------------------------------------
    cleaned = _post_cleanup(raw_notes, config, inst_range)

    # -- Compute stats for debug/evaluation ---------------------------------
    avg_dur = 0.0
    avg_interval = 0.0
    avg_selected_pitch = 0.0
    selected_pitches: List[int] = []
    if cleaned:
        avg_dur = sum(n.duration_s for n in cleaned) / len(cleaned)
        selected_pitches = [n.midi_pitch for n in cleaned]
        avg_selected_pitch = sum(selected_pitches) / len(selected_pitches)
    if len(cleaned) > 1:
        intervals = [abs(cleaned[i+1].midi_pitch - cleaned[i].midi_pitch)
                      for i in range(len(cleaned) - 1)]
        avg_interval = sum(intervals) / len(intervals)

    mean_clutter = 0.0
    if selected_ids_set:
        mean_clutter = sum(clutter[i] for i in selected_ids_set) / len(selected_ids_set)

    # Stream switch stats
    stream_switches = 0
    if len(cleaned) > 1:
        for i in range(len(cleaned) - 1):
            if abs(cleaned[i+1].midi_pitch - cleaned[i].midi_pitch) > 7:
                stream_switches += 1

    # Local register stats
    pct_below_local_median = 0.0
    pct_in_target_band = 0.0
    if local_contexts and selected_ids_set:
        below_count = 0
        in_band_count = 0
        for sid in selected_ids_set:
            ctx = local_contexts[sid]
            p = candidates[sid].pitch_midi
            if p < ctx.local_median:
                below_count += 1
            if ctx.local_upper_quartile - 3 <= p <= ctx.local_upper_quartile + 7:
                in_band_count += 1
        n_sel = len(selected_ids_set)
        pct_below_local_median = round(below_count / n_sel * 100, 1)
        pct_in_target_band = round(in_band_count / n_sel * 100, 1)

    # Selected-note pitch percentiles
    sel_median_pitch = 0.0
    sel_upper_quartile_pitch = 0.0
    if selected_pitches:
        sp_sorted = sorted(selected_pitches)
        sel_median_pitch = float(sp_sorted[len(sp_sorted) // 2])
        sel_upper_quartile_pitch = float(sp_sorted[int(len(sp_sorted) * 0.75)])

    # Sustain / phrase metrics
    notes_merged = max(0, notes_before_cleanup - len(cleaned))

    stats = {
        "total_candidates": len(candidates) + pitch_floor_filtered,
        "pitch_floor_filtered": pitch_floor_filtered,
        "pitch_floor_midi_used": effective_floor,
        "auto_floor_applied": auto_floor_applied,
        "candidates_after_floor": len(candidates),
        "global_register_corrections": global_register_corrections,
        "lattice_columns": len(lattice),
        "path_length": len(best_path),
        "gap_states_in_path": sum(1 for x in best_path if x == GAP_STATE),
        "unique_selected": len(selected_ids_set),
        "notes_before_cleanup": notes_before_cleanup,
        "after_cleanup": len(cleaned),
        "notes_merged": notes_merged,
        "avg_duration_before_cleanup_s": round(avg_dur_before, 4),
        "median_pitch": median_pitch,
        "upper_quartile_pitch": upper_q_pitch,
        "avg_note_duration_s": round(avg_dur, 4),
        "avg_melodic_interval": round(avg_interval, 2),
        "mean_clutter_of_selected": round(mean_clutter, 2),
        "avg_selected_pitch": round(avg_selected_pitch, 1),
        "selected_median_pitch": sel_median_pitch,
        "selected_upper_quartile_pitch": sel_upper_quartile_pitch,
        "stream_switches": stream_switches,
        "pct_below_local_median": pct_below_local_median,
        "pct_in_target_band": pct_in_target_band,
    }
    logger.info(
        "Melody path: %d candidates -> %d selected -> %d after cleanup  "
        "(avg_dur=%.3fs, avg_interval=%.1f st, clutter=%.1f, "
        "sel_median=%.0f, sel_uq=%.0f, stream_sw=%d)",
        len(candidates), len(selected_ids_set), len(cleaned),
        avg_dur, avg_interval, mean_clutter,
        sel_median_pitch, sel_upper_quartile_pitch, stream_switches,
    )

    return PathResult(
        selected_notes=cleaned,
        selected_candidate_ids=sorted(selected_ids_set),
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Lattice construction
# --------------------------------------------------------------------------- #

def _build_lattice(
    candidates: List[CandidateNote],
    config: MelodyConfig,
) -> List[LatticeColumn]:
    """
    Build the DP lattice by collecting all unique time boundaries and
    determining which candidates are active at each boundary.
    """
    boundaries: Set[float] = set()
    for c in candidates:
        boundaries.add(round(c.start_s, 6))
        boundaries.add(round(c.end_s, 6))

    sorted_times = sorted(boundaries)

    if len(sorted_times) < 2:
        return []

    lattice: List[LatticeColumn] = []
    for t in sorted_times:
        active_ids: List[int] = []
        for i, c in enumerate(candidates):
            if c.start_s <= t < c.end_s:
                active_ids.append(i)
            elif abs(c.start_s - t) < 1e-6:
                if i not in active_ids:
                    active_ids.append(i)

        if active_ids or config.allow_gap_state:
            lattice.append(LatticeColumn(
                time_s=t,
                candidate_ids=active_ids,
                has_gap=config.allow_gap_state,
            ))

    return lattice


# --------------------------------------------------------------------------- #
# Viterbi-style DP
# --------------------------------------------------------------------------- #

def _viterbi_decode(
    lattice: List[LatticeColumn],
    candidates: List[CandidateNote],
    intrinsic_scores: List[float],
    config: MelodyConfig,
    inst_range: InstrumentRange,
    median_pitch: float,
) -> List[int]:
    """
    Find the globally optimal path through the lattice.

    State at each column is either a candidate_id or GAP_STATE.
    Score = sum of intrinsic note scores + transition scores between
    consecutive columns.

    Uses standard Viterbi with backpointer tracking.
    """
    n_cols = len(lattice)
    if n_cols == 0:
        return []

    dp: List[Dict[int, Tuple[float, int]]] = [{} for _ in range(n_cols)]

    # -- Initialise first column --
    col0 = lattice[0]
    for cid in col0.candidate_ids:
        dp[0][cid] = (intrinsic_scores[cid], -1)
    if col0.has_gap:
        dp[0][GAP_STATE] = (0.0, -1)

    # -- Forward pass --
    for col_idx in range(1, n_cols):
        col = lattice[col_idx]
        prev_col = lattice[col_idx - 1]
        dt = col.time_s - prev_col.time_s

        current_states: List[int] = list(col.candidate_ids)
        if col.has_gap:
            current_states.append(GAP_STATE)

        prev_states = list(dp[col_idx - 1].keys())
        if not prev_states:
            for cid in col.candidate_ids:
                dp[col_idx][cid] = (intrinsic_scores[cid], -1)
            if col.has_gap:
                dp[col_idx][GAP_STATE] = (0.0, -1)
            continue

        for cur_state in current_states:
            best_score = -1e18
            best_prev = prev_states[0] if prev_states else -1

            cur_intrinsic = (
                intrinsic_scores[cur_state] if cur_state != GAP_STATE else 0.0
            )

            for prev_state in prev_states:
                prev_score = dp[col_idx - 1][prev_state][0]
                trans = _transition_score(
                    prev_state, cur_state, candidates, config, dt,
                )
                total = prev_score + trans + cur_intrinsic
                if total > best_score:
                    best_score = total
                    best_prev = prev_state

            dp[col_idx][cur_state] = (best_score, best_prev)

    # -- Backtrace --
    last_col = dp[n_cols - 1]
    if not last_col:
        return []

    best_final_state = max(last_col, key=lambda s: last_col[s][0])
    path: List[int] = [best_final_state]
    for col_idx in range(n_cols - 1, 0, -1):
        _, prev_state = dp[col_idx][path[-1]]
        path.append(prev_state)

    path.reverse()
    return path


def _transition_score(
    prev_state: int,
    cur_state: int,
    candidates: List[CandidateNote],
    config: MelodyConfig,
    dt: float,
) -> float:
    """Compute the transition score between two states in the lattice."""
    # gap -> gap
    if prev_state == GAP_STATE and cur_state == GAP_STATE:
        return score_gap_continuation(dt, config)

    # note -> gap
    if prev_state != GAP_STATE and cur_state == GAP_STATE:
        prev_note = candidates[prev_state]
        return score_gap_entry(prev_note, prev_note.end_s, config)

    # gap -> note (re-entering melody costs less than leaving)
    if prev_state == GAP_STATE and cur_state != GAP_STATE:
        return -config.gap_entry_cost * 0.3

    # note -> note (same note continuing = stability bonus)
    if prev_state == cur_state:
        return config.same_note_continuation_bonus

    # note -> different note = full melodic transition scoring
    prev_note = candidates[prev_state]
    cur_note = candidates[cur_state]
    return score_transition(prev_note, cur_note, config)


# --------------------------------------------------------------------------- #
# Post-selection cleanup
# --------------------------------------------------------------------------- #

_SEGMENT_GAP_S = 1.0  # time gap that defines a phrase/segment boundary


def _global_register_correction(
    notes: List[NoteEvent],
    inst_range: InstrumentRange,
) -> Tuple[List[NoteEvent], int]:
    """
    Detect phrase segments whose median pitch is too low and shift each
    such segment up by 12 semitones (one octave).

    By operating on *whole segments*, contour within each segment is
    perfectly preserved.  Only segments whose median is more than 8
    semitones below the instrument center are shifted.
    """
    if not notes:
        return notes, 0

    # Split into phrase segments based on time gaps
    segments: List[List[NoteEvent]] = []
    seg: List[NoteEvent] = [notes[0]]
    for i in range(1, len(notes)):
        gap = notes[i].start_s - notes[i - 1].end_s
        if gap > _SEGMENT_GAP_S:
            segments.append(seg)
            seg = [notes[i]]
        else:
            seg.append(notes[i])
    segments.append(seg)

    center = inst_range.center
    corrections = 0

    for seg in segments:
        pitches = sorted(n.midi_pitch for n in seg)
        seg_median = pitches[len(pitches) // 2]

        if center - seg_median <= 14:
            continue

        if not all(inst_range.contains(n.midi_pitch + 12) for n in seg):
            continue

        for n in seg:
            n.midi_pitch += 12
            corrections += 1

        logger.debug(
            "Register correction: shifted segment of %d notes +12 "
            "(seg_median was %d, center=%d)",
            len(seg), seg_median, center,
        )

    if corrections:
        logger.info(
            "Global register correction: shifted %d notes across %d "
            "segments (center=%d)",
            corrections, sum(
                1 for s in segments
                if sorted(n.midi_pitch for n in s)[len(s) // 2] >= center - 8
                # count segments that were shifted (their medians are now higher)
            ), center,
        )

    return notes, corrections


def _deduplicate_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Remove duplicates from selected notes (same start/pitch)."""
    seen: Set[Tuple[float, int]] = set()
    result: List[NoteEvent] = []
    for n in notes:
        key = (round(n.start_s, 4), n.midi_pitch)
        if key not in seen:
            seen.add(key)
            result.append(n)
    result.sort(key=lambda n: n.start_s)
    return result


def _post_cleanup(
    notes: List[NoteEvent],
    config: MelodyConfig,
    inst_range: InstrumentRange,
) -> List[NoteEvent]:
    """
    Apply post-selection cleanup to produce a clean, phrase-like melody.

    Steps:
      1. Merge tiny gaps between same-pitch notes
      2. Merge tiny gaps between nearby-pitch notes (extend prev)
      3. Consolidate repeated same-pitch notes separated by small gaps
      4. Pitch-stability grouping (merge jittery fragments)
      5. Remove micro-notes
      6. Phrase-segment smoothing (collapse runs of short notes)
      7. Octave smoothing (fix +/-12 artifacts)
      8. Monophonic enforcement
      9. Re-attack suppression (final same-pitch merge)
     10. Instrument range enforcement (transpose, not clip)
    """
    if not notes:
        return notes

    notes = sorted(notes, key=lambda n: n.start_s)

    # 1. Merge tiny gaps (same pitch)
    notes = _merge_tiny_gaps(notes, config.merge_gap_ms)

    # 2. Merge tiny gaps (nearby pitch — extend previous note)
    notes = _merge_nearby_pitch_gaps(
        notes, config.merge_nearby_pitch_gap_ms, config.nearby_pitch_semitones,
    )

    # 3. Consolidate repeated same-pitch segments
    if config.consolidate_repeated_pitches:
        notes = _consolidate_repeated(notes, config.consolidate_max_gap_ms)

    # 4. Pitch-stability grouping (merge ±N semitone jitter fragments)
    notes = group_by_pitch_stability(
        notes, config.pitch_stability_max_gap_ms,
        config.pitch_stability_semitones,
    )

    # 5. Remove micro-notes
    min_dur = config.micro_note_ms / 1000.0
    notes = [n for n in notes if n.duration_s >= min_dur]

    # 6. Phrase-segment smoothing (collapse runs of short near-pitch notes)
    if config.phrase_smoothing_enabled:
        notes = smooth_phrase_segments(
            notes, config.phrase_max_short_note_ms,
            config.phrase_max_pitch_diff,
        )

    # 7. Octave smoothing
    if config.octave_smoothing:
        notes = _octave_smooth(notes)

    # 8. Monophonic enforcement
    notes = _enforce_monophonic(notes)

    # 9. Re-attack suppression (catch same-pitch re-attacks after octave smoothing)
    if config.reattack_suppression_enabled:
        notes = suppress_reattacks(notes, config.reattack_max_gap_ms)

    # 10. Instrument range
    if config.range_enforcement:
        notes = _enforce_range(notes, inst_range, config.range_enforcement_mode)

    return notes


def _merge_tiny_gaps(notes: List[NoteEvent], gap_ms: float) -> List[NoteEvent]:
    """Extend a note to fill a tiny gap if the next note has the same pitch."""
    if len(notes) < 2:
        return notes
    gap_s = gap_ms / 1000.0
    result = [notes[0]]
    for n in notes[1:]:
        prev = result[-1]
        g = n.start_s - prev.end_s
        if prev.midi_pitch == n.midi_pitch and 0 < g <= gap_s:
            prev.end_s = n.end_s
        else:
            result.append(n)
    return result


def _merge_nearby_pitch_gaps(
    notes: List[NoteEvent], gap_ms: float, semitone_tolerance: int,
) -> List[NoteEvent]:
    """
    When two consecutive notes have nearly the same pitch and a tiny gap,
    extend the previous note to fill the gap.  This reduces fragmentation
    while preserving the melodic contour.
    """
    if len(notes) < 2:
        return notes
    gap_s = gap_ms / 1000.0
    result = [notes[0]]
    for n in notes[1:]:
        prev = result[-1]
        g = n.start_s - prev.end_s
        pitch_diff = abs(n.midi_pitch - prev.midi_pitch)
        if 0 < g <= gap_s and pitch_diff <= semitone_tolerance:
            prev.end_s = n.start_s
        result.append(n)
    return result


def _consolidate_repeated(notes: List[NoteEvent], max_gap_ms: float) -> List[NoteEvent]:
    """
    Merge adjacent notes of the same pitch separated by small gaps
    into a single sustained note.  This is the "sustain modelling" step:
    Basic Pitch often splits a single held note into multiple short events.
    """
    if len(notes) < 2:
        return notes
    max_gap_s = max_gap_ms / 1000.0
    result = [notes[0]]
    for n in notes[1:]:
        prev = result[-1]
        g = n.start_s - prev.end_s
        if prev.midi_pitch == n.midi_pitch and -0.01 <= g <= max_gap_s:
            prev.end_s = max(prev.end_s, n.end_s)
            prev.confidence = max(prev.confidence, n.confidence)
        else:
            result.append(n)
    return result


def _octave_smooth(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Fix octave-detection artifacts: if a note jumps +/-12 from both
    neighbours, pull it back to the surrounding register.
    """
    if len(notes) < 3:
        return notes

    for i in range(1, len(notes) - 1):
        dp = notes[i].midi_pitch - notes[i - 1].midi_pitch
        dn = notes[i].midi_pitch - notes[i + 1].midi_pitch
        if abs(dp) in (12, 24) and abs(dn) in (12, 24):
            if (dp > 0 and dn > 0) or (dp < 0 and dn < 0):
                direction = -1 if dp > 0 else 1
                notes[i].midi_pitch += 12 * direction

    return notes


def _enforce_monophonic(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Ensure no two notes overlap.  Later note truncates earlier."""
    if len(notes) < 2:
        return notes
    result = [notes[0]]
    for n in notes[1:]:
        prev = result[-1]
        if n.start_s < prev.end_s:
            prev.end_s = n.start_s
            if prev.duration_s < 0.01:
                result.pop()
        result.append(n)
    return result


def _enforce_range(
    notes: List[NoteEvent],
    inst_range: InstrumentRange,
    mode: str,
) -> List[NoteEvent]:
    """Transpose notes outside the instrument range into the nearest octave."""
    for n in notes:
        if mode == "transpose":
            while n.midi_pitch < inst_range.low:
                n.midi_pitch += 12
            while n.midi_pitch > inst_range.high:
                n.midi_pitch -= 12
        else:  # clip
            n.midi_pitch = max(inst_range.low, min(inst_range.high, n.midi_pitch))
    return notes
