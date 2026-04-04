"""
Melody scoring features — musically informed scoring for notes and transitions.

Provides intrinsic note scores (how melody-like a single note is) and
transition scores (how musically plausible is the step from one note to
the next in a melodic line).

All weights come from MelodyConfig so nothing is hard-coded here.

Design philosophy: think like a trained musician following the sung
melody by ear.  The vocal line is typically:
  - in the upper-mid register (above most accompaniment)
  - sustained, phrase-like, with mostly stepwise or small-leap motion
  - NOT buried in dense chordal texture
  - NOT jumping erratically between octaves
"""

import logging
import math
from typing import List, Optional

from ..config.melody_config import MelodyConfig, InstrumentRange
from .candidate_notes import CandidateNote
from .local_register import LocalRegisterContext

logger = logging.getLogger("mp3_to_sheet.melody_features")


def score_note_intrinsic(
    note: CandidateNote,
    config: MelodyConfig,
    inst_range: InstrumentRange,
    clutter_density: int = 0,
    median_pitch: float = 72.0,
    upper_quartile_pitch: float = 78.0,
    local_context: Optional[LocalRegisterContext] = None,
) -> float:
    """
    Compute how melody-like a single candidate note is, independent of
    its neighbours.

    Higher score = more likely to be part of the melody.
    """
    score = 0.0

    # 1. Confidence bonus — still matters, but reduced so it doesn't
    #    let accompaniment notes win by confidence alone
    score += note.confidence * config.confidence_weight

    # 2. Duration bonus — longer notes are strongly preferred as melody.
    #    Use a steeper curve so the advantage of 0.3s over 0.08s is decisive.
    dur_bonus = math.log1p(note.duration_s * 10.0) * 0.7
    dur_bonus = min(dur_bonus, 3.0)
    score += dur_bonus * config.duration_weight

    # 3. Global register bonus (coarse whole-song bias)
    register_bonus = _compute_register_bonus(
        note.pitch_midi, config, inst_range, median_pitch, upper_quartile_pitch,
    )
    score += register_bonus * config.register_weight

    # 4. Local register bonus (adaptive, time-windowed)
    if local_context is not None and config.local_register_mode != "off":
        score += _compute_local_register_bonus(
            note.pitch_midi, config, local_context,
        )

    # 5. Salience/energy bonus
    score += note.salience * config.salience_weight

    # 6. Instrument plausibility — notes in comfortable range score higher
    if inst_range.contains(note.pitch_midi):
        dist_to_center = abs(note.pitch_midi - inst_range.center) / 12.0
        plausibility = max(0.0, 1.0 - dist_to_center * 0.25)
    else:
        plausibility = -1.0
    score += plausibility * config.instrument_plausibility_weight

    # -- Penalties --

    # 7. Short note penalty — micro-notes are rarely melody
    if note.duration_s * 1000.0 < config.short_note_threshold_ms:
        fraction = note.duration_s * 1000.0 / config.short_note_threshold_ms
        penalty = (1.0 - fraction) ** 1.5
        score -= penalty * config.short_note_penalty_weight

    # 8. Dense clutter penalty — many simultaneous notes = accompaniment texture
    if clutter_density >= config.clutter_count_threshold:
        excess = clutter_density - config.clutter_count_threshold
        penalty = min(excess * 0.5, 4.0)
        score -= penalty * config.clutter_penalty_weight

    return score


def _compute_local_register_bonus(
    pitch_midi: int,
    config: MelodyConfig,
    ctx: LocalRegisterContext,
) -> float:
    """
    Score a note based on its position within the local pitch texture.

    Three components:
      a) Local register target bonus (like global register but time-local)
      b) Below-local-median penalty (likely accompaniment)
      c) Above-local-median bonus (likely melody)
      d) High-outlier penalty (absurdly high harmonics)
    """
    score = 0.0

    # a) Local register target bonus
    if config.local_register_mode == "adaptive_vocal":
        target = ctx.local_upper_quartile
        offset = (pitch_midi - target) / 12.0
        if config.vocals_mode:
            if -0.25 <= offset <= 0.5:
                local_bonus = 1.5 + offset * 0.4
            elif offset > 0.5:
                local_bonus = max(-0.3, 1.7 - (offset - 0.5) * 0.6)
            else:
                local_bonus = max(-3.0, 1.5 + (offset + 0.25) * 2.5)
        else:
            if -0.25 <= offset <= 0.5:
                local_bonus = 1.0 + offset * 0.4
            elif offset < -0.25:
                local_bonus = max(-2.0, 1.0 + (offset + 0.25) * 1.2)
            else:
                local_bonus = max(-0.3, 1.2 - (offset - 0.5) * 0.6)
    elif config.local_register_mode == "upper_quartile":
        target = ctx.local_upper_quartile
        offset = (pitch_midi - target) / 12.0
        local_bonus = max(-1.5, min(1.5, offset * 0.5 + 0.5))
    elif config.local_register_mode == "median_plus_offset":
        target = ctx.local_median + 7
        offset = (pitch_midi - target) / 12.0
        local_bonus = max(-1.5, min(1.5, 1.0 - abs(offset) * 0.8))
    else:
        local_bonus = 0.0

    score += local_bonus * config.local_register_weight

    # b) Below-local-median penalty
    if pitch_midi < ctx.local_median:
        below_semitones = ctx.local_median - pitch_midi
        penalty = min(below_semitones / 12.0, 2.0)
        score -= penalty * config.below_local_median_penalty_weight

    # c) Above-local-median bonus (only within plausible vocal range,
    #    not for extreme high harmonics)
    if pitch_midi > ctx.local_median and pitch_midi <= ctx.local_upper_quartile + 12:
        above_semitones = pitch_midi - ctx.local_median
        bonus = min(above_semitones / 12.0, 1.5)
        score += bonus * config.above_local_median_bonus_weight

    # d) High-outlier penalty — notes far above local upper quartile
    if pitch_midi > ctx.local_upper_quartile + 7:
        excess = (pitch_midi - ctx.local_upper_quartile - 7) / 12.0
        score -= min(excess, 2.5) * config.high_outlier_penalty_weight

    return score


def _compute_register_bonus(
    pitch_midi: int,
    config: MelodyConfig,
    inst_range: InstrumentRange,
    median_pitch: float,
    upper_quartile_pitch: float,
) -> float:
    """
    Compute register bonus for a note.  Different modes for different use cases.
    """
    if config.register_bias_mode == "vocal":
        target = upper_quartile_pitch
        offset = (pitch_midi - target) / 12.0
        if config.vocals_mode:
            # Aggressive curve for vocals: steep penalty below target so
            # lower-octave sub-harmonics cannot compete with real vocal notes.
            if -0.25 <= offset <= 0.5:
                return 1.5 + offset * 0.4
            elif offset > 0.5:
                return max(-0.3, 1.7 - (offset - 0.5) * 0.6)
            else:
                # Steeper falloff: each octave below target costs ~2.5 points
                return max(-3.0, 1.5 + (offset + 0.25) * 2.5)
        else:
            if -0.5 <= offset <= 1.0:
                return 1.0 + offset * 0.3
            elif offset < -0.5:
                return max(-1.5, 1.0 + (offset + 0.5) * 0.8)
            else:
                return max(-0.5, 1.3 - (offset - 1.0) * 0.5)

    elif config.register_bias_mode == "above_median":
        offset = (pitch_midi - median_pitch) / 12.0
        return max(-1.0, min(1.5, offset * 0.6))

    elif config.register_bias_mode == "center":
        dist = abs(pitch_midi - inst_range.center) / 12.0
        return max(0.0, 1.2 - dist * 0.4)

    return 0.0


def score_transition(
    prev_note: CandidateNote,
    next_note: CandidateNote,
    config: MelodyConfig,
) -> float:
    """
    Score the melodic plausibility of transitioning from prev_note to next_note.

    A good melody transition has temporal continuity, small intervals,
    smooth contour, and phrase coherence.  Large leaps, octave flips,
    and temporal gaps are penalised heavily.
    """
    score = 0.0

    gap_s = next_note.start_s - prev_note.end_s
    interval = abs(next_note.pitch_midi - prev_note.pitch_midi)

    # 1. Temporal continuity — reward notes that follow closely in time
    if gap_s <= 0.02:
        continuity = 1.2
    elif gap_s < 0.3:
        continuity = max(0.0, 1.2 - gap_s * 4.0)
    elif gap_s < 1.0:
        continuity = max(-0.5, -gap_s * 0.5)
    else:
        continuity = max(-1.0, -0.5 - (gap_s - 1.0) * 0.2)
    score += continuity * config.temporal_continuity_weight

    # 2. Leap penalty — heavily penalise large melodic intervals.
    #    A trained musician following the melody rarely hears the tune
    #    jumping more than a 5th without deliberate intent.
    if interval <= 2:
        leap_pen = 0.0
    elif interval <= config.leap_mild_threshold:
        leap_pen = (interval - 2) / max(1, config.leap_mild_threshold - 2) * 0.2
    elif interval <= config.leap_large_threshold:
        leap_pen = 0.2 + (interval - config.leap_mild_threshold) / \
                   max(1, config.leap_large_threshold - config.leap_mild_threshold) * 0.6
    elif interval <= config.leap_extreme_threshold:
        leap_pen = 0.8 + (interval - config.leap_large_threshold) / \
                   max(1, config.leap_extreme_threshold - config.leap_large_threshold) * 1.0
    else:
        leap_pen = 2.0
    score -= leap_pen * config.leap_penalty_weight

    # 3. Octave flip penalty — ±12 semitone jumps are almost always
    #    detection artefacts, not real melodic motion.  Apply broadly
    #    (gap < 0.3s) and very heavily.
    if interval in (12, 24) and gap_s < 0.3:
        score -= 2.0 * config.octave_flip_penalty_weight

    # 4. Contour smoothness — stepwise motion is the backbone of melody
    if interval <= 2:
        score += 0.8 * config.contour_smoothness_weight
    elif interval <= 4:
        score += 0.3 * config.contour_smoothness_weight
    elif interval <= 5:
        score += 0.1 * config.contour_smoothness_weight

    # 5. Phrase continuity — reward staying within a musical phrase
    if 0.0 <= gap_s <= 0.3 and interval <= config.leap_mild_threshold:
        score += 0.8 * config.phrase_continuity_weight
    elif 0.0 <= gap_s <= 0.5 and interval <= config.leap_large_threshold:
        score += 0.3 * config.phrase_continuity_weight

    # 6. Monophonic overlap penalty
    if gap_s < 0.0:
        overlap_s = min(-gap_s, prev_note.duration_s)
        score -= min(overlap_s * 5.0, 2.5) * config.monophonic_overlap_penalty

    # 7. Burst penalty — if both notes are very short, this is likely
    #    accompaniment texture, not melody.
    if (prev_note.duration_s * 1000 < config.burst_threshold_ms and
            next_note.duration_s * 1000 < config.burst_threshold_ms):
        score -= 0.5 * config.burst_penalty_weight

    # 8. Stream persistence — reward staying within the same pitch band
    #    and penalise switching to a distant register.
    if interval <= config.same_band_semitones:
        score += config.same_band_bonus
    else:
        excess = (interval - config.same_band_semitones) / 12.0
        score -= excess * config.stream_switch_penalty_weight

    return score


def score_gap_entry(
    prev_note: Optional[CandidateNote],
    gap_start_s: float,
    config: MelodyConfig,
) -> float:
    """Cost of entering a gap/rest state from a note."""
    return -config.gap_entry_cost


def score_gap_continuation(
    gap_duration_s: float,
    config: MelodyConfig,
) -> float:
    """Cost of staying in gap/rest for continued duration."""
    if gap_duration_s > config.gap_max_duration_s:
        return -5.0
    return -config.gap_continuation_cost * gap_duration_s


def compute_median_pitch(candidates: List[CandidateNote]) -> float:
    """Weighted median pitch across all candidates (weighted by duration)."""
    if not candidates:
        return 72.0
    weighted = sorted(candidates, key=lambda n: n.pitch_midi)
    total_dur = sum(n.duration_s for n in weighted)
    if total_dur <= 0:
        return 72.0
    cumulative = 0.0
    for n in weighted:
        cumulative += n.duration_s
        if cumulative >= total_dur / 2.0:
            return float(n.pitch_midi)
    return float(weighted[-1].pitch_midi)


def compute_upper_quartile_pitch(candidates: List[CandidateNote]) -> float:
    """
    Weighted 75th-percentile pitch.  This approximates where the vocal
    melody sits — above most accompaniment but not at the absolute top.
    """
    if not candidates:
        return 78.0
    weighted = sorted(candidates, key=lambda n: n.pitch_midi)
    total_dur = sum(n.duration_s for n in weighted)
    if total_dur <= 0:
        return 78.0
    cumulative = 0.0
    for n in weighted:
        cumulative += n.duration_s
        if cumulative >= total_dur * 0.75:
            return float(n.pitch_midi)
    return float(weighted[-1].pitch_midi)
