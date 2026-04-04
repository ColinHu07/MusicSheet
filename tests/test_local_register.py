"""
Tests for the local register modeling, stream persistence, relative
register scoring, pitch-floor heuristic, and regression scenarios that
reproduce the observed failure mode (selecting low accompaniment instead
of the vocal melody).

Run with:  pytest tests/test_local_register.py -v
"""

import pytest

from src.config.melody_config import MelodyConfig, InstrumentRange, INSTRUMENT_RANGES
from src.pipeline.candidate_notes import CandidateNote
from src.pipeline.local_register import (
    LocalRegisterContext,
    compute_local_register_contexts,
)
from src.pipeline.melody_features import (
    score_note_intrinsic,
    score_transition,
    _compute_local_register_bonus,
    compute_median_pitch,
    compute_upper_quartile_pitch,
)
from src.pipeline.melody_path import resolve_melody
from src.pipeline.utils import NoteEvent


# =========================================================================== #
# A — Local register modeling
# =========================================================================== #

class TestLocalRegisterContextComputation:
    """Verify that local register contexts are computed correctly in
    sliding windows and adapt to register shifts over time."""

    def test_single_note_context(self):
        candidates = [CandidateNote(0.0, 0.5, 72, confidence=0.7)]
        contexts = compute_local_register_contexts(candidates, window_s=2.0)
        assert len(contexts) == 1
        assert contexts[0].local_median == 72.0
        assert contexts[0].notes_in_window == 1

    def test_window_captures_neighbors(self):
        candidates = [
            CandidateNote(0.0, 0.3, 48),
            CandidateNote(0.0, 0.3, 60),
            CandidateNote(0.0, 0.3, 72),
        ]
        contexts = compute_local_register_contexts(candidates, window_s=2.0)
        assert all(c.notes_in_window == 3 for c in contexts)

    def test_distant_notes_excluded_from_window(self):
        candidates = [
            CandidateNote(0.0, 0.3, 72),
            CandidateNote(10.0, 10.3, 48),
        ]
        contexts = compute_local_register_contexts(candidates, window_s=2.0)
        assert contexts[0].notes_in_window == 1
        assert contexts[1].notes_in_window == 1

    def test_local_context_adapts_to_register_shift(self):
        """Simulate a verse (low) → chorus (high) register shift.
        Local contexts should reflect the shift."""
        verse = [CandidateNote(i * 0.5, i * 0.5 + 0.4, 60 + (i % 3))
                 for i in range(8)]
        chorus = [CandidateNote(5.0 + i * 0.5, 5.0 + i * 0.5 + 0.4, 72 + (i % 3))
                  for i in range(8)]
        all_notes = verse + chorus
        all_notes.sort(key=lambda n: n.start_s)
        contexts = compute_local_register_contexts(all_notes, window_s=2.0)
        verse_medians = [contexts[i].local_median for i in range(4)]
        chorus_medians = [contexts[i].local_median for i in range(12, 16)]
        avg_verse_median = sum(verse_medians) / len(verse_medians)
        avg_chorus_median = sum(chorus_medians) / len(chorus_medians)
        assert avg_chorus_median > avg_verse_median + 5, \
            f"Chorus local median ({avg_chorus_median:.1f}) should be " \
            f"clearly above verse ({avg_verse_median:.1f})"

    def test_relative_position_correct(self):
        candidates = [
            CandidateNote(0.0, 0.5, 48),
            CandidateNote(0.0, 0.5, 60),
            CandidateNote(0.0, 0.5, 72),
        ]
        contexts = compute_local_register_contexts(candidates, window_s=2.0)
        assert contexts[0].relative_position < 0.1   # lowest
        assert 0.4 < contexts[1].relative_position < 0.6  # middle
        assert contexts[2].relative_position > 0.9   # highest

    def test_upper_quartile_above_median(self):
        candidates = [
            CandidateNote(0.0, 1.0, 48),
            CandidateNote(0.0, 1.0, 55),
            CandidateNote(0.0, 1.0, 60),
            CandidateNote(0.0, 1.0, 72),
        ]
        contexts = compute_local_register_contexts(candidates, window_s=2.0)
        for ctx in contexts:
            assert ctx.local_upper_quartile >= ctx.local_median

    def test_empty_candidates(self):
        contexts = compute_local_register_contexts([], window_s=2.0)
        assert contexts == []


class TestLocalRegisterScoring:
    """Verify that local register scoring correctly favours notes in the
    local melody band over accompaniment."""

    def _make_local_ctx(self, **overrides):
        defaults = dict(
            local_median=55.0, local_upper_quartile=65.0,
            local_min=40.0, local_max=80.0,
            relative_position=0.5, notes_in_window=10,
        )
        defaults.update(overrides)
        return LocalRegisterContext(**defaults)

    def test_note_at_local_target_scores_high(self):
        cfg = MelodyConfig()
        ctx = self._make_local_ctx(local_upper_quartile=68.0)
        vocal = CandidateNote(0.0, 0.5, 68, confidence=0.7)
        bonus = _compute_local_register_bonus(vocal.pitch_midi, cfg, ctx)
        assert bonus > 2.0, f"Note at local target should score high, got {bonus:.2f}"

    def test_note_below_local_median_penalized(self):
        cfg = MelodyConfig()
        ctx = self._make_local_ctx(local_median=60.0, local_upper_quartile=68.0)
        bass = CandidateNote(0.0, 0.5, 48, confidence=0.7)
        bonus = _compute_local_register_bonus(bass.pitch_midi, cfg, ctx)
        assert bonus < -2.0, \
            f"Note far below local median should be penalized, got {bonus:.2f}"

    def test_local_vocal_scores_higher_than_accompaniment(self):
        """In a local window, a melody-like note should score decisively
        higher than a low accompaniment note, even with equal conf/dur."""
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        ctx = self._make_local_ctx(local_median=55.0, local_upper_quartile=65.0)
        vocal = CandidateNote(0.0, 0.5, 68, confidence=0.6)
        accomp = CandidateNote(0.0, 0.5, 48, confidence=0.6)
        s_vocal = score_note_intrinsic(
            vocal, cfg, inst, 0, 55.0, 65.0, local_context=ctx)
        s_accomp = score_note_intrinsic(
            accomp, cfg, inst, 0, 55.0, 65.0, local_context=ctx)
        diff = s_vocal - s_accomp
        assert diff > 8.0, \
            f"Vocal note should outscore accompaniment by >8, got {diff:.2f}"

    def test_global_accompaniment_does_not_suppress_local_vocal(self):
        """Even when global stats are dominated by low accompaniment
        (median=46, uq=58), local register scoring should still prefer
        vocal-range notes."""
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        ctx = self._make_local_ctx(local_median=58.0, local_upper_quartile=68.0)
        vocal = CandidateNote(0.0, 0.5, 70, confidence=0.6)
        accomp = CandidateNote(0.0, 0.5, 46, confidence=0.7)
        s_vocal = score_note_intrinsic(
            vocal, cfg, inst, 0, 46.0, 58.0, local_context=ctx)
        s_accomp = score_note_intrinsic(
            accomp, cfg, inst, 0, 46.0, 58.0, local_context=ctx)
        assert s_vocal > s_accomp, \
            f"Vocal ({s_vocal:.2f}) should outscore accompaniment ({s_accomp:.2f}) " \
            f"even with globally skewed stats"

    def test_high_outlier_penalized(self):
        cfg = MelodyConfig()
        ctx = self._make_local_ctx(local_upper_quartile=68.0)
        outlier = CandidateNote(0.0, 0.5, 95, confidence=0.3)
        bonus = _compute_local_register_bonus(outlier.pitch_midi, cfg, ctx)
        assert bonus < 0, "Extreme high outlier should be penalized"

    def test_local_register_off_has_no_effect(self):
        cfg = MelodyConfig(local_register_mode="off")
        inst = cfg.get_instrument_range()
        ctx = self._make_local_ctx()
        note = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        s_with = score_note_intrinsic(
            note, cfg, inst, 0, 60.0, 70.0, local_context=ctx)
        s_without = score_note_intrinsic(
            note, cfg, inst, 0, 60.0, 70.0, local_context=None)
        assert s_with == pytest.approx(s_without), \
            "local_register_mode='off' should make local_context irrelevant"


# =========================================================================== #
# B — Stream persistence
# =========================================================================== #

class TestStreamPersistence:
    """Verify that the transition scoring rewards staying in the same
    melodic stream and penalises abrupt register switches."""

    def test_same_band_continuation_rewarded(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        near = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        far = CandidateNote(0.5, 1.0, 60, confidence=0.7)
        s_near = score_transition(prev, near, cfg)
        s_far = score_transition(prev, far, cfg)
        assert s_near > s_far, \
            f"Near-band continuation ({s_near:.2f}) should beat " \
            f"register switch ({s_far:.2f})"

    def test_same_band_bonus_applied(self):
        """Two notes within same_band_semitones should receive extra bonus
        compared to notes just outside the band."""
        cfg = MelodyConfig(same_band_semitones=5, same_band_bonus=1.2)
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        within = CandidateNote(0.5, 1.0, 76, confidence=0.7)  # 4 st
        outside = CandidateNote(0.5, 1.0, 79, confidence=0.7)  # 7 st
        s_within = score_transition(prev, within, cfg)
        s_outside = score_transition(prev, outside, cfg)
        assert s_within > s_outside

    def test_stream_switch_penalty_proportional(self):
        """Larger register jumps should receive proportionally larger
        stream switch penalties."""
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        jump_8 = CandidateNote(0.5, 1.0, 64, confidence=0.7)   # 8 st
        jump_15 = CandidateNote(0.5, 1.0, 57, confidence=0.7)  # 15 st
        s_8 = score_transition(prev, jump_8, cfg)
        s_15 = score_transition(prev, jump_15, cfg)
        assert s_8 > s_15, \
            "Smaller register jump should score higher"

    def test_brief_accompaniment_does_not_steal_path(self):
        """A continuous vocal line should not be interrupted by a brief
        low accompaniment note. The DP should prefer staying on the
        vocal even if a single bass note has decent confidence."""
        vocal = [
            CandidateNote(0.0, 0.6, 72, confidence=0.65),
            CandidateNote(0.6, 1.2, 74, confidence=0.6),
            CandidateNote(1.2, 1.8, 72, confidence=0.65),
            CandidateNote(1.8, 2.4, 76, confidence=0.6),
        ]
        brief_bass = [
            CandidateNote(0.6, 0.9, 48, confidence=0.75),
        ]
        all_notes = vocal + brief_bass
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        low_count = sum(1 for p in pitches if p < 55)
        assert low_count == 0, \
            f"Should not select any bass notes, got pitches: {pitches}"


# =========================================================================== #
# C — Relative register scoring
# =========================================================================== #

class TestRelativeRegisterScoring:
    """Verify that notes are scored relative to local accompaniment texture."""

    def test_below_local_median_penalized(self):
        cfg = MelodyConfig()
        ctx = LocalRegisterContext(
            local_median=60.0, local_upper_quartile=68.0,
            local_min=40.0, local_max=80.0,
            relative_position=0.3, notes_in_window=8)
        bonus = _compute_local_register_bonus(48, cfg, ctx)
        assert bonus < -3.0, \
            f"Note 12 semitones below local median should be heavily penalized, got {bonus:.2f}"

    def test_above_local_median_rewarded(self):
        cfg = MelodyConfig()
        ctx = LocalRegisterContext(
            local_median=55.0, local_upper_quartile=65.0,
            local_min=40.0, local_max=80.0,
            relative_position=0.7, notes_in_window=8)
        bonus = _compute_local_register_bonus(68, cfg, ctx)
        assert bonus > 3.0, \
            f"Note above local melody band should score high, got {bonus:.2f}"

    def test_moderate_above_preferred_over_extreme_low(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        ctx = LocalRegisterContext(
            local_median=55.0, local_upper_quartile=65.0,
            local_min=40.0, local_max=80.0,
            relative_position=0.5, notes_in_window=10)
        mid_high = CandidateNote(0.0, 0.5, 68, confidence=0.5)
        very_low = CandidateNote(0.0, 0.5, 42, confidence=0.7)
        s_high = score_note_intrinsic(
            mid_high, cfg, inst, 0, 55.0, 65.0, local_context=ctx)
        ctx_low = LocalRegisterContext(
            local_median=55.0, local_upper_quartile=65.0,
            local_min=40.0, local_max=80.0,
            relative_position=0.05, notes_in_window=10)
        s_low = score_note_intrinsic(
            very_low, cfg, inst, 0, 55.0, 65.0, local_context=ctx_low)
        assert s_high > s_low, \
            f"Mid-high note ({s_high:.2f}) should outscore " \
            f"very low note ({s_low:.2f}) despite lower confidence"


# =========================================================================== #
# D — Pitch floor heuristic
# =========================================================================== #

class TestPitchFloor:
    """Verify that the optional pitch-floor filter works correctly."""

    def test_pitch_floor_filters_low_candidates(self):
        candidates = [
            CandidateNote(0.0, 0.5, 48, confidence=0.8),
            CandidateNote(0.0, 0.5, 55, confidence=0.6),
            CandidateNote(0.0, 0.5, 60, confidence=0.7),
            CandidateNote(0.5, 1.0, 65, confidence=0.6),
        ]
        cfg = MelodyConfig(pitch_floor_midi=55)
        result = resolve_melody(candidates, cfg)
        assert result.stats["pitch_floor_filtered"] == 1
        pitches = [n.midi_pitch for n in result.selected_notes]
        assert all(p >= 55 for p in pitches), \
            f"All selected pitches should be >= 55, got {pitches}"

    def test_pitch_floor_disabled_backward_compatible(self):
        """With pitch_floor_midi=None, behaviour should be identical to
        the old pipeline (no candidates removed)."""
        candidates = [
            CandidateNote(0.0, 0.5, 48, confidence=0.9),
            CandidateNote(0.0, 0.5, 72, confidence=0.7),
            CandidateNote(0.5, 1.0, 74, confidence=0.7),
        ]
        cfg = MelodyConfig(pitch_floor_midi=None)
        result = resolve_melody(candidates, cfg)
        assert result.stats["pitch_floor_filtered"] == 0

    def test_pitch_floor_count_in_stats(self):
        candidates = [
            CandidateNote(0.0, 0.5, 30, confidence=0.8),
            CandidateNote(0.0, 0.5, 40, confidence=0.8),
            CandidateNote(0.0, 0.5, 50, confidence=0.8),
            CandidateNote(0.5, 1.0, 60, confidence=0.7),
            CandidateNote(0.5, 1.0, 70, confidence=0.7),
        ]
        cfg = MelodyConfig(pitch_floor_midi=50)
        result = resolve_melody(candidates, cfg)
        assert result.stats["pitch_floor_filtered"] == 2
        assert result.stats["candidates_after_floor"] == 3

    def test_pitch_floor_removes_all_candidates(self):
        candidates = [
            CandidateNote(0.0, 0.5, 30, confidence=0.8),
            CandidateNote(0.5, 1.0, 40, confidence=0.7),
        ]
        cfg = MelodyConfig(pitch_floor_midi=60)
        result = resolve_melody(candidates, cfg)
        assert result.selected_notes == []
        assert result.stats["pitch_floor_filtered"] == 2


# =========================================================================== #
# E — Regression scenarios (reproducing observed failure)
# =========================================================================== #

class TestRegressionLowAccompanimentBed:
    """Reproduce the observed failure: a low-register accompaniment bed
    dominates the candidate pool, causing the global upper quartile to
    sit in the accompaniment range.  The resolver must still select the
    higher vocal melody."""

    def _make_failure_scenario(self):
        """Construct a scenario mimicking the actual 'Raise Your Glass'
        failure: many low accompaniment notes (MIDI 40-55), fewer vocal
        melody notes (MIDI 65-78), global median ~46, global uq ~58."""
        accomp = []
        for beat in range(12):
            t = beat * 0.4
            accomp.extend([
                CandidateNote(t, t + 0.35, 44, confidence=0.55),
                CandidateNote(t, t + 0.35, 48, confidence=0.5),
                CandidateNote(t, t + 0.35, 52, confidence=0.45),
                CandidateNote(t, t + 0.2, 55, confidence=0.4),
            ])
        vocal = [
            CandidateNote(0.0, 0.7,  67, confidence=0.6),
            CandidateNote(0.7, 1.2,  69, confidence=0.55),
            CandidateNote(1.2, 1.9,  67, confidence=0.6),
            CandidateNote(1.9, 2.5,  65, confidence=0.55),
            CandidateNote(2.5, 3.2,  69, confidence=0.6),
            CandidateNote(3.2, 3.9,  72, confidence=0.55),
            CandidateNote(3.9, 4.8,  70, confidence=0.6),
        ]
        all_notes = accomp + vocal
        all_notes.sort(key=lambda n: (n.start_s, n.pitch_midi))
        return all_notes

    def test_selects_vocal_over_accompaniment(self):
        all_notes = self._make_failure_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        notes = result.selected_notes
        pitches = [n.midi_pitch for n in notes]
        vocal_found = sum(1 for p in pitches if p >= 63)
        # Only count accompaniment intrusions during the vocal-active period
        # (boundary notes after the vocal ends don't count as stream failures)
        accomp_during_vocal = sum(
            1 for n in notes
            if n.midi_pitch < 57 and n.start_s < 4.6
        )
        assert vocal_found >= 4, \
            f"Should select >= 4 vocal notes, got {vocal_found}: {pitches}"
        assert accomp_during_vocal == 0, \
            f"Should select 0 accompaniment notes during vocal, got " \
            f"{accomp_during_vocal}: {pitches}"

    def test_selected_median_pitch_in_vocal_range(self):
        all_notes = self._make_failure_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        sel_median = result.stats.get("selected_median_pitch", 0)
        assert sel_median >= 63, \
            f"Selected median pitch should be >= 63 (vocal range), got {sel_median}"

    def test_few_stream_switches(self):
        all_notes = self._make_failure_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        sw = result.stats.get("stream_switches", 999)
        assert sw <= 3, \
            f"Should have few stream switches, got {sw}"

    def test_low_pct_below_local_median(self):
        all_notes = self._make_failure_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pct = result.stats.get("pct_below_local_median", 100)
        assert pct <= 30, \
            f"At most 30% of selected notes should be below local median, got {pct}%"


class TestRegressionMelodyRegisterChanges:
    """Verify the resolver tracks melody across register changes (e.g.
    verse at MIDI 60-65, chorus at MIDI 70-76)."""

    def _make_register_change_scenario(self):
        verse_melody = [
            CandidateNote(0.0, 0.6, 62, confidence=0.65),
            CandidateNote(0.6, 1.2, 64, confidence=0.6),
            CandidateNote(1.2, 1.8, 62, confidence=0.65),
            CandidateNote(1.8, 2.4, 60, confidence=0.6),
        ]
        chorus_melody = [
            CandidateNote(3.0, 3.6, 72, confidence=0.7),
            CandidateNote(3.6, 4.2, 74, confidence=0.65),
            CandidateNote(4.2, 4.8, 72, confidence=0.7),
            CandidateNote(4.8, 5.4, 76, confidence=0.65),
        ]
        accompaniment = []
        for t_start in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]:
            accompaniment.extend([
                CandidateNote(t_start, t_start + 0.4, 44, confidence=0.5),
                CandidateNote(t_start, t_start + 0.4, 48, confidence=0.45),
            ])
        all_notes = verse_melody + chorus_melody + accompaniment
        all_notes.sort(key=lambda n: (n.start_s, n.pitch_midi))
        return all_notes

    def test_tracks_both_verse_and_chorus_register(self):
        all_notes = self._make_register_change_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        verse_pitches = [p for p in pitches if p < 66]
        chorus_pitches = [p for p in pitches if p >= 70]
        assert len(verse_pitches) >= 2, \
            f"Should track verse melody, got verse pitches: {verse_pitches}"
        assert len(chorus_pitches) >= 2, \
            f"Should track chorus melody, got chorus pitches: {chorus_pitches}"

    def test_no_accompaniment_selected(self):
        all_notes = self._make_register_change_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        accomp = [p for p in pitches if p < 50]
        assert len(accomp) == 0, \
            f"Should not select any accompaniment, got: {accomp}"


class TestRegressionAccompanimentDominatesGlobalDistribution:
    """When accompaniment notes vastly outnumber vocal notes, global
    pitch stats are poisoned.  Local register should still correctly
    identify the vocal."""

    def _make_dominated_scenario(self):
        accomp = []
        for i in range(40):
            t = i * 0.3
            accomp.append(CandidateNote(t, t + 0.25, 46 + (i % 5),
                                         confidence=0.5))
        vocal = [
            CandidateNote(0.0, 0.8, 70, confidence=0.55),
            CandidateNote(0.8, 1.6, 72, confidence=0.5),
            CandidateNote(1.6, 2.4, 70, confidence=0.55),
            CandidateNote(2.4, 3.2, 74, confidence=0.5),
            CandidateNote(3.2, 4.0, 72, confidence=0.55),
        ]
        all_notes = accomp + vocal
        all_notes.sort(key=lambda n: (n.start_s, n.pitch_midi))
        return all_notes, vocal

    def test_vocal_selected_despite_dominated_distribution(self):
        all_notes, vocal = self._make_dominated_scenario()
        global_median = compute_median_pitch(all_notes)
        assert global_median < 55, \
            f"Global median should be in accompaniment range, got {global_median}"
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        vocal_found = sum(1 for p in pitches if p >= 65)
        assert vocal_found >= 3, \
            f"Should select >= 3 vocal notes despite dominated distribution, " \
            f"got {vocal_found}: {pitches}"

    def test_selected_pitch_well_above_global_median(self):
        all_notes, _ = self._make_dominated_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        global_med = compute_median_pitch(all_notes)
        sel_median = result.stats.get("selected_median_pitch", 0)
        assert sel_median > global_med + 10, \
            f"Selected median ({sel_median}) should be well above " \
            f"global median ({global_med})"


# =========================================================================== #
# Stats expansion
# =========================================================================== #

class TestExpandedStats:
    """Verify that new debug stats are present in PathResult."""

    def test_new_stats_present(self):
        candidates = [
            CandidateNote(0.0, 0.5, 72, confidence=0.8),
            CandidateNote(0.5, 1.0, 74, confidence=0.7),
            CandidateNote(1.0, 1.5, 76, confidence=0.7),
        ]
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        for key in [
            "pitch_floor_filtered", "candidates_after_floor",
            "avg_selected_pitch", "selected_median_pitch",
            "selected_upper_quartile_pitch", "stream_switches",
            "pct_below_local_median", "pct_in_target_band",
        ]:
            assert key in result.stats, f"Missing stat: {key}"
