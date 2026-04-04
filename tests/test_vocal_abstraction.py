"""
Tests for vocal abstraction — vibrato suppression, phrase consolidation,
representative pitch selection, and micro-rest absorption.

Covers:
  A. Vibrato suppression
  B. Phrase consolidation
  C. Representative pitch selection
  D. Micro-rest absorption
  E. Full abstraction pipeline
  F. Edge cases
  G. Regression — contour preservation

Run with:  pytest tests/test_vocal_abstraction.py -v
"""

import pytest

from src.pipeline.vocal_abstraction import (
    suppress_vibrato,
    consolidate_sustained,
    consolidate_phrases,
    absorb_micro_rests,
    abstract_vocal_notes,
    VocalAbstractionConfig,
    AbstractionStats,
    _merge_with_robust_pitch,
    _weighted_median,
    _is_oscillating,
    _has_directional_contour,
)
from src.pipeline.utils import NoteEvent


# =========================================================================== #
# A — Vibrato suppression
# =========================================================================== #

class TestVibratoSuppression:
    """Oscillating pitch clusters should collapse to one note."""

    def test_classic_vibrato_pattern(self):
        """C-D-C-D-C (oscillation ±1) -> 1 note."""
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.1, 0.2, 73),
            NoteEvent(0.2, 0.3, 72),
            NoteEvent(0.3, 0.4, 73),
            NoteEvent(0.4, 0.5, 72),
        ]
        result, count = suppress_vibrato(notes, max_span=2, min_notes=3)
        assert len(result) == 1
        assert count == 1
        assert result[0].start_s == pytest.approx(0.0)
        assert result[0].end_s == pytest.approx(0.5)

    def test_wide_vibrato_within_span(self):
        """72-74-72-74-72 (±2 semitones) -> 1 note with span=2."""
        notes = [
            NoteEvent(0.0, 0.08, 72),
            NoteEvent(0.08, 0.16, 74),
            NoteEvent(0.16, 0.24, 72),
            NoteEvent(0.24, 0.32, 74),
            NoteEvent(0.32, 0.4, 72),
        ]
        result, count = suppress_vibrato(notes, max_span=2, min_notes=3)
        assert len(result) == 1

    def test_ascending_scale_not_vibrato(self):
        """C-D-E-F-G (monotonic ascent) should NOT be collapsed."""
        notes = [
            NoteEvent(0.0, 0.1, 60),
            NoteEvent(0.1, 0.2, 62),
            NoteEvent(0.2, 0.3, 64),
            NoteEvent(0.3, 0.4, 65),
            NoteEvent(0.4, 0.5, 67),
        ]
        result, count = suppress_vibrato(notes, max_span=7, min_notes=3)
        assert len(result) == 5
        assert count == 0

    def test_large_interval_not_vibrato(self):
        """Notes spanning >max_span should not merge."""
        notes = [
            NoteEvent(0.0, 0.1, 60),
            NoteEvent(0.1, 0.2, 64),
            NoteEvent(0.2, 0.3, 60),
        ]
        result, count = suppress_vibrato(notes, max_span=2, min_notes=3)
        assert len(result) == 3

    def test_gap_too_large_breaks_cluster(self):
        """Large gap splits what would be one vibrato cluster into two
        sub-clusters, each too small to qualify on its own."""
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.1, 0.2, 73),
            NoteEvent(0.5, 0.6, 72),  # 300ms gap — splits cluster
            NoteEvent(0.6, 0.7, 73),
        ]
        result, count = suppress_vibrato(
            notes, max_span=2, min_notes=3, max_gap_ms=150)
        assert len(result) == 4
        assert count == 0

    def test_too_few_notes(self):
        """Fewer than min_notes should not merge."""
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.1, 0.2, 73),
        ]
        result, count = suppress_vibrato(notes, max_span=2, min_notes=3)
        assert len(result) == 2
        assert count == 0

    def test_empty_input(self):
        result, count = suppress_vibrato([], max_span=2, min_notes=3)
        assert result == []
        assert count == 0


class TestIsOscillating:
    def test_oscillating(self):
        notes = [NoteEvent(0, 0.1, p) for p in [60, 61, 60, 61, 60]]
        assert _is_oscillating(notes) is True

    def test_monotonic_ascending(self):
        notes = [NoteEvent(0, 0.1, p) for p in [60, 61, 62, 63, 64]]
        assert _is_oscillating(notes) is False

    def test_flat(self):
        notes = [NoteEvent(0, 0.1, p) for p in [60, 60, 60]]
        assert _is_oscillating(notes) is True

    def test_two_notes(self):
        notes = [NoteEvent(0, 0.1, 60), NoteEvent(0.1, 0.2, 61)]
        assert _is_oscillating(notes) is False


# =========================================================================== #
# A2 — Sustained-pitch consolidation
# =========================================================================== #

class TestSustainedConsolidation:
    """Same-pitch fragments should merge regardless of individual duration."""

    def test_same_pitch_fragments_merge(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.52, 1.0, 72),
            NoteEvent(1.02, 1.5, 72),
        ]
        result, count = consolidate_sustained(notes, max_gap_ms=300, pitch_tolerance=1)
        assert len(result) == 1
        assert count == 1

    def test_stepwise_ascent_preserved(self):
        notes = [
            NoteEvent(0.0, 0.4, 60),
            NoteEvent(0.42, 0.82, 61),
            NoteEvent(0.84, 1.24, 62),
        ]
        result, count = consolidate_sustained(notes, max_gap_ms=300, pitch_tolerance=1)
        assert len(result) == 3

    def test_large_pitch_jump_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.52, 1.0, 65),
        ]
        result, count = consolidate_sustained(notes, max_gap_ms=300, pitch_tolerance=1)
        assert len(result) == 2

    def test_large_gap_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(1.0, 1.5, 72),
        ]
        result, count = consolidate_sustained(notes, max_gap_ms=300, pitch_tolerance=1)
        assert len(result) == 2


class TestDirectionalContour:
    """Contour detection must distinguish motion from jitter."""

    def test_ascending_pair_is_contour(self):
        notes = [NoteEvent(0, 0.3, 60), NoteEvent(0.3, 0.6, 62)]
        assert _has_directional_contour(notes, pitch_tolerance=2) is True

    def test_same_pitch_is_not_contour(self):
        notes = [NoteEvent(0, 0.3, 60), NoteEvent(0.3, 0.6, 60)]
        assert _has_directional_contour(notes, pitch_tolerance=2) is False

    def test_small_oscillation_not_contour(self):
        notes = [NoteEvent(0, 0.1, 60), NoteEvent(0.1, 0.2, 61),
                 NoteEvent(0.2, 0.3, 60)]
        assert _has_directional_contour(notes, pitch_tolerance=2) is False

    def test_clear_descent_is_contour(self):
        notes = [NoteEvent(0, 0.2, 67), NoteEvent(0.2, 0.4, 65),
                 NoteEvent(0.4, 0.6, 64)]
        assert _has_directional_contour(notes, pitch_tolerance=2) is True


# =========================================================================== #
# B — Phrase consolidation
# =========================================================================== #

class TestPhraseConsolidation:
    """Short notes belonging to one sung syllable should merge."""

    def test_short_notes_at_same_pitch_merge(self):
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.12, 0.22, 72),
            NoteEvent(0.24, 0.34, 72),
        ]
        result, count = consolidate_phrases(
            notes, max_gap_ms=200, pitch_tolerance=2, max_short_note_ms=250)
        assert len(result) == 1
        assert count == 1

    def test_short_notes_nearby_pitch_merge(self):
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.12, 0.22, 73),
            NoteEvent(0.24, 0.34, 72),
        ]
        result, count = consolidate_phrases(
            notes, max_gap_ms=200, pitch_tolerance=2, max_short_note_ms=250)
        assert len(result) == 1

    def test_long_notes_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.52, 1.0, 73),
        ]
        result, count = consolidate_phrases(
            notes, max_gap_ms=200, pitch_tolerance=2, max_short_note_ms=250)
        assert len(result) == 2
        assert count == 0

    def test_large_pitch_jump_breaks_phrase(self):
        notes = [
            NoteEvent(0.0, 0.1, 60),
            NoteEvent(0.12, 0.22, 60),
            NoteEvent(0.24, 0.34, 70),  # +10 semitones
        ]
        result, count = consolidate_phrases(
            notes, max_gap_ms=200, pitch_tolerance=2, max_short_note_ms=250)
        assert len(result) >= 2

    def test_large_gap_breaks_phrase(self):
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.5, 0.6, 72),  # 400ms gap
        ]
        result, count = consolidate_phrases(
            notes, max_gap_ms=200, pitch_tolerance=2, max_short_note_ms=250)
        assert len(result) == 2

    def test_empty_input(self):
        result, count = consolidate_phrases([])
        assert result == []


# =========================================================================== #
# C — Representative pitch selection
# =========================================================================== #

class TestRepresentativePitch:
    """Merged pitch should reflect the pitch center, not outliers."""

    def test_majority_pitch_wins(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),  # 500ms at 72
            NoteEvent(0.5, 0.55, 74), # 50ms at 74 (outlier)
            NoteEvent(0.55, 1.0, 72), # 450ms at 72
        ]
        merged = _merge_with_robust_pitch(notes)
        assert merged.midi_pitch == 72

    def test_trimmed_median_ignores_blips(self):
        notes = [
            NoteEvent(0.0, 0.02, 60),  # 20ms blip (trimmed)
            NoteEvent(0.02, 0.5, 72),
            NoteEvent(0.5, 1.0, 72),
            NoteEvent(1.0, 1.02, 80),  # 20ms blip (trimmed)
        ]
        merged = _merge_with_robust_pitch(notes)
        assert merged.midi_pitch == 72

    def test_single_note_unchanged(self):
        notes = [NoteEvent(0.0, 1.0, 65)]
        merged = _merge_with_robust_pitch(notes)
        assert merged.midi_pitch == 65

    def test_weighted_median_basic(self):
        assert _weighted_median({60: 1.0, 72: 5.0, 80: 1.0}) == 72

    def test_weighted_median_single(self):
        assert _weighted_median({65: 3.0}) == 65

    def test_timing_preserved(self):
        notes = [
            NoteEvent(1.0, 1.5, 72),
            NoteEvent(1.5, 2.0, 73),
        ]
        merged = _merge_with_robust_pitch(notes)
        assert merged.start_s == pytest.approx(1.0)
        assert merged.end_s == pytest.approx(2.0)


# =========================================================================== #
# D — Micro-rest absorption
# =========================================================================== #

class TestMicroRestAbsorption:
    """Tiny gaps between phrase notes should be filled."""

    def test_small_gap_absorbed(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.6, 1.0, 73),  # 100ms gap
        ]
        result, count = absorb_micro_rests(notes, max_gap_ms=150)
        assert result[0].end_s == pytest.approx(0.6)
        assert count == 1

    def test_large_gap_preserved(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(1.0, 1.5, 73),  # 500ms gap
        ]
        result, count = absorb_micro_rests(notes, max_gap_ms=150)
        assert result[0].end_s == pytest.approx(0.5)
        assert count == 0

    def test_large_pitch_gap_preserved(self):
        """Gap between notes with large pitch difference is kept."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.6, 1.0, 72),  # +12 semitones
        ]
        result, count = absorb_micro_rests(notes, max_gap_ms=150)
        assert count == 0

    def test_empty_input(self):
        result, count = absorb_micro_rests([])
        assert result == []
        assert count == 0


# =========================================================================== #
# E — Full abstraction pipeline
# =========================================================================== #

class TestFullAbstraction:
    """End-to-end vocal abstraction."""

    def _make_vibrato_vocal(self):
        """Simulate a vocal with vibrato and phrase fragmentation."""
        vibrato_phrase = [
            NoteEvent(0.0, 0.08, 72),
            NoteEvent(0.08, 0.16, 73),
            NoteEvent(0.16, 0.24, 72),
            NoteEvent(0.24, 0.32, 73),
            NoteEvent(0.32, 0.4, 72),
        ]
        short_fragments = [
            NoteEvent(0.5, 0.6, 67),
            NoteEvent(0.62, 0.72, 67),
            NoteEvent(0.74, 0.84, 68),
            NoteEvent(0.86, 0.96, 67),
        ]
        held_note = [NoteEvent(1.1, 2.0, 65)]
        return vibrato_phrase + short_fragments + held_note

    def test_reduces_note_count(self):
        notes = self._make_vibrato_vocal()
        result, stats = abstract_vocal_notes(notes)
        assert stats.notes_final < stats.notes_before
        assert stats.notes_final <= 5

    def test_stats_populated(self):
        notes = self._make_vibrato_vocal()
        _, stats = abstract_vocal_notes(notes)
        assert stats.notes_before == 10
        assert stats.notes_after_vibrato <= 10
        assert stats.avg_duration_before > 0
        assert stats.avg_duration_after >= stats.avg_duration_before

    def test_preserves_held_note(self):
        notes = self._make_vibrato_vocal()
        result, _ = abstract_vocal_notes(notes)
        pitches = [n.midi_pitch for n in result]
        assert 65 in pitches

    def test_config_override(self):
        notes = self._make_vibrato_vocal()
        cfg = VocalAbstractionConfig(
            vibrato_merge_enabled=False,
            phrase_merge_enabled=False,
        )
        result, stats = abstract_vocal_notes(notes, cfg)
        assert stats.vibrato_groups_merged == 0
        assert stats.phrase_groups_merged == 0

    def test_empty_input(self):
        result, stats = abstract_vocal_notes([])
        assert result == []
        assert stats.notes_final == 0


# =========================================================================== #
# F — Regression: real melodic motion preserved
# =========================================================================== #

class TestContourRegression:
    """Real melodic steps must survive abstraction."""

    def test_stepwise_melody_preserved(self):
        """C-D-E-F-G as quarter notes should not be collapsed."""
        notes = [
            NoteEvent(0.0, 0.4, 60),
            NoteEvent(0.42, 0.82, 62),
            NoteEvent(0.84, 1.24, 64),
            NoteEvent(1.26, 1.66, 65),
            NoteEvent(1.68, 2.08, 67),
        ]
        cfg = VocalAbstractionConfig()
        result, _ = abstract_vocal_notes(notes, cfg)
        assert len(result) >= 4
        pitches = [n.midi_pitch for n in result]
        assert len(set(pitches)) >= 4

    def test_large_intervals_preserved(self):
        """Notes with large pitch jumps should not merge."""
        notes = [
            NoteEvent(0.0, 0.3, 60),
            NoteEvent(0.32, 0.62, 72),
            NoteEvent(0.64, 0.94, 60),
        ]
        result, _ = abstract_vocal_notes(notes)
        assert len(result) == 3
