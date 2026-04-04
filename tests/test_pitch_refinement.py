"""
Tests for pitch refinement — segment splitting, pitch re-estimation,
octave correction, interval smoothing, and neighbor consistency.

Run with:  pytest tests/test_pitch_refinement.py -v
"""

import pytest

from src.pipeline.pitch_refinement import (
    split_multi_pitch_segments,
    reestimate_pitches,
    correct_octave_errors,
    smooth_intervals,
    correct_neighbor_outliers,
    refine_pitches,
    PitchRefinementConfig,
    PitchRefinementStats,
    _find_overlapping,
    _find_pitch_clusters,
    _stability_weighted_pitch,
    _is_register_shift_supported,
)
from src.pipeline.utils import NoteEvent


# =========================================================================== #
# A — Segment splitting
# =========================================================================== #

class TestSegmentSplitting:
    """Merged segments with multiple pitches should be split."""

    def test_split_two_pitch_regions(self):
        """A merged note spanning C4 and E4 regions should split."""
        merged = [NoteEvent(0.0, 1.0, 62)]
        originals = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 64),
        ]
        result, count, added = split_multi_pitch_segments(
            merged, originals, min_range=3, min_dur_ms=100)
        assert count == 1
        assert len(result) == 2

    def test_no_split_when_range_small(self):
        """Notes within 2 semitones should not split."""
        merged = [NoteEvent(0.0, 1.0, 60)]
        originals = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 61),
        ]
        result, count, _ = split_multi_pitch_segments(
            merged, originals, min_range=3)
        assert count == 0
        assert len(result) == 1

    def test_no_split_without_originals(self):
        """Without originals, no splitting happens."""
        merged = [NoteEvent(0.0, 1.0, 60)]
        result, count, _ = split_multi_pitch_segments(
            merged, [], min_range=3)
        assert count == 0

    def test_finds_overlapping_notes(self):
        target = NoteEvent(0.5, 1.5, 60)
        candidates = [
            NoteEvent(0.0, 0.4, 58),
            NoteEvent(0.3, 0.8, 60),
            NoteEvent(0.8, 1.2, 62),
            NoteEvent(1.6, 2.0, 64),
        ]
        found = _find_overlapping(target, candidates)
        assert len(found) == 2

    def test_pitch_clusters_two_regions(self):
        notes = [
            NoteEvent(0.0, 0.3, 60),
            NoteEvent(0.3, 0.6, 60),
            NoteEvent(0.6, 0.9, 67),
            NoteEvent(0.9, 1.2, 67),
        ]
        clusters = _find_pitch_clusters(notes)
        assert len(clusters) == 2


# =========================================================================== #
# B — Pitch re-estimation
# =========================================================================== #

class TestPitchReestimation:
    """Re-estimated pitch should reflect stability, not just duration."""

    def test_sustained_pitch_wins(self):
        """A pitch held for one long run beats scattered short fragments."""
        originals = [
            NoteEvent(0.0, 0.05, 62),
            NoteEvent(0.05, 0.55, 60),
            NoteEvent(0.55, 0.6, 62),
            NoteEvent(0.6, 0.65, 62),
        ]
        pitch = _stability_weighted_pitch(originals)
        assert pitch == 60

    def test_single_note(self):
        originals = [NoteEvent(0.0, 1.0, 72)]
        pitch = _stability_weighted_pitch(originals)
        assert pitch == 72

    def test_reestimate_corrects_drift(self):
        notes = [NoteEvent(0.0, 1.0, 62)]
        originals = [
            NoteEvent(0.0, 0.8, 60),
            NoteEvent(0.8, 1.0, 62),
        ]
        result = reestimate_pitches(notes, originals)
        assert result[0].midi_pitch == 60


# =========================================================================== #
# C — Octave correction
# =========================================================================== #

class TestOctaveCorrection:
    """Notes jumping ±12 from both neighbors should be corrected."""

    def test_octave_spike_corrected(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 72),  # +12 spike
            NoteEvent(1.0, 1.5, 60),
        ]
        result, count = correct_octave_errors(notes)
        assert result[1].midi_pitch == 60
        assert count == 1

    def test_real_register_shift_preserved(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 72),
            NoteEvent(1.0, 1.5, 72),  # stays in new register
        ]
        result, count = correct_octave_errors(notes)
        assert result[1].midi_pitch == 72

    def test_octave_drop_corrected(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.5, 1.0, 60),  # -12 drop
            NoteEvent(1.0, 1.5, 72),
        ]
        result, count = correct_octave_errors(notes)
        assert result[1].midi_pitch == 72
        assert count == 1

    def test_no_correction_when_short(self):
        notes = [NoteEvent(0.0, 0.5, 60)]
        result, count = correct_octave_errors(notes)
        assert count == 0


# =========================================================================== #
# D — Interval smoothing
# =========================================================================== #

class TestIntervalSmoothing:
    """Implausible isolated jumps should be corrected."""

    def test_isolated_spike_corrected(self):
        """A single note jumping far from both neighbors is an outlier."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 80),  # +20 from both
            NoteEvent(1.0, 1.5, 60),
        ]
        result, count = smooth_intervals(notes, max_jump=7)
        assert abs(result[1].midi_pitch - 60) <= 3
        assert count == 1

    def test_real_register_shift_preserved(self):
        """If melody stays in new register, the jump is real."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 72),
            NoteEvent(1.0, 1.5, 74),
            NoteEvent(1.5, 2.0, 72),
        ]
        result, count = smooth_intervals(notes, max_jump=7)
        assert result[1].midi_pitch == 72
        assert count == 0

    def test_stepwise_motion_preserved(self):
        """Small intervals should never be corrected."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 62),
            NoteEvent(1.0, 1.5, 64),
        ]
        result, count = smooth_intervals(notes)
        assert count == 0


# =========================================================================== #
# E — Neighbor consistency
# =========================================================================== #

class TestNeighborConsistency:
    """Notes slightly off from both neighbors should be nudged."""

    def test_above_both_neighbors(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 62),  # +2 from both
            NoteEvent(1.0, 1.5, 60),
        ]
        result, count = correct_neighbor_outliers(notes, max_correction=2)
        assert result[1].midi_pitch == 60
        assert count == 1

    def test_below_both_neighbors(self):
        notes = [
            NoteEvent(0.0, 0.5, 65),
            NoteEvent(0.5, 1.0, 63),  # -2 from both
            NoteEvent(1.0, 1.5, 65),
        ]
        result, count = correct_neighbor_outliers(notes, max_correction=2)
        assert result[1].midi_pitch == 65
        assert count == 1

    def test_real_step_not_corrected(self):
        """A genuine stepwise movement should not be corrected."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 62),
            NoteEvent(1.0, 1.5, 64),
        ]
        result, count = correct_neighbor_outliers(notes)
        assert count == 0

    def test_large_deviation_not_corrected(self):
        """Deviations larger than max_correction are not touched."""
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 67),  # +7, too large
            NoteEvent(1.0, 1.5, 60),
        ]
        result, count = correct_neighbor_outliers(notes, max_correction=2)
        assert count == 0


# =========================================================================== #
# F — Full pipeline
# =========================================================================== #

class TestFullPitchRefinement:
    """End-to-end pitch refinement."""

    def test_stats_populated(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 72),  # octave spike
            NoteEvent(1.0, 1.5, 60),
        ]
        result, stats = refine_pitches(notes)
        assert stats.notes_before == 3
        assert stats.octave_corrections >= 1

    def test_no_change_on_clean_input(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 62),
            NoteEvent(1.0, 1.5, 64),
        ]
        result, stats = refine_pitches(notes)
        assert stats.octave_corrections == 0
        assert stats.interval_corrections == 0
        pitches = [n.midi_pitch for n in result]
        assert pitches == [60, 62, 64]

    def test_empty_input(self):
        result, stats = refine_pitches([])
        assert result == []
        assert stats.notes_final == 0

    def test_with_originals_splits(self):
        notes = [NoteEvent(0.0, 1.0, 62)]
        originals = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 65),
        ]
        result, stats = refine_pitches(notes, original_notes=originals)
        assert stats.segments_split >= 1


# =========================================================================== #
# G — Regression
# =========================================================================== #

class TestRegressions:
    """Pitch refinement must not re-introduce note explosion."""

    def test_note_count_does_not_explode(self):
        """Refinement should add only a few splits, not double the count."""
        notes = [NoteEvent(i * 0.5, i * 0.5 + 0.4, 60 + (i % 5))
                 for i in range(20)]
        result, stats = refine_pitches(notes)
        assert len(result) <= len(notes) + 5

    def test_stepwise_melody_unchanged(self):
        """A clean stepwise melody should survive refinement intact."""
        notes = [
            NoteEvent(0.0, 0.4, 60),
            NoteEvent(0.5, 0.9, 62),
            NoteEvent(1.0, 1.4, 64),
            NoteEvent(1.5, 1.9, 65),
            NoteEvent(2.0, 2.4, 67),
        ]
        result, _ = refine_pitches(notes)
        pitches = [n.midi_pitch for n in result]
        assert pitches == [60, 62, 64, 65, 67]
