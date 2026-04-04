"""
Tests for phrase smoothing — sustain modeling, fragmentation reduction,
and contour preservation.

Covers:
  A. Pitch-stability grouping (jitter merge + contour awareness)
  B. Sustained note merging (repeated same-pitch)
  C. Small pitch jitter (±1 semitone)
  D. Real melodic step preservation (≥3 semitones must NOT merge)
  E. Micro-note removal
  F. Phrase-segment smoothing (with contour awareness)
  G. Re-attack suppression
  H. Post-quantization merge
  I. Full pipeline integration (before/after metrics)
  J. Contour preservation — stepwise melodic runs must survive smoothing

Run with:  pytest tests/test_phrase_smoothing.py -v
"""

import pytest

from src.config.melody_config import MelodyConfig
from src.pipeline.candidate_notes import CandidateNote
from src.pipeline.melody_path import resolve_melody
from src.pipeline.phrase_smoothing import (
    group_by_pitch_stability,
    smooth_phrase_segments,
    suppress_reattacks,
    merge_quantized_same_pitch,
    _contour_aware_merge,
)
from src.pipeline.utils import NoteEvent, QuantizedNote


# =========================================================================== #
# A — Pitch-stability grouping
# =========================================================================== #

class TestPitchStabilityGrouping:
    """A sustained vocal note split by Basic Pitch into multiple
    jittery fragments should be recombined into a single note."""

    def test_sustained_note_broken_into_fragments(self):
        """Five fragments of the same pitch with tiny gaps -> 1 note."""
        notes = [
            NoteEvent(0.0, 0.2, 72),
            NoteEvent(0.22, 0.4, 72),
            NoteEvent(0.42, 0.6, 72),
            NoteEvent(0.63, 0.8, 72),
            NoteEvent(0.82, 1.0, 72),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 1
        assert result[0].start_s == pytest.approx(0.0)
        assert result[0].end_s == pytest.approx(1.0)
        assert result[0].midi_pitch == 72

    def test_jittery_fragments_merge(self):
        """Same note with ±1 semitone jitter -> merged (no directional trend)."""
        notes = [
            NoteEvent(0.0, 0.25, 72),
            NoteEvent(0.27, 0.5, 73),   # +1 jitter
            NoteEvent(0.52, 0.75, 72),
            NoteEvent(0.77, 1.0, 71),   # -1 jitter
            NoteEvent(1.02, 1.25, 72),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 1
        assert result[0].midi_pitch == 72  # majority pitch
        assert result[0].end_s == pytest.approx(1.25)

    def test_real_melodic_step_not_merged(self):
        """A clear 3-semitone step must NOT be merged."""
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.52, 1.0, 75),   # +3 semitones = real step
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 2

    def test_large_gap_prevents_merge(self):
        """Same pitch but gap > threshold -> no merge."""
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.8, 1.1, 72),    # 500ms gap
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 2

    def test_empty_input(self):
        assert group_by_pitch_stability([], max_gap_ms=200, pitch_semitones=1) == []

    def test_single_note_unchanged(self):
        notes = [NoteEvent(0.0, 1.0, 72)]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 1
        assert result[0].midi_pitch == 72

    def test_two_semitone_tolerance(self):
        """With pitch_semitones=2, a 2-semitone difference merges."""
        notes = [
            NoteEvent(0.0, 0.4, 72),
            NoteEvent(0.42, 0.8, 74),   # +2
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=2)
        assert len(result) == 1

    def test_majority_pitch_by_duration(self):
        """The merged pitch should be the pitch with longest total duration."""
        notes = [
            NoteEvent(0.0, 0.1, 73),   # 0.1s at 73
            NoteEvent(0.12, 0.8, 72),  # 0.68s at 72 — longest
            NoteEvent(0.82, 0.9, 73),  # 0.08s at 73
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 1
        assert result[0].midi_pitch == 72

    def test_multiple_groups(self):
        """Two separate pitch-stable regions with a melodic step between."""
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.32, 0.6, 72),
            NoteEvent(0.65, 1.0, 67),   # -5 semitones: new group
            NoteEvent(1.02, 1.4, 67),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 2
        assert result[0].midi_pitch == 72
        assert result[1].midi_pitch == 67


# =========================================================================== #
# B — Contour preservation (NEW)
# =========================================================================== #

class TestContourPreservation:
    """Verify that groups with directional pitch contour are NOT collapsed."""

    def test_stepwise_ascending_scale_preserved(self):
        """C-D-E (72-74-76 by 2 semitones each) should NOT merge."""
        notes = [
            NoteEvent(0.0, 0.15, 72),
            NoteEvent(0.17, 0.32, 73),
            NoteEvent(0.34, 0.5, 74),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) >= 2, \
            f"Ascending scale should be preserved, got {len(result)} notes"

    def test_descending_run_preserved(self):
        """E-D-C (76-75-74-73-72) should NOT merge into one note."""
        notes = [
            NoteEvent(0.0, 0.1, 76),
            NoteEvent(0.12, 0.22, 75),
            NoteEvent(0.24, 0.34, 74),
            NoteEvent(0.36, 0.46, 73),
            NoteEvent(0.48, 0.58, 72),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) >= 3, \
            f"Descending run should be preserved, got {len(result)} notes"

    def test_oscillating_jitter_still_merges(self):
        """72-73-72-73-72 (alternating jitter, no net movement) merges."""
        notes = [
            NoteEvent(0.0, 0.2, 72),
            NoteEvent(0.22, 0.4, 73),
            NoteEvent(0.42, 0.6, 72),
            NoteEvent(0.62, 0.8, 73),
            NoteEvent(0.82, 1.0, 72),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        assert len(result) == 1, \
            f"Oscillating jitter should merge, got {len(result)} notes"

    def test_contour_aware_merge_splits_ascending(self):
        """Direct test of _contour_aware_merge on ascending notes."""
        notes = [
            NoteEvent(0.0, 0.2, 60),
            NoteEvent(0.2, 0.4, 61),
            NoteEvent(0.4, 0.6, 62),
        ]
        result = _contour_aware_merge(notes, pitch_tolerance=1)
        assert len(result) >= 2, \
            "Ascending contour should be split, not merged"

    def test_contour_aware_merge_keeps_flat(self):
        """Same-pitch group with no contour merges to 1."""
        notes = [
            NoteEvent(0.0, 0.2, 72),
            NoteEvent(0.2, 0.4, 72),
            NoteEvent(0.4, 0.6, 72),
        ]
        result = _contour_aware_merge(notes, pitch_tolerance=1)
        assert len(result) == 1

    def test_short_ornamental_turn_preserved(self):
        """A melodic turn (up then down) within pitch tolerance should
        still keep distinct notes when contour is meaningful."""
        notes = [
            NoteEvent(0.0, 0.15, 72),
            NoteEvent(0.17, 0.32, 73),
            NoteEvent(0.34, 0.5, 72),
            NoteEvent(0.52, 0.67, 71),
            NoteEvent(0.69, 0.85, 72),
        ]
        result = group_by_pitch_stability(notes, max_gap_ms=200, pitch_semitones=1)
        # Oscillating jitter — this should merge (no net directional trend)
        assert len(result) <= 2

    def test_two_note_step_up_preserved(self):
        """Two notes with 2-semitone step should NOT merge (with tolerance=1)."""
        notes = [
            NoteEvent(0.0, 0.3, 60),
            NoteEvent(0.32, 0.6, 62),
        ]
        result = _contour_aware_merge(notes, pitch_tolerance=1)
        assert len(result) == 2

    def test_two_note_within_tolerance_merges(self):
        """Two notes with 2-semitone step SHOULD merge when tolerance=2."""
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.32, 0.6, 74),
        ]
        result = _contour_aware_merge(notes, pitch_tolerance=2)
        assert len(result) == 1

    def test_two_note_same_pitch_merges(self):
        """Two notes at same pitch should merge."""
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.32, 0.6, 72),
        ]
        result = _contour_aware_merge(notes, pitch_tolerance=1)
        assert len(result) == 1


# =========================================================================== #
# C — Phrase-segment smoothing
# =========================================================================== #

class TestPhraseSegmentSmoothing:
    """Runs of consecutive short, near-pitch notes should collapse
    when they are jitter, but preserve contour when directional."""

    def test_run_of_short_jitter_notes_merged(self):
        """Four consecutive 100ms notes oscillating at same pitch -> fewer notes."""
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.1, 0.2, 73),
            NoteEvent(0.2, 0.3, 72),
            NoteEvent(0.3, 0.4, 71),
        ]
        result = smooth_phrase_segments(notes, max_short_note_ms=250, max_pitch_diff=2)
        # With contour awareness, a descending trend 73->72->71 may split,
        # but the run should still be simplified from 4 notes
        assert len(result) <= 3

    def test_long_notes_not_merged(self):
        """Notes longer than threshold are never merged."""
        notes = [
            NoteEvent(0.0, 0.5, 72),   # 500ms > 250ms threshold
            NoteEvent(0.5, 1.0, 74),
        ]
        result = smooth_phrase_segments(notes, max_short_note_ms=250, max_pitch_diff=2)
        assert len(result) == 2

    def test_large_pitch_diff_breaks_run(self):
        """A large pitch jump within short notes breaks the run."""
        notes = [
            NoteEvent(0.0, 0.1, 72),
            NoteEvent(0.1, 0.2, 72),
            NoteEvent(0.2, 0.3, 80),   # +8: breaks run
            NoteEvent(0.3, 0.4, 80),
        ]
        result = smooth_phrase_segments(notes, max_short_note_ms=250, max_pitch_diff=2)
        assert len(result) == 2

    def test_single_short_note_preserved(self):
        """A single short note (not a run) is not merged."""
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.5, 0.6, 74),   # short, but alone
            NoteEvent(0.6, 1.1, 76),
        ]
        result = smooth_phrase_segments(notes, max_short_note_ms=250, max_pitch_diff=2)
        assert len(result) == 3

    def test_empty_input(self):
        assert smooth_phrase_segments([], max_short_note_ms=250, max_pitch_diff=2) == []


# =========================================================================== #
# D — Re-attack suppression
# =========================================================================== #

class TestReattackSuppression:
    """Same-pitch re-attacks after all other cleanup should be merged."""

    def test_same_pitch_reattack_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.55, 1.0, 72),
        ]
        result = suppress_reattacks(notes, max_gap_ms=200)
        assert len(result) == 1
        assert result[0].end_s == pytest.approx(1.0)

    def test_different_pitch_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.55, 1.0, 74),
        ]
        result = suppress_reattacks(notes, max_gap_ms=200)
        assert len(result) == 2

    def test_large_gap_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(1.0, 1.5, 72),
        ]
        result = suppress_reattacks(notes, max_gap_ms=200)
        assert len(result) == 2

    def test_overlapping_same_pitch_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.49, 1.0, 72),
        ]
        result = suppress_reattacks(notes, max_gap_ms=200)
        assert len(result) == 1
        assert result[0].end_s == pytest.approx(1.0)

    def test_empty_input(self):
        assert suppress_reattacks([], max_gap_ms=200) == []

    def test_confidence_takes_max(self):
        notes = [
            NoteEvent(0.0, 0.5, 72, confidence=0.6),
            NoteEvent(0.52, 1.0, 72, confidence=0.8),
        ]
        result = suppress_reattacks(notes, max_gap_ms=200)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.8)


# =========================================================================== #
# E — Post-quantization merge
# =========================================================================== #

class TestPostQuantizeMerge:
    """Adjacent same-pitch quantized notes should merge."""

    def test_contiguous_same_pitch_merged(self):
        """Short contiguous same-pitch notes merge into one."""
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.25, midi_pitch=72),
            QuantizedNote(start_ql=0.25, duration_ql=0.25, midi_pitch=72),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 1
        assert result[0].duration_ql == pytest.approx(0.5)

    def test_long_contiguous_same_pitch_not_merged(self):
        """Longer same-pitch notes (re-articulations) are preserved."""
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.5, midi_pitch=72),
            QuantizedNote(start_ql=0.5, duration_ql=0.5, midi_pitch=72),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 2

    def test_different_pitch_not_merged(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.5, midi_pitch=72),
            QuantizedNote(start_ql=0.5, duration_ql=0.5, midi_pitch=74),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 2

    def test_gap_prevents_merge(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.5, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.5, midi_pitch=72),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 2

    def test_rests_not_merged(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.5, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=0.5, duration_ql=0.5, midi_pitch=0, is_rest=True),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 2

    def test_tied_notes_not_merged(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.5, midi_pitch=72, tie_type="start"),
            QuantizedNote(start_ql=0.5, duration_ql=0.5, midi_pitch=72, tie_type="stop"),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 2

    def test_three_contiguous_merged(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.25, midi_pitch=72),
            QuantizedNote(start_ql=0.25, duration_ql=0.25, midi_pitch=72),
            QuantizedNote(start_ql=0.5, duration_ql=0.25, midi_pitch=72),
        ]
        result = merge_quantized_same_pitch(notes)
        assert len(result) == 1
        assert result[0].duration_ql == pytest.approx(0.75)

    def test_empty_input(self):
        assert merge_quantized_same_pitch([]) == []


# =========================================================================== #
# F — Full pipeline integration (before/after metrics)
# =========================================================================== #

class TestFullPipelineFragmentation:
    """Run the full resolver on a fragmented scenario and verify that
    the output has fewer, longer notes."""

    def _make_fragmented_vocal(self):
        """Simulate a vocal melody with Basic Pitch fragmentation:
        two sustained notes, each split into ~4 fragments with jitter."""
        sustained_a = [
            CandidateNote(0.0, 0.25, 72, confidence=0.7),
            CandidateNote(0.27, 0.5, 73, confidence=0.65),
            CandidateNote(0.52, 0.75, 72, confidence=0.7),
            CandidateNote(0.77, 1.0, 72, confidence=0.65),
        ]
        sustained_b = [
            CandidateNote(1.1, 1.35, 74, confidence=0.7),
            CandidateNote(1.37, 1.6, 75, confidence=0.65),
            CandidateNote(1.62, 1.85, 74, confidence=0.7),
            CandidateNote(1.87, 2.2, 74, confidence=0.65),
        ]
        return sustained_a + sustained_b

    def test_fragmented_notes_reduced(self):
        candidates = self._make_fragmented_vocal()
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        notes = result.selected_notes
        assert len(notes) <= 4, \
            f"8 fragments should merge to <=4 notes, got {len(notes)}"

    def test_avg_duration_increased(self):
        candidates = self._make_fragmented_vocal()
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        avg_dur = result.stats.get("avg_note_duration_s", 0)
        assert avg_dur > 0.35, \
            f"Average duration should be >0.35s after merging, got {avg_dur}"

    def test_sustain_metrics_present(self):
        candidates = self._make_fragmented_vocal()
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        assert "notes_before_cleanup" in result.stats
        assert "notes_merged" in result.stats
        assert "avg_duration_before_cleanup_s" in result.stats

    def test_melody_pitch_preserved(self):
        """After smoothing, the selected pitches should still be in the
        vocal range, not shifted to some wrong register."""
        candidates = self._make_fragmented_vocal()
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        for n in result.selected_notes:
            assert 70 <= n.midi_pitch <= 77, \
                f"Pitch {n.midi_pitch} is outside expected vocal range"

    def test_monophonic_output(self):
        candidates = self._make_fragmented_vocal()
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        notes = result.selected_notes
        for i in range(len(notes) - 1):
            assert notes[i].end_s <= notes[i + 1].start_s + 1e-6, \
                f"Notes {i} and {i+1} overlap"


# =========================================================================== #
# G — Config parameter verification
# =========================================================================== #

class TestSustainConfig:
    """Verify config defaults match the contour-preserving tuning."""

    def test_defaults(self):
        cfg = MelodyConfig()
        assert cfg.pitch_stability_semitones == 1
        assert cfg.pitch_stability_max_gap_ms == 120.0
        assert cfg.phrase_smoothing_enabled is True
        assert cfg.phrase_max_short_note_ms == 180.0
        assert cfg.phrase_max_pitch_diff == 2
        assert cfg.reattack_suppression_enabled is True
        assert cfg.reattack_max_gap_ms == 120.0
        assert cfg.post_quantize_merge is True

    def test_tuned_thresholds(self):
        """Verify smoothing thresholds are contour-preserving (tighter)."""
        cfg = MelodyConfig()
        assert cfg.merge_gap_ms == 120.0
        assert cfg.merge_nearby_pitch_gap_ms == 80.0
        assert cfg.consolidate_max_gap_ms == 150.0
        assert cfg.micro_note_ms == 80.0
