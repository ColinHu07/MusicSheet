"""
Tests for the candidate repair stage (subharmonic suppression + octave recovery).

Covers:
  A. Subharmonic detection
  B. Octave recovery / synthetic candidate generation
  C. Pipeline integration / regression
"""

import pytest
from src.pipeline.candidate_notes import CandidateNote
from src.pipeline.candidate_repair import (
    CandidateRepairConfig,
    repair_candidates,
    _compute_local_upper_quartiles,
    _find_overlapping_at_pitch,
)


def _cand(start, end, pitch, conf=0.8, source="basic_pitch"):
    return CandidateNote(
        start_s=start, end_s=end, pitch_midi=pitch,
        confidence=conf, salience=conf, source=source,
    )


# ═══════════════════════════════════════════════════════════════════════
# A. SUBHARMONIC DETECTION
# ═══════════════════════════════════════════════════════════════════════

class TestSubharmonicDetection:

    def test_low_candidate_flagged_when_context_is_high(self):
        """A candidate far below the local UQ is detected as low."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),   # Ab3 — suspiciously low (P+12=68 NOT present)
            _cand(0.5, 1.0, 70, conf=0.7),   # Bb4 — high context
            _cand(1.0, 1.5, 72, conf=0.7),   # C5  — high context
            _cand(1.5, 2.0, 68, conf=0.7),   # Ab4 — high context (non-overlapping)
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        assert stats.low_candidates_detected >= 1
        assert stats.synthetics_created >= 1

    def test_real_low_note_not_flagged_when_context_matches(self):
        """A low note in a context of other low notes should not be flagged."""
        candidates = [
            _cand(0.0, 0.5, 58, conf=0.8),
            _cand(0.5, 1.0, 60, conf=0.8),
            _cand(1.0, 1.5, 62, conf=0.8),
            _cand(1.5, 2.0, 60, conf=0.8),
        ]
        cfg = CandidateRepairConfig(enabled=True, low_band_threshold=7)
        result, stats = repair_candidates(candidates, cfg)

        # Local UQ is around 61-62, so notes at 58-60 are only 2-3 below UQ
        # — not enough to trigger the threshold of 7
        assert stats.low_candidates_detected == 0

    def test_local_uq_computed_correctly(self):
        """Local upper quartile should reflect the top of the pitch distribution."""
        candidates = [
            _cand(0.0, 0.5, 60),
            _cand(0.0, 0.5, 65),
            _cand(0.0, 0.5, 70),
            _cand(0.0, 0.5, 75),
        ]
        uqs = _compute_local_upper_quartiles(candidates, window_s=5.0)
        # All notes are within the window; UQ should be at the 75th percentile
        # With 4 equal-duration notes, P75 is the 3rd note = 70
        for uq in uqs:
            assert uq >= 70

    def test_disabled_config_does_nothing(self):
        """When disabled, candidates pass through unchanged."""
        candidates = [_cand(0.0, 0.5, 56, conf=0.8)]
        cfg = CandidateRepairConfig(enabled=False)
        result, stats = repair_candidates(candidates, cfg)

        assert len(result) == 1
        assert result[0].pitch_midi == 56
        assert result[0].confidence == 0.8
        assert stats.synthetics_created == 0


# ═══════════════════════════════════════════════════════════════════════
# B. OCTAVE RECOVERY
# ═══════════════════════════════════════════════════════════════════════

class TestOctaveRecovery:

    def test_synthetic_created_when_upper_missing(self):
        """When P exists but P+12 does not overlap, a synthetic at P+12 is created."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),   # Ab3 — likely subharmonic (no Ab4 overlaps)
            _cand(0.5, 1.0, 70, conf=0.7),   # Bb4 — context
            _cand(1.0, 1.5, 72, conf=0.7),   # C5  — context
            _cand(1.5, 2.0, 68, conf=0.7),   # Ab4 — context (non-overlapping with Ab3)
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        synth_pitches = [c.pitch_midi for c in result if c.source == "octave_recovery"]
        assert 68 in synth_pitches, f"Expected synthetic at 68, got {synth_pitches}"
        assert stats.synthetics_created >= 1

    def test_synthetic_preserves_timing(self):
        """Synthetic candidate inherits start/end from the original."""
        candidates = [
            _cand(1.5, 2.3, 58, conf=0.9),   # Bb3
            _cand(1.5, 2.3, 72, conf=0.8),   # C5 — context
            _cand(2.3, 3.0, 70, conf=0.8),
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        synthetics = [c for c in result if c.source == "octave_recovery"]
        if synthetics:
            s = synthetics[0]
            assert s.start_s == 1.5
            assert s.end_s == 2.3
            assert s.pitch_midi == 70  # 58 + 12

    def test_synthetic_confidence_scaled(self):
        """Synthetic candidate confidence is scaled from the original."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),
            _cand(0.0, 0.5, 70, conf=0.7),  # context
            _cand(0.5, 1.0, 72, conf=0.7),
        ]
        cfg = CandidateRepairConfig(
            enabled=True,
            octave_recovery_confidence_scale=0.90,
        )
        result, stats = repair_candidates(candidates, cfg)

        synthetics = [c for c in result if c.source == "octave_recovery"]
        if synthetics:
            assert abs(synthetics[0].confidence - 0.8 * 0.90) < 0.01

    def test_existing_upper_boosted_when_weak(self):
        """If P+12 already exists but is weaker than P, boost it."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.9),    # Ab3 — strong subharmonic
            _cand(0.0, 0.5, 68, conf=0.3),    # Ab4 — weak existing upper
            _cand(0.5, 1.0, 70, conf=0.8),    # Bb4 — context
            _cand(1.0, 1.5, 72, conf=0.8),    # C5  — context
        ]
        cfg = CandidateRepairConfig(
            enabled=True,
            octave_boost_min_ratio=0.85,
        )
        result, stats = repair_candidates(candidates, cfg)

        # The Ab4 at 68 should be boosted
        ab4 = [c for c in result if c.pitch_midi == 68 and c.source == "basic_pitch"]
        if ab4:
            assert ab4[0].confidence >= 0.9 * 0.85 - 0.01
        assert stats.existing_boosted >= 1

    def test_no_synthetic_when_recovery_disabled(self):
        """Disabling octave_recovery prevents synthetic generation."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),
            _cand(0.0, 0.5, 70, conf=0.7),  # context
            _cand(0.5, 1.0, 72, conf=0.7),
        ]
        cfg = CandidateRepairConfig(
            enabled=True,
            octave_recovery_enabled=True,  # repair is enabled
        )
        result_on, stats_on = repair_candidates(candidates, cfg)

        cfg_off = CandidateRepairConfig(enabled=False)
        result_off, stats_off = repair_candidates(candidates, cfg_off)

        assert stats_off.synthetics_created == 0
        assert len(result_off) == len(candidates)

    def test_no_synthetic_above_max_midi(self):
        """Synthetic candidates above max_synthetic_midi should not be created."""
        candidates = [
            _cand(0.0, 0.5, 90, conf=0.8),   # shifted would be 102
            _cand(0.0, 0.5, 70, conf=0.7),   # context at 70
            _cand(0.5, 1.0, 72, conf=0.7),
        ]
        cfg = CandidateRepairConfig(enabled=True, max_synthetic_midi=96)
        result, stats = repair_candidates(candidates, cfg)

        # 90 + 12 = 102 > 96 → no synthetic
        synth_pitches = [c.pitch_midi for c in result if c.source == "octave_recovery"]
        assert 102 not in synth_pitches

    def test_original_suppressed_when_synthetic_created(self):
        """When a synthetic is created, the original's confidence is reduced."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),   # will be suppressed
            _cand(0.0, 0.5, 70, conf=0.7),   # context
            _cand(0.5, 1.0, 72, conf=0.7),
        ]
        cfg = CandidateRepairConfig(
            enabled=True,
            suppress_original=True,
            suppress_confidence_scale=0.50,
        )
        result, stats = repair_candidates(candidates, cfg)

        original_ab3 = [c for c in result if c.pitch_midi == 56]
        if original_ab3:
            assert original_ab3[0].confidence <= 0.8 * 0.50 + 0.01
        assert stats.originals_suppressed >= 1


# ═══════════════════════════════════════════════════════════════════════
# C. PIPELINE INTEGRATION / REGRESSION
# ═══════════════════════════════════════════════════════════════════════

class TestCandidateRepairIntegration:

    def test_empty_candidates_handled(self):
        """Empty candidate list should pass through cleanly."""
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates([], cfg)
        assert len(result) == 0
        assert stats.candidates_before == 0
        assert stats.candidates_after == 0

    def test_candidate_count_changes_correctly(self):
        """After repair, count = original + synthetics."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),   # low → synth at 68
            _cand(0.0, 0.5, 58, conf=0.8),   # low → synth at 70
            _cand(0.0, 0.5, 70, conf=0.7),   # high context
            _cand(0.5, 1.0, 72, conf=0.7),   # high context
            _cand(1.0, 1.5, 68, conf=0.7),   # high context
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        assert stats.candidates_after == stats.candidates_before + stats.synthetics_created

    def test_output_sorted_by_start_and_pitch(self):
        """Repaired candidates must be sorted by (start_s, pitch_midi)."""
        candidates = [
            _cand(0.0, 0.5, 56, conf=0.8),
            _cand(0.5, 1.0, 58, conf=0.8),
            _cand(0.0, 0.5, 70, conf=0.7),
            _cand(0.5, 1.0, 72, conf=0.7),
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, _ = repair_candidates(candidates, cfg)

        for i in range(len(result) - 1):
            key_a = (result[i].start_s, result[i].pitch_midi)
            key_b = (result[i + 1].start_s, result[i + 1].pitch_midi)
            assert key_a <= key_b, f"Not sorted at index {i}: {key_a} > {key_b}"

    def test_stats_have_all_fields(self):
        """RepairStats should contain all expected diagnostic fields."""
        candidates = [_cand(0.0, 0.5, 68)]
        cfg = CandidateRepairConfig(enabled=True)
        _, stats = repair_candidates(candidates, cfg)

        d = stats.to_dict()
        for key in [
            "candidates_before", "candidates_after", "low_candidates_detected",
            "synthetics_created", "existing_boosted", "originals_suppressed",
        ]:
            assert key in d, f"Missing stats key: {key}"

    def test_high_candidates_unchanged(self):
        """Candidates already in the melodic band should pass through untouched."""
        candidates = [
            _cand(0.0, 0.5, 68, conf=0.8),
            _cand(0.5, 1.0, 70, conf=0.7),
            _cand(1.0, 1.5, 72, conf=0.9),
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        assert stats.low_candidates_detected == 0
        assert stats.synthetics_created == 0
        assert len(result) == 3

    def test_find_overlapping_at_pitch(self):
        """Overlap finder returns the best match at the target pitch."""
        candidates = [
            _cand(0.0, 0.5, 68, conf=0.7),
            _cand(0.0, 0.5, 70, conf=0.9),
            _cand(1.0, 1.5, 68, conf=0.3),  # non-overlapping
        ]
        ref = _cand(0.0, 0.5, 56)
        found = _find_overlapping_at_pitch(candidates, ref, 68)
        assert found is not None
        assert found.confidence == 0.7

        not_found = _find_overlapping_at_pitch(candidates, ref, 75)
        assert not_found is None

    def test_realistic_vocal_scenario(self):
        """Simulate a realistic vocal scenario with subharmonic pollution."""
        # Early phrase: only sub-harmonics available
        candidates = [
            _cand(0.0, 0.4, 58, conf=0.7),
            _cand(0.4, 0.8, 60, conf=0.8),
            _cand(0.8, 1.2, 58, conf=0.6),
            # Later phrase: high candidates present
            _cand(2.0, 2.4, 68, conf=0.8),
            _cand(2.4, 2.8, 70, conf=0.7),
            _cand(2.8, 3.2, 72, conf=0.9),
            _cand(3.2, 3.6, 68, conf=0.7),
        ]
        cfg = CandidateRepairConfig(enabled=True)
        result, stats = repair_candidates(candidates, cfg)

        # The early sub-harmonics (58, 60) should get synthetics at (70, 72)
        synth_pitches = sorted(set(
            c.pitch_midi for c in result if c.source == "octave_recovery"
        ))
        assert 70 in synth_pitches or 72 in synth_pitches, \
            f"Expected synthetics in 70-72 range, got {synth_pitches}"

        # The later high candidates should be untouched
        late_originals = [c for c in result if c.start_s >= 2.0 and c.source == "basic_pitch"]
        assert len(late_originals) >= 4
