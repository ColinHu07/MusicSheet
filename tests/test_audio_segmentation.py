"""Tests for the audio-based note segmentation module."""

import pytest
from src.pipeline.audio_segmentation import (
    AudioSegmentationConfig,
    SegmentationStats,
    _apply_splits,
    _find_split_points,
)
from src.pipeline.utils import NoteEvent


class TestApplySplits:
    """Test the note splitting logic."""

    def test_no_split_points_returns_original(self):
        note = NoteEvent(0.0, 1.0, 60, 0.9)
        result = _apply_splits(note, [], 0.1)
        assert len(result) == 1
        assert result[0].start_s == 0.0
        assert result[0].end_s == 1.0

    def test_single_split_produces_two_fragments(self):
        note = NoteEvent(0.0, 1.0, 60, 0.9)
        result = _apply_splits(note, [(0.5, "onset")], 0.1)
        assert len(result) == 2
        assert result[0].start_s == 0.0
        assert result[0].end_s == pytest.approx(0.5)
        assert result[1].start_s == pytest.approx(0.5)
        assert result[1].end_s == 1.0
        assert result[0].midi_pitch == 60
        assert result[1].midi_pitch == 60

    def test_split_too_close_to_boundary_ignored(self):
        note = NoteEvent(0.0, 1.0, 60, 0.9)
        result = _apply_splits(note, [(0.05, "energy")], 0.12)
        assert len(result) == 1

    def test_multiple_splits(self):
        note = NoteEvent(0.0, 2.0, 60, 0.9)
        splits = [(0.5, "onset"), (1.0, "energy"), (1.5, "pitch")]
        result = _apply_splits(note, splits, 0.1)
        assert len(result) == 4
        assert result[0].end_s == pytest.approx(0.5)
        assert result[1].start_s == pytest.approx(0.5)
        assert result[1].end_s == pytest.approx(1.0)

    def test_close_splits_merged(self):
        note = NoteEvent(0.0, 2.0, 60, 0.9)
        splits = [(0.50, "onset"), (0.55, "energy")]
        result = _apply_splits(note, splits, 0.12)
        assert len(result) == 2

    def test_fragment_too_short_dropped(self):
        note = NoteEvent(0.0, 0.5, 60, 0.9)
        splits = [(0.1, "onset"), (0.15, "pitch")]
        result = _apply_splits(note, splits, 0.12)
        assert all(f.duration_s >= 0.12 for f in result)

    def test_pitch_preserved_in_fragments(self):
        note = NoteEvent(0.0, 1.0, 72, 0.8)
        result = _apply_splits(note, [(0.4, "onset")], 0.1)
        for f in result:
            assert f.midi_pitch == 72
            assert f.confidence == 0.8


class TestConfig:
    """Test segmentation configuration defaults."""

    def test_default_config(self):
        cfg = AudioSegmentationConfig()
        assert cfg.enabled is True
        assert cfg.min_note_duration_to_split_s == 0.40
        assert cfg.min_fragment_duration_s == 0.12
        assert cfg.energy_enabled is True
        assert cfg.onset_enabled is True
        assert cfg.pitch_reset_enabled is True
        assert cfg.min_evidence_count == 1

    def test_disabled_config(self):
        cfg = AudioSegmentationConfig(enabled=False)
        assert cfg.enabled is False


class TestStats:
    """Test stats serialization."""

    def test_to_dict(self):
        s = SegmentationStats(
            notes_before=100, notes_after=150,
            notes_split=30, splits_by_energy=10,
            splits_by_onset=15, splits_by_pitch=5,
        )
        d = s.to_dict()
        assert d["notes_before"] == 100
        assert d["notes_after"] == 150
        assert d["notes_split"] == 30


class TestDoubleTimeDetection:
    """Test tempo halving logic in beat_grid."""

    def test_tempo_above_100_gets_halved(self):
        from src.pipeline.beat_grid import _correct_double_time
        import numpy as np

        y = np.random.randn(22050 * 10).astype(np.float32) * 0.01
        bt = np.linspace(0.5, 9.5, 20)

        new_tempo, new_beats = _correct_double_time(140.0, bt, y, 22050)
        assert 60 < new_tempo < 80
        assert len(new_beats) == 10

    def test_tempo_below_100_unchanged(self):
        from src.pipeline.beat_grid import _correct_double_time
        import numpy as np

        y = np.random.randn(22050 * 10).astype(np.float32)
        bt = np.linspace(0.5, 9.5, 20)

        new_tempo, new_beats = _correct_double_time(85.0, bt, y, 22050)
        assert new_tempo == 85.0
        assert len(new_beats) == 20

    def test_halved_too_low_unchanged(self):
        from src.pipeline.beat_grid import _correct_double_time
        import numpy as np

        y = np.random.randn(22050 * 10).astype(np.float32) * 0.01
        bt = np.linspace(0.5, 9.5, 20)

        # 204/2 = 102, which is above 100 — stays unchanged
        new_tempo, new_beats = _correct_double_time(204.0, bt, y, 22050)
        assert new_tempo == 204.0
