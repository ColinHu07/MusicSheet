"""
Tests for manual meter override (time signature).

Covers:
  A. 2/2 override changes measure construction
  B. Config propagation
  C. BeatGrid measure_duration_ql correctness
  D. MusicXML reflects override
  E. Quantizer uses the override
"""

import os
import pytest
import tempfile

from src.pipeline.beat_grid import BeatGrid, _parse_time_signature
from src.pipeline.quantize import quantize_notes, split_at_barlines
from src.pipeline.utils import NoteEvent, QuantizedNote, TranscriptionConfig


class TestParseTimeSignature:
    def test_auto_defaults_to_4_4(self):
        assert _parse_time_signature("auto") == (4, 4)

    def test_4_4(self):
        assert _parse_time_signature("4/4") == (4, 4)

    def test_3_4(self):
        assert _parse_time_signature("3/4") == (3, 4)

    def test_2_2(self):
        assert _parse_time_signature("2/2") == (2, 2)

    def test_6_8(self):
        assert _parse_time_signature("6/8") == (6, 8)

    def test_2_4(self):
        assert _parse_time_signature("2/4") == (2, 4)

    def test_custom_fraction(self):
        assert _parse_time_signature("5/4") == (5, 4)

    def test_unknown_defaults(self):
        assert _parse_time_signature("xyz") == (4, 4)


class TestBeatGridMeasureDuration:
    def _grid(self, ts_num, ts_den, bpm=120):
        import numpy as np
        return BeatGrid(
            tempo_bpm=bpm,
            beat_times=np.array([0.0, 0.5, 1.0]),
            time_sig_num=ts_num,
            time_sig_den=ts_den,
        )

    def test_4_4_measure_is_4_ql(self):
        g = self._grid(4, 4)
        assert g.measure_duration_ql == pytest.approx(4.0)

    def test_2_2_measure_is_4_ql(self):
        """2/2 = 2 half notes = 4 quarter-note lengths per measure."""
        g = self._grid(2, 2)
        assert g.measure_duration_ql == pytest.approx(4.0)

    def test_3_4_measure_is_3_ql(self):
        g = self._grid(3, 4)
        assert g.measure_duration_ql == pytest.approx(3.0)

    def test_6_8_measure_is_3_ql(self):
        g = self._grid(6, 8)
        assert g.measure_duration_ql == pytest.approx(3.0)


class TestQuantizationWithMeterOverride:
    def _grid_22(self):
        import numpy as np
        return BeatGrid(
            tempo_bpm=120,
            beat_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            time_sig_num=2,
            time_sig_den=2,
        )

    def test_2_2_split_at_barlines(self):
        """A note spanning 5 ql in 2/2 (4 ql per measure) should be split."""
        grid = self._grid_22()
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=5.0, midi_pitch=72),
        ]
        result = split_at_barlines(notes, grid)
        assert len(result) == 2
        assert result[0].duration_ql == pytest.approx(4.0)
        assert result[1].duration_ql == pytest.approx(1.0)

    def test_2_2_note_within_measure(self):
        """A note within one 2/2 measure should not be split."""
        grid = self._grid_22()
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=3.0, midi_pitch=72),
        ]
        result = split_at_barlines(notes, grid)
        assert len(result) == 1


class TestConfigTimeSigPropagation:
    def test_config_holds_time_sig(self):
        cfg = TranscriptionConfig(time_signature="2/2")
        assert cfg.time_signature == "2/2"

    def test_config_defaults_to_auto(self):
        cfg = TranscriptionConfig()
        assert cfg.time_signature == "auto"


class TestMusicXMLTimeSignature:
    """Verify MusicXML export includes the correct time signature."""

    def test_2_2_in_musicxml(self):
        import numpy as np
        from src.pipeline.beat_grid import BeatGrid
        from src.pipeline.musicxml_export import export_musicxml

        grid = BeatGrid(
            tempo_bpm=120,
            beat_times=np.array([0.0, 0.5, 1.0]),
            time_sig_num=2,
            time_sig_den=2,
        )
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=2.0, midi_pitch=72),
            QuantizedNote(start_ql=2.0, duration_ql=2.0, midi_pitch=74),
        ]

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            out = f.name

        try:
            export_musicxml(
                notes=notes, key_name="C", mode="major", sharps=0,
                grid=grid, output_path=out,
            )
            with open(out, "r", encoding="utf-8") as fh:
                xml = fh.read()

            assert "<beats>2</beats>" in xml
            assert "<beat-type>2</beat-type>" in xml
        finally:
            os.unlink(out)

    def test_4_4_in_musicxml(self):
        import numpy as np
        from src.pipeline.beat_grid import BeatGrid
        from src.pipeline.musicxml_export import export_musicxml

        grid = BeatGrid(
            tempo_bpm=120,
            beat_times=np.array([0.0, 0.5, 1.0]),
            time_sig_num=4,
            time_sig_den=4,
        )
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=4.0, midi_pitch=72),
        ]

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            out = f.name

        try:
            export_musicxml(
                notes=notes, key_name="C", mode="major", sharps=0,
                grid=grid, output_path=out,
            )
            with open(out, "r", encoding="utf-8") as fh:
                xml = fh.read()

            assert "<beats>4</beats>" in xml
            assert "<beat-type>4</beat-type>" in xml
        finally:
            os.unlink(out)
