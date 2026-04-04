"""
Tests for source separation, source mode routing, and the separation wrapper.

Covers:
  A. Source separation wrapper (StemPaths, availability)
  B. Command builder (--mp3 flag, --two-stems, module mode)
  C. Backend selection (CLI > module > library)
  D. Cache behaviour (mp3 and wav detection)
  E. Source mode routing in pipeline
  F. Graceful fallback when separator unavailable
  G. MP3 stem output acceptance

Run with:  pytest tests/test_source_separation.py -v
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline.source_separation import (
    StemPaths,
    is_available,
    separate,
    build_demucs_command,
    _check_cache,
    _find_stem_file,
    _demucs_cli_available,
    _demucs_module_available,
    _demucs_library_available,
    _select_backend,
)
from src.pipeline.utils import TranscriptionConfig


class TestStemPaths:
    def test_defaults(self):
        sp = StemPaths()
        assert sp.vocals is None
        assert sp.accompaniment is None
        assert sp.model_name == ""

    def test_populated(self):
        sp = StemPaths(vocals="/tmp/v.wav", model_name="htdemucs")
        assert sp.vocals == "/tmp/v.wav"
        assert sp.model_name == "htdemucs"

    def test_mp3_path(self):
        sp = StemPaths(vocals="/tmp/v.mp3", model_name="htdemucs")
        assert sp.vocals == "/tmp/v.mp3"


class TestBuildDemucsCommand:
    """Verify the command builder produces the correct invocation."""

    def test_includes_mp3_flag(self):
        cmd = build_demucs_command("/audio/song.wav", "/out", mp3_output=True)
        assert "--mp3" in cmd

    def test_includes_two_stems(self):
        cmd = build_demucs_command("/audio/song.wav", "/out")
        assert "--two-stems" in cmd
        idx = cmd.index("--two-stems")
        assert cmd[idx + 1] == "vocals"

    def test_includes_model(self):
        cmd = build_demucs_command("/audio/song.wav", "/out", model="htdemucs_ft")
        assert "-n" in cmd
        idx = cmd.index("-n")
        assert cmd[idx + 1] == "htdemucs_ft"

    def test_includes_output_dir(self):
        cmd = build_demucs_command("/audio/song.wav", "/my/output")
        assert "-o" in cmd
        idx = cmd.index("-o")
        assert cmd[idx + 1] == "/my/output"

    def test_audio_path_is_last(self):
        cmd = build_demucs_command("/audio/song.wav", "/out")
        assert cmd[-1] == "/audio/song.wav"

    def test_no_mp3_flag_when_disabled(self):
        cmd = build_demucs_command("/audio/song.wav", "/out", mp3_output=False)
        assert "--mp3" not in cmd

    def test_default_model(self):
        cmd = build_demucs_command("/audio/song.wav", "/out")
        idx = cmd.index("-n")
        assert cmd[idx + 1] == "htdemucs"

    def test_module_mode_uses_python(self):
        """When use_module=True, command starts with python -m demucs.separate."""
        cmd = build_demucs_command("/audio/song.wav", "/out", use_module=True)
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "demucs.separate"
        assert "--mp3" in cmd
        assert cmd[-1] == "/audio/song.wav"

    def test_non_module_mode_uses_bare_demucs(self):
        cmd = build_demucs_command("/audio/song.wav", "/out", use_module=False)
        assert cmd[0] == "demucs"


class TestAvailabilityChecks:
    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=False)
    def test_not_available_when_no_demucs(self, mock_lib, mock_mod, mock_cli):
        assert is_available() is False

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=True)
    def test_available_when_cli_found(self, mock_cli):
        assert is_available() is True

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=True)
    def test_available_when_module_found(self, mock_mod, mock_cli):
        assert is_available() is True

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=True)
    def test_available_when_library_found(self, mock_lib, mock_mod, mock_cli):
        assert is_available() is True


class TestSelectBackend:
    """Verify backend selection priority: CLI > module > library > none."""

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=True)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=True)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=True)
    def test_prefers_cli_when_available(self, mock_lib, mock_mod, mock_cli):
        backend, use_module = _select_backend()
        assert backend == "cli"
        assert use_module is False

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=True)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=True)
    def test_prefers_module_over_library(self, mock_lib, mock_mod, mock_cli):
        backend, use_module = _select_backend()
        assert backend == "cli"
        assert use_module is True

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=True)
    def test_falls_back_to_library(self, mock_lib, mock_mod, mock_cli):
        backend, use_module = _select_backend()
        assert backend == "library"
        assert use_module is False

    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=False)
    def test_returns_none_when_nothing_available(self, mock_lib, mock_mod, mock_cli):
        backend, use_module = _select_backend()
        assert backend == "none"


class TestSeparateGracefulFallback:
    @patch("src.pipeline.source_separation._demucs_cli_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_module_available", return_value=False)
    @patch("src.pipeline.source_separation._demucs_library_available", return_value=False)
    def test_raises_when_no_separator(self, mock_lib, mock_mod, mock_cli, tmp_path):
        dummy_wav = tmp_path / "test.wav"
        dummy_wav.write_bytes(b"RIFF" + b"\x00" * 100)
        with pytest.raises(RuntimeError, match="No source separator"):
            separate(str(dummy_wav), output_dir=str(tmp_path))


class TestFindStemFile:
    """Verify stem file discovery supports mp3, wav, and flac."""

    def test_finds_mp3(self, tmp_path):
        (tmp_path / "vocals.mp3").write_bytes(b"mp3data")
        result = _find_stem_file(tmp_path, "vocals")
        assert result is not None
        assert result.endswith("vocals.mp3")

    def test_finds_wav(self, tmp_path):
        (tmp_path / "vocals.wav").write_bytes(b"wavdata")
        result = _find_stem_file(tmp_path, "vocals")
        assert result is not None
        assert result.endswith("vocals.wav")

    def test_prefers_mp3_over_wav_by_order(self, tmp_path):
        """mp3 comes first in STEM_EXTENSIONS, so it's found first."""
        (tmp_path / "vocals.mp3").write_bytes(b"mp3data")
        (tmp_path / "vocals.wav").write_bytes(b"wavdata")
        result = _find_stem_file(tmp_path, "vocals")
        assert result.endswith("vocals.mp3")

    def test_returns_none_when_missing(self, tmp_path):
        result = _find_stem_file(tmp_path, "vocals")
        assert result is None


class TestCacheCheck:
    def test_returns_none_when_no_cache(self, tmp_path):
        result = _check_cache(tmp_path, "htdemucs")
        assert result is None

    def test_returns_stems_when_wav_cached(self, tmp_path):
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"dummy")
        result = _check_cache(tmp_path, "htdemucs")
        assert result is not None
        assert result.vocals == str(vocals)
        assert result.model_name == "htdemucs"

    def test_returns_stems_when_mp3_cached(self, tmp_path):
        vocals = tmp_path / "vocals.mp3"
        vocals.write_bytes(b"dummy")
        result = _check_cache(tmp_path, "htdemucs")
        assert result is not None
        assert result.vocals == str(vocals)

    def test_includes_accompaniment_if_present(self, tmp_path):
        (tmp_path / "vocals.wav").write_bytes(b"dummy")
        (tmp_path / "no_vocals.wav").write_bytes(b"dummy")
        result = _check_cache(tmp_path, "htdemucs")
        assert result.accompaniment is not None


class TestSourceModeConfig:
    def test_default_is_mix(self):
        cfg = TranscriptionConfig()
        assert cfg.source_mode == "mix"

    def test_can_set_vocals(self):
        cfg = TranscriptionConfig(source_mode="vocals")
        assert cfg.source_mode == "vocals"

    def test_readable_rhythm_default_on(self):
        cfg = TranscriptionConfig()
        assert cfg.readable_rhythm is True
