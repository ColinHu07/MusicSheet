"""
Tests for the YouTube provider (mocked — no network required).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.providers.base_provider import SearchResult, FetchResult
from src.providers.youtube_provider import (
    YouTubeProvider,
    _sanitize_filename,
    _find_ytdlp,
)


class TestSanitizeFilename:

    def test_basic(self):
        assert _sanitize_filename("my song") == "my song"

    def test_strips_invalid_chars(self):
        result = _sanitize_filename('song: the <best> "remix" | v2')
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result
        assert "|" not in result

    def test_truncates_long(self):
        result = _sanitize_filename("a" * 200, max_len=80)
        assert len(result) <= 80

    def test_empty_returns_untitled(self):
        assert _sanitize_filename("") == "untitled"

    def test_unicode(self):
        result = _sanitize_filename("告白氣球 Love Confession")
        assert "告白氣球" in result


class TestYouTubeProviderAvailability:

    @patch("src.providers.youtube_provider._find_ytdlp", return_value=None)
    def test_unavailable_when_no_ytdlp(self, mock):
        provider = YouTubeProvider()
        assert not provider.is_available()

    @patch("src.providers.youtube_provider._find_ytdlp", return_value="/usr/bin/yt-dlp")
    def test_available_when_ytdlp_found(self, mock):
        provider = YouTubeProvider()
        assert provider.is_available()


class TestYouTubeSearch:

    @patch("src.providers.youtube_provider._find_ytdlp", return_value=None)
    def test_search_raises_when_no_ytdlp(self, mock):
        provider = YouTubeProvider()
        with pytest.raises(RuntimeError, match="yt-dlp not found"):
            provider.search("test query")

    @patch("src.providers.youtube_provider._find_ytdlp", return_value="/usr/bin/yt-dlp")
    @patch("subprocess.run")
    def test_search_parses_results(self, mock_run, mock_find):
        search_result = {
            "title": "Test Song",
            "url": "https://www.youtube.com/watch?v=abc123",
            "duration": 240,
            "channel": "Test Channel",
            "thumbnail": "https://img.youtube.com/test.jpg",
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(search_result),
            stderr="",
        )
        provider = YouTubeProvider()
        results = provider.search("test")
        assert len(results) == 1
        assert results[0].title == "Test Song"
        assert results[0].duration_s == 240

    @patch("src.providers.youtube_provider._find_ytdlp", return_value="/usr/bin/yt-dlp")
    @patch("subprocess.run")
    def test_search_handles_error(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        provider = YouTubeProvider()
        results = provider.search("test")
        assert results == []


class TestYouTubeFetch:

    @patch("src.providers.youtube_provider._find_ytdlp", return_value=None)
    def test_fetch_raises_when_no_ytdlp(self, mock):
        provider = YouTubeProvider()
        with pytest.raises(RuntimeError):
            provider.fetch_audio("https://youtube.com/watch?v=test")

    @patch("src.providers.youtube_provider._find_ytdlp", return_value="/usr/bin/yt-dlp")
    @patch("subprocess.run")
    def test_fetch_success(self, mock_run, mock_find):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            f.write(b"fake wav data")

        try:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=f"{tmp_path}\nTest Song\n240",
                stderr="",
            )
            provider = YouTubeProvider()
            result = provider.fetch_audio("https://youtube.com/watch?v=test",
                                          output_dir=tempfile.gettempdir())
            assert result.audio_path == tmp_path
            assert result.title == "Test Song"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch("src.providers.youtube_provider._find_ytdlp", return_value="/usr/bin/yt-dlp")
    @patch("subprocess.run")
    def test_fetch_error(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Video unavailable"
        )
        provider = YouTubeProvider()
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            provider.fetch_audio("https://youtube.com/watch?v=invalid")


class TestSearchResult:

    def test_duration_display(self):
        r = SearchResult(title="test", url="", duration_s=125)
        assert r.duration_display == "2:05"

    def test_duration_display_none(self):
        r = SearchResult(title="test", url="", duration_s=None)
        assert r.duration_display == ""
