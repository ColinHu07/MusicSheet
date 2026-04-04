"""
YouTube audio provider using yt-dlp.

Handles both search-by-query and direct-URL fetch.
Audio is downloaded as best-quality audio and converted to WAV.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .base_provider import AudioProvider, FetchResult, SearchResult

logger = logging.getLogger("mp3_to_sheet.youtube")

_YTDLP_NAMES = ("yt-dlp", "yt-dlp.exe")


def _sanitize_filename(title: str, max_len: int = 80) -> str:
    """Make a string safe for use as a filename."""
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', title)
    cleaned = cleaned.strip('. ')
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip('. ')
    return cleaned or "untitled"


def _find_ytdlp() -> Optional[str]:
    for name in _YTDLP_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


class YouTubeProvider(AudioProvider):
    """YouTube audio provider backed by yt-dlp."""

    def __init__(self, cache_dir: str = "out/cache/youtube"):
        self._cache_dir = cache_dir

    def is_available(self) -> bool:
        return _find_ytdlp() is not None

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        """
        Search YouTube for videos matching a query.
        Uses yt-dlp's ytsearch feature.
        """
        ytdlp = _find_ytdlp()
        if not ytdlp:
            raise RuntimeError(
                "yt-dlp not found. Install it:\n  pip install yt-dlp\nor download from https://github.com/yt-dlp/yt-dlp"
            )

        cmd = [
            ytdlp,
            f"ytsearch{max_results}:{query}",
            "--dump-json",
            "--flat-playlist",
            "--no-download",
            "--no-warnings",
        ]

        logger.info("YouTube search: %s", query)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, encoding="utf-8",
            )
        except FileNotFoundError:
            raise RuntimeError("yt-dlp executable not found")
        except subprocess.TimeoutExpired:
            raise RuntimeError("YouTube search timed out")

        if result.returncode != 0:
            logger.error("yt-dlp search error: %s", result.stderr[:500])
            return []

        import json
        results: List[SearchResult] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                url = data.get("url") or data.get("webpage_url") or ""
                if url and not url.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={data.get('id', '')}"
                results.append(SearchResult(
                    title=data.get("title", "Unknown"),
                    url=url,
                    duration_s=data.get("duration"),
                    channel=data.get("channel", data.get("uploader", "")),
                    thumbnail_url=data.get("thumbnail", ""),
                    provider="youtube",
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        logger.info("YouTube search returned %d results", len(results))
        return results

    def fetch_audio(self, url: str, output_dir: str = "") -> FetchResult:
        """
        Download audio from a YouTube URL.
        Returns the path to the downloaded audio file.
        """
        ytdlp = _find_ytdlp()
        if not ytdlp:
            raise RuntimeError("yt-dlp not found")

        cache_dir = output_dir or self._cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        output_template = os.path.join(cache_dir, f"{url_hash}_%(title)s.%(ext)s")

        cmd = [
            ytdlp,
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", output_template,
            "--no-playlist",
            "--print", "after_move:filepath",
            "--print", "%(title)s",
            "--print", "%(duration)s",
            url,
        ]

        logger.info("Fetching audio from: %s", url)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("YouTube download timed out (5 min)")

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            raise RuntimeError(f"yt-dlp failed: {error_msg}")

        lines = result.stdout.strip().split("\n")
        audio_path = ""
        title = "Unknown"
        duration_s = 0.0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if os.path.isfile(line):
                audio_path = line
            elif not title or title == "Unknown":
                title = line
            else:
                try:
                    duration_s = float(line)
                except ValueError:
                    if os.path.isfile(line):
                        audio_path = line

        if not audio_path:
            candidates = list(Path(cache_dir).glob(f"{url_hash}_*"))
            if candidates:
                audio_path = str(candidates[0])

        if not audio_path or not os.path.isfile(audio_path):
            raise RuntimeError("Download completed but audio file not found")

        logger.info("Audio downloaded: %s (%s)", audio_path, title)
        return FetchResult(
            audio_path=audio_path,
            title=title,
            source_url=url,
            duration_s=duration_s,
            provider="youtube",
            metadata={"url_hash": url_hash},
        )
