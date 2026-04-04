"""
Base provider interface for audio sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    """One search result from an audio provider."""
    title: str
    url: str
    duration_s: Optional[float] = None
    channel: str = ""
    thumbnail_url: str = ""
    provider: str = ""

    @property
    def duration_display(self) -> str:
        if self.duration_s is None:
            return ""
        m, s = divmod(int(self.duration_s), 60)
        return f"{m}:{s:02d}"


@dataclass
class FetchResult:
    """Result of fetching audio from a provider."""
    audio_path: str
    title: str
    source_url: str = ""
    duration_s: float = 0.0
    provider: str = ""
    metadata: dict = field(default_factory=dict)


class AudioProvider(ABC):
    """Abstract base for audio source providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        ...

    @abstractmethod
    def fetch_audio(self, url: str, output_dir: str) -> FetchResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
