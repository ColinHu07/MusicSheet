"""
Stage A — Audio ingest.

Load audio (MP3 / WAV / FLAC), convert to mono WAV, peak-normalize,
optionally truncate, and cache the result.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .utils import file_hash, ensure_dir

logger = logging.getLogger("mp3_to_sheet.ingest")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
TARGET_SR = 22050


def ingest_audio(
    input_path: str,
    output_dir: str,
    max_duration_s: float = 45.0,
    target_sr: int = TARGET_SR,
) -> Tuple[str, int]:
    """
    Load *input_path*, convert to mono WAV at *target_sr*, peak-normalize.

    Returns
    -------
    (wav_path, sample_rate)
    """
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ext = Path(input_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # ---- cache check ----
    fhash = file_hash(input_path)
    cache_dir = ensure_dir(os.path.join(output_dir, "cache"))
    cached_wav = os.path.join(cache_dir, f"{fhash}.wav")

    if os.path.exists(cached_wav):
        logger.info("Using cached WAV: %s", cached_wav)
        return cached_wav, target_sr

    logger.info("Loading audio: %s", input_path)

    audio: Optional[np.ndarray] = None
    sr: Optional[int] = None

    # 1) Try soundfile (WAV / FLAC / OGG)
    if ext in (".wav", ".flac", ".ogg"):
        try:
            audio, sr = sf.read(input_path, dtype="float32")
            logger.info("Loaded with soundfile: sr=%d, shape=%s", sr, audio.shape)
        except Exception as exc:
            logger.warning("soundfile failed (%s), falling back", exc)

    # 2) Try librosa (handles MP3 via audioread / ffmpeg)
    if audio is None:
        try:
            import librosa
            audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
            logger.info("Loaded with librosa: sr=%d, samples=%d", sr, len(audio))
        except Exception as exc:
            logger.warning("librosa failed (%s), trying ffmpeg directly", exc)

    # 3) Last-resort: ffmpeg subprocess
    if audio is None:
        audio, sr = _load_with_ffmpeg(input_path, target_sr)

    if audio is None:
        raise RuntimeError(
            f"Cannot decode '{input_path}'.\n"
            "• For MP3 support install ffmpeg: https://ffmpeg.org/download.html\n"
            "• Then make sure 'ffmpeg' is on your PATH."
        )

    # ---- mono ----
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # ---- resample ----
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # ---- truncate ----
    max_samples = int(max_duration_s * sr)
    if len(audio) > max_samples:
        logger.info(
            "Truncating from %.1fs to %.1fs", len(audio) / sr, max_duration_s
        )
        audio = audio[:max_samples]

    # ---- peak normalize ----
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95
        logger.info("Peak-normalized (original peak=%.4f)", peak)

    # ---- save ----
    sf.write(cached_wav, audio, sr, subtype="FLOAT")
    logger.info("Saved processed WAV: %s", cached_wav)
    return cached_wav, sr


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _load_with_ffmpeg(
    input_path: str, target_sr: int
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Decode any audio via an ffmpeg subprocess → WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.error(
            "ffmpeg not found on PATH — cannot decode this format."
        )
        return None, None

    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        cmd = [
            ffmpeg, "-y",
            "-i", input_path,
            "-ar", str(target_sr),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-f", "wav",
            tmp_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            logger.error("ffmpeg error:\n%s", result.stderr[:500])
            return None, None

        audio, sr = sf.read(tmp_path, dtype="float32")
        return audio, sr

    except Exception as exc:
        logger.error("ffmpeg conversion failed: %s", exc)
        return None, None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
