"""Local full-mix audio registration and score-independent transcription features.

The caller supplies a manifest of explicitly authorized audio/PDF pairs. This
script copies audio without modifying the originals and never trains a neural
model, opens a microphone, or reads score-derived labels. It uses CPU DSP only.
Run with the repository's existing bundled DSP Python environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SR = 16000
HOP = 320
MIDI = np.arange(24, 108, dtype=np.int16)
SCHEMA_VERSION = "fullmix_cpu_20ms_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1",
    ])
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if len(audio) < SR or not np.isfinite(audio).all():
        raise ValueError("Audio must contain at least one second of finite samples")
    return audio


def extract(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Return centered, time-major features on one exact 20 ms grid.

    CQT and STFT use centered windows (including symmetric edge padding), so
    frame t describes audio near t*HOP/SR. The final timestamp is strictly
    within the decoded duration. All amplitudes retain absolute scale; learned
    feature normalization must be fit on training songs only.
    """
    import librosa
    from scipy.ndimage import median_filter

    count = int(np.ceil(len(audio) / HOP))
    times = np.arange(count, dtype=np.float64) * (HOP / SR)
    cqt = np.abs(librosa.cqt(
        y=audio, sr=SR, hop_length=HOP, fmin=float(librosa.midi_to_hz(MIDI[0])),
        n_bins=len(MIDI), bins_per_octave=12, tuning=0.0, pad_mode="reflect",
    )).astype(np.float32)[:, :count]
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=HOP,
                                   center=True, pad_mode="reflect")).astype(np.float32)[:, :count]
    mel = librosa.feature.melspectrogram(S=spectrum ** 2, sr=SR, n_fft=2048,
                                        n_mels=80, fmin=30.0, fmax=7900.0)
    # Local pitch-axis subtraction suppresses broadband energy. The weighted
    # harmonic comb supplements direct CQT energy; neither is ground truth.
    whitened = np.maximum(cqt - 0.60 * median_filter(cqt, size=(7, 1), mode="nearest"), 0)
    salience = np.zeros_like(cqt)
    denominators = np.zeros((len(MIDI), 1), dtype=np.float32)
    bins = np.arange(len(MIDI))
    for harmonic in range(1, 9):
        locations = bins + 12.0 * np.log2(harmonic)
        left = np.floor(locations).astype(int)
        fraction = locations - left
        valid = left + 1 < len(MIDI)
        weight = 1.0 / harmonic
        salience[valid] += weight * (
            (1.0 - fraction[valid, None]) * whitened[left[valid]] +
            fraction[valid, None] * whitened[left[valid] + 1]
        )
        denominators[valid] += weight
    # Include the final CQT pitch (which otherwise has no interpolation pair).
    salience[-1] = whitened[-1]
    denominators[-1] = 1.0
    salience /= np.maximum(denominators, 1e-6)
    chroma = np.zeros((12, count), dtype=np.float32)
    for index, midi in enumerate(MIDI):
        chroma[int(midi) % 12] += cqt[index]
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
    log_mel = np.log1p(10.0 * mel)
    onset_flux = np.concatenate((np.zeros(1, dtype=np.float32),
                                np.maximum(np.diff(log_mel, axis=1), 0).mean(axis=0)))
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP,
                              center=True, pad_mode="reflect")[0, :count]
    result = {
        "times_s": times,
        "midi": MIDI.copy(),
        "cqt_magnitude": cqt.T.astype(np.float32),
        "log_cqt": np.log1p(10.0 * cqt).T.astype(np.float32),
        "harmonic_salience": np.log1p(10.0 * salience).T.astype(np.float32),
        "log_mel": log_mel.T.astype(np.float32),
        "chroma": chroma.T.astype(np.float32),
        "rms": rms.astype(np.float32),
        "onset_flux": onset_flux.astype(np.float32),
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite feature: " + name)
        if name != "midi" and len(value) != count:
            raise ValueError("Frame mismatch: " + name)
    return result
