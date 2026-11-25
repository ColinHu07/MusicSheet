"""Add authorized, local vocal evidence using existing frozen separation/F0 models.

No RVC voice model is loaded or trained. Full-mix features remain the primary
evidence for score-covered instrumental introductions. GPU inference is bounded
to 12-second cores with one-second context and all outputs retain source timing.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  RVC_CUDA_GRAPH="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

import numpy as np

ROOT = Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", Path(__file__).resolve().parents[3])).resolve()
RVC = ROOT / "vendor/rvc-2.3.260718/RVC20260718Nvidia50x0"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_features import HOP, SR, decode, extract, save_json, sha256

ASSETS = {
    "assets/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt":
        "414b7ea8e3f57485282886f0bae273a898e02b9ca14524cdefa0fb9f5784e4d1",
    "assets/rmvpe/rmvpe.pt":
        "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
}


def decode_stereo(path: Path) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(path),
        "-vn", "-ac", "2", "-ar", "44100", "-f", "f32le", "pipe:1",
    ])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()


def separate(mix, separator, torch, source_id):
    rate, core, context = 44100, 12 * 44100, 44100
    vocal = np.empty(mix.shape[1], dtype=np.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, mix.shape[1], core):
            end = min(start + core, mix.shape[1])
            left, right = max(0, start - context), min(mix.shape[1], end + context)
            chunk = mix[:, left:right].copy()
            gain = min(1.0, .95 / max(float(np.max(np.abs(chunk))), 1e-9))
            with open(os.devnull, "w") as progress_sink, contextlib.redirect_stderr(progress_sink):
                result = separator.demix_track(separator.model, torch.from_numpy(chunk * gain),
                                               torch.device("cuda:0"))["vocals"]
            result = np.asarray(result, dtype=np.float32) / gain
            if result.shape != chunk.shape or not np.isfinite(result).all():
                raise ValueError("Invalid separator output")
            vocal[start:end] = result[:, start-left:end-left].mean(axis=0)
            chunks.append(dict(start_s=start/rate, end_s=end/rate, gain=gain))
            if start == 0 or (len(chunks) % 5 == 0) or end == mix.shape[1]:
                print(json.dumps(dict(id=source_id, stage="separating", seconds=round(end/rate, 2))), flush=True)
    return vocal, chunks


def rmvpe_evidence(vocal, model, frame_count):
    hz = np.zeros(frame_count, np.float32)
    confidence = np.zeros(frame_count, np.float32)
    # Integer-second context boundaries preserve the original 10 ms RMVPE grid.
    for start in range(0, len(vocal), 12 * SR):
        end = min(start + 12 * SR, len(vocal))
        left, right = max(0, start - SR), min(len(vocal), end + SR)
        hidden = model.mel2hidden(model.extract_mel(vocal[left:right], center=True))
        hidden = hidden.squeeze(0).detach().float().cpu().numpy()
        f0 = model.decode(hidden, thred=0.03)
        first = (start + HOP - 1) // HOP
        last = min(frame_count, (end + HOP - 1) // HOP)
        frame_ids = np.arange(first, last)
        local_ids = ((frame_ids * HOP - left) // 160).astype(int)
        if local_ids[-1] >= len(f0):
            raise ValueError("RMVPE frame grid overrun")
        hz[frame_ids] = f0[local_ids]
        confidence[frame_ids] = hidden[local_ids].max(axis=1)
    midi = np.zeros_like(hz)
    voiced = hz > 0
    midi[voiced] = 69.0 + 12.0 * np.log2(hz[voiced] / 440.0)
    return dict(rmvpe_f0_hz=hz, rmvpe_confidence=confidence, rmvpe_midi=midi,
                rmvpe_voiced=voiced.astype(np.float32))
