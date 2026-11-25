"""Prepare score-free audio features for inference or a development demonstration.

This separate wrapper leaves the frozen training-corpus scripts and data intact.
Every record is explicitly ineligible for model fitting. Existing, aligned
source-derived stems may be reused; otherwise frozen BS-Roformer is run locally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_features import ROOT, SR, HOP, MIDI, decode, extract, save_json, sha256
from audio_stems import ASSETS, RVC, decode_stereo, rmvpe_evidence, separate
import numpy as np


def save_features(path, arrays, common, **extra):
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite feature: " + name)
    np.savez_compressed(path, **arrays)
    metadata = dict(common, status="complete", feature_path=str(path), feature_sha256=sha256(path),
                    feature_shapes={name:list(value.shape) for name,value in arrays.items()},
                    feature_dtypes={name:str(value.dtype) for name,value in arrays.items()}, **extra)
    save_json(path.with_suffix(".json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id", default="user_audio")
    parser.add_argument("--split", choices=("inference", "development_demo"), default="inference")
    parser.add_argument("--vocal-stem", type=Path)
    parser.add_argument("--accompaniment-stem", type=Path)
    args = parser.parse_args()
    source, base = args.audio.resolve(), args.output_dir.resolve()
    vocal_input = args.vocal_stem.resolve() if args.vocal_stem else None
    accompaniment_input = args.accompaniment_stem.resolve() if args.accompaniment_stem else None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
        parser.error("ID must contain letters, digits, underscores or hyphens")
    if not source.is_file() or not base.is_relative_to(ROOT / "data"):
        parser.error("Audio must exist and output must be inside repository data/")
    folder = base / args.id
    if folder.exists():
        parser.error("Use a fresh track directory: " + str(folder))
    if accompaniment_input and not vocal_input:
        parser.error("An accompaniment reference requires --vocal-stem")
    if any(path and not path.is_file() for path in (vocal_input, accompaniment_input)):
        parser.error("Supplied stem file does not exist")
    started = time.monotonic()
    audio_dir, feature_dir = folder / "audio", folder / "features"
    audio_dir.mkdir(parents=True)
    feature_dir.mkdir()
    local = audio_dir / ("source" + source.suffix.lower())
    source_hash = sha256(source)
    shutil.copy2(source, local)
    if sha256(local) != source_hash:
        raise ValueError("Copied source hash differs")
    authorization = dict(
        id=args.id, split=args.split, purpose="inference_only", training_eligible=False,
        voice_conversion_training_authorized=False, model_fitting_authorized=False,
        source_sha256=source_hash, sha256=source_hash, original_path=str(source), local_path=str(local),
        authorization_scope="User-requested local melody transcription inference or development demonstration only",
        originals_modified=False, score_used=False, uploads=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    save_json(audio_dir / "provenance.json", authorization)
    save_json(base / (args.id + "_inference_manifest.json"), dict(
        schema_version=1, purpose="inference_only", training_eligible=False, tracks=[dict(
            id=args.id, audio_path=str(source), split=args.split, purpose="inference_only", training_eligible=False)]))
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(local)]))
    save_json(audio_dir / "probe.json", probe)
    fullmix = decode(local)
    full_arrays = extract(fullmix)
    common = dict(authorization, script_sha256=sha256(Path(__file__)), source_path=str(local),
                  feature_script_sha256=sha256(Path(__file__).with_name("audio_features.py")),
                  frames=len(full_arrays["times_s"]), decoded_samples=len(fullmix), duration_s=len(fullmix)/SR,
                  sample_rate=SR, hop_samples=HOP, hop_s=HOP/SR, midi_min=int(MIDI[0]), midi_max=int(MIDI[-1]),
                  time_origin_s=0.0, time_semantics="Centered windows; no source trimming; exact 20ms timestamps")
    save_features(feature_dir / "audio_features.npz", full_arrays, common, schema_version="fullmix_cpu_20ms_v1")
    print(json.dumps(dict(id=args.id, stage="fullmix_complete", frames=common["frames"])), flush=True)
    assets = {relative:sha256(RVC / relative) for relative in ASSETS}
    if assets != ASSETS:
        raise ValueError("Frozen inference model hash mismatch")
    sys.path.insert(0, str(RVC))
    os.chdir(RVC)
    sys.argv = [sys.argv[0]]
    import torch
    import soundfile as sf
    import librosa
    import types
    inference_namespace = types.ModuleType("infer")
    inference_namespace.__path__ = [str(RVC / "infer")]
    sys.modules["infer"] = inference_namespace
    from infer.rmvpe import RMVPE

    torch.set_num_threads(2)
    stem_details = {}
    if vocal_input:
        vocal = decode(vocal_input)
        if abs(len(vocal)-len(fullmix)) > SR // 10:
            raise ValueError("Existing vocal duration differs from source by over100ms")
        stem_details.update(reused_vocal_path=str(vocal_input), reused_vocal_sha256=sha256(vocal_input),
                            method="Existing source-derived vocal stem decoded by FFmpeg with source-matched mono gain")
        if accompaniment_input:
            backing_check = decode(accompaniment_input)
            if len(vocal) != len(backing_check) or abs(len(backing_check)-len(fullmix)) > SR // 10:
                raise ValueError("Existing stem durations do not match")
            count = min(len(fullmix),len(vocal),len(backing_check))
            residual = fullmix[:count].astype(np.float64) - (vocal[:count].astype(np.float64)+backing_check[:count].astype(np.float64))
            reconstruction_error = float(np.linalg.norm(residual)/max(np.linalg.norm(fullmix[:count].astype(np.float64)),1e-12))
            if reconstruction_error > 0.01:
                raise ValueError("Reused stems fail source timing/amplitude reconstruction check")
            stem_details.update(reused_accompaniment_path=str(accompaniment_input),
                                reused_accompaniment_sha256=sha256(accompaniment_input),
                                reused_stem_reconstruction_relative_rms_error=reconstruction_error)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Local BS-Roformer stem preparation requires CUDA")
        from tools.uvr5.bsroformer import Roformer_Loader
        asset = RVC / "assets/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        separator = Roformer_Loader(str(asset), str(asset.with_suffix(".yaml")), torch.device("cuda:0"), True)
        separator.config["inference"]["batch_size"] = 1
        separator.config["audio"]["chunk_size"] = 88200
        separator.model.eval()
        mix = decode_stereo(local)
        raw_vocal, chunks = separate(mix, separator, torch, args.id)
        vocal = librosa.resample(raw_vocal * np.sqrt(2.0), orig_sr=44100, target_sr=SR)
        stem_details.update(method="Frozen BS-Roformer,12s cores/1s context,FFmpeg-aligned mono gain", chunks=chunks)
        del raw_vocal, mix, separator
        torch.cuda.empty_cache()
    vocal = librosa.util.fix_length(vocal, size=len(fullmix)).astype(np.float32)
    accompaniment = fullmix-vocal
    vocal_path, accompaniment_path = audio_dir / "vocals.wav", audio_dir / "accompaniment.wav"
    sf.write(vocal_path,vocal,SR,subtype="FLOAT")
    sf.write(accompaniment_path,accompaniment,SR,subtype="FLOAT")
    stem_metadata = dict(authorization, assets=assets, **stem_details,
                         vocal_sha256=sha256(vocal_path), accompaniment_sha256=sha256(accompaniment_path),
                         sample_rate=SR, samples=len(vocal), ffmpeg_mono_gain_aligned=True)
    save_json(audio_dir / "stems.json",stem_metadata)
    vocal_arrays = extract(vocal)
    if not np.array_equal(vocal_arrays["times_s"],full_arrays["times_s"]):
        raise ValueError("Fullmix and vocal time grids differ")
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    pitch_model=RMVPE(str(RVC / "assets/rmvpe/rmvpe.pt"),is_half=device.startswith("cuda"),device=device)
    vocal_arrays.update(rmvpe_evidence(vocal,pitch_model,len(vocal_arrays["times_s"])))
    vocal_arrays["vocal_fullmix_rms_ratio"] = (vocal_arrays["rms"]/np.maximum(full_arrays["rms"],1e-6)).astype(np.float32)
    save_features(feature_dir / "vocal_features.npz",vocal_arrays,common,
                  schema_version="fullmix_cpu_20ms_v1_plus_rmvpe",assets=assets,
                  vocal_source_path=str(vocal_path),vocal_sha256=stem_metadata["vocal_sha256"],
                  verification=dict(all_finite=True,exact_fullmix_time_grid=True),
                  pretrained_models_frozen=True,neural_training=False)
    verification=dict(id=args.id,split=args.split,training_eligible=False,original_unchanged=sha256(source)==source_hash,
                      frames=common["frames"],duration_s=common["duration_s"],all_finite=True,
                      exact_fullmix_vocal_time_grid=True,assets_unchanged=all(sha256(RVC/k)==v for k,v in ASSETS.items()),
                      elapsed_s=time.monotonic()-started,score_used=False,neural_training=False)
    save_json(folder / "audio_verification.json",verification)
    print(json.dumps(dict(verification,feature_directory=str(feature_dir))),flush=True)


if __name__=="__main__":
    main()
