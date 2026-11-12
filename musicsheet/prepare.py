"""Prepare features from aligned local stems and an independently computed F0 track."""
from pathlib import Path
import numpy as np
from .audio import decode, extract, save_json, sha256
from .engine import arrays


def prepare(audio_path, vocal_path, pitch_path, output):
    audio_path, vocal_path, pitch_path, output = map(Path, (audio_path, vocal_path, pitch_path, output))
    if output.exists():
        raise FileExistsError('Use a fresh feature directory')
    mix, vocal = decode(audio_path), decode(vocal_path)
    if len(mix) != len(vocal):
        raise ValueError('Stems must preserve the full source timeline and sample count; do not trim leading silence')
    full_features, vocal_features = extract(mix), extract(vocal)
    track = arrays(pitch_path)
    if not np.array_equal(track['times_s'], full_features['times_s']):
        raise ValueError('Pitch evidence must use the exact 20ms source-time grid')
    for key in ('rmvpe_f0_hz', 'rmvpe_midi', 'rmvpe_confidence', 'rmvpe_voiced'):
        a = track[key]
        if a.shape != full_features['times_s'].shape or np.isinf(a).any() or (key != 'rmvpe_midi' and not np.isfinite(a).all()):
            raise ValueError('Invalid pitch evidence: ' + key)
        vocal_features[key] = a.astype(np.float32)
    vocal_features['vocal_fullmix_rms_ratio'] = (vocal_features['rms'] / np.maximum(full_features['rms'], 1e-6)).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('audio', full_features), ('vocal', vocal_features)]:
        path = output / (name + '_features.npz')
        np.savez_compressed(path, **values)
        save_json(path.with_suffix('.json'), dict(feature_sha256=sha256(path), source_sha256=sha256(audio_path),
                 vocal_sha256=sha256(vocal_path), pitch_track_sha256=sha256(pitch_path),
                 sample_rate=16000, hop_s=.02, score_read=False, audio_uploaded=False,
                 stem_alignment='Caller supplies aligned stems; equal length is checked, content identity is not inferred'))
