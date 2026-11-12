"""Portable local inference using explicitly selected, checksum-verified assets."""
from pathlib import Path
import json
import re
import numpy as np
import torch

from . import acoustic, events
from .audio import sha256, save_json
from .register import ArticulationHead, RegisterHead, correct_register
from .decoder import decode
from .event_fusion import fuse_event_evidence
from .lyric_inputs import lyric_input
from .midi import midi_bytes
from .beats import estimate_beat_candidates


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def feature_pair(directory):
    directory = Path(directory)
    mix = arrays(directory / 'audio_features.npz')
    vocal = arrays(directory / 'vocal_features.npz')
    times = mix['times_s']
    if times.ndim != 1 or not len(times) or not np.allclose(times, np.arange(len(times)) * .02, atol=1e-8, rtol=0):
        raise ValueError('Features require a nonempty 20ms source-time grid starting at zero')
    if not np.array_equal(times, vocal['times_s']):
        raise ValueError('Mix and vocal feature grids differ')
    for source in (mix, vocal):
        for key, width in [('log_cqt', 84), ('harmonic_salience', 84), ('log_mel', 80), ('rms', None), ('onset_flux', None)]:
            shape = (len(times), width) if width else (len(times),)
            if source[key].shape != shape or not np.isfinite(source[key]).all():
                raise ValueError('Invalid acoustic feature: ' + key)
    for key in ('rmvpe_midi', 'rmvpe_voiced', 'rmvpe_confidence', 'vocal_fullmix_rms_ratio'):
        if vocal[key].shape != (len(times),):
            raise ValueError('Invalid vocal feature length: ' + key)
        if np.isinf(vocal[key]).any() or (key != 'rmvpe_midi' and not np.isfinite(vocal[key]).all()):
            raise ValueError('Invalid vocal evidence: ' + key)
    return mix, vocal


def probabilities(path, times):
    data = arrays(path)
    if 'times_s' not in data or not np.array_equal(data['times_s'], times):
        raise ValueError('Cached probabilities require the identical feature time grid')
    pp, bb = data['pitch'], data['boundaries']
    if pp.shape != (61, len(times)) or bb.shape != (2, len(times)):
        raise ValueError('Expected pitch=(61,T) and boundaries=(2,T)')
    if not all(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in (pp, bb)):
        raise ValueError('Invalid probability values')
    if not np.allclose(pp.sum(0), 1, atol=1e-4):
        raise ValueError('Pitch probabilities must sum to one')
    return pp.astype(np.float32), bb.astype(np.float32)


class Engine:
    """CPU-portable model inference. No scores, annotations or tempo are inputs."""
    def __init__(self, asset_manifest, *, threads=2):
        torch.set_num_threads(threads)
        self.path = Path(asset_manifest).resolve()
        self.assets = json.loads(self.path.read_text(encoding='utf-8'))
        self.verified = {}
        self.base = None
        self.articulation = None
        self.register = None
        self.joint_norm = None
        self.joint_config = None

    def asset(self, section, name):
        try:
            entry = self.assets[section][name]
            path = (self.path.parent / entry['path']).resolve()
            expected = entry['sha256'].lower()
        except KeyError as exc:
            raise ValueError(f'Missing asset {section}.{name}; see docs/assets.example.json') from exc
        if not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError(f'{section}.{name} requires a real SHA-256 checksum')
        if sha256(path) != expected:
            raise ValueError(f'Asset checksum mismatch: {section}.{name}')
        self.verified[f'{section}.{name}'] = expected
        return path

    def load_joint(self):
        if self.articulation is not None:
            return
        state = torch.load(self.asset('joint', 'checkpoint'), map_location='cpu', weights_only=True)
        self.articulation, self.register = ArticulationHead().eval(), RegisterHead().eval()
        self.articulation.load_state_dict(state['articulation'])
        self.register.load_state_dict(state['register'])
        self.joint_norm = arrays(self.asset('joint', 'normalization'))
        self.joint_config = json.loads(self.asset('joint', 'decoder').read_text(encoding='utf-8'))
        validate_norm(self.joint_norm, 164)
        validate_norm(self.joint_norm, 255, 'event_')

    def item(self, feature_directory, cached_probabilities=None, *, joint=True):
        mix, vocal = feature_pair(feature_directory)
        if cached_probabilities:
            pp, bb = probabilities(cached_probabilities, mix['times_s'])
        else:
            if self.base is None:
                self.base = acoustic.MelodyNetwork().eval()
                self.base.load_state_dict(torch.load(self.asset('base', 'checkpoint'), map_location='cpu', weights_only=True))
                self.base_norm = arrays(self.asset('base', 'normalization'))
                validate_norm(self.base_norm, 425)
            pp, bb = acoustic.predict(self.base, acoustic.acoustic_features(mix, vocal),
                                      self.base_norm['mean'], self.base_norm['scale'])
        item = dict(id=Path(feature_directory).parent.name, mix=mix, vocal=vocal, pp=pp, bb=bb)
        if joint:
            self.load_joint()
            item['bb'] = events.previous_boundaries(item, self.articulation, self.joint_norm)
        item['ex'] = events.features(item)
        return item

    def infer(self, feature_directory, *, mode='joint', cached_probabilities=None,
              lyric_directory=None, vocal_audio=None):
        if mode not in ('base', 'joint', 'note-events', 'lyric'):
            raise ValueError('Unknown inference mode: ' + mode)
        item = self.item(feature_directory, cached_probabilities, joint=mode != 'base')
        pp, bb, aux = item['pp'], item['bb'], None
        fusion_audit = None
        if mode in ('note-events', 'lyric'):
            section = 'events' if mode == 'note-events' else 'lyric'
            if mode == 'lyric':
                if lyric_directory is None or vocal_audio is None:
                    raise ValueError('Lyric inference requires --lyrics and --vocal-audio for identity checks')
                verify_lyrics(Path(lyric_directory), Path(vocal_audio))
                feature_metadata = json.loads((Path(feature_directory) / 'vocal_features.json').read_text(encoding='utf-8'))
                if feature_metadata.get('vocal_sha256') != sha256(Path(vocal_audio)):
                    raise ValueError('Vocal features and lyric inputs refer to different audio')
                item['ex'] = np.concatenate([item['ex'], lyric_input(Path(lyric_directory), item['vocal']['times_s'])])
            norm = arrays(self.asset(section, 'normalization'))
            validate_norm(norm, item['ex'].shape[0])
            model = events.NoteEventNetwork(inputs=item['ex'].shape[0]).eval()
            model.load_state_dict(torch.load(self.asset(section, 'checkpoint'), map_location='cpu', weights_only=True))
            pp, bb, aux = events.predict(model, item, norm['mean'], norm['scale'])
            if 'decoder' in self.assets[section]:
                config = json.loads(self.asset(section, 'decoder').read_text(encoding='utf-8'))
                pp, bb, fusion_audit = fuse_event_evidence(pp, bb, aux, item['vocal'], **config)
            notes, audit = events.decode_events(item, pp, bb, self.register, self.joint_norm, aux)
        else:
            threshold = self.joint_config['onset_threshold'] if mode == 'joint' else .67
            notes, audit = decode(pp, bb, mix=item['mix'], vocal=item['vocal'], onset_threshold=threshold, return_audit=True)
            if mode == 'joint':
                notes, changes = correct_register(self.register, item, notes, self.joint_norm['event_mean'],
                                                  self.joint_norm['event_scale'], self.joint_config['pitch_confidence_threshold'])
                audit['register_changes'] = changes
        for note in notes:
            note.setdefault('quantized', False)
            note.setdefault('linguistic_relation', 'unknown')
        metadata = dict(mode=mode, model_files_sha256=dict(self.verified),
                        input_features_sha256={name: sha256(Path(feature_directory) / name) for name in ('audio_features.npz', 'vocal_features.npz')},
                        cached_probabilities_sha256=sha256(Path(cached_probabilities)) if cached_probabilities else None,
                        score_read_at_inference=False, annotations_read_at_inference=False,
                        lyrics_read_at_inference=mode == 'lyric', tempo_used=False,
                        experimental=mode in ('note-events', 'lyric'), musical_accuracy_verified=False)
        return dict(notes=notes, metadata=metadata, audit=audit, pitch=pp, boundaries=bb,
                    auxiliary=aux, times_s=item['mix']['times_s'], fusion_audit=fusion_audit,
                    beat_candidates=estimate_beat_candidates(notes, audio_features=item['mix']))


def validate_norm(norm, channels, prefix=''):
    mean, scale = norm[prefix + 'mean'], norm[prefix + 'scale']
    allowed = ((channels, 1),) if not prefix else ((channels,), (1, channels))
    if mean.shape not in allowed or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid model normalization: ' + prefix)


def verify_lyrics(directory, vocal_audio):
    path = directory / 'lyrics_asr.json'
    asr = json.loads(path.read_text(encoding='utf-8'))
    alignment = json.loads((directory / 'syllable_alignment.json').read_text(encoding='utf-8'))
    if alignment['asr_sha256'] != sha256(path):
        raise ValueError('Lyrics changed after alignment')
    if asr['source_sha256'] != sha256(vocal_audio):
        raise ValueError('Lyrics came from a different vocal recording')
    if 'phonetics_sha256' in alignment and alignment['phonetics_sha256'] != sha256(directory / 'phonetics.npz'):
        raise ValueError('Phonetic evidence changed after alignment')


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / 'predictions.json', dict(notes=result['notes'], **result['metadata']))
    save_json(directory / 'decoder_audit.json', result['audit'])
    save_json(directory / 'beat_candidates.json', result['beat_candidates'])
    if result['fusion_audit'] is not None:
        save_json(directory / 'event_fusion_audit.json', result['fusion_audit'])
    values = {key: result[key] for key in ('pitch', 'boundaries', 'times_s')}
    if result['auxiliary'] is not None:
        values['auxiliary'] = result['auxiliary']
    np.savez_compressed(directory / 'probabilities.npz', **values)
    # 120 is a serialization scale only; timestamps stay in source seconds.
    (directory / 'raw.mid').write_bytes(midi_bytes(result['notes'], 120))
