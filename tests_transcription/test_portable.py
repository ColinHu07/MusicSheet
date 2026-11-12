import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from musicsheet.audio import save_json
from musicsheet.engine import Engine, probabilities, feature_pair, validate_norm
from musicsheet.notation import export_score
from musicsheet.rhythm import read_musicxml, spell_pitch, quantize_rhythm
from musicsheet.training import selected_records, fit
from musicsheet.cli import main


def make_features(folder, length=200, source_hash='a' * 64):
    folder.mkdir(parents=True)
    t = np.arange(length) * .02
    common = dict(times_s=t, log_cqt=np.zeros((length, 84), np.float32),
                  harmonic_salience=np.zeros((length, 84), np.float32), log_mel=np.zeros((length, 80), np.float32),
                  rms=np.ones(length, np.float32) * .1, onset_flux=np.zeros(length, np.float32))
    vocal = dict(common, rmvpe_midi=np.full(length, 60., np.float32),
                 rmvpe_voiced=np.ones(length, np.float32), rmvpe_confidence=np.ones(length, np.float32),
                 vocal_fullmix_rms_ratio=np.ones(length, np.float32))
    for name, data in [('audio', common), ('vocal', vocal)]:
        path = folder / (name + '_features.npz')
        np.savez_compressed(path, **data)
        save_json(path.with_suffix('.json'), dict(source_sha256=source_hash, feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return t


def test_features_reject_shifted_grid(tmp_path):
    folder = tmp_path / 'features'
    t = make_features(folder)
    path = folder / 'vocal_features.npz'
    with np.load(path) as z:
        data = dict(z)
    data['times_s'] = t + .02
    np.savez(path, **data)
    with pytest.raises(ValueError, match='grids differ'):
        feature_pair(folder)


def test_cached_probabilities_reject_wrong_time_or_distribution(tmp_path):
    t = np.arange(10) * .02
    path = tmp_path / 'probabilities.npz'
    np.savez(path, times_s=t + .02, pitch=np.ones((61, 10)) / 61, boundaries=np.zeros((2, 10)))
    with pytest.raises(ValueError, match='identical'):
        probabilities(path, t)
    np.savez(path, times_s=t, pitch=np.ones((61, 10)), boundaries=np.zeros((2, 10)))
    with pytest.raises(ValueError, match='sum to one'):
        probabilities(path, t)


def test_changed_asset_is_rejected_before_deserialization(tmp_path):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'original')
    manifest = tmp_path / 'assets.json'
    manifest.write_text(json.dumps(dict(base=dict(checkpoint=dict(path='weights.pt', sha256=hashlib.sha256(path.read_bytes()).hexdigest())))))
    engine = Engine(manifest)
    assert engine.asset('base', 'checkpoint') == path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        engine.asset('base', 'checkpoint')


def test_norm_rejects_zero_scale():
    with pytest.raises(ValueError, match='normalization'):
        validate_norm(dict(mean=np.zeros((4, 1)), scale=np.zeros((4, 1))), 4)


def test_triplets_and_repeated_pitches_survive_export(tmp_path):
    notes = [dict(start_s=i / 6, end_s=(i + 1) / 6, midi=60) for i in range(6)]
    original = copy.deepcopy(notes)
    path = tmp_path / 'predictions.json'
    path.write_text(json.dumps(dict(notes=notes)))
    out = tmp_path / 'score'
    export_score(path, out, bpm=120)
    parsed = read_musicxml(out / 'melody.musicxml')['parts']['P1']
    assert len(parsed) == len(notes)
    assert [n['midi'] for n in parsed] == [60] * 6
    assert abs(parsed[-1]['end_s'] - 1.) < 1e-6
    assert '<actual-notes>3</actual-notes>' in (out / 'melody.musicxml').read_text()
    assert json.loads(path.read_text())['notes'] == original
    with pytest.raises(FileExistsError):
        export_score(path, out, bpm=120)


def test_real_staggered_pattern_is_retained():
    notes = [dict(start_s=a, end_s=b, midi=p) for a, b, p in [(0, .125, 72), (.125, .375, 72), (.375, .5, 72)]]
    output, audit = quantize_rhythm(notes, 120)
    assert [n['tick_end'] - n['tick_start'] for n in output] == [12, 24, 12]
    assert audit['attacks_merged'] == 0


@pytest.mark.parametrize('fifths', range(-7, 8))
def test_key_spelling_preserves_pitch(fifths):
    pc = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
    for midi in range(36, 96):
        letter, alter, octave = spell_pitch(midi, fifths)
        assert 12 * (octave + 1) + pc[letter] + alter == midi
    if fifths == -5:
        assert spell_pitch(61, fifths) == ('D', -1, 4)


def test_split_overlap_rejected_and_test_paths_never_opened(tmp_path):
    path = tmp_path / 'manifest.json'
    tracks = [dict(id='a', split='train', source_sha256='a' * 64),
              dict(id='b', split='validation', source_sha256='b' * 64),
              dict(id='c', split='test', source_sha256='c' * 64, features='does-not-exist', alignment='does-not-exist')]
    path.write_text(json.dumps(dict(tracks=tracks)))
    assert len(selected_records(path)) == 2
    tracks[2]['source_sha256'] = 'a' * 64
    path.write_text(json.dumps(dict(tracks=tracks)))
    with pytest.raises(ValueError, match='multiple splits'):
        selected_records(path)


@pytest.mark.parametrize('mode', ['base', 'events'])
def test_training_updates_weights_and_preserves_test_holdout(tmp_path, mode):
    records = []
    for identifier, split, digest in [('train', 'train', 'a' * 64), ('val', 'validation', 'b' * 64)]:
        folder = tmp_path / identifier
        times = make_features(folder / 'features', source_hash=digest)
        pp = np.full((61, len(times)), .01 / 60, np.float32)
        pp[25] = .99
        np.savez(folder / 'probabilities.npz', times_s=times, pitch=pp,
                 boundaries=np.zeros((2, len(times)), np.float32))
        alignment = dict(source_sha256=digest, notes=[dict(start_s=.5, end_s=3.5, midi=60,
                            supervise=True, alignment_confidence=.9, register_supervision_weight=1.)])
        save_json(folder / 'alignment.json', alignment)
        records.append(dict(id=identifier, split=split, source_sha256=digest,
                            features=identifier + '/features', alignment=identifier + '/alignment.json',
                            probabilities=identifier + '/probabilities.npz'))
    records.append(dict(id='heldout', split='test', source_sha256='c' * 64,
                        features='must-not-open', alignment='must-not-open'))
    manifest = tmp_path / 'manifest.json'
    save_json(manifest, dict(tracks=records))
    out = tmp_path / 'model'
    assets = None
    if mode == 'events':
        import torch
        from musicsheet.register import ArticulationHead, RegisterHead
        joint = tmp_path / 'joint'
        joint.mkdir()
        torch.save(dict(articulation=ArticulationHead().state_dict(), register=RegisterHead().state_dict()), joint / 'checkpoint.pt')
        np.savez(joint / 'normalization.npz', mean=np.zeros((164, 1), np.float32), scale=np.ones((164, 1), np.float32),
                 event_mean=np.zeros(255, np.float32), event_scale=np.ones(255, np.float32))
        save_json(joint / 'decoder.json', dict(onset_threshold=.67, pitch_confidence_threshold=.99))
        files = dict(checkpoint='checkpoint.pt', normalization='normalization.npz', decoder='decoder.json')
        spec = {key: dict(path=str(joint / name), sha256=hashlib.sha256((joint / name).read_bytes()).hexdigest()) for key, name in files.items()}
        assets = tmp_path / 'assets.json'
        save_json(assets, dict(joint=spec))
    fit(manifest, out, mode=mode, assets=assets, epochs=1, batch_size=1, chunk_length=192)
    result = json.loads((out / 'selection.json').read_text())
    assert result['optimizer_steps'] > 0 and result['parameter_delta_l2'] > 0
    assert not json.loads((out / 'provenance.json').read_text())['test_data_loaded']


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert 'infer' in capsys.readouterr().out
