"""Manifest-driven note-model training with train-only normalization and explicit splits.

This runner ports the residual event objective, not the historical experiment's
entire sampling/selection schedule. Published experiment metrics are not claims
about a new run. No automatic lyric hypotheses are converted to note labels.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from scipy.ndimage import maximum_filter1d
from . import acoustic, events
from .audio import save_json, sha256
from .engine import Engine, feature_pair, verify_lyrics
from .lyric_inputs import lyric_input
from .targets import reviewed_targets
from .decoder import decode
from .evaluation import evaluate_notes


def selected_records(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    records, identities, ids = [], {}, set()
    for record in document['tracks']:
        if record['id'] in ids or record['split'] not in ('train', 'validation', 'test'):
            raise ValueError('IDs must be unique and splits must be train/validation/test')
        ids.add(record['id'])
        source = record['source_sha256']
        if len(source) != 64 or any(c not in '0123456789abcdef' for c in source.lower()):
            raise ValueError('Each track requires its actual source SHA-256')
        source = source.lower()
        if source in identities and identities[source] != record['split']:
            raise ValueError('The same source audio appears in multiple splits')
        identities[source] = record['split']
        if record['split'] == 'test':
            continue  # Do not open held-out features, lyrics or labels while fitting.
        r = dict(record)
        for key in ('features', 'alignment', 'probabilities', 'lyrics', 'vocal_audio'):
            if key in r:
                r[key] = (path.parent / r[key]).resolve()
        records.append(r)
    if not {'train', 'validation'} <= {r['split'] for r in records}:
        raise ValueError('Separate train and validation songs are required')
    return records


def _load(records, engine, mode):
    items = []
    for record in records:
        if mode == 'base':
            mix, vocal = feature_pair(record['features'])
            item = dict(mix=mix, vocal=vocal, ex=acoustic.acoustic_features(mix, vocal))
        else:
            item = engine.item(record['features'], record.get('probabilities'))
            if mode in ('lyric', 'lyric-control'):
                verify_lyrics(record['lyrics'], record['vocal_audio'])
                lyric = lyric_input(record['lyrics'], item['vocal']['times_s'])
                item['ex'] = np.concatenate([item['ex'], lyric if mode == 'lyric' else np.zeros_like(lyric)])
        alignment = json.loads(record['alignment'].read_text(encoding='utf-8'))
        if alignment.get('source_sha256') != record['source_sha256']:
            raise ValueError('Alignment source hash differs from the manifest')
        for name in ('audio_features.npz', 'vocal_features.npz'):
            metadata = json.loads((record['features'] / name).with_suffix('.json').read_text(encoding='utf-8'))
            if metadata.get('source_sha256') != record['source_sha256'] or metadata.get('feature_sha256') != sha256(record['features'] / name):
                raise ValueError('Features and labels must refer to the same verified source')
        item.update(id=record['id'], split=record['split'], alignment=alignment)
        item['target'] = reviewed_targets(alignment, item['ex'].shape[1])
        if mode != 'base':
            item['event_target'] = events.targets(item)
        if not np.isfinite(item['ex']).all() or not np.any(item['target']['weights'] > 0):
            raise ValueError('Each training/validation track needs finite features and some supervised notes')
        items.append(item)
    return items


def fit(manifest, output, *, mode='events', assets=None, epochs=12, seed=20261112,
        learning_rate=.00015, batch_size=8, chunk_length=384):
    if mode not in ('base', 'events', 'lyric', 'lyric-control'):
        raise ValueError('Unknown training mode')
    if epochs < 1 or batch_size < 1 or chunk_length < 160 or learning_rate <= 0:
        raise ValueError('Invalid training parameters; chunks must exceed the receptive field')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Training requires a new versioned directory')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    engine = Engine(assets) if mode != 'base' else None
    records = selected_records(manifest)
    items = _load(records, engine, mode)
    train = [i for i in items if i['split'] == 'train']
    validation = [i for i in items if i['split'] == 'validation']
    stack = np.concatenate([i['ex'] for i in train], axis=1)
    mean = stack.mean(1, keepdims=True).astype(np.float32)
    scale = np.maximum(stack.std(1, keepdims=True), .05).astype(np.float32)
    del stack
    model = acoustic.MelodyNetwork() if mode == 'base' else events.NoteEventNetwork(inputs=mean.shape[0])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.015)
    segments = []
    for item in train:
        for start in range(0, item['ex'].shape[1], chunk_length // 2):
            stop = min(start + chunk_length, item['ex'].shape[1])
            if stop - start > 128 and np.count_nonzero(item['target']['weights'][start + 64:stop - 64]) > 0:
                segments.append((item, start))
    if not segments:
        raise ValueError('No supervised interiors after 64-frame boundary masking; supply longer labeled phrases')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'normalization.npz', mean=mean, scale=scale)
    save_json(output / 'provenance.json', dict(mode=mode, seed=seed,
              train_ids=[i['id'] for i in train], validation_ids=[i['id'] for i in validation],
              test_data_loaded=False, normalization_fit_on='train_only', lyrics_are_targets=False,
              timing_quantization=False, manifest_sha256=sha256(Path(manifest)),
              training_recipe='portable masked note-event objective; historical sampling schedule not replicated'))
    best, selected_epoch, steps, history = -1., 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(segments)
        losses = []
        for offset in range(0, len(segments), batch_size):
            rows = []
            for item, start in segments[offset:offset + batch_size]:
                stop = min(start + chunk_length, item['ex'].shape[1])
                sl = slice(start, stop)
                target = item['target']
                pw = (target['weights'][sl] * target['register'][sl]).copy()
                pw[:64] = pw[-64:] = 0
                if mode == 'base':
                    y = target['boundary'][:, sl]
                    w = np.broadcast_to(target['boundary_weight'][sl], y.shape).copy()
                else:
                    y = item['event_target']['events'][:, sl]
                    w = item['event_target']['event_weights'][:, sl].copy()
                w[:, :64] = w[:, -64:] = 0
                values = [(item['ex'][:, sl] - mean) / scale, target['pitch'][sl], pw, y, w]
                if mode != 'base':
                    values.extend([np.log(np.maximum(item['pp'][:, sl], 1e-7)), events.logit(item['bb'][:, sl])])
                padding = chunk_length - (stop - start)
                rows.append([np.pad(a, ((0, 0), (0, padding)) if a.ndim == 2 else (0, padding)) for a in values])
            batch = [torch.from_numpy(np.stack(a)) for a in zip(*rows)]
            x, py, pw, y, w = batch[:5]
            optimizer.zero_grad(set_to_none=True)
            if mode == 'base':
                p, boundary = model(x)
                logits, correction = boundary, None
                channel_weights = torch.tensor([.6, .25])
                positive_weights = torch.tensor([4., 2.])
            else:
                p, boundary, auxiliary, correction = model(x, batch[5], batch[6])
                logits = torch.cat([boundary, auxiliary], 1)
                channel_weights = torch.tensor([.6, .25, .2, .2, .2])
                positive_weights = torch.tensor([4., 2., 4., 4., 1.])
            pitch_loss = (nn.functional.cross_entropy(p, py, reduction='none') * pw).sum() / pw.sum().clamp_min(1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y,
                    pos_weight=positive_weights.view(1, -1, 1), reduction='none')
            event_loss = (bce * w).sum((0, 2)) / w.sum((0, 2)).clamp_min(1)
            loss = pitch_loss + (event_loss * channel_weights).sum()
            if correction is not None:
                loss = loss + .025 * correction.square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        model.eval()
        scores = []
        for item in validation:
            if mode == 'base':
                pp, bb = acoustic.predict(model, item['ex'], mean, scale)
                notes = decode(pp, bb, mix=item['mix'], vocal=item['vocal'], onset_threshold=.67)
            else:
                pp, bb, aux = events.predict(model, item, mean, scale)
                notes, _ = events.decode_events(item, pp, bb, engine.register, engine.joint_norm, aux)
            reference = [n for n in item['alignment']['notes'] if n['supervise'] and n.get('register_supervision_weight', 1) > .5 and 36 <= n['midi'] <= 95]
            if not reference:
                raise ValueError('Validation needs some notes with supervised register')
            known = maximum_filter1d(((item['target']['weights'] > 0) & (item['target']['register'] > .5)).astype(int), 11) > 0
            scoped = [n for n in notes if known[min(len(known) - 1, max(0, round(n['start_s'] / .02)))]]
            metric = evaluate_notes(reference, scoped, tolerances=(.1,))
            scores.append(metric['sensitivity'][0]['exact_pitch_onset']['f1'])
        score = float(np.mean(scores))
        if score > best:
            best, selected_epoch = score, epoch
            torch.save(model.state_dict(), output / 'checkpoint.pt')
        history.append(dict(epoch=epoch, optimizer_steps=steps, loss=float(np.mean(losses)),
                            validation_exact_pitch_onset_f1=score, selected_epoch=selected_epoch))
        save_json(output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    selected = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    delta = float(np.sqrt(sum(float(((selected[key] - value) ** 2).sum()) for key, value in initial.items())))
    save_json(output / 'selection.json', dict(selected_epoch=selected_epoch, optimizer_steps=steps,
              parameter_delta_l2=delta, validation_f1=best, promoted=False,
              checkpoint_sha256=sha256(output / 'checkpoint.pt'), normalization_sha256=sha256(output / 'normalization.npz')))
