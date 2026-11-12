"""Unquantized note-event residual model; experimental, not promoted."""
import numpy as np
import torch
from torch import nn
from .acoustic import ChannelNorm
from .targets import reviewed_targets
from .register import ArticulationHead, RegisterHead, features as old_features, event_features, SHIFTS
from .decoder import decode
HOP = .02
HEADS = ["attack", "ending", "same_pitch_reattack", "pitch_transition", "continuation"]

def logit(p):
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def previous_boundaries(item, art, norm):
    x = old_features(item); old = logit(item['bb'][0]); chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[1], 2048):
            a, b = max(0, start-40), min(x.shape[1], start+2088)
            y, _ = art(torch.from_numpy(((x[:, a:b]-norm['mean'])/norm['scale'])[None]),
                       torch.from_numpy(old[None, a:b]))
            chunks.append(torch.sigmoid(y)[0, start-a:start-a+min(2048, x.shape[1]-start)].numpy())
    out = item['bb'].copy(); out[0] = np.concatenate(chunks)
    return out


def features(item):
    v = item['vocal']; f = np.nan_to_num(v['rmvpe_midi'], nan=60.)
    extra = np.stack([(f-60)/24, v['rmvpe_voiced'],
                      np.nan_to_num(v['rmvpe_confidence']), item['pp'][0]])
    return np.concatenate([old_features(item), np.log(np.maximum(item['pp'], 1e-5))/5, extra]).astype(np.float32)


class NoteEventNetwork(nn.Module):
    def __init__(self, inputs=229, hidden=48):
        super().__init__()
        self.input = nn.Conv1d(inputs, hidden, 5, padding=2)
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2*d, dilation=d),
                       ChannelNorm(hidden), nn.GELU(), nn.Dropout(.12)) for d in (1, 2, 4, 8, 16)])
        self.output = nn.Conv1d(hidden, 66, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, pitch_log, boundary_log):
        h = nn.functional.gelu(self.input(x))
        for layer in self.layers:
            h = h+layer(h)
        raw = self.output(h)
        correction = 4*torch.tanh(raw[:, :63]/4)
        return pitch_log+correction[:, :61], boundary_log+correction[:, 61:63], raw[:, 63:], correction


def targets(item, regions=()):
    """Silver note labels; unknown spans are masked, never fabricated rests."""
    t = np.arange(item['pp'].shape[1])*HOP
    base = reviewed_targets(item['alignment'], len(t))
    y = np.zeros((5, len(t)), np.float32); w = np.zeros_like(y)
    y[:2] = base['boundary']; w[:2] = base['boundary_weight']*.3
    known = base['weights'] > 0
    w[2:] = base['weights'][None]*.3
    notes = item['alignment']['notes']
    for j, note in enumerate(notes):
        if not note['supervise'] or not 36 <= note['midi'] <= 95:
            continue
        a, b = note['start_s'], note['end_s']
        # A brief consonant energy dip inside an accepted note is continuation.
        y[4, (t >= a+.06) & (t < b-.06)] = 1
        if j and notes[j-1]['supervise'] and a-notes[j-1]['end_s'] <= .5:
            channel = 2 if note['midi'] == notes[j-1]['midi'] else 3
            pulse = np.exp(-.5*((t-a)/.04)**2); pulse[abs(t-a) > .12] = 0
            y[channel] = np.maximum(y[channel], pulse)
    # Reviewed attack counts have approximate acoustic times, not phone times.
    for region in regions:
        if region['track'] != item['id']:
            continue
        scope = (t >= region['events'][0]['reference_estimate_s']-.08) & (t <= region['events'][-1]['reference_estimate_s']+.1)
        y[[0, 2, 3], :] *= 1  # independent targets; no silent/rest inference
        for channel in (0, 2, 3):
            y[channel, scope] = 0; w[channel, scope] = 2
        # Approximate reviewed boundaries supersede incompatible silver hold labels.
        w[4, scope] = 0
        for j, event in enumerate(region['events']):
            if not event['acoustic_supported']:
                w[:, abs(t-event['reference_estimate_s']) <= .18] = 0
                continue
            pulse = np.exp(-.5*((t-event['start_s'])/.035)**2)
            y[0, scope] = np.maximum(y[0, scope], pulse[scope])
            if j:
                channel = 2 if event['midi'] == region['events'][j-1]['midi'] else 3
                y[channel, scope] = np.maximum(y[channel, scope], pulse[scope])
    return dict(events=y, event_weights=w, pitch=base['pitch'],
                pitch_weights=base['weights'], register=base['register'], known=known)


def predict(network, item, mean, scale):
    x = item['ex']; pp, bb, aux = [], [], []
    network.eval()
    with torch.no_grad():
        for start in range(0, x.shape[1], 2048):
            # Receptive-field radius 64; retain larger context for all outputs.
            a, b = max(0, start-80), min(x.shape[1], start+2128)
            p, q, r, _ = network(torch.from_numpy(((x[:, a:b]-mean)/scale)[None]),
                    torch.from_numpy(np.log(np.maximum(item['pp'][:, a:b], 1e-7))[None]),
                    torch.from_numpy(logit(item['bb'][:, a:b])[None]))
            sl = slice(start-a, start-a+min(2048, x.shape[1]-start))
            pp.append(torch.softmax(p, 1)[0, :, sl].numpy())
            bb.append(torch.sigmoid(q)[0, :, sl].numpy())
            aux.append(torch.sigmoid(r)[0, :, sl].numpy())
    return np.concatenate(pp, 1), np.concatenate(bb, 1), np.concatenate(aux, 1)


def decode_events(item, pp, bb, register, norm, aux=None):
    raw, audit = decode(pp, bb, mix=item['mix'], vocal=item['vocal'], onset_threshold=.67, return_audit=True)
    if not raw:
        return [], audit
    with torch.no_grad():
        probabilities = torch.softmax(register(torch.from_numpy((event_features(dict(item, pp=pp), raw)-norm['event_mean'])/norm['event_scale'])), 1).numpy()
    notes = []
    for j, (n, probs) in enumerate(zip(raw, probabilities)):
        choice = int(probs.argmax()); shift = int(SHIFTS[choice]) if probs[choice] >= .99 else 0
        if not 36 <= n['midi']+shift <= 95:
            shift = 0
        a = min(pp.shape[1]-1, max(0, round(n['start_s']/HOP)))
        b = min(pp.shape[1], max(a+1, round(n['end_s']/HOP)))
        avg = pp[:, a:b].mean(1); alternatives = np.argsort(avg[1:])[-3:][::-1]+1
        c = dict(n, midi=n['midi']+shift, pre_register_midi=n['midi'], register_shift=shift,
                 register_confidence=float(probs[choice]), timing_resolution_s=HOP,
                 linguistic_relation='unknown', quantized=False,
                 pitch_alternatives=[dict(midi=int(k+35), acoustic_probability=float(avg[k])) for k in alternatives],
                 uncertainty=dict(pitch=bool(avg[1:].max() < .6), boundary_probability=float(bb[0, a]),
                                  calibration='model scores, not calibrated correctness probabilities'))
        if aux is not None:
            c['event_evidence'] = dict(zip(HEADS[2:], [float(aux[0, max(0,a-2):a+3].max()),
                                         float(aux[1, max(0,a-2):a+3].max()), float(aux[2, a:b].mean())]))
            c['boundary_kind'] = 'same_pitch_reattack' if j and c['midi'] == notes[-1]['midi'] else 'pitch_change_or_phrase_start'
        notes.append(c)
    return notes, audit
