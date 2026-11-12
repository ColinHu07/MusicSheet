"""Learned rhythm probabilities from audio-derived note events; NumPy inference.

No score, track identity, absolute pitch, or reference alignment is an input.
The separate IOI and duration heads retain the distinction between reattacks
and sustained notes. This component never inserts, deletes, or repitches notes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

VERSION = 'context-rhythm-network-r1'
TICKS = np.array([3, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144, 192], dtype=np.float32)
BEATS = TICKS / 48

def features(events, bpm):
    period = 60 / bpm
    starts = np.array([n['start_s'] for n in events], dtype=float)
    ends = np.array([n['end_s'] for n in events], dtype=float)
    pitches = np.array([n['midi'] for n in events], dtype=float)
    if not len(events): return np.empty((0, 63), np.float32), np.empty((0, 2, len(TICKS)), np.float32)
    durations = np.maximum((ends-starts)/period, .03125)
    intervals = np.r_[np.diff(starts)/period, durations[-1]]
    gaps = np.maximum(intervals-durations, 0)
    repeat = np.r_[pitches[1:]==pitches[:-1], False]
    confidence = np.array([n.get('model_confidence', .5) for n in events])
    row = np.stack([np.clip(np.log2(intervals), -5, 5), np.clip(np.log2(durations), -5, 5),
                    np.clip(gaps/intervals, 0, 1), repeat, confidence,
                    np.r_[np.minimum(abs(np.diff(pitches)), 12)/12, 0], np.ones(len(events))], axis=1)
    padded = np.pad(row, ((4,4),(0,0)))
    x = np.stack([padded[i:i+9].reshape(-1) for i in range(len(events))]).astype(np.float32)
    values = np.stack([intervals, durations], axis=1)
    # Physical timing prior, shared across songs; learned residual weights adapt it.
    base = -np.minimum(((values[:,:,None]-BEATS)/(0.16+0.035*BEATS))**2, 30)
    return x, base.astype(np.float32)

class RhythmModel:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        self.path = directory/'weights.npz'
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != self.meta['weights_sha256'] or self.meta['feature_version'] != VERSION:
            raise ValueError('Rhythm model checksum/version mismatch')
        self.weights = dict(np.load(self.path, allow_pickle=False))

    def probabilities(self, events, bpm):
        x, base = features(events, bpm)
        w = self.weights
        h = np.tanh(((x-w['mean'])/w['scale']) @ w['w1'].T + w['b1'])
        h = np.tanh(h @ w['w2'].T + w['b2'])
        logits = base + (h @ w['w3'].T+w['b3']).reshape(-1, 2, len(TICKS))
        logits -= logits.max(axis=-1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=-1, keepdims=True)

def class_index(beats):
    index = int(np.argmin(abs(BEATS-beats)))
    return index if abs(float(BEATS[index])-beats)<.002 else -1
