"""Audio-only note decoding with conservative octave continuity and event audit.

This module does not import the network, read scores, quantize rhythm, or fold
pitches into an instrument range. Inputs use the frozen model's 20 ms grid:
61 probability rows (silence, MIDI 36..95) and two onset/offset rows. It only
uses optional, independently estimated vocal F0 as *soft* register evidence.

``gate_strength`` controls rejection of weak events, not a mandatory
onset gate: stable legato changes remain valid. Below 80 ms, notes require
a clear learned onset plus corroborating acoustic pitch and attack evidence.
Same
pitch attacks are split after continuity repairs so pitch smoothing cannot
silently combine repeated notes. Defaults are hypotheses for validation-song
selection, not claims of better accuracy on any evaluation song.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.signal import find_peaks

MIN_MIDI = 36
MAX_MIDI = 95
HOP = .02
VERSION = 'decoder_v3.1'


def _runs(labels):
    if not len(labels):
        return []
    edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def _vector(source, key, length):
    if source is None or key not in source:
        return None
    value = np.asarray(source[key], dtype=np.float64)
    if value.shape != (length,):
        raise ValueError(f'{key} must have shape ({length},), got {value.shape}')
    return value


def _register_evidence(vocal, length, confidence_threshold):
    """Missing confidence is unknown evidence, never an implicitly good F0."""
    pitch = _vector(vocal, 'rmvpe_midi', length)
    voiced = _vector(vocal, 'rmvpe_voiced', length)
    confidence = _vector(vocal, 'rmvpe_confidence', length)
    reliable = np.zeros(length, dtype=bool)
    if pitch is None or voiced is None or confidence is None:
        return np.full(length, np.nan), reliable
    reliable = (np.isfinite(pitch) & np.isfinite(confidence) & np.isfinite(voiced)
                & (voiced > .5) & (confidence >= confidence_threshold)
                & (pitch >= MIN_MIDI - .5) & (pitch <= MAX_MIDI + .5))
    ratio = _vector(vocal, 'vocal_fullmix_rms_ratio', length)
    if ratio is not None:
        reliable &= np.isfinite(ratio) & (ratio >= .12)
    return pitch, reliable


def _sequence(probabilities, boundaries, f0, reliable, transition, octave_cost,
              acoustic_register_weight, missing):
    states, length = probabilities.shape
    cost = -np.log(np.maximum(probabilities, 1e-9))
    pitches = np.arange(MIN_MIDI, MAX_MIDI + 1)
    if acoustic_register_weight > 0 and np.any(reliable):
        distance = np.abs(pitches[:, None] - f0[None, :])
        # Only compare octave alternatives in the estimator's pitch class.
        # A contradictory F0 pitch class cannot pull the melody to that class.
        pc_distance = np.abs((pitches[:, None] - f0[None, :] + 6) % 12 - 6)
        same_class = (pc_distance < .65) & reliable[None, :]
        direct = same_class & (distance < .65)
        wrong_octave = same_class & (distance > 11)
        cost[1:] += acoustic_register_weight * (wrong_octave.astype(float) - direct.astype(float))
    cost[:, missing] = 0.  # Missing evidence must not invent a silence label.
    change = np.full((states, states), transition, dtype=np.float64)
    change[0, :] *= .65
    change[:, 0] *= .65
    interval = np.abs(pitches[:, None] - pitches[None, :])
    change[1:, 1:] += .045 * np.minimum(interval, 24)
    octave = (interval >= 12) & (interval % 12 == 0)
    change[1:, 1:] += octave_cost * octave
    np.fill_diagonal(change, 0.)
    previous = cost[:, 0].copy()
    back = np.zeros((length, states), dtype=np.int16)
    state_indices = np.arange(states)
    for frame in range(1, length):
        local = change * (1 - .45 * boundaries[0, frame])
        local[:, 0] *= 1 - .7 * boundaries[1, frame]
        candidates = previous[:, None] + local
        ancestors = candidates.argmin(axis=0)
        back[frame] = ancestors
        previous = candidates[ancestors, state_indices] + cost[:, frame]
        # Normalization avoids precision loss on unusually long recordings.
        previous -= previous.min()
    labels = np.empty(length, dtype=np.int16)
    state = int(previous.argmin())
    for frame in range(length - 1, -1, -1):
        labels[frame] = state
        state = int(back[frame, state])
    return labels


def decode(probabilities, boundaries, mix: Mapping | None = None,
           vocal: Mapping | None = None, *, onset_threshold=.65, transition=2.,
           gate_strength=.5, octave_cost=2., minimum_duration=.08,
           acoustic_register_weight=.25, f0_confidence_threshold=.75,
           bridge_duration=.18, repeat_gap=.08, repeat_context=.08,
           short_onset_threshold=.90, short_confidence=.95, short_margin=.85,
           return_audit=False,
           on_threshold=None):
    """Return source-time events, or ``(events, audit)`` when requested.

    Main validation knobs are onset_threshold, gate_strength (0 disables the
    ordinary weak-event filter) and octave_cost. The short-event corroboration
    requirement always applies. No estimate is forced into a violin register.
    Below ``minimum_duration``, onset plus independently estimated acoustic
    evidence is required. A posterior spike alone cannot authorize a short note.
    ``mix`` is reserved for compatible callers; no mix energy gate is applied.
    Optional acoustic arrays must share the exact probability time grid.
    """
    if on_threshold is not None:
        onset_threshold = float(on_threshold)
    knobs = dict(onset_threshold=onset_threshold, transition=transition,
                 gate_strength=gate_strength, octave_cost=octave_cost,
                 minimum_duration=minimum_duration,
                 acoustic_register_weight=acoustic_register_weight,
                 f0_confidence_threshold=f0_confidence_threshold,
                 bridge_duration=bridge_duration, repeat_gap=repeat_gap,
                 repeat_context=repeat_context,
                 short_onset_threshold=short_onset_threshold,
                 short_confidence=short_confidence, short_margin=short_margin)
    if not all(np.isfinite(v) for v in knobs.values()):
        raise ValueError('Decoder settings must be finite')
    if not 0 <= onset_threshold <= 1 or not 0 <= gate_strength <= 1:
        raise ValueError('onset_threshold and gate_strength must be in [0, 1]')
    if any(knobs[k] < 0 for k in knobs if k not in ('onset_threshold', 'gate_strength')):
        raise ValueError('Decoder costs, durations and evidence thresholds must be nonnegative')
    original = np.asarray(probabilities, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != MAX_MIDI - MIN_MIDI + 2:
        raise ValueError('Expected 61 probability rows: silence then MIDI 36..95')
    if np.any(np.isfinite(original) & (original < 0)):
        raise ValueError('Probabilities cannot be negative')
    length = original.shape[1]
    usable = np.isfinite(original)
    pp = np.where(usable, original, 0.)
    totals = pp.sum(axis=0)
    missing = totals <= 0
    partial = ~usable.all(axis=0) & ~missing
    pp /= np.maximum(totals[None, :], 1e-300)
    pp[:, missing] = 1. / pp.shape[0]
    if boundaries is None:
        bb = np.zeros((2, length), dtype=np.float64)
        missing_boundaries = length
    else:
        bb = np.asarray(boundaries, dtype=np.float64)
        if bb.shape != (2, length):
            raise ValueError(f'Expected onset/offset shape (2, {length}), got {bb.shape}')
        missing_boundaries = int(np.sum(~np.isfinite(bb).all(axis=0)))
        bb = np.clip(np.nan_to_num(bb, nan=0., posinf=0., neginf=0.), 0, 1)
    f0, reliable = _register_evidence(vocal, length, f0_confidence_threshold)
    flux = _vector(vocal, 'onset_flux', length)
    acoustic_attacks = np.zeros(length, dtype=np.float64)
    if flux is not None and length:
        safe_flux = np.maximum(np.nan_to_num(flux, nan=0., posinf=0., neginf=0.), 0.)
        scale = float(np.percentile(safe_flux, 95))
        if scale > 1e-9:
            normalized_flux = safe_flux / scale
            acoustic_peaks, _ = find_peaks(normalized_flux, height=.85, prominence=.25, distance=2)
            acoustic_attacks[acoustic_peaks] = np.minimum(normalized_flux[acoustic_peaks], 1.)
    audit = dict(version=VERSION, time_grid_s=HOP, config={k: float(v) for k, v in knobs.items()},
                 input_quality=dict(frames=length, missing_probability_frames=int(missing.sum()),
                                    partial_probability_frames=int(partial.sum()),
                                    missing_boundary_frames=missing_boundaries,
                                    reliable_f0_frames=int(reliable.sum()),
                                    independent_acoustic_attack_peaks=int(np.count_nonzero(acoustic_attacks))),
                 changes=[], events=[], summary={})
    if length == 0:
        audit['summary'] = dict(accepted_notes=0, rejected_events=0, removed_events=0,
                                bridged_events=0, merged_events=0, octave_changed_frames=0,
                                repeated_attacks=0, uncertain_kept=0)
        return ([], audit) if return_audit else []
    labels = _sequence(pp, bb, f0, reliable, transition, octave_cost,
                       acoustic_register_weight, missing)
    sequence_labels = labels.copy()
    raw = np.argmax(pp, axis=0)
    raw[missing] = labels[missing]
    competitors = np.empty_like(pp)
    best_state = pp.argmax(axis=0)
    best = pp.max(axis=0)
    runner_up = np.partition(pp, -2, axis=0)[-2]
    for state in range(pp.shape[0]):
        competitors[state] = np.where(best_state == state, runner_up, best)

    # Onsets are an independent event stream. A pitch change does not create
    # an onset peak, and absence of one does not rule out a legato transition.
    peaks, _ = find_peaks(bb[0], height=onset_threshold,
                          distance=max(1, int(math.ceil(repeat_gap / HOP))),
                          prominence=.10)

    def evidence(a, b, state):
        known = ~missing[a:b]
        values = pp[state, a:b][known]
        margins = (pp[state, a:b] - competitors[state, a:b])[known]
        available = int(known.sum())
        onset_a, onset_b = max(0, a - 2), min(length, a + 3, b)
        onset = float(bb[0, onset_a:onset_b].max(initial=0))
        pitch = state + MIN_MIDI - 1
        acoustic = reliable[a:b] & ~missing[a:b]
        f0_matches = acoustic & (np.abs(f0[a:b] - pitch) < .65)
        return dict(confidence=float(values.mean()) if available else 0.,
                    margin=float(margins.mean()) if available else 0.,
                    winning_fraction=float(np.mean(margins > 0)) if available else 0.,
                    known_fraction=available / max(1, b - a),
                    duration_s=(b - a) * HOP, onset=onset,
                    acoustic_coverage=float(acoustic.mean()),
                    acoustic_support=float(f0_matches.sum() / max(1, acoustic.sum())),
                    acoustic_attack=float(acoustic_attacks[onset_a:onset_b].max(initial=0)))

    def short_support(e, repeat=False):
        if e['duration_s'] < .04 or e['onset'] < max(onset_threshold, short_onset_threshold):
            return []
        if e['confidence'] < short_confidence or e['margin'] < short_margin:
            return []
        # F0 alone does not establish a new attack; a transient alone can be a
        # drum leak. The short exception requires agreement of both channels.
        if (e['acoustic_attack'] >= .85 and e['acoustic_coverage'] >= .5
                and e['acoustic_support'] >= .8):
            return ['short_onset_acoustic_attack_and_pitch']
        return []

    def supports(e):
        confidence, margin, duration = e['confidence'], e['margin'], e['duration_s']
        reasons = []
        if duration + 1e-9 < minimum_duration:
            reasons = short_support(e)
            if e['known_fraction'] < .5:
                reasons.append('missing_evidence_preserved')
            return reasons
        floor = .16 + .16 * gate_strength
        # Strong pitch evidence allows short notes without a piano-like attack.
        if confidence >= .55 and margin >= .25:
            reasons.append('strong_pitch')
        if e['onset'] >= onset_threshold and confidence >= floor and margin >= .035:
            reasons.append('onset_and_pitch')
        if (duration >= minimum_duration and confidence >= floor
                and margin >= .04 + .12 * gate_strength
                and e['winning_fraction'] >= .55 + .15 * gate_strength):
            reasons.append('stable_pitch_transition')
        if (e['acoustic_coverage'] >= .5 and e['acoustic_support'] >= .8
                and confidence >= .12 and margin >= -.15 and duration >= .04):
            reasons.append('acoustic_pitch_support')
        # A long ambiguous note must remain inspectable. Deletion is justified
        # by weak brief evidence, not merely by missing estimator information.
        if (duration >= max(.30, 3 * minimum_duration) and confidence >= .12
                and margin >= -.01 and e['winning_fraction'] >= .5):
            reasons.append('sustained_uncertain_pitch')
        if e['known_fraction'] < .5:
            reasons.append('missing_evidence_preserved')
        if gate_strength == 0:
            reasons.append('event_filter_disabled')
        return reasons

    # Hysteresis: bridge only a brief, weak excursion surrounded by the same
    # pitch. Octave confusion warrants more continuity than a stepwise ornament.
    if gate_strength > 0:
        initial_runs = _runs(labels)
        for index, (a, b, state) in enumerate(initial_runs):
            if state == 0 or index == 0 or index == len(initial_runs) - 1:
                continue
            left, right = initial_runs[index - 1][2], initial_runs[index + 1][2]
            if left == 0 or left != right or (b - a) * HOP > bridge_duration:
                continue
            if labels[a - 1] != left or labels[b] != right:
                continue  # A previous repair changed this proposed context.
            e = evidence(a, b, state)
            known = ~missing[a:b]
            if not known.any():
                continue
            is_octave = abs(state - left) >= 12 and (state - left) % 12 == 0
            threshold = (.08 + .24 * gate_strength) if is_octave else (.02 + .12 * gate_strength)
            advantage = float((pp[state, a:b] - pp[left, a:b])[known].mean())
            attack_supported = ('onset_and_pitch' in supports(e))
            acoustic_supported = e['acoustic_coverage'] >= .5 and e['acoustic_support'] >= .8
            if advantage <= threshold and not attack_supported and not acoustic_supported:
                labels[a:b] = left
                audit['changes'].append(dict(action='bridge', start_s=round(a * HOP, 6),
                    end_s=round(b * HOP, 6), from_midi=state + MIN_MIDI - 1,
                    to_midi=left + MIN_MIDI - 1, octave=is_octave,
                    reason='brief_octave_excursion' if is_octave else 'brief_unsupported_excursion',
                    posterior_advantage=advantage, evidence=e))

    # Reject unsupported events while keeping explicit uncertainty. Rejection
    # is represented as silence; it is never disguised as an invented pitch.
    for a, b, state in _runs(labels):
        if state == 0:
            continue
        e = evidence(a, b, state)
        reasons = supports(e)
        if not reasons:
            labels[a:b] = 0
            audit['events'].append(dict(status='rejected', start_s=round(a * HOP, 6),
                end_s=round(b * HOP, 6), midi=state + MIN_MIDI - 1,
                reasons=['unsupported_short_candidate' if e['duration_s'] + 1e-9 < minimum_duration
                         else 'weak_pitch_without_attack_or_stability'], evidence=e))

    notes = []
    for a, b, state in _runs(labels):
        if state == 0:
            continue
        edges = [a, b]
        context_frames = max(2, int(math.ceil(repeat_context / HOP - 1e-9)))
        for at in sorted(peaks[(peaks >= a + 2) & (peaks <= b - 2)],
                         key=lambda x: bb[0, x], reverse=True):
            distance = min(abs(int(at) - edge) for edge in edges)
            if distance < 2:
                continue
            before = evidence(max(a, int(at) - 3), int(at), state)
            after = evidence(int(at), min(b, int(at) + 3), state)
            if distance < context_frames and not short_support(after, repeat=True):
                audit['events'].append(dict(status='rejected_split',
                    start_s=round(int(at) * HOP, 6), end_s=round(b * HOP, 6),
                    midi=state + MIN_MIDI - 1,
                    reasons=['insufficient_repeat_context_without_acoustic_attack'],
                    evidence=after))
                continue
            if min(before['confidence'], after['confidence']) >= .12:
                edges.append(int(at))
        edges.sort()
        for left, right in zip(edges[:-1], edges[1:]):
            e = evidence(left, right, state)
            reasons = supports(e)
            repeat = left != a
            if repeat:
                reasons.append('independent_repeated_attack')
            if not reasons:
                # Once a supported long note is split at a real attack, a short
                # tail must not disappear merely because it is now shorter.
                reasons.append('part_of_supported_pitch_run')
            uncertain = (e['known_fraction'] < 1 or e['confidence'] < .45
                         or e['margin'] < .10)
            event = dict(start_s=round(left * HOP, 6), end_s=round(right * HOP, 6),
                         midi=state + MIN_MIDI - 1, velocity=82,
                         model_confidence=e['confidence'], confidence_margin=e['margin'],
                         repeated_attack=repeat, extra_review=bool(uncertain),
                         decoder_reasons=reasons)
            notes.append(event)
            audit['events'].append(dict(status='accepted', **event, evidence=e))

    changed = (raw != labels) & ~missing
    octave_changed = changed & (raw > 0) & (labels > 0) & (np.abs(raw - labels) % 12 == 0)
    rejected = sum(e['status'] == 'rejected' for e in audit['events'])
    rejected_splits = sum(e['status'] == 'rejected_split' for e in audit['events'])
    bridged = len(audit['changes'])
    audit['summary'] = dict(accepted_notes=len(notes), rejected_events=rejected,
        removed_events=rejected, rejected_repeat_splits=rejected_splits,
        rejected_short_candidates=sum('unsupported_short_candidate' in e.get('reasons', [])
                                      for e in audit['events']),
        bridged_events=bridged, merged_events=bridged,
        repeated_attacks=sum(n['repeated_attack'] for n in notes),
        uncertain_kept=sum(n['extra_review'] for n in notes),
        raw_argmax_note_runs=sum(s > 0 for _, _, s in _runs(raw)),
        sequence_note_runs=sum(s > 0 for _, _, s in _runs(sequence_labels)),
        changed_frames=int(changed.sum()), octave_changed_frames=int(octave_changed.sum()),
        changed_frame_comparator='independent_frame_argmax_before_sequence_decoding')
    # Audit ordinary sequence smoothing as well as explicit post-filter edits.
    for a, b, state in _runs(np.where(raw != sequence_labels, raw + 1, 0)):
        if state:
            targets = sorted(set(int(x + MIN_MIDI - 1) if x else None for x in sequence_labels[a:b]),
                             key=lambda x: -999 if x is None else x)
            audit['changes'].append(dict(action='sequence_continuity', start_s=round(a * HOP, 6),
                end_s=round(b * HOP, 6), from_midi=state + MIN_MIDI - 2 if state > 1 else None,
                to_midi=targets, reason='soft_transition_and_optional_register_evidence'))
    return (notes, audit) if return_audit else notes
