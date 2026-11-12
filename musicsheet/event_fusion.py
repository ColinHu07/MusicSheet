"""Conservative, score-free use of learned note-event auxiliary evidence.

This optional decoder adapter changes frame probabilities, never note timestamps
or score grids. Its two gains must be selected on validation songs only. It is a
decoder policy, not additional trained weights or verified syllable recognition.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np


def _neighbor_max(values, radius, *, side=None):
    """Maximum of existing neighbor frames; no wraparound or edge replication."""
    result = np.zeros_like(values)
    shifts = range(1, radius + 1)
    for shift in shifts:
        if shift >= len(values):
            break
        if side in (None, "left"):
            result[shift:] = np.maximum(result[shift:], values[:-shift])
        if side in (None, "right"):
            result[:-shift] = np.maximum(result[:-shift], values[shift:])
    return result


def fuse_event_evidence(pp, bb, aux, vocal: Mapping | None, *, attack_mix=0., continuation_strength=0.):
    """Return ``(pitch_probabilities, boundaries, audit)`` without input mutation.

    Shapes: pp=(61,T), bb=(2,T), aux=(3,T), with auxiliary rows same-pitch
    reattack, pitch transition, continuation. ``vocal`` supplies rmvpe_voiced and
    rmvpe_confidence on the same 20-ms grid. Missing/invalid evidence is masked;
    it does not become silence, an attack, or an invented pitch.

    Attack mixing only raises existing onset probabilities toward the stronger
    auxiliary prediction, where current/nearby reliable voicing supports it.
    Continuation requires reliable voicing on BOTH sides within 60 ms, a learned
    continuation score above .5, and no supported attack/transition within40 ms.
    It gently transfers silence mass to the existing conditional pitch
    distribution (unchanged relative pitches), and reduces offset evidence.
    All-silence columns remain all-silence. No phoneme/word quota is imposed.
    """
    if not np.isfinite([attack_mix, continuation_strength]).all() or not 0 <= attack_mix <= 1 or not 0 <= continuation_strength <= 1:
        raise ValueError("Fusion strengths must be finite values in [0,1]")
    p = np.asarray(pp)
    b = np.asarray(bb)
    if p.ndim != 2 or p.shape[0] != 61 or b.shape != (2, p.shape[1]):
        raise ValueError("Expected pp=(61,T) and bb=(2,T)")
    pitch, boundaries = p.copy(), b.copy()
    if not np.issubdtype(pitch.dtype, np.floating):
        pitch = pitch.astype(float)
    if not np.issubdtype(boundaries.dtype, np.floating):
        boundaries = boundaries.astype(float)
    audit = dict(
        policy="conservative_acoustic_event_fusion_v1", attack_mix=float(attack_mix),
        continuation_strength=float(continuation_strength),
        attack_frames_reinforced=0, continuation_frames_adjusted=0,
        raw_note_times_edited=False, rhythm_grid_used=False,
        pitch_ratios_preserved=True, linguistic_classification="unavailable",
        uncertainty_policy="Unknown evidence unchanged; gains are not correctness probabilities",
    )
    if attack_mix == 0 and continuation_strength == 0:
        return pitch, boundaries, dict(audit, status="identity_zero_gains")
    if aux is None or vocal is None or "rmvpe_voiced" not in vocal or "rmvpe_confidence" not in vocal:
        return pitch, boundaries, dict(audit, status="identity_missing_evidence")
    a = np.asarray(aux, dtype=float)
    voiced = np.asarray(vocal["rmvpe_voiced"], dtype=float)
    confidence = np.asarray(vocal["rmvpe_confidence"], dtype=float)
    frames = p.shape[1]
    if a.shape != (3, frames) or voiced.shape != (frames,) or confidence.shape != (frames,):
        raise ValueError("Auxiliary/voicing evidence must match the posterior time grid")
    if frames == 0:
        return pitch, boundaries, dict(audit, status="identity_empty")
    # Keep malformed/unknown columns unchanged. Do not silently normalize a
    # malformed posterior into apparent evidence.
    valid_p = np.isfinite(p).all(0) & (p >= 0).all(0) & (p <= 1).all(0) & np.isclose(p.sum(0), 1, rtol=1e-4, atol=1e-6)
    valid_b = np.isfinite(b).all(0) & (b >= 0).all(0) & (b <= 1).all(0)
    valid_a = np.isfinite(a).all(0) & (a >= 0).all(0) & (a <= 1).all(0)
    voice_known = np.isfinite(voiced) & np.isfinite(confidence) & (voiced >= 0) & (voiced <= 1) & (confidence >= 0) & (confidence <= 1)
    reliable = np.where(voice_known & (voiced >= .5) & (confidence >= .75), np.minimum(voiced, confidence), 0.)
    eligible = valid_p & valid_b & valid_a
    onset_support = np.maximum(reliable, _neighbor_max(reliable, 3))
    learned_attack = np.where(valid_a, np.maximum(a[0], a[1]), 0.)
    if attack_mix:
        mask = eligible & (onset_support > 0)
        # Monotonic convex interpolation cannot suppress an established repeat.
        target = np.maximum(b[0, mask], learned_attack[mask])
        boundaries[0, mask] = (1-attack_mix) * b[0, mask] + attack_mix * target
        audit["attack_frames_reinforced"] = int(np.count_nonzero(boundaries[0, mask] > b[0, mask]))
    if continuation_strength:
        left, right = _neighbor_max(reliable, 3, side="left"), _neighbor_max(reliable, 3, side="right")
        context = np.minimum(left, right)
        # Protect original and auxiliary new-note evidence, including nearby
        # boundary frames. Unknown neighbors are conservatively protected too.
        attacks = np.maximum(np.where(valid_b, b[0], 1.), np.where(valid_a, learned_attack, 1.))
        protected = np.maximum(attacks, _neighbor_max(attacks, 2)) >= .5
        voiced_mass = p[1:].sum(0)
        mask = eligible & (context > 0) & ~protected & (voiced_mass > 1e-8)
        activation = np.maximum(0, 2*a[2]-1)
        amount = np.where(mask, continuation_strength * context * activation, 0.)
        mask &= amount > 0
        # exp(-amount) is strictly positive even at maximum gain: uncertain
        # silence/ending evidence is weakened, never hard-deleted.
        retained = np.exp(-amount[mask])
        silence = p[0, mask].astype(float) * retained
        removed = p[0, mask].astype(float) - silence
        pitch[1:, mask] = p[1:, mask] * (1 + removed / voiced_mass[mask])[None, :]
        pitch[0, mask] = silence
        boundaries[1, mask] = b[1, mask] * retained
        audit["continuation_frames_adjusted"] = int(np.count_nonzero(mask))
        audit["total_silence_mass_redistributed"] = float(removed.sum())
    audit["unknown_or_invalid_columns_unchanged"] = int(np.count_nonzero(~eligible))
    audit["voicing_supported_frames"] = int(np.count_nonzero(reliable))
    audit["status"] = "fused"
    return pitch, boundaries, audit
