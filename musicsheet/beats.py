"""Score-free rhythm diagnostics for the separate note-event experiment.

Raw events are immutable here. Candidate grids are explanations of their timing,
not a quantizer: no attack, release, pitch, or event count is changed. Full-mix
onset flux supplies an independent periodicity cue when available. Neither that
cue nor note spacing determines meter, downbeat, or a unique tempo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


GRIDS = {2: "eighths", 3: "eighth_triplets", 4: "sixteenths", 6: "sixteenth_triplets"}


def _audio_periodicity(features):
    if features is None or "onset_flux" not in features or "times_s" not in features:
        return None
    flux = np.asarray(features["onset_flux"], dtype=float)
    times = np.asarray(features["times_s"], dtype=float)
    if flux.ndim != 1 or times.shape != flux.shape or len(times) < 8:
        raise ValueError("Audio times_s and onset_flux must be matching 1-D arrays")
    if not np.isfinite(flux).all() or not np.isfinite(times).all():
        raise ValueError("Audio features must be finite")
    differences = np.diff(times)
    hop = float(np.median(differences))
    if hop <= 0 or not np.allclose(differences, hop, rtol=.001, atol=1e-7):
        raise ValueError("Audio features must have a uniform increasing time grid")
    # Mean removal prevents a large DC component from looking like a beat.
    centered = np.maximum(flux, 0) - np.mean(np.maximum(flux, 0))
    if np.dot(centered, centered) < 1e-12:
        return None
    size = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:len(centered)]
    # Correct overlap length, retaining normalization against zero-lag energy.
    correlation /= np.arange(len(centered), 0, -1)
    correlation /= max(float(correlation[0]), 1e-12)
    return hop, correlation


def _grid_fit(starts, period, divisor):
    step = period / divisor
    # Uniform phase hypotheses plus observed phases avoid search-grid artifacts.
    phases = np.unique(np.r_[np.arange(48) * step / 48, np.mod(starts[:32], step)])
    residuals = (starts[None, :] - phases[:, None] + step / 2) % step - step / 2
    # Limit isolated wrong attacks' influence; do not delete them.
    loss = np.mean(np.minimum(np.abs(residuals), .12), axis=1)
    index = int(np.argmin(loss))
    residual = residuals[index]
    phase = float(phases[index])
    return dict(
        subdivision=divisor, name=GRIDS[divisor], subdivision_period_s=float(step),
        phase_mod_subdivision_s=phase,
        median_absolute_residual_ms=float(1000 * np.median(np.abs(residual))),
        mean_absolute_residual_ms=float(1000 * np.mean(np.abs(residual))),
        p90_absolute_residual_ms=float(1000 * np.percentile(np.abs(residual), 90)),
        within_30ms_fraction=float(np.mean(np.abs(residual) <= .030)),
        # Mild complexity cost only ranks explanations, never erases real short notes.
        diagnostic_cost_s=float(loss[index] + .001 * (divisor - 2)),
    )


def _patterns(starts, period, tolerance_s=.035):
    intervals = np.diff(starts)
    triplets = []
    staggered = []
    # Report complete groups of three equally spaced attacks, with a following
    # boundary when available. A pickup can start on any third of the beat.
    for index in range(max(0, len(starts) - 2)):
        portion = intervals[index:index + 2]
        if len(portion) == 2 and np.all(np.abs(portion - period / 3) <= tolerance_s):
            triplets.append([index, index + 1, index + 2])
    # Four consecutive boundaries expose the three durations 1/4, 1/2, 1/4 beat.
    # This uses attack intervals, not acoustic releases, which may be legato.
    for index in range(max(0, len(starts) - 3)):
        if np.all(np.abs(intervals[index:index + 3] - period * np.array([.25, .5, .25])) <= tolerance_s):
            staggered.append([index, index + 1, index + 2, index + 3])
    return dict(
        triplet_attack_groups=triplets,
        sixteenth_eighth_sixteenth_boundary_groups=staggered,
        interpretation="Timing support only; no syllable, articulation, or notated-duration ground truth",
        tolerance_s=tolerance_s,
    )


def estimate_beat_candidates(
    notes: Sequence[Mapping], audio_features=None, *, bpm_candidates=None,
    min_bpm=40., max_bpm=220., max_candidates=8,
):
    """Return uncertain beat explanations without modifying any input note.

    ``notes`` contains source-time ``start_s``, ``end_s``, and ``midi``. Optional
    ``audio_features`` is an NPZ/mapping with ``times_s`` and ``onset_flux``.
    ``bpm_candidates`` restricts diagnosis to explicitly supplied hypotheses;
    otherwise a 0.5-BPM search is used. Ranking is a heuristic, not a calibrated
    probability or a learned model, and must not select checkpoints on test data.
    """
    if not (0 < min_bpm <= max_bpm) or max_candidates < 1:
        raise ValueError("Invalid tempo bounds or candidate count")
    events = []
    for index, note in enumerate(notes):
        values = [float(note[k]) for k in ("start_s", "end_s", "midi")]
        if not np.isfinite(values).all() or values[0] < 0 or values[1] <= values[0]:
            raise ValueError("Every note needs finite pitch and nonnegative increasing times")
        events.append((values[0], index))
    events.sort()
    starts = np.asarray([event[0] for event in events], dtype=float)
    original_indices = [event[1] for event in events]
    common = dict(
        schema_version=1, raw_events_modified=False, note_count=len(notes),
        original_indices_in_time_order=original_indices,
        confidence="uncalibrated diagnostic alternatives", meter=None, downbeat_s=None,
        limitations=[
            "Half/double tempo and subdivision alternatives may explain identical note times",
            "A global constant-tempo fit does not model local tempo changes or expressive timing",
            "No event is added, deleted, merged, retimed, transposed, or engraved",
        ],
    )
    if len(starts) < 3:
        return dict(common, status="insufficient_note_attacks", evidence="unavailable", candidates=[])
    periodicity = _audio_periodicity(audio_features)
    bpms = np.asarray(list(bpm_candidates) if bpm_candidates is not None else np.arange(min_bpm, max_bpm + .001, .5), dtype=float)
    if bpms.ndim != 1 or not len(bpms) or not np.isfinite(bpms).all() or np.any(bpms <= 0):
        raise ValueError("BPM hypotheses must be finite and positive")
    bpms = np.unique(bpms[(bpms >= min_bpm) & (bpms <= max_bpm)])
    if not len(bpms):
        raise ValueError("No BPM hypotheses inside tempo bounds")
    records = []
    for bpm in bpms:
        period = 60 / bpm
        grids = [_grid_fit(starts, period, divisor) for divisor in GRIDS]
        best = min(grids, key=lambda grid: grid["diagnostic_cost_s"])
        audio_score = None
        if periodicity is not None:
            hop, autocorrelation = periodicity
            lag = period / hop
            # Interpolation respects fractional sample lags; whole-second/grid
            # rounding must not invent precision in the tempo estimate.
            audio_score = float(np.interp(lag, np.arange(len(autocorrelation)), autocorrelation, left=0, right=0))
        fit_score = float(np.exp(-best["diagnostic_cost_s"] / .035))
        score = fit_score if audio_score is None else .35 * fit_score + .65 * max(0, audio_score)
        records.append(dict(bpm=float(bpm), beat_period_s=float(period), rank_score=float(score),
                            audio_periodicity=audio_score, note_fit_score=fit_score,
                            best_grid=best["name"], grids=grids,
                            beat_phase_alternatives_s=[float(best["phase_mod_subdivision_s"] + i * period / best["subdivision"]) for i in range(best["subdivision"])],
                            phase_is_downbeat=False))
    ranked = sorted(records, key=lambda result: (-result["rank_score"], result["bpm"]))
    # Suppress near-identical tempi while explicitly preserving metric relatives.
    chosen = []
    for candidate in ranked:
        if all(abs(np.log2(candidate["bpm"] / old["bpm"])) > .025 for old in chosen):
            chosen.append(candidate)
        if len(chosen) >= max_candidates:
            break
    if bpm_candidates is None and chosen and max_candidates > 1:
        primary = chosen[0]["bpm"]
        for relative in (primary / 2, primary * 2):
            if min_bpm <= relative <= max_bpm:
                candidate = min(records, key=lambda result: abs(result["bpm"] - relative))
                if not any(abs(np.log2(candidate["bpm"] / old["bpm"])) <= .025 for old in chosen):
                    if len(chosen) >= max_candidates:
                        chosen.pop()
                    chosen.append(candidate)
    chosen.sort(key=lambda result: (-result["rank_score"], result["bpm"]))
    for rank, candidate in enumerate(chosen, 1):
        candidate["rank"] = rank
        candidate["patterns"] = _patterns(starts, candidate["beat_period_s"])
        candidate["tempo_relatives_bpm"] = [float(value) for value in (candidate["bpm"] / 2, candidate["bpm"] * 2) if min_bpm <= value <= max_bpm]
    return dict(common, status="diagnostic_only",
                evidence="fullmix_onset_flux_and_note_attacks" if periodicity is not None else "note_attacks_only",
                search_resolution_bpm=.5 if bpm_candidates is None else None, candidates=chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--audio-features", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.notes.read_text(encoding="utf-8"))
    notes = payload if isinstance(payload, list) else payload.get("notes", payload.get("events"))
    if notes is None:
        parser.error("JSON must contain a note list, notes, or events")
    if args.output.exists():
        parser.error("Use a fresh output path")
    if args.audio_features:
        with np.load(args.audio_features) as features:
            result = estimate_beat_candidates(notes, features)
    else:
        result = estimate_beat_candidates(notes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
