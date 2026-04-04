"""
Candidate repair — fix subharmonic artifacts before melody-path resolution.

Basic Pitch on separated vocal stems often detects subharmonic candidates
one octave below the true sung pitch, while MISSING the actual fundamental.
This module identifies likely subharmonics and creates synthetic candidates
at +12 semitones so the downstream DP resolver has correct-register options.

Designed for vocals mode; disabled by default in mix mode.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .candidate_notes import CandidateNote

logger = logging.getLogger("mp3_to_sheet.candidate_repair")


@dataclass
class CandidateRepairConfig:
    """All tuneable parameters for the candidate repair stage."""

    enabled: bool = True

    # Local context window for computing the melodic band.
    context_window_s: float = 3.0

    # A candidate is "suspiciously low" if it is this many semitones
    # below the local upper-quartile pitch of overlapping candidates.
    low_band_threshold: int = 7

    # --- Octave recovery ---
    # When a low candidate lacks an overlapping P+12 counterpart,
    # create a synthetic candidate at P+12.
    octave_recovery_enabled: bool = True

    # Confidence assigned to the synthetic candidate, relative to the
    # original.  Values > 1.0 boost the recovered candidate above the
    # subharmonic so the DP prefers it.
    octave_recovery_confidence_scale: float = 0.90

    # If both P and P+12 already exist and P+12 is weaker, boost P+12
    # to at least this fraction of P's confidence.
    octave_boost_min_ratio: float = 0.85

    # --- Subharmonic suppression ---
    # When a synthetic P+12 is created, optionally penalise the original
    # low candidate so the DP is less likely to choose it.
    suppress_original: bool = True
    suppress_confidence_scale: float = 0.50

    # Safety: do not create synthetic candidates above this MIDI pitch.
    max_synthetic_midi: int = 96  # C7


@dataclass
class RepairStats:
    """Diagnostic counters for the repair pass."""
    candidates_before: int = 0
    candidates_after: int = 0
    low_candidates_detected: int = 0
    synthetics_created: int = 0
    existing_boosted: int = 0
    originals_suppressed: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "low_candidates_detected": self.low_candidates_detected,
            "synthetics_created": self.synthetics_created,
            "existing_boosted": self.existing_boosted,
            "originals_suppressed": self.originals_suppressed,
        }


def repair_candidates(
    candidates: List[CandidateNote],
    config: CandidateRepairConfig,
) -> Tuple[List[CandidateNote], RepairStats]:
    """
    Repair the candidate pool by detecting likely subharmonics and
    recovering missing upper-octave fundamentals.

    Returns the repaired candidate list and diagnostic stats.
    """
    stats = RepairStats(candidates_before=len(candidates))

    if not config.enabled or not candidates:
        stats.candidates_after = len(candidates)
        return candidates, stats

    # Pre-compute a fast lookup: for each candidate, what is the local
    # upper-quartile pitch in a time window around it?
    local_uqs = _compute_local_upper_quartiles(candidates, config.context_window_s)

    # Build a temporal index for fast overlap queries
    by_pitch_class: Dict[int, List[CandidateNote]] = {}
    for c in candidates:
        by_pitch_class.setdefault(c.pitch_class, []).append(c)

    new_candidates: List[CandidateNote] = []
    synthetics: List[CandidateNote] = []

    for i, cand in enumerate(candidates):
        local_uq = local_uqs[i]

        is_low = (local_uq - cand.pitch_midi) >= config.low_band_threshold
        if not is_low:
            new_candidates.append(cand)
            continue

        stats.low_candidates_detected += 1
        target_pitch = cand.pitch_midi + 12

        if target_pitch > config.max_synthetic_midi:
            new_candidates.append(cand)
            continue

        # Check if an overlapping candidate at P+12 already exists
        existing_upper = _find_overlapping_at_pitch(
            candidates, cand, target_pitch,
        )

        if existing_upper is not None:
            # P+12 already exists — boost it if weak
            if existing_upper.confidence < cand.confidence * config.octave_boost_min_ratio:
                existing_upper.confidence = cand.confidence * config.octave_boost_min_ratio
                existing_upper.salience = max(existing_upper.salience, cand.salience * 0.9)
                stats.existing_boosted += 1
            # Keep the original (possibly suppressed)
            if config.suppress_original:
                cand.confidence *= config.suppress_confidence_scale
                cand.salience *= config.suppress_confidence_scale
                stats.originals_suppressed += 1
            new_candidates.append(cand)
        else:
            # No P+12 exists — create a synthetic candidate
            synthetic = CandidateNote(
                start_s=cand.start_s,
                end_s=cand.end_s,
                pitch_midi=target_pitch,
                confidence=cand.confidence * config.octave_recovery_confidence_scale,
                salience=cand.salience * config.octave_recovery_confidence_scale,
                source="octave_recovery",
                raw_index=cand.raw_index,
            )
            synthetics.append(synthetic)
            stats.synthetics_created += 1

            # Suppress the original so the DP prefers the recovered one
            if config.suppress_original:
                cand.confidence *= config.suppress_confidence_scale
                cand.salience *= config.suppress_confidence_scale
                stats.originals_suppressed += 1

            new_candidates.append(cand)

    # Merge original + synthetic candidates, sorted by (start_s, pitch_midi)
    result = new_candidates + synthetics
    result.sort(key=lambda n: (n.start_s, n.pitch_midi))

    stats.candidates_after = len(result)

    logger.info(
        "Candidate repair: %d → %d candidates  "
        "(low_detected=%d, synthetics=%d, boosted=%d, suppressed=%d)",
        stats.candidates_before, stats.candidates_after,
        stats.low_candidates_detected, stats.synthetics_created,
        stats.existing_boosted, stats.originals_suppressed,
    )

    return result, stats


def _compute_local_upper_quartiles(
    candidates: List[CandidateNote],
    window_s: float,
) -> List[float]:
    """
    For each candidate, compute the duration-weighted 75th-percentile
    pitch of all candidates within a time window.
    """
    n = len(candidates)
    uqs: List[float] = []

    for i in range(n):
        note = candidates[i]
        mid_t = (note.start_s + note.end_s) / 2.0
        win_start = mid_t - window_s
        win_end = mid_t + window_s

        neighbors: List[Tuple[int, float]] = []
        for j in range(n):
            other = candidates[j]
            other_mid = (other.start_s + other.end_s) / 2.0
            if win_start <= other_mid <= win_end:
                neighbors.append((other.pitch_midi, max(other.duration_s, 0.001)))

        if not neighbors:
            uqs.append(float(note.pitch_midi))
            continue

        sorted_nb = sorted(neighbors, key=lambda x: x[0])
        total_w = sum(w for _, w in sorted_nb)
        target = total_w * 0.75
        cumulative = 0.0
        uq = float(sorted_nb[-1][0])
        for pitch, weight in sorted_nb:
            cumulative += weight
            if cumulative >= target:
                uq = float(pitch)
                break
        uqs.append(uq)

    return uqs


def _find_overlapping_at_pitch(
    candidates: List[CandidateNote],
    reference: CandidateNote,
    target_pitch: int,
) -> Optional[CandidateNote]:
    """
    Find a candidate at exactly `target_pitch` that overlaps in time
    with `reference`.  Returns the best (highest confidence) match or None.
    """
    best: Optional[CandidateNote] = None
    for c in candidates:
        if c.pitch_midi != target_pitch:
            continue
        if c.start_s < reference.end_s and reference.start_s < c.end_s:
            if best is None or c.confidence > best.confidence:
                best = c
    return best
