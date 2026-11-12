import numpy as np
import pytest

from musicsheet.event_fusion import fuse_event_evidence


def fixture(frames=24):
    p = np.zeros((61, frames), np.float32)
    p[0], p[25], p[27] = .4, .4, .2
    b = np.full((2, frames), .1, np.float32)
    a = np.zeros((3, frames), np.float32)
    a[2] = .95
    v = dict(rmvpe_voiced=np.ones(frames), rmvpe_confidence=np.full(frames, .9))
    return p, b, a, v


def test_zero_gains_are_exact_identity_with_no_input_mutation():
    p, b, a, v = fixture()
    p[0, 3] = np.nan  # Identity must not sanitize unknown evidence.
    before_p, before_b = p.copy(), b.copy()
    out_p, out_b, audit = fuse_event_evidence(p, b, a, v)
    np.testing.assert_array_equal(out_p, before_p)
    np.testing.assert_array_equal(out_b, before_b)
    out_b[0, 0] = 0
    np.testing.assert_array_equal(b, before_b)
    assert audit["status"] == "identity_zero_gains"


def test_repeated_attack_is_never_erased_by_continuation():
    p, b, a, v = fixture()
    b[0, 10] = .8
    a[0, 10] = .9
    out_p, out_b, audit = fuse_event_evidence(p, b, a, v, attack_mix=.5, continuation_strength=1)
    assert out_b[0, 10] >= b[0, 10]
    np.testing.assert_array_equal(out_p[:, 8:13], p[:, 8:13])
    np.testing.assert_array_equal(out_b[1, 8:13], b[1, 8:13])
    assert out_p[0, 5] < p[0, 5]
    assert audit["continuation_frames_adjusted"] > 0


def test_weak_auxiliary_attack_cannot_lower_strong_original_attack():
    p, b, a, v = fixture()
    b[0, 10] = .9
    a[:2] = .1
    _, out_b, _ = fuse_event_evidence(p, b, a, v, attack_mix=1)
    assert out_b[0, 10] == b[0, 10]


def test_unvoiced_false_auxiliary_positive_has_no_effect():
    p, b, a, v = fixture()
    v["rmvpe_voiced"][:] = 0
    a[:] = .99
    out_p, out_b, audit = fuse_event_evidence(p, b, a, v, attack_mix=1, continuation_strength=1)
    np.testing.assert_array_equal(out_p, p)
    np.testing.assert_array_equal(out_b, b)
    assert audit["attack_frames_reinforced"] == audit["continuation_frames_adjusted"] == 0


def test_continuation_can_bridge_short_consonant_dip_but_not_trailing_rest():
    p, b, a, v = fixture()
    v["rmvpe_voiced"][10] = 0
    v["rmvpe_voiced"][16:] = 0
    out_p, out_b, _ = fuse_event_evidence(p, b, a, v, continuation_strength=.7)
    assert out_p[0, 10] < p[0, 10]
    np.testing.assert_array_equal(out_p[:, 16:], p[:, 16:])
    np.testing.assert_array_equal(out_b[:, 16:], b[:, 16:])


def test_probability_mass_and_pitch_ratios_remain_finite_and_normalized():
    p, b, a, v = fixture()
    out_p, out_b, _ = fuse_event_evidence(p, b, a, v, attack_mix=.3, continuation_strength=.8)
    assert np.isfinite(out_p).all() and np.isfinite(out_b).all()
    np.testing.assert_allclose(out_p.sum(0), 1, atol=1e-6)
    np.testing.assert_allclose(out_p[25] / out_p[27], p[25] / p[27], atol=1e-6)
    np.testing.assert_array_equal(out_p[np.r_[1:25, 26, 28:61]], p[np.r_[1:25, 26, 28:61]])
    np.testing.assert_array_equal(p[0], np.full(24, .4, np.float32))


def test_all_silence_has_no_pitch_mass_invented():
    p, b, a, v = fixture()
    p[:] = 0
    p[0] = 1
    out_p, _, _ = fuse_event_evidence(p, b, a, v, continuation_strength=1)
    np.testing.assert_array_equal(out_p, p)


def test_unknown_auxiliary_column_and_neighbors_are_not_bridged():
    p, b, a, v = fixture()
    a[:, 10] = np.nan
    out_p, out_b, _ = fuse_event_evidence(p, b, a, v, attack_mix=.8, continuation_strength=1)
    np.testing.assert_array_equal(out_p[:, 8:13], p[:, 8:13])
    np.testing.assert_array_equal(out_b[:, 8:13], b[:, 8:13])


@pytest.mark.parametrize("kwargs", [dict(attack_mix=-1), dict(continuation_strength=2), dict(attack_mix=float("nan"))])
def test_invalid_gains_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fuse_event_evidence(*fixture(), **kwargs)
