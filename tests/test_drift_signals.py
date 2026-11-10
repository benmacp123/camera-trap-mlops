"""Tests for the label-free signals: each against a case worked by hand."""

import numpy as np
import pytest

from cameratrap.drift import signals


def test_entropy_of_one_hot_and_uniform_rows():
    """A certain row has entropy 0 and a uniform row over k classes has log k."""
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    assert signals.entropy(probs) == pytest.approx([0.0, np.log(4)])


def test_atc_threshold_matches_share_above_to_accuracy():
    """With 3 of 5 correct, exactly 3 of the 5 scores lie above the threshold."""
    scores = np.array([0.9, 0.5, 0.8, 0.6, 0.7])
    correct = np.array([True, False, True, True, False])
    threshold = signals.atc_threshold(scores, correct)
    assert threshold == pytest.approx(0.65)
    assert signals.atc_estimate(scores, threshold) == pytest.approx(0.6)


def test_atc_threshold_at_the_extremes():
    """All correct puts every score above the threshold; none correct, none."""
    scores = np.array([0.4, 0.9])
    everything = signals.atc_threshold(scores, [True, True])
    nothing = signals.atc_threshold(scores, [False, False])
    assert signals.atc_estimate(scores, everything) == 1.0
    assert signals.atc_estimate(scores, nothing) == 0.0


def test_atc_threshold_refuses_no_data():
    """An empty labelled set raises instead of returning an arbitrary threshold."""
    with pytest.raises(ValueError):
        signals.atc_threshold([], [])


def test_atc_estimate_falls_when_confidence_falls():
    """A less confident site gets a lower predicted accuracy at the same threshold."""
    assert signals.atc_estimate([0.9, 0.9, 0.3, 0.3], 0.5) == 0.5
    assert signals.atc_estimate([0.9, 0.3, 0.3, 0.3], 0.5) == 0.25


def test_knn_distance_is_zero_on_the_reference_and_one_when_orthogonal():
    """A query equal to its neighbours is at 0; one orthogonal to them is at 1."""
    reference = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert signals.knn_distance(reference, query, 2) == pytest.approx([0.0, 1.0])


def test_knn_distance_averages_only_the_nearest_k():
    """Far reference rows beyond the k nearest do not change the distance."""
    reference = signals.normalise(np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]]))
    query = np.array([[1.0, 0.0]])
    expected = 1 - (1 + reference[1, 0]) / 2
    assert signals.knn_distance(reference, query, 2) == pytest.approx([expected])


def test_knn_distance_refuses_too_small_a_reference():
    """Asking for more neighbours than exist raises."""
    with pytest.raises(ValueError):
        signals.knn_distance(np.eye(2), np.eye(2), 3)


def test_median_bandwidth_on_known_points():
    """Three points on a line at 0, 1 and 3 have pairwise distances 1, 2, 3."""
    points = np.array([[0.0], [1.0], [3.0]])
    assert signals.median_bandwidth(points) == pytest.approx(2.0)


def test_mmd_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    """Two samples of one distribution give about 0; a shifted one gives more."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 5))
    same = rng.normal(size=(150, 5))
    shifted = rng.normal(size=(150, 5)) + 1.5
    bandwidth = signals.median_bandwidth(reference)
    assert abs(signals.mmd2(same, reference, bandwidth)) < 0.01
    assert signals.mmd2(shifted, reference, bandwidth) > 0.1


def test_mmd_does_not_grow_just_because_the_sample_is_small():
    """The unbiased estimate stays near 0 for a small same-distribution sample."""
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(400, 5))
    bandwidth = signals.median_bandwidth(reference)
    small = [
        signals.mmd2(rng.normal(size=(20, 5)), reference, bandwidth) for _ in range(50)
    ]
    assert abs(np.mean(small)) < 0.01


def test_mmd_refuses_a_single_row():
    """One row has no off-diagonal pair, so MMD raises."""
    with pytest.raises(ValueError):
        signals.mmd2(np.ones((1, 2)), np.ones((3, 2)), 1.0)


def test_class_shares_and_total_variation():
    """Shares count classes, absent ones included; the distance is half the L1 gap."""
    shares = signals.class_shares([0, 0, 2, 2], 3)
    assert shares.tolist() == [0.5, 0.0, 0.5]
    assert signals.total_variation(shares, [0.5, 0.5, 0.0]) == pytest.approx(0.5)
    assert signals.total_variation(shares, shares) == 0.0
