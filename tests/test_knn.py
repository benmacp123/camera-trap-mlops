"""Tests for the kNN arm: weighting, smoothing, choosing k and fold isolation."""

import numpy as np
import pytest

from cameratrap.arms import knn

CLASSES = ["a", "b", "c"]
# The real grid reaches 200, more than this data's smallest inner split.
SMALL_KS = (1, 5, 15)


def make_data(seed=0):
    """Return a separable three-class problem over six sites in three folds.

    Args:
        seed: Seed for the noise.

    Returns:
        A tuple ``(features, labels, sites, folds)``, rows L2-normalised.
    """
    rng = np.random.default_rng(seed)
    labels = np.tile(CLASSES, 60)
    features = np.eye(3)[[CLASSES.index(c) for c in labels]] * 4
    features = features + rng.normal(size=(180, 3))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    sites = np.repeat([f"S{i}" for i in range(6)], 30)
    folds = np.repeat(["fold0", "fold1", "fold2"], 60)
    return features, labels, sites, folds


def test_class_weights_match_balanced():
    """Weights are n / (classes * count), so each class's total weight is equal."""
    y = np.array([0] * 9 + [1] * 1)
    weights = knn.class_weights(y, 2)
    assert np.allclose(weights, [10 / 18, 10 / 2])
    assert weights[0] * 9 == pytest.approx(weights[1] * 1)


def test_class_weights_refuse_a_missing_class():
    """A class with no training rows would get an infinite weight."""
    with pytest.raises(ValueError):
        knn.class_weights(np.array([0, 0]), 2)


def test_neighbours_are_most_similar_first():
    """Neighbours come back in descending cosine similarity."""
    train = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    near = knn.neighbours(train, np.array([[0.8, 0.6]]), 3)
    # Similarities to the three training rows: 0.8, 0.6 and 0.96.
    assert near.tolist() == [[2, 0, 1]]


def test_weighting_lets_a_rare_class_win():
    """Two rare-class neighbours outvote eight common ones once weighted."""
    near = np.array([[0] * 8 + [1] * 2])
    plain = knn.vote_logits(near, np.array([1.0, 1.0]), 10)
    weighted = knn.vote_logits(near, np.array([0.55, 5.0]), 10)
    assert plain.argmax() == 0
    assert weighted.argmax() == 1


def test_pseudo_count_keeps_logits_finite_and_confidence_below_one():
    """A class with no votes gets a finite logit, so the softmax is not saturated."""
    near = np.array([[0] * 10])
    logits = knn.vote_logits(near, np.array([1.0, 1.0]), 10)
    assert np.isfinite(logits).all()
    shares = np.exp(logits)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0, 0] == pytest.approx(10.5 / 11)


def test_choose_k_returns_a_candidate_and_its_curve():
    """The chosen k is a candidate with the best inner score."""
    features, labels, sites, _ = make_data()
    y = np.array([CLASSES.index(c) for c in labels])
    k, curve = knn.choose_k(features, y, sites, 3, ks=(1, 5, 15))
    assert set(curve) == {1, 5, 15}
    assert curve[k] == max(curve.values())


def test_knn_learns_separable_classes():
    """Out-of-fold predictions recover well-separated classes."""
    features, labels, sites, folds = make_data()
    logits, chosen, curves = knn.out_of_fold_logits(
        features, labels, sites, folds, CLASSES, ks=SMALL_KS
    )
    preds = np.array(CLASSES)[logits.argmax(axis=1)]
    assert np.mean(preds == labels) > 0.95
    assert set(chosen) == set(curves) == {"fold0", "fold1", "fold2"}
    assert all(k in SMALL_KS for k in chosen.values())


def test_scored_fold_is_never_a_neighbour():
    """Changing one fold's labels leaves that fold's logits unchanged."""
    features, labels, sites, folds = make_data()
    first, _, _ = knn.out_of_fold_logits(
        features, labels, sites, folds, CLASSES, ks=SMALL_KS
    )
    shuffled = labels.copy()
    in_fold = folds == "fold0"
    shuffled[in_fold] = np.random.default_rng(1).permutation(labels[in_fold])
    second, _, _ = knn.out_of_fold_logits(
        features, shuffled, sites, folds, CLASSES, ks=SMALL_KS
    )
    assert np.allclose(first[in_fold], second[in_fold])
