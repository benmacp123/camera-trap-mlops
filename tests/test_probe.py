"""Tests for the out-of-fold probe: it learns a separable problem and never
scores a fold with a model trained on it.
"""

import numpy as np
import pytest

from cameratrap.arms import probe

CLASSES = ["a", "b", "c"]


def make_data(seed=0):
    """Return a separable three-class problem over six sites in three folds.

    Args:
        seed: Seed for the noise.

    Returns:
        A tuple ``(features, labels, sites, folds)``.
    """
    rng = np.random.default_rng(seed)
    centres = np.eye(3) * 4
    labels = np.tile(CLASSES, 60)
    features = centres[[CLASSES.index(c) for c in labels]] + rng.normal(size=(180, 3))
    sites = np.repeat([f"S{i}" for i in range(6)], 30)
    folds = np.repeat(["fold0", "fold1", "fold2"], 60)
    return features, labels, sites, folds


def test_probe_learns_separable_classes():
    """Out-of-fold predictions recover well-separated classes."""
    features, labels, sites, folds = make_data()
    logits, chosen = probe.out_of_fold_logits(features, labels, sites, folds, CLASSES)
    preds = np.array(CLASSES)[logits.argmax(axis=1)]
    assert np.mean(preds == labels) > 0.95
    assert set(chosen) == {"fold0", "fold1", "fold2"}


def test_scored_fold_is_never_trained_on():
    """Changing one fold's labels leaves that fold's logits unchanged."""
    features, labels, sites, folds = make_data()
    first, _ = probe.out_of_fold_logits(features, labels, sites, folds, CLASSES)
    shuffled = labels.copy()
    in_fold = folds == "fold0"
    shuffled[in_fold] = np.random.default_rng(1).permutation(labels[in_fold])
    second, _ = probe.out_of_fold_logits(features, shuffled, sites, folds, CLASSES)
    assert np.allclose(first[in_fold], second[in_fold])


def test_missing_class_in_training_is_refused():
    """A class absent from the training folds would leave its logit undefined."""
    features, labels, sites, folds = make_data()
    labels = labels.copy()
    labels[(labels == "c") & (folds != "fold0")] = "a"
    with pytest.raises(ValueError):
        probe.out_of_fold_logits(features, labels, sites, folds, CLASSES)
