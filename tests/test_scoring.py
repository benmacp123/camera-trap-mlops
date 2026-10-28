"""Tests for sequence-level scoring: logits are averaged before the softmax,
and balanced accuracy counts only the classes present at a site.
"""

import numpy as np
import polars as pl
import pytest

from cameratrap.arms import scoring


def test_group_mean_averages_rows_per_key():
    """Rows sharing a key are averaged; keys come back sorted."""
    keys, means = scoring.group_mean(["b", "a", "b"], np.array([[1.0], [5.0], [3.0]]))
    assert keys == ["a", "b"]
    assert means[:, 0].tolist() == [5.0, 2.0]


def test_sequence_averages_logits_not_votes():
    """One confident frame can outweigh two weak ones, as logit averaging intends."""
    logits = np.array([[0.0, 0.4], [0.0, 0.4], [3.0, 0.0]])
    preds = scoring.sequence_predictions(["s"] * 3, logits, ["a", "b"])
    assert preds["pred"].to_list() == ["a"]
    assert preds["msp"][0] == pytest.approx(
        scoring.softmax(np.array([[1.0, 0.8 / 3]]))[0, 0]
    )


def test_softmax_rows_sum_to_one():
    """Large logits do not overflow and each row sums to one."""
    probs = scoring.softmax(np.array([[1000.0, 1000.0], [0.0, 1.0]]))
    assert probs.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert probs[0].tolist() == [0.5, 0.5]


def test_balanced_accuracy_ignores_absent_classes():
    """A site with two classes is scored on those two; abstentions count as wrong."""
    labels = ["a", "a", "a", "b"]
    preds = ["a", "a", None, "b"]
    assert scoring.balanced_accuracy(labels, preds) == pytest.approx((2 / 3 + 1) / 2)


def test_per_site_drops_small_sites():
    """Sites below the minimum sequence count are not scored."""
    scored = pl.DataFrame(
        {
            "site": ["A"] * 3 + ["B"],
            "label": ["x", "x", "y", "x"],
            "pred": ["x", "y", "y", "x"],
        }
    )
    sites = scoring.per_site(scored, min_sequences=2)
    assert sites["site"].to_list() == ["A"]
    assert sites["balanced_accuracy"][0] == pytest.approx(0.75)
    assert sites["accuracy"][0] == pytest.approx(2 / 3)
