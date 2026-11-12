"""Tests for the probe head applied as plain arrays."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cameratrap.arms import head


def test_logits_match_scikit_learn(tmp_path):
    """The numpy head gives the logits scikit-learn's decision_function gives."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 5))
    labels = np.arange(60) % 3
    model = LogisticRegression().fit(features, labels)
    path = tmp_path / "head.npz"
    np.savez(
        path,
        coef=model.coef_,
        intercept=model.intercept_,
        classes=np.array(["a", "b", "c"]),
    )
    loaded = head.load(path)
    assert loaded["classes"] == ["a", "b", "c"]
    np.testing.assert_allclose(
        head.logits(loaded, features), model.decision_function(features)
    )


def test_wrong_embedding_width_is_refused():
    """Embeddings from another backbone cannot be scored by mistake."""
    loaded = {
        "coef": np.zeros((3, 5)),
        "intercept": np.zeros(3),
        "classes": list("abc"),
    }
    with pytest.raises(ValueError, match="dimensions"):
        head.logits(loaded, np.zeros((2, 4)))


def test_committed_head_matches_the_class_list():
    """The promoted head on disk has one row per species and DINOv2's width."""
    loaded = head.load()
    assert loaded["coef"].shape == (9, 768)
    assert len(set(loaded["classes"])) == 9
