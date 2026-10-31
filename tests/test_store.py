"""Tests for the embedding cache store, which every downstream stage reads."""

import numpy as np
import pytest

from cameratrap.embed import store


def sample_cache(tmp_path):
    """Write a three-row cache and return its path and contents.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        A tuple ``(path, paths, vectors, meta)``.
    """
    path = store.cache_path("toy", root=tmp_path)
    paths = ["A01/1.JPG", "A01/2.JPG", "B02/3.JPG"]
    vectors = np.eye(3, dtype=np.float32)
    meta = {"backbone": "toy", "index_sha256": "abc"}
    store.write(path, paths, vectors, meta, extras={"scale": np.float32(2.0)})
    return path, paths, vectors, meta


def test_round_trip(tmp_path):
    """A written cache reads back with the same paths, vectors, meta and extras."""
    path, paths, vectors, meta = sample_cache(tmp_path)
    cache = store.read(path)
    assert cache["paths"] == paths
    np.testing.assert_array_equal(cache["vectors"], vectors)
    assert cache["meta"] == meta
    assert float(cache["scale"]) == 2.0
    assert not list(path.parent.glob("*.part*"))


def test_write_rejects_mismatch_and_duplicates(tmp_path):
    """Paths and vectors must line up one to one, with no repeated path."""
    path = tmp_path / "x.npz"
    with pytest.raises(ValueError, match="2 paths but 1 vectors"):
        store.write(path, ["a", "b"], np.zeros((1, 2)), {})
    with pytest.raises(ValueError, match="duplicate"):
        store.write(path, ["a", "a"], np.zeros((2, 2)), {})


def test_rows_for_follows_requested_order(tmp_path):
    """Rows come back in the caller's order, and an unknown path is an error."""
    path, _, _, _ = sample_cache(tmp_path)
    cache = store.read(path)
    rows = store.rows_for(cache, ["B02/3.JPG", "A01/1.JPG"])
    np.testing.assert_array_equal(rows, [[0, 0, 1], [1, 0, 0]])
    with pytest.raises(KeyError):
        store.rows_for(cache, ["missing.JPG"])


def test_check_meta_names_the_mismatch():
    """A cache made from a different crop index is refused, naming the key."""
    store.check_meta({"index_sha256": "abc"}, {"index_sha256": "abc"})
    with pytest.raises(ValueError, match="index_sha256"):
        store.check_meta({"index_sha256": "abc"}, {"index_sha256": "def"})


def test_cosine_agreement_ignores_scale():
    """Agreement is cosine, so rescaled vectors agree and orthogonal ones do not."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[3.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(store.cosine_agreement(a, b), [1.0, 0.0])


def test_file_sha256(tmp_path):
    """The file hash matches the known SHA-256 of a short string."""
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert store.file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
