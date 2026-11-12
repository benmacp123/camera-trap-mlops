"""Tests for the Lambda handler, run without torch or AWS."""

import io

import numpy as np
import pytest

from cloud import handler


def fake_predict(files):
    """Return logits that encode each file's first byte.

    Args:
        files: Local crop files.

    Returns:
        Array of shape ``(len(files), 2)``.
    """
    return np.array([[f.read_bytes()[0], len(files)] for f in files], dtype=float)


@pytest.fixture(name="store")
def fixture_store(tmp_path):
    """A folder store holding three one-byte crops.

    Args:
        tmp_path: Pytest's temporary folder.

    Returns:
        The ``FolderStore``.
    """
    for i, key in enumerate(["A/1.JPG", "A/2.JPG", "B/1.JPG"]):
        path = tmp_path / "bucket" / "crops" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([i + 1]))
    return handler.FolderStore(tmp_path)


def event(keys):
    """Build a handler event.

    Args:
        keys: Crop keys.

    Returns:
        The event dict.
    """
    return {
        "bucket": "bucket",
        "prefix": "crops/",
        "keys": keys,
        "output_key": "runs/t/shard_00000.npz",
    }


def test_process_stores_logits_in_key_order(store, tmp_path):
    """Each key's logits land in its own row, even when file names repeat."""
    keys = ["B/1.JPG", "A/1.JPG", "A/2.JPG"]
    result = handler.process(event(keys), store, fake_predict)
    assert result["n"] == 3
    with np.load(tmp_path / "bucket" / "runs/t/shard_00000.npz") as data:
        assert data["keys"].tolist() == keys
        assert data["logits"][:, 0].tolist() == [3.0, 1.0, 2.0]


def test_process_reports_its_timings(store):
    """The result carries the three timings the cost report separates."""
    result = handler.process(event(["A/1.JPG"]), store, fake_predict)
    for name in ("download_seconds", "model_seconds", "upload_seconds"):
        assert result[name] >= 0


def test_missing_crop_fails_the_whole_shard(store, tmp_path):
    """A shard with a missing crop writes nothing, never a partial result."""
    with pytest.raises(FileNotFoundError):
        handler.process(event(["A/1.JPG", "A/9.JPG"]), store, fake_predict)
    assert not (tmp_path / "bucket" / "runs").exists()


def test_empty_shard_is_refused(store):
    """An event with no crops is an error, not an empty result."""
    with pytest.raises(ValueError, match="no crops"):
        handler.process(event([]), store, fake_predict)


def test_scratch_files_are_removed(store, tmp_path):
    """Downloaded crops are deleted, so a warm container's /tmp never fills."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    handler.process(event(["A/1.JPG"]), store, fake_predict, scratch=scratch)
    assert not list(scratch.iterdir())


def test_handler_loads_the_model_once(store, monkeypatch):
    """Only the first invocation of a container is cold."""
    loads = []
    monkeypatch.setattr(handler, "_STATE", {})
    monkeypatch.setattr(handler, "default_store", lambda: store)
    monkeypatch.setattr(handler, "load_model", lambda: loads.append(1) or fake_predict)
    first = handler.handler(event(["A/1.JPG"]), None)
    second = handler.handler(event(["A/2.JPG"]), None)
    assert (first["cold"], second["cold"]) == (True, False)
    assert len(loads) == 1


def test_default_store_follows_local_root(monkeypatch, tmp_path):
    """LOCAL_ROOT switches the handler to the folder store for the emulator."""
    monkeypatch.setenv("LOCAL_ROOT", str(tmp_path))
    assert isinstance(handler.default_store(), handler.FolderStore)


def test_pack_round_trips():
    """A packed result reads back with its keys and logits."""
    body = handler.pack(["a", "b"], np.eye(2))
    with np.load(io.BytesIO(body)) as data:
        assert data["keys"].tolist() == ["a", "b"]
        np.testing.assert_array_equal(data["logits"], np.eye(2))
