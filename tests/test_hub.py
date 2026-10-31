"""Tests for publishing and fetching the Phase 2 caches, without touching the network."""

import json

import numpy as np
import pytest

from cameratrap.arms import speciesnet_arm
from cameratrap.data import orinoquia
from cameratrap.embed import hub, store

TAXA = {c: orinoquia.TAXA[c] for c in ("black_agouti", "spixs_guan")}
AGOUTI, GUAN = TAXA["black_agouti"][2], TAXA["spixs_guan"][2]


def full_predictions(image_dir):
    """Return a SpeciesNet output with absolute paths and fields the arms never read.

    Args:
        image_dir: Local image root to prefix the paths with.

    Returns:
        A dict in SpeciesNet's output format.
    """
    detection = {
        "category": "1",
        "label": "animal",
        "conf": 0.9,
        "bbox": [0.1, 0.1, 0.2, 0.2],
        "raw_classifications": {
            "classes": ["many", "labels"],
            "scores": [0.9, 0.1],
            "target_classes": [GUAN, AGOUTI],
            "target_logits": [1.0, 5.0],
        },
        "processed_classification": {"class": AGOUTI, "source": "classifier"},
    }
    return {
        "info": {"format_version": "x"},
        "predictions": [
            {
                "filepath": str(image_dir.resolve() / "A" / "1.JPG"),
                "country": "COL",
                "detections": [detection],
            },
            {"filepath": str(image_dir.resolve() / "B" / "2.JPG"), "detections": []},
        ],
    }


def test_slim_speciesnet_reads_back_the_same(tmp_path):
    """The slimmed, relative-path file gives the arms exactly what the full one does."""
    image_dir = tmp_path / "img"
    full = tmp_path / "full.json"
    full.write_text(json.dumps(full_predictions(image_dir)))
    slim = tmp_path / "slim.json"
    slim.write_text(json.dumps(hub.slim_speciesnet(full, image_dir)))

    # Read from a different image root, as a fresh clone would.
    elsewhere = tmp_path / "clone" / "img"
    a = speciesnet_arm.read_predictions(full, image_dir, TAXA)
    b = speciesnet_arm.read_predictions(slim, elsewhere, TAXA)
    assert list(a) == list(b) == ["A/1.JPG"]
    np.testing.assert_array_equal(a["A/1.JPG"]["logits"], b["A/1.JPG"]["logits"])
    assert a["A/1.JPG"]["open_label"] == b["A/1.JPG"]["open_label"]
    assert "scores" not in slim.read_text()


def test_card_names_licence_and_cautions():
    """The card carries the dataset licence, the revisions and both cautions."""
    meta = {
        "model_id": "imageomics/bioclip-2",
        "revision": "abc123",
        "input_size": 224,
        "index_sha256": "f00d",
    }
    text = hub.card({"bioclip2": meta})
    assert text.startswith("---\nlicense: cdla-permissive-1.0\n")
    assert "`abc123`" in text and "f00d" in text
    assert "Hold-out sites are included" in text
    assert "probably seen these images" in text


def test_download_keeps_matching_and_spares_differing(tmp_path, monkeypatch):
    """Matching files are kept, a differing local file is spared, a missing one is fetched."""
    cache = tmp_path / "cache"
    names = ["a.npz", "b.json", "c.json"]
    for name, body in zip(names[:2], (b"same", b"local")):
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(body)
    published = {"a.npz": b"same", "b.json": b"remote", "c.json": b"new"}
    record = {
        "repo_id": "u/r",
        "repo_type": "dataset",
        "revision": "deadbeef",
        "files": {},
    }
    for name, body in published.items():
        (tmp_path / "x").write_bytes(body)
        record["files"][name] = store.file_sha256(tmp_path / "x")
    record_path = tmp_path / "hub.json"
    record_path.write_text(json.dumps(record))

    calls = []

    def fake_download(repo_id, name, **kwargs):
        """Stand in for the Hub: write the published bytes and log the call.

        Args:
            repo_id: The repo asked for.
            name: The file asked for.
            **kwargs: ``repo_type``, ``revision`` and ``local_dir``.

        Returns:
            The local path written.
        """
        calls.append((repo_id, name, kwargs["revision"]))
        (kwargs["local_dir"] / name).write_bytes(published[name])
        return str(kwargs["local_dir"] / name)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    status = hub.download(record_path, cache)
    assert status == {
        "a.npz": "kept",
        "b.json": "skipped (differs locally)",
        "c.json": "downloaded",
    }
    assert calls == [("u/r", "c.json", "deadbeef")]
    assert hub.download(record_path, cache, force=True)["b.json"] == "downloaded"


def test_download_refuses_a_corrupt_file(tmp_path, monkeypatch):
    """A downloaded file whose hash differs from the record is an error."""
    record_path = tmp_path / "hub.json"
    record_path.write_text(
        json.dumps(
            {
                "repo_id": "u/r",
                "repo_type": "dataset",
                "revision": "x",
                "files": {"a.npz": "0" * 64},
            }
        )
    )
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda repo_id, name, **kw: (kw["local_dir"] / name).write_bytes(b"bad"),
    )
    (tmp_path / "cache").mkdir()
    with pytest.raises(ValueError, match="does not match"):
        hub.download(record_path, tmp_path / "cache")


def test_stage_refuses_cache_from_another_index(tmp_path):
    """Staging stops if a cache was embedded from a different crop index."""
    index = tmp_path / "index.csv"
    index.write_text("path\n")
    cache = tmp_path / "cache"
    for name in hub.EMBEDDINGS.values():
        store.write(cache / name, ["a"], np.ones((1, 2)), {"index_sha256": "other"})
    with pytest.raises(ValueError, match="index_sha256"):
        hub.stage(tmp_path / "stage", cache, index)
