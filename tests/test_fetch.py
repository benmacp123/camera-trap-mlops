"""Tests for the image fetcher, using ``file://`` URLs so no network is needed.

They protect resumption (files on disk are hashed, not fetched again),
failure reporting, and the rule that a recorded checksum is never
silently overwritten.
"""

import hashlib
import json

import polars as pl
import pytest

from cameratrap.data import fetch, manifest


@pytest.fixture(name="source")
def fixture_source(tmp_path):
    """Return a ``file://`` base URL serving two small images.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A pair ``(base_url, contents)`` with contents keyed by manifest path.
    """
    contents = {"A/1.JPG": b"first image", "B/2.JPG": b"second image"}
    root = tmp_path / "mirror"
    for path, data in contents.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    return root.as_uri() + "/", contents


def sha(data):
    """Return the SHA-256 hex digest of bytes.

    Args:
        data: The bytes to hash.

    Returns:
        The digest as a hex string.
    """
    return hashlib.sha256(data).hexdigest()


def test_download_writes_file_and_hash(source, tmp_path):
    """A download lands under its manifest path with its hash and no partial file."""
    base_url, contents = source
    dest = tmp_path / "images" / "A" / "1.JPG"
    assert fetch.download(base_url + "A/1.JPG", dest) == sha(contents["A/1.JPG"])
    assert dest.read_bytes() == contents["A/1.JPG"]
    assert not list(dest.parent.glob("*.part"))


def test_existing_file_is_hashed_not_fetched(tmp_path):
    """A file already on disk is hashed, so an interrupted fetch resumes."""
    image_dir = tmp_path / "images"
    (image_dir / "A").mkdir(parents=True)
    (image_dir / "A" / "1.JPG").write_bytes(b"local")
    unreachable = (tmp_path / "nowhere").as_uri() + "/"
    assert fetch.fetch_one("A/1.JPG", unreachable, image_dir) == sha(b"local")


def test_failures_are_reported_not_raised(source, tmp_path, monkeypatch):
    """A missing source image is listed as a failure; the others still arrive."""
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    base_url, contents = source
    hashes, failures = fetch.fetch(
        ["A/1.JPG", "B/2.JPG", "C/missing.JPG"],
        base_url,
        tmp_path / "images",
        workers=2,
    )
    assert hashes == {p: sha(d) for p, d in contents.items()}
    assert list(failures) == ["C/missing.JPG"]


def manifest_frame(recorded):
    """Return a two-image manifest with the given recorded checksums.

    Args:
        recorded: Checksums for ``A/1.JPG`` and ``B/2.JPG``, or ``None``.

    Returns:
        A DataFrame with the manifest schema.
    """
    rows = [
        {
            "path": path,
            "sha256": digest,
            "site": path[0],
            "timestamp": "2020-02-01 10:00:00+00:00",
            "seq_id": path,
            "frame_num": 0,
            "label": "black_agouti",
            "image_label": "black_agouti",
            "md_conf": 0.9,
            "split": "fold0",
        }
        for path, digest in zip(("A/1.JPG", "B/2.JPG"), recorded)
    ]
    return pl.DataFrame(rows, schema=manifest.SCHEMA, orient="row")


def test_record_fills_empty_checksums():
    """Fresh hashes fill empty checksums and leave unfetched images empty."""
    frame, mismatches = fetch.record_checksums(
        manifest_frame([None, None]), {"A/1.JPG": "aa"}
    )
    assert frame["sha256"].to_list() == ["aa", None]
    assert frame.columns == list(manifest.SCHEMA)
    assert not mismatches


def test_record_keeps_and_reports_mismatch():
    """A hash that disagrees with the manifest is reported and not written."""
    frame, mismatches = fetch.record_checksums(
        manifest_frame(["aa", "bb"]), {"A/1.JPG": "aa", "B/2.JPG": "zz"}
    )
    assert frame["sha256"].to_list() == ["aa", "bb"]
    assert mismatches == ["B/2.JPG"]


def test_main_records_checksums(source, tmp_path):
    """The script fetches the selection, fills the manifest and notes the fetch."""
    base_url, contents = source
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest_frame([None, None]).write_csv(manifest_dir / "orinoquia.csv")
    (manifest_dir / "orinoquia.json").write_text("{}")
    selection = tmp_path / "selection.csv"
    selection.write_text("path\nA/1.JPG\nB/2.JPG\n")
    argv = [
        "--manifest-dir",
        str(manifest_dir),
        "--selection",
        str(selection),
        "--image-dir",
        str(tmp_path / "images"),
        "--base-url",
        base_url,
    ]

    assert fetch.main(argv) == 0
    loaded = manifest.load(manifest_dir / "orinoquia.csv")
    assert loaded["sha256"].to_list() == [sha(d) for d in contents.values()]
    record = json.loads((manifest_dir / "orinoquia.json").read_text())
    assert record["images_fetched"]["checksums_recorded"] == 2

    # A second run only verifies, so the record does not change.
    assert fetch.main(argv) == 0
    assert json.loads((manifest_dir / "orinoquia.json").read_text()) == record
