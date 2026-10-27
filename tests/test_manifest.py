"""Tests for the manifest builder: which sequences and images it keeps, and
that the committed manifest still matches its recorded split freeze.

The builder runs on a small synthetic metadata set, one sequence per rule,
so CI needs neither the real metadata nor any images.
"""

import collections
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from cameratrap.data import manifest, orinoquia

COMMITTED = Path(__file__).resolve().parents[1] / "data" / "manifest"

# Tiny sites and classes, so every site is eligible and any draw passes.
LOOSE_SPLITS = {
    "n_holdout": 1,
    "n_folds": 2,
    "seed": 0,
    "min_site_sequences": 1,
    "min_sites_per_class": 0,
    "min_sequences_per_class": 0,
}

# seq_id -> (site, [(file, labels, has_file, datetime)]).
SEQUENCES = {
    "clean": (
        "A",
        [
            ("A/1.JPG", ["collared_peccary"], True, "2020-02-01 10:00:00+00:00"),
            ("A/2.JPG", ["empty"], True, "2020-02-01 10:00:01+00:00"),
        ],
    ),
    "human": (
        "A",
        [
            ("A/3.JPG", ["human"], False, "2020-02-02 10:00:00+00:00"),
            ("A/4.JPG", ["black_agouti"], True, "2020-02-02 10:00:01+00:00"),
        ],
    ),
    "empty": ("B", [("B/1.JPG", ["empty"], True, "2020-02-03 10:00:00+00:00")]),
    "multi": (
        "B",
        [
            (
                "B/2.JPG",
                ["black_agouti", "spixs_guan"],
                True,
                "2020-02-04 10:00:00+00:00",
            )
        ],
    ),
    "other": ("B", [("B/3.JPG", ["rodent"], True, "2020-02-05 10:00:00+00:00")]),
    "partial": (
        "C",
        [
            ("C/1.JPG", ["black_agouti"], True, "2020-03-01 10:00:00+00:00"),
            ("C/2.JPG", ["black_agouti"], False, "2020-03-01 10:00:01+00:00"),
            ("C/3.JPG", ["unknown"], True, "2019-03-01 10:00:02+00:00"),
        ],
    ),
    "stale": ("C", [("C/4.JPG", ["spixs_guan"], True, "2019-06-01 10:00:00+00:00")]),
    "second": ("B", [("B/4.JPG", ["spixs_guan"], True, "2020-04-01 10:00:00+00:00")]),
}


def make_inputs():
    """Return synthetic metadata and MDv5a results covering every rule.

    Returns:
        A pair ``(metadata, detections)`` shaped like LILA's files.
    """
    names = sorted(
        {lab for _, frames in SEQUENCES.values() for f in frames for lab in f[1]}
    )
    category = {name: i for i, name in enumerate(names)}
    metadata = {"images": [], "annotations": [], "categories": []}
    metadata["categories"] = [{"id": i, "name": n} for n, i in category.items()]
    detections = {
        "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "images": [],
    }
    for seq_id, (site, frames) in SEQUENCES.items():
        for frame_num, (path, labels, has_file, when) in enumerate(frames):
            metadata["images"].append(
                {
                    "id": path,
                    "file_name": path,
                    "location": site,
                    "seq_id": seq_id,
                    "frame_num": frame_num,
                    "datetime": when,
                }
            )
            metadata["annotations"].extend(
                {"image_id": path, "category_id": category[lab]} for lab in labels
            )
            if has_file:
                # A confident person box must not count as an animal.
                found = [
                    {"category": "1", "conf": 0.4},
                    {"category": "1", "conf": 0.7},
                    {"category": "2", "conf": 0.99},
                ]
                detections["images"].append(
                    {"file": path, "detections": found if path == "A/1.JPG" else []}
                )
    return metadata, detections


@pytest.fixture(name="built")
def fixture_built():
    """Return the manifest and report built from the synthetic inputs.

    Returns:
        The ``(frame, report)`` pair from ``manifest.build``.
    """
    metadata, detections = make_inputs()
    return manifest.build(metadata, detections, split_settings=LOOSE_SPLITS)


def test_keeps_only_clean_target_sequences(built):
    """Sequences of one target class survive with that label; the rest go."""
    frame, _ = built
    labels = dict(frame.select("seq_id", "label").unique().iter_rows())
    assert labels == {
        "clean": "collared_peccary",
        "partial": "black_agouti",
        "second": "spixs_guan",
    }


def test_reports_why_each_sequence_was_excluded(built):
    """Every excluded sequence is counted under its reason."""
    _, report = built
    assert report["sequences_excluded"] == {
        "contains human": 1,
        "multi-species": 1,
        "no animal label": 1,
        "other class": 1,
        "no usable image": 1,
    }
    assert report["images_dropped"] == {
        "no image file": 1,
        "timestamp out of bounds": 2,
    }


def test_human_sequence_dropped_whole(built):
    """A sequence that ever showed a person loses its frames that still have files."""
    frame, _ = built
    assert "A/4.JPG" not in frame["path"].to_list()


def test_drops_missing_and_out_of_bounds_frames(built):
    """Frames with no file or a timestamp outside the study period are dropped."""
    frame, _ = built
    assert frame.filter(pl.col("seq_id") == "partial")["path"].to_list() == ["C/1.JPG"]


def test_keeps_empty_frames_of_a_kept_sequence(built):
    """Empty frames stay, with their own image label, for detector-miss counts."""
    frame, _ = built
    row = frame.filter(pl.col("path") == "A/2.JPG").row(0, named=True)
    assert (row["label"], row["image_label"]) == ("collared_peccary", "empty")


def test_confidence_is_best_animal_box(built):
    """md_conf is the highest animal confidence, ignoring person boxes; 0 with none."""
    frame, _ = built
    conf = dict(frame.select("path", "md_conf").iter_rows())
    assert conf["A/1.JPG"] == pytest.approx(0.7)
    assert conf["A/2.JPG"] == 0.0


def test_each_site_has_one_split(built):
    """Every image at a site shares that site's split; never split by image."""
    frame, report = built
    per_site = frame.group_by("site").agg(pl.col("split").n_unique())
    assert per_site["split"].to_list() == [1] * per_site.height
    assert sorted(frame.filter(pl.col("split") == "holdout")["site"].unique()) == (
        report["holdout_sites"]
    )


def test_load_round_trips(built, tmp_path):
    """A written manifest loads back unchanged, empty checksums as nulls."""
    frame, _ = built
    path = tmp_path / "m.csv"
    frame.write_csv(path)
    loaded = manifest.load(path)
    assert loaded.equals(frame)
    assert loaded["sha256"].null_count() == loaded.height


def test_load_rejects_wrong_columns(tmp_path):
    """A CSV with other columns is refused rather than half-read."""
    path = tmp_path / "bad.csv"
    path.write_text("path,label\nA/1.JPG,x\n")
    with pytest.raises(ValueError):
        manifest.load(path)


def test_in_bounds_is_half_open():
    """The start of the window is inside it and the end is outside."""
    bounds = ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00")
    assert manifest.in_bounds("2020-01-01 00:00:00+00:00", bounds)
    assert not manifest.in_bounds("2021-01-01 00:00:00+00:00", bounds)


def write_zip(path, members):
    """Write a zip archive of JSON members.

    Args:
        path: Archive path.
        members: Mapping of member name to a JSON-serialisable object.

    Returns:
        The archive path.
    """
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, json.dumps(content))
    return path


def test_read_zipped_json_needs_one_member(tmp_path):
    """An archive with two JSON members is ambiguous and refused."""
    path = write_zip(tmp_path / "two.zip", {"a.json": {}, "b.json": {}})
    with pytest.raises(ValueError):
        orinoquia.read_zipped_json(path)


@pytest.fixture(name="archives")
def fixture_archives(tmp_path, monkeypatch):
    """Point the recorded archives at synthetic zips with their real checksums.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        The directory holding the archives.
    """
    metadata, detections = make_inputs()
    raw = tmp_path / "raw"
    raw.mkdir()
    records = {}
    for key, content in (("metadata", metadata), ("detections", detections)):
        path = write_zip(raw / f"{key}.zip", {f"{key}.json": content})
        records[key] = (f"https://example.org/{key}.zip", manifest.sha256_file(path))
    monkeypatch.setattr(orinoquia, "ARCHIVES", records)
    monkeypatch.setattr(manifest, "SPLIT_SETTINGS", LOOSE_SPLITS)
    return raw


def test_main_writes_manifest_and_provenance(archives, tmp_path):
    """The script writes the CSV and a provenance record with source and licence."""
    out = tmp_path / "out"
    assert manifest.main(["--raw-dir", str(archives), "--out-dir", str(out)]) == 0
    assert manifest.load(out / "orinoquia.csv").height == 4
    record = json.loads((out / "orinoquia.json").read_text())
    assert record["licence"] == orinoquia.LICENCE
    assert record["sequences"] == 3


def test_main_stops_on_checksum_mismatch(archives, tmp_path):
    """A changed upstream archive stops the build before anything is written."""
    with zipfile.ZipFile(archives / "metadata.zip", "a") as archive:
        archive.writestr("extra.txt", "changed")
    out = tmp_path / "out"
    assert manifest.main(["--raw-dir", str(archives), "--out-dir", str(out)]) == 1
    assert not out.exists()


def test_committed_manifest_matches_its_freeze():
    """The committed manifest's site splits hash to the recorded value.

    Moving a site between hold-out and folds after the freeze would fail
    here, as would a manifest edited without rebuilding its record.
    """
    frame = manifest.load(COMMITTED / "orinoquia.csv")
    record = json.loads((COMMITTED / "orinoquia.json").read_text())
    splits = dict(frame.select("site", "split").unique().iter_rows())
    assert manifest.splits_digest(splits) == record["splits_sha256"]
    sequences = collections.Counter(frame.unique("seq_id")["label"].to_list())
    assert dict(sequences) == record["sequences_per_class"]
    assert frame.height == record["images"]
