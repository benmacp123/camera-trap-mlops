"""Tests for crops: padded squares that stay inside the image, cut around
the most confident animal box; the index keeps each sequence's best frames
and drops person frames; the store holds exactly the indexed crops.
"""

from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from cameratrap.data import crops, manifest, orinoquia

COMMITTED = Path(__file__).resolve().parents[1] / "data" / "manifest"


def test_square_box_is_padded_square():
    """A wide box becomes a square on its long side plus padding."""
    left, top, right, bottom = crops.square_box(
        [0.4, 0.4, 0.2, 0.1], 1000, 1000, pad=0.1
    )
    assert right - left == bottom - top == 240
    assert (left + right) / 2 == 500 and (top + bottom) / 2 == 450


def test_square_box_shifts_inside_image():
    """A box at the edge is shifted, not cut, so the crop stays square."""
    box = crops.square_box([0.0, 0.9, 0.1, 0.1], 1000, 800, pad=0.0)
    assert box == (0, 700, 100, 800)


def test_square_box_never_exceeds_short_side():
    """A box larger than the image shrinks to the image's short side."""
    left, top, right, bottom = crops.square_box([0.0, 0.0, 1.0, 1.0], 1920, 1440)
    assert (right - left, bottom - top) == (1440, 1440)
    assert 0 <= left and right <= 1920


def test_top_animal_ignores_people_and_threshold():
    """The crop is the most confident animal box at or above the threshold."""
    dets = [
        {"category": "2", "conf": 0.99},
        {"category": "1", "conf": 0.3},
        {"category": "1", "conf": 0.6},
    ]
    assert crops.top_animal(dets)["conf"] == 0.6
    assert crops.top_animal(dets, min_conf=0.7) is None


def test_crop_cuts_the_box(tmp_path):
    """The crop has the square box's size and is RGB."""
    path = tmp_path / "x.jpg"
    Image.new("L", (400, 300)).save(path)
    image = crops.crop(path, {"bbox": [0.25, 0.25, 0.25, 0.25]})
    assert image.mode == "RGB"
    assert image.size == (120, 120)


def make_manifest():
    """Return a small manifest: sequence s1 of four frames, s2 of one.

    Returns:
        A DataFrame with the manifest schema.
    """
    rows = [
        (f"A/s1_{n}.JPG", "A", "s1", n, "black_agouti", "fold0") for n in range(4)
    ] + [("H/s2_0.JPG", "H", "s2", 0, "lowland_tapir", "holdout")]
    return pl.DataFrame(
        [
            {
                "path": path,
                "sha256": None,
                "site": site,
                "timestamp": "2020-02-01 10:00:00+00:00",
                "seq_id": seq,
                "frame_num": num,
                "label": label,
                "image_label": label,
                "md_conf": 0.0,
                "split": split,
            }
            for path, site, seq, num, label, split in rows
        ],
        schema=manifest.SCHEMA,
        orient="row",
    )


def animal(conf, bbox=(0.1, 0.1, 0.2, 0.2)):
    """Return one MDv5a animal detection.

    Args:
        conf: Detection confidence.
        bbox: Normalised ``[x, y, w, h]`` box.

    Returns:
        The detection dict.
    """
    return {"category": "1", "conf": conf, "bbox": list(bbox)}


DETECTIONS = {
    # s1: frame 1 is the best animal frame but also shows a person, frame 3
    # is below the threshold, so frames 0 and 2 survive plus nothing else.
    "A/s1_0.JPG": [animal(0.5), animal(0.3)],
    "A/s1_1.JPG": [animal(0.95), {"category": "2", "conf": 0.4, "bbox": [0] * 4}],
    "A/s1_2.JPG": [animal(0.7), {"category": "2", "conf": 0.1, "bbox": [0] * 4}],
    "A/s1_3.JPG": [animal(0.1)],
    "H/s2_0.JPG": [animal(0.9, (0.5, 0.5, 0.1, 0.1))],
}


@pytest.fixture(name="selected")
def fixture_selected():
    """Return the index and excluded frames from the small manifest.

    Returns:
        The pair ``(index, excluded)`` from ``crops.select``.
    """
    return crops.select(make_manifest(), DETECTIONS, **crops.CROP_SETTINGS)


def test_person_frame_dropped_before_ranking(selected):
    """A frame with a person box is excluded and the next-best frames fill in."""
    index, excluded = selected
    assert excluded["path"].to_list() == ["A/s1_1.JPG"]
    assert index.filter(pl.col("seq_id") == "s1")["path"].to_list() == [
        "A/s1_0.JPG",
        "A/s1_2.JPG",
    ]


def test_index_keeps_top_box_and_schema(selected):
    """Each row carries the frame's most confident animal box, in index columns."""
    index, _ = selected
    assert index.columns == list(crops.INDEX_SCHEMA)
    row = index.filter(pl.col("path") == "A/s1_0.JPG").row(0, named=True)
    assert row["conf"] == 0.5 and row["person_conf"] == 0.0
    tapir = index.filter(pl.col("seq_id") == "s2").row(0, named=True)
    assert [tapir[k] for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")] == [
        0.5,
        0.5,
        0.1,
        0.1,
    ]


def test_frames_per_sequence_cap():
    """No sequence gives more crops than the cap, best frames first."""
    settings = {**crops.CROP_SETTINGS, "frames_per_sequence": 1}
    index, _ = crops.select(make_manifest(), DETECTIONS, **settings)
    assert index["path"].to_list() == ["A/s1_2.JPG", "H/s2_0.JPG"]


def test_write_crop_caps_size_and_round_trips(tmp_path):
    """A large crop is shrunk to the cap, saved as JPEG and indexed crops are found."""
    source = tmp_path / "images" / "A" / "big.JPG"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (2000, 1500), "green").save(source)
    index = pl.DataFrame(
        [("A/big.JPG", 0.1, 0.1, 0.5, 0.5), ("A/gone.JPG", 0.1, 0.1, 0.5, 0.5)],
        schema=["path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        orient="row",
    )
    written, failures = crops.write_all(index, tmp_path / "images", tmp_path / "c")
    assert written == 1 and list(failures) == ["A/gone.JPG"]
    with Image.open(tmp_path / "c" / "A" / "big.JPG") as stored:
        assert stored.format == "JPEG"
        assert stored.size == (crops.MAX_SIDE, crops.MAX_SIDE)
    assert crops.missing_crops(index, tmp_path / "c") == ["A/gone.JPG"]
    assert not list((tmp_path / "c").rglob("*.part"))


def test_write_skips_crops_on_disk(tmp_path):
    """A re-run writes nothing for crops already stored."""
    source = tmp_path / "images" / "x.JPG"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (300, 200)).save(source)
    index = pl.DataFrame(
        [("x.JPG", 0.0, 0.0, 0.5, 0.5)],
        schema=["path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        orient="row",
    )
    assert crops.write_all(index, tmp_path / "images", tmp_path / "c") == (1, {})
    assert crops.write_all(index, tmp_path / "images", tmp_path / "c") == (0, {})


def test_load_index_rejects_wrong_columns(tmp_path):
    """An index with different columns is refused rather than misread."""
    path = tmp_path / "bad.csv"
    pl.DataFrame({"path": ["a"]}).write_csv(path)
    with pytest.raises(ValueError):
        crops.load_index(path)


@pytest.fixture(name="committed")
def fixture_committed():
    """Return the committed manifest, crop index and excluded frames.

    Returns:
        A triple ``(manifest, index, excluded)`` of DataFrames.
    """
    return (
        manifest.load(COMMITTED / "orinoquia.csv"),
        crops.load_index(COMMITTED / "orinoquia_crops.csv"),
        crops.load_index(COMMITTED / "orinoquia_crops_excluded.csv"),
    )


def test_committed_index_has_no_human(committed):
    """No crop comes from a frame or sequence with a human label or a person box."""
    frame, index, _ = committed
    joined = index.join(frame, on="path", suffix="_manifest")
    assert joined.height == index.height
    assert set(joined["label"]) <= set(orinoquia.CLASSES)
    assert orinoquia.HUMAN not in set(joined["image_label"])
    assert index["person_conf"].max() < crops.CROP_SETTINGS["person_conf"]


def test_committed_index_follows_settings(committed):
    """The committed index obeys the frame cap and threshold and matches the manifest."""
    frame, index, excluded = committed
    settings = crops.CROP_SETTINGS
    assert index["conf"].min() >= settings["min_conf"]
    assert (
        index.group_by("seq_id").len()["len"].max() == settings["frames_per_sequence"]
    )
    assert excluded["person_conf"].min() >= settings["person_conf"]
    assert not set(index["path"]) & set(excluded["path"])
    # Each crop's label, site and split are the manifest's, not a stale copy.
    joined = index.join(frame, on="path", suffix="_manifest")
    for column in ("seq_id", "site", "split", "label", "frame_num"):
        assert joined[column].equals(joined[f"{column}_manifest"], check_names=False)


@pytest.mark.skipif(
    not (manifest.RAW_DIR / "orinoquia_camera_traps_metadata.zip").exists(),
    reason="raw LILA archives not downloaded (make manifest)",
)
def test_committed_index_rebuilds_from_raw(committed):
    """The committed index is what the manifest and raw detections give."""
    frame, index, excluded = committed
    rebuilt, rebuilt_excluded = crops.select(
        frame, manifest.load_detections(), **crops.CROP_SETTINGS
    )
    assert rebuilt.equals(index) and rebuilt_excluded.equals(excluded)
