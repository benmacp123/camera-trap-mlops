"""Tests for the detector-miss report: misses are counted per sequence and
per frame, person losses are kept apart from detector misses, and the
committed outputs reconcile with the crop index.
"""

import polars as pl
import pytest

from cameratrap.data import crops
from cameratrap.report import detector


def make_frames():
    """Return a small manifest slice: a found, a missed and a person-lost sequence.

    Returns:
        A DataFrame with the manifest columns the report reads.
    """
    rows = [
        # s1 at A: found, but frame 1 carries the label and no box.
        ("A/1.JPG", "s1", "A", "fold0", "spotted_paca", "spotted_paca", 0.9),
        ("A/2.JPG", "s1", "A", "fold0", "spotted_paca", "spotted_paca", 0.05),
        ("A/3.JPG", "s1", "A", "fold0", "spotted_paca", "empty", 0.0),
        # s2 at A: no frame reaches the threshold.
        ("A/4.JPG", "s2", "A", "fold0", "lowland_tapir", "lowland_tapir", 0.15),
        # s3 at B: a box, but its only frame showed a person.
        ("B/1.JPG", "s3", "B", "holdout", "black_agouti", "black_agouti", 0.8),
    ]
    return pl.DataFrame(
        rows,
        schema=["path", "seq_id", "site", "split", "label", "image_label", "md_conf"],
        orient="row",
    )


@pytest.fixture(name="sequences")
def fixture_sequences():
    """Return the sequence table for the small slice.

    Returns:
        Output of ``detector.sequence_table``.
    """
    index = pl.DataFrame({"path": ["A/1.JPG"], "seq_id": ["s1"]})
    excluded = pl.DataFrame({"path": ["B/1.JPG"], "seq_id": ["s3"]})
    return detector.sequence_table(make_frames(), index, excluded, 0.2)


def test_sequence_outcomes(sequences):
    """Each sequence is found, missed or lost to a person box, never two of these."""
    rows = {r["seq_id"]: r for r in sequences.iter_rows(named=True)}
    assert (rows["s1"]["missed"], rows["s1"]["crops"]) == (False, 1)
    assert rows["s2"]["missed"] and not rows["s2"]["lost_to_person"]
    assert rows["s3"]["lost_to_person"] and not rows["s3"]["missed"]


def test_frame_misses_count_only_labelled_frames(sequences):
    """An empty frame inside an animal sequence is not a detector miss."""
    s1 = sequences.filter(pl.col("seq_id") == "s1").row(0, named=True)
    assert (s1["frames_labelled"], s1["frames_missed"]) == (2, 1)


def test_per_site_rates(sequences):
    """Site rates divide misses by that site's sequences and labelled frames."""
    per_site = detector.summarise(sequences, ["site", "split"])
    site_a = per_site.filter(pl.col("site") == "A").row(0, named=True)
    assert site_a["sequence_miss_rate"] == 0.5
    assert site_a["frame_miss_rate"] == pytest.approx(2 / 3)
    record = detector.summary(sequences, per_site, 0.2)
    assert record["worst_site"] == "A" and record["sites_with_a_miss"] == 1


def test_committed_report_reconciles(tmp_path):
    """Every manifest sequence is found, missed or lost, and crops match the index."""
    assert detector.main(["--out-dir", str(tmp_path)]) == 0
    record = (tmp_path / "summary.json").read_text()
    summary = pl.read_json(tmp_path / "summary.json").row(0, named=True)
    assert summary["sequences"] == (
        summary["sequences_with_crop"]
        + summary["sequences_missed"]
        + summary["sequences_lost_to_person"]
    )
    assert summary["crops"] == crops.load_index().height
    committed = detector.OUT_DIR / "summary.json"
    assert record == committed.read_text()
