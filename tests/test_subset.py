"""Tests for the spike selection: hold-out sites stay unseen, the caps hold,
and the committed selection is what the committed manifest produces.
"""

from pathlib import Path

import polars as pl
import pytest

from cameratrap.data import manifest, subset

COMMITTED = Path(__file__).resolve().parents[1] / "data" / "manifest"

SETTINGS = {
    "sequences_per_class_site": 2,
    "frames_per_sequence": 2,
    "min_conf": 0.2,
    "n_detector_check": 4,
    "seed": 1,
}


def make_manifest():
    """Return a small manifest: three sequences of three frames at site A,
    one held-out sequence at site H.

    Returns:
        A DataFrame with the manifest schema.
    """
    rows = []
    for seq, site, split in (
        ("s1", "A", "fold0"),
        ("s2", "A", "fold0"),
        ("s3", "A", "fold0"),
        ("h1", "H", "holdout"),
    ):
        for frame_num, conf in enumerate((0.9, 0.1, 0.5)):
            rows.append(
                {
                    "path": f"{site}/{seq}_{frame_num}.JPG",
                    "sha256": None,
                    "site": site,
                    "timestamp": "2020-02-01 10:00:00+00:00",
                    "seq_id": seq,
                    "frame_num": frame_num,
                    "label": "black_agouti",
                    "image_label": "black_agouti",
                    "md_conf": conf,
                    "split": split,
                }
            )
    return pl.DataFrame(rows, schema=manifest.SCHEMA, orient="row")


@pytest.fixture(name="chosen")
def fixture_chosen():
    """Return the selection from the small manifest, joined back to it.

    Returns:
        The selection with every manifest column attached.
    """
    frame = make_manifest()
    return subset.select(frame, **SETTINGS).join(frame, on="path")


def test_holdout_never_selected(chosen):
    """No image from a hold-out site is fetched before the freeze."""
    assert "holdout" not in chosen["split"].to_list()


def test_caps_sequences_per_class_and_site(chosen):
    """At most the capped number of sequences of one class at one site."""
    assert chosen.filter("classify")["seq_id"].n_unique() == 2


def test_keeps_most_confident_frames(chosen):
    """Each sequence keeps its most confident frames, never one below the threshold."""
    confs = chosen.filter("classify")["md_conf"].to_list()
    assert sorted(set(confs)) == [0.5, 0.9]


def test_detector_check_includes_low_confidence(chosen):
    """The detector check is random over all fold images, empty-looking ones included."""
    check = chosen.filter("detector_check")
    assert check.height == 4
    assert set(check["split"].to_list()) == {"fold0"}


def test_selection_fixed_by_seed():
    """The same seed gives the same selection."""
    frame = make_manifest()
    assert subset.select(frame, **SETTINGS).equals(subset.select(frame, **SETTINGS))


def test_committed_selection_matches_manifest():
    """The committed spike selection is what the committed manifest and settings give."""
    frame = manifest.load(COMMITTED / "orinoquia.csv")
    committed = pl.read_csv(COMMITTED / "orinoquia_spike.csv")
    assert subset.select(frame, **subset.SPIKE_SETTINGS).equals(committed)
