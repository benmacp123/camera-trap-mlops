"""Tests for the replay stream: site and date order, halves, and its checks."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from cameratrap.replay import stream

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def make_inputs(sequences):
    """Build a crop index and manifest from ``(seq_id, site, day)`` tuples.

    Args:
        sequences: One tuple per sequence; ``day`` offsets its first image.

    Returns:
        A pair ``(index, manifest)`` with two frames per sequence, the
        second a minute after the first.
    """
    index, images = [], []
    for seq_id, site, day in sequences:
        for frame in range(2):
            index.append(
                {"seq_id": seq_id, "site": site, "split": "fold0", "label": "tapir"}
            )
            images.append(
                {
                    "seq_id": seq_id,
                    "timestamp": START + timedelta(days=day, minutes=frame),
                }
            )
    return pl.DataFrame(index), pl.DataFrame(images)


def test_stream_is_site_then_date_ordered():
    """Sequences come out by site, then first timestamp, whatever the input order."""
    index, manifest = make_inputs(
        [("c", "B", 1), ("a", "A", 5), ("b", "A", 2), ("d", "B", 0)]
    )
    out = stream.sequence_stream(index, manifest)
    assert out["seq_id"].to_list() == ["b", "a", "d", "c"]
    assert out["position"].to_list() == [0, 1, 0, 1]
    assert out["timestamp"][0] == START + timedelta(days=2)


def test_stream_does_not_depend_on_input_order():
    """Shuffling both inputs gives the same stream: the replay is deterministic."""
    sequences = [(f"s{i}", "AB"[i % 2], (i * 7) % 5) for i in range(12)]
    index, manifest = make_inputs(sequences)
    first = stream.sequence_stream(index, manifest)
    second = stream.sequence_stream(
        index.sample(fraction=1.0, shuffle=True, seed=1),
        manifest.sample(fraction=1.0, shuffle=True, seed=2),
    )
    assert first.equals(second)


def test_ties_in_time_break_by_sequence_id():
    """Two sequences at the same instant are ordered by ID, not by chance."""
    index, manifest = make_inputs([("z", "A", 1), ("y", "A", 1)])
    assert stream.sequence_stream(index, manifest)["seq_id"].to_list() == ["y", "z"]


def test_odd_site_gives_the_extra_sequence_to_late():
    """A site with five sequences splits two early, three late."""
    index, manifest = make_inputs([(f"s{i}", "A", i) for i in range(5)])
    out = stream.sequence_stream(index, manifest)
    assert out["half"].to_list() == ["early"] * 2 + ["late"] * 3


def test_sequence_without_timestamp_is_refused():
    """A classified sequence missing from the manifest raises, never sorts last."""
    index, manifest = make_inputs([("a", "A", 0), ("b", "A", 1)])
    with pytest.raises(ValueError, match="no timestamp"):
        stream.sequence_stream(index, manifest.filter(pl.col("seq_id") != "b"))


def test_sequence_spanning_two_sites_is_refused():
    """Frames of one sequence that disagree on site raise."""
    index, manifest = make_inputs([("a", "A", 0)])
    bad = index.with_columns(pl.Series("site", ["A", "B"]))
    with pytest.raises(ValueError, match="disagree"):
        stream.sequence_stream(bad, manifest)
