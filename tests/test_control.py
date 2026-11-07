"""Tests for the CNN control's report: reference verdicts, and no effect on aliases."""

import json

import numpy as np
import polars as pl
import pytest

from cameratrap.gate import rule
from cameratrap.report import control

CLASSES = ["agouti", "tapir"]
SITES = [f"S{i}" for i in range(6)]
FAST_GATE = rule.GateConfig(null_draws=200, resamples=999)


def make_tables(per_site=60):
    """Build a synthetic crop index and size table: one frame per sequence.

    Args:
        per_site: Sequences per site.

    Returns:
        A pair ``(index, sizes)``.
    """
    rows = [
        {
            "path": f"{site}/{i:03d}.jpg",
            "seq_id": f"{site}-{i:03d}",
            "site": site,
            "split": f"fold{s % 3}",
            "label": CLASSES[i % 2],
        }
        for s, site in enumerate(SITES)
        for i in range(per_site)
    ]
    held = {
        "path": "H0/000.jpg",
        "seq_id": "H0-000",
        "site": "H0",
        "split": "holdout",
        "label": CLASSES[0],
    }
    index = pl.DataFrame(rows + [held]).sort("path")
    sizes = pl.DataFrame({"path": index["path"], "side": [224] * index.height})
    return index, sizes


def predictions(frames, wrong_every):
    """Return per-sequence predictions wrong at a fixed rate.

    Args:
        frames: The fold-site frames.
        wrong_every: Every n-th sequence is wrong; 0 means none are.

    Returns:
        A per-sequence DataFrame.
    """
    labels = frames["label"].to_list()
    preds = [
        (
            CLASSES[1 - CLASSES.index(label)]
            if wrong_every and i % wrong_every == 0
            else label
        )
        for i, label in enumerate(labels)
    ]
    return pl.DataFrame(
        {
            "seq_id": frames["seq_id"],
            "site": frames["site"],
            "label": labels,
            "pred": preds,
            "msp": [0.9] * len(preds),
        }
    )


def logits_for(frames, wrong_every):
    """Return frame logits that predict at a fixed error rate.

    Args:
        frames: The fold-site frames.
        wrong_every: Every n-th frame is wrong; 0 means none are.

    Returns:
        An array of shape ``(n, 2)``.
    """
    table = predictions(frames, wrong_every)
    return np.array(
        [[2.0, 0.0] if p == "agouti" else [0.0, 2.0] for p in table["pred"]]
    )


def score(control_wrong, opponent_wrong):
    """Run the report on synthetic predictions.

    Args:
        control_wrong: The control's error spacing.
        opponent_wrong: Both opponents' error spacing.

    Returns:
        The output of ``control.run``.
    """
    index, sizes = make_tables()
    frames = index.filter(pl.col("split") != "holdout").sort("path")
    opponents = {arm: predictions(frames, opponent_wrong) for arm in control.OPPONENTS}
    return control.run(
        index, sizes, logits_for(frames, control_wrong), CLASSES, opponents, FAST_GATE
    )


def test_a_losing_control_gets_a_failing_verdict():
    """A control much worse than each opponent would not pass the gate."""
    summary, *_ = score(control_wrong=3, opponent_wrong=20)
    for verdict in summary["reference_verdicts"].values():
        assert verdict["would_pass"] is False
        assert verdict["mean_gain"] < 0


def test_a_winning_control_is_reported_as_passing():
    """A control better than its opponents is reported as such, not forced to fail."""
    summary, *_ = score(control_wrong=20, opponent_wrong=3)
    for verdict in summary["reference_verdicts"].values():
        assert verdict["would_pass"] is True
        assert verdict["mean_gain"] > 0


def test_holdout_is_neither_scored_nor_counted():
    """No hold-out sequence or site reaches any output."""
    summary, per_site, _, scored, gains = score(control_wrong=3, opponent_wrong=20)
    assert "H0" not in scored["site"].to_list()
    assert "H0" not in per_site["site"].to_list()
    assert "H0" not in gains["site"].to_list()
    assert summary["sequences"] == len(SITES) * 60
    assert "hold-out neither" in summary["scope"]


def test_opponent_on_other_sequences_is_refused():
    """Arms scored on different sequences cannot be compared."""
    index, sizes = make_tables()
    frames = index.filter(pl.col("split") != "holdout").sort("path")
    opponents = {arm: predictions(frames, 3) for arm in control.OPPONENTS}
    opponents["bioclip2"] = opponents["bioclip2"].head(10)
    with pytest.raises(ValueError, match="other sequences"):
        control.run(index, sizes, logits_for(frames, 3), CLASSES, opponents, FAST_GATE)


def test_load_logits_refuses_other_frames(tmp_path):
    """Logits for other frames, or in another order, are refused."""
    index, _ = make_tables()
    frames = index.filter(pl.col("split") != "holdout").sort("path")
    path = tmp_path / "logits.npz"
    np.savez(
        path,
        paths=np.array(frames["path"].to_list()[::-1]),
        logits=logits_for(frames, 3),
        classes=np.array(CLASSES),
        meta=np.array(json.dumps({})),
    )
    with pytest.raises(ValueError, match="fold-site frames"):
        control.load_logits(path, frames)


def test_plot_gain_writes_a_figure(tmp_path):
    """The figure is drawn from the gains table."""
    *_, gains = score(control_wrong=3, opponent_wrong=20)
    path = tmp_path / "gain.png"
    control.plot_gain(gains, path)
    assert path.stat().st_size > 0
