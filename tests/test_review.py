"""Tests for the simulated review loop and its report.

The synthetic deployment has three folds and a hold-out of small sites
whose frames sit near their class's mean, so a probe learns them quickly
and the tests run in seconds.
"""

import numpy as np
import polars as pl
import pytest

from cameratrap.gate import rule
from cameratrap.report import review
from cameratrap.review import queue, simulate

CLASSES = ["agouti", "paca", "tapir"]
SETTINGS = simulate.SETTINGS | {
    "budgets": [2, 4],
    "random_seeds": [0, 1],
    "min_half_sequences": 5,
    "bootstrap_resamples": 50,
    "primary": {"strategy": queue.DIVERSE, "budget": 4, "weight": 1.0},
}
GATE = rule.GateConfig(null_draws=200, resamples=200, min_groups=2)


def make_data(sites_per_split=2, sequences=14, seed=0):
    """Build a crop index, embeddings and replay table for a toy deployment.

    Args:
        sites_per_split: Sites in each of three folds and the hold-out.
        sequences: Sequences per site, two frames each.
        seed: Seed for the embeddings and the champion's confidences.

    Returns:
        A tuple ``(index, features, table)`` shaped like the real inputs;
        the champion's predictions are right except for every fifth
        sequence.
    """
    rng = np.random.default_rng(seed)
    means = np.eye(3, 6) * 4
    frames, rows, vectors = [], [], []
    for s, split in enumerate(["fold0", "fold1", "fold2", "holdout"]):
        for k in range(sites_per_split):
            site = f"S{s}{k}"
            for i in range(sequences):
                label = i % 3
                seq_id = f"{site}-{i:02d}"
                wrong = i % 5 == 0
                rows.append(
                    {
                        "seq_id": seq_id,
                        "site": site,
                        "split": split,
                        "label": CLASSES[label],
                        "position": i,
                        "half": "early" if i < sequences // 2 else "late",
                        "pred": CLASSES[(label + wrong) % 3],
                        "msp": float(rng.uniform(0.4, 1.0)),
                    }
                )
                for _ in range(2):
                    frames.append(
                        {
                            "path": f"{seq_id}-{len(frames)}",
                            "seq_id": seq_id,
                            "site": site,
                            "split": split,
                            "label": CLASSES[label],
                        }
                    )
                    vectors.append(means[label] + rng.normal(0, 1, 6))
    return pl.DataFrame(frames), np.array(vectors), pl.DataFrame(rows)


def cs():
    """Return a C for every split of the toy deployment."""
    return {s: 1.0 for s in ("fold0", "fold1", "fold2", "holdout")}


def test_cells_cover_the_grid_and_both_references():
    """One refit-only cell, a ceiling per weight, and every grid cell."""
    grid = simulate.cells(simulate.SETTINGS)
    assert len(grid) == 1 + 2 * (1 + (5 + 1 + 1) * 4)
    assert len(set(grid)) == len(grid)
    assert simulate.Cell(**simulate.SETTINGS["primary"], seed=0) in grid


def test_queues_need_no_labels_and_rank_the_whole_pool():
    """Queues build from predictions alone and each ranks every sequence once."""
    index, features, table = make_data()
    pool = table.filter(pl.col("half") == "early").drop("label")
    vectors = review.sequence_embeddings(index, features, table)
    early = (table["half"] == "early").to_numpy()
    queues = simulate.build_queues(pool, vectors[early], seeds=[0, 1])
    sizes = queues.group_by("site", "strategy", "seed").agg(
        pl.col("rank").max(), pl.col("seq_id").n_unique().alias("distinct")
    )
    assert sizes.height == 8 * 4
    assert (sizes["rank"] == 7).all() and (sizes["distinct"] == 7).all()


def test_budgets_are_nested_and_a_small_site_gives_all_it_has():
    """A larger budget contains a smaller one and never exceeds the pool."""
    index, features, table = make_data()
    pool = table.filter(pl.col("half") == "early")
    vectors = review.sequence_embeddings(index, features, table)
    early = (table["half"] == "early").to_numpy()
    queues = simulate.build_queues(pool, vectors[early], seeds=[0])
    small = simulate.reviewed(queues, pool, simulate.Cell(queue.DIVERSE, 0, 2, 1.0))
    large = simulate.reviewed(queues, pool, simulate.Cell(queue.DIVERSE, 0, 4, 1.0))
    huge = simulate.reviewed(queues, pool, simulate.Cell(queue.DIVERSE, 0, 99, 1.0))
    assert small < large and len(small) == 8 * 2
    assert huge == set(pool["seq_id"].to_list())
    assert simulate.reviewed(queues, pool, simulate.Cell("none", 0, 0, 1.0)) == set()
    assert simulate.reviewed(queues, pool, simulate.Cell("all", 0, None, 1.0)) == huge


def test_training_on_an_evaluation_frame_is_refused():
    """A frame that is both reviewed and scored raises before any fit."""
    features = np.zeros((4, 2))
    labels = np.array(["a", "b", "a", "b"])
    base = np.array([True, True, False, False])
    added = np.array([False, False, True, False])
    score = np.array([False, False, True, True])
    with pytest.raises(ValueError, match="evaluation frame"):
        simulate.retrain_logits(
            features, labels, (base, added, score), ["a", "b"], 1.0, 1.0
        )


def test_evaluation_sequences_never_enter_a_queue_or_training():
    """Queues hold only early sequences; predictions cover exactly the late ones."""
    index, features, table = make_data()
    vectors = review.sequence_embeddings(index, features, table)
    queues, labelled, predictions = simulate.deployment(
        index, features, (table, vectors), "fold1", 1.0, SETTINGS
    )
    here = table.filter(pl.col("split") == "fold1")
    early = set(here.filter(pl.col("half") == "early")["seq_id"].to_list())
    late = set(here.filter(pl.col("half") == "late")["seq_id"].to_list())
    assert set(queues["seq_id"].to_list()) == early
    assert set(predictions["seq_id"].to_list()) == late
    grid = labelled.filter(~pl.col("strategy").is_in(["none", "all"]))
    assert (grid["labels"] == grid["budget"].cast(pl.Int64)).all()
    assert set(labelled["site"].to_list()) == set(here["site"].to_list())


def test_new_labels_teach_a_class_seen_differently_at_the_new_sites():
    """When the new sites move a class, labelling them all recovers it."""
    index, features, table = make_data(sequences=30)
    # At fold0's sites tapir frames look like nothing the base model saw.
    moved = ((index["split"] == "fold0") & (index["label"] == "tapir")).to_numpy()
    features = features.copy()
    features[moved] = np.array([0, 0, 0, 6.0, 0, 0]) + features[moved] * 0.1
    vectors = review.sequence_embeddings(index, features, table)
    _, _, predictions = simulate.deployment(
        index, features, (table, vectors), "fold0", 1.0, SETTINGS
    )
    truth = table.select("seq_id", "label")

    def accuracy(strategy):
        """Return one reference cell's late-half accuracy."""
        part = predictions.filter(
            (pl.col("strategy") == strategy) & (pl.col("weight") == 1.0)
        ).join(truth, on="seq_id")
        return (part["pred"] == part["label"]).mean()

    assert accuracy("all") > accuracy("none") + 0.1


def test_eligible_sites_need_enough_in_both_halves():
    """A site short in either half is left out of scoring."""
    table = pl.DataFrame(
        {
            "site": ["A"] * 6 + ["B"] * 5,
            "half": ["early"] * 3 + ["late"] * 3 + ["early"] * 3 + ["late"] * 2,
        }
    )
    assert review.eligible_sites(table, 3) == ["A"]


def test_gate_cell_keeps_the_champion_on_a_tie_and_promotes_a_clear_win():
    """Identical predictions are rejected; fixing every error is promoted."""
    rng = np.random.default_rng(0)
    truth = {s: rng.integers(0, 3, 40) for s in "ABCD"}
    before = {
        s: np.where(np.arange(40) % 4 == 0, (t + 1) % 3, t) for s, t in truth.items()
    }
    labels = {s: 5 for s in truth}
    key = ("diverse", 1.0, 0, "5")
    tie, _ = review.gate_cell(
        key, {s: (truth[s], before[s], before[s]) for s in truth}, labels, GATE
    )
    win, sites = review.gate_cell(
        key, {s: (truth[s], before[s], truth[s]) for s in truth}, labels, GATE
    )
    assert not tie["promote"] and tie["mean_gain"] == 0
    assert win["promote"] and win["sequences_fixed"] == 40 and win["labels"] == 20
    assert [r["site"] for r in sites] == list("ABCD")


def test_comparison_with_random_averages_its_seeds():
    """A queue's edge is its gain minus the random queue's mean over seeds."""
    rows = [
        {"strategy": "random", "weight": 1.0, "seed": seed, "budget": "5"}
        | {"site": site, "gain": gain}
        for seed, gains in ((0, (0.0, 0.02)), (1, (0.02, 0.04)))
        for site, gain in zip("AB", gains)
    ]
    rows += [
        {"strategy": name, "weight": 1.0, "seed": 0, "budget": "5"}
        | {"site": site, "gain": 0.05}
        for name in ("uncertainty", "diverse")
        for site in "AB"
    ]
    out = review.compare_with_random(pl.DataFrame(rows), SETTINGS)
    assert out["strategy"].to_list() == ["diverse", "uncertainty"]
    assert out["mean_difference"].to_list() == pytest.approx([0.03, 0.03])
    assert out["sites_ahead"].to_list() == [2, 2]


def test_run_reports_both_sources_and_is_deterministic(tmp_path):
    """An end-to-end run covers fold and hold-out sites and repeats exactly."""
    index, features, table = make_data()
    first = review.run(index, features, table, cs(), GATE, SETTINGS)
    second = review.run(index, features, table, cs(), GATE, SETTINGS)
    for name in ("queues", "cells", "per_site", "comparisons", "primary"):
        assert first[name].equals(second[name])
    cells = first["cells"]
    grid = len(simulate.cells(SETTINGS))
    assert (
        cells["source"].value_counts().sort("source")["count"].to_list() == [grid] * 2
    )
    assert cells.filter(pl.col("source") == "fold")["sites"].unique().to_list() == [6]
    assert first["summary"]["holdout"]["sites_scored"] == 2
    primary = first["summary"]["fold"]["primary"]
    assert (primary["strategy"], primary["budget"]) == ("diverse", "4")
    # The late half is scored, never the early one a reviewer drew from.
    late = set(table.filter(pl.col("half") == "late")["seq_id"].to_list())
    assert set(first["primary"]["seq_id"].to_list()) == late
    assert (first["queues"]["rank"] <= review.QUEUE_ROWS).all()

    path = tmp_path / "curves.png"
    review.plot_curves(cells.filter(pl.col("source") == "fold"), SETTINGS, path)
    assert path.stat().st_size > 0
