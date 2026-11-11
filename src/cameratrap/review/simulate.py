"""Simulated review loop: label a few sequences at new sites, then retrain.

A deployment is a set of sites the champion has never seen. Each site's
sequences arrive in date order; the reviewer may label only from the early
half, and the retrained model is judged on the late half. Splitting by time
rather than at random matters because camera traps fire in bursts: a third
of sequences follow a same-species sequence at the same camera within ten
minutes, so a random evaluation sample would hold near-copies of the
training labels.

The reviewer is simulated from labels that were held back: asking for a
sequence reveals its true label, with no reviewer error. Retraining keeps
the champion's recipe and regularisation strength and only adds the
reviewed frames, optionally upweighted, so any change in accuracy comes
from the new labels alone.

A *cell* is one retraining: a queue strategy, a seed (for the random
queue), a budget of labels per site and a weight for the new frames. Two
reference cells sit beside the grid: ``none`` (retrain with no new labels,
which shows how much refitting alone moves predictions) and ``all`` (every
early sequence labelled, the most these sites could give).
"""

from typing import NamedTuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from cameratrap.arms import probe, scoring
from cameratrap.replay import stream
from cameratrap.review import queue

NONE = "none"
ALL = "all"
HOLDOUT = "holdout"

# Fixed in the plan on 2026-11-11, before any cell was scored.
SETTINGS = {
    "budgets": [5, 10, 20, 40],
    "weights": [1.0, 10.0],
    "random_seeds": [0, 1, 2, 3, 4],
    "min_half_sequences": 20,
    "window": queue.WINDOW,
    "primary": {"strategy": queue.DIVERSE, "budget": 20, "weight": 1.0},
    "bootstrap_resamples": 2000,
    "confidence": 0.95,
    "seed": 0,
}


class Cell(NamedTuple):
    """One retraining to run.

    Attributes:
        strategy: A queue strategy, or ``none`` or ``all``.
        seed: The random queue's repetition; 0 for the others.
        budget: Labels per site; 0 for ``none`` and ``None`` for ``all``.
        weight: Weight of each reviewed frame against a base frame's 1.
    """

    strategy: str
    seed: int
    budget: int | None
    weight: float

    @property
    def budget_label(self):
        """Return the budget as text, ``all`` when there is no limit."""
        return ALL if self.budget is None else str(self.budget)


def cells(settings):
    """List every cell: the two references, then the grid.

    Args:
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A list of ``Cell``; ``none`` once (a weight on no frames changes
        nothing), ``all`` once per weight, then every strategy, seed and
        budget at every weight.
    """
    out = [Cell(NONE, 0, 0, 1.0)]
    for weight in settings["weights"]:
        out.append(Cell(ALL, 0, None, weight))
        for strategy in queue.STRATEGIES:
            seeds = settings["random_seeds"] if strategy == queue.RANDOM else [0]
            out.extend(
                Cell(strategy, seed, budget, weight)
                for seed in seeds
                for budget in settings["budgets"]
            )
    return out


def build_queues(pool, embeddings, seeds):
    """Rank every site's reviewable sequences under every strategy.

    Args:
        pool: The reviewable sequences, with ``seq_id``, ``site``, ``pred``
            and ``msp``. Labels, if present, are never read.
        embeddings: Unit-norm sequence embeddings in ``pool`` row order.
        seeds: Repetitions of the random queue.

    Returns:
        A DataFrame with ``site``, ``strategy``, ``seed``, ``rank`` (1 is
        reviewed first), ``seq_id``, ``pred`` and ``msp``, sorted by those
        first four columns. Every strategy ranks the whole pool.
    """
    parts = []
    sites = pool["site"].to_numpy()
    for site in sorted(set(sites.tolist())):
        rows = np.flatnonzero(sites == site)
        part = pool[rows]
        args = (
            part["seq_id"].to_numpy(),
            part["msp"].to_numpy(),
            part["pred"].to_numpy(),
            embeddings[rows],
            site,
        )
        for strategy in queue.STRATEGIES:
            for seed in seeds if strategy == queue.RANDOM else [0]:
                ranked = part[queue.order(strategy, *args, seed=seed)]
                parts.append(
                    ranked.select(
                        pl.lit(site).alias("site"),
                        pl.lit(strategy).alias("strategy"),
                        pl.lit(seed, dtype=pl.Int64).alias("seed"),
                        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("rank"),
                        "seq_id",
                        "pred",
                        "msp",
                    )
                )
    return pl.concat(parts).sort("site", "strategy", "seed", "rank")


def reviewed(queues, pool, cell):
    """Return the sequences a cell's reviewer labels.

    Args:
        queues: From ``build_queues``.
        pool: The reviewable sequences.
        cell: The cell.

    Returns:
        A set of sequence IDs: none for ``none``, the whole pool for
        ``all``, otherwise each site's first ``budget`` in the cell's
        queue (all of a site that has fewer).
    """
    if cell.strategy == NONE:
        return set()
    if cell.strategy == ALL:
        return set(pool["seq_id"].to_list())
    chosen = queues.filter(
        (pl.col("strategy") == cell.strategy)
        & (pl.col("seed") == cell.seed)
        & (pl.col("rank") <= cell.budget)
    )
    return set(chosen["seq_id"].to_list())


def retrain_logits(features, labels, masks, classes, c, weight):
    """Refit the probe with the reviewed frames added and score new frames.

    Args:
        features: Frame embeddings, shape ``(n, d)``.
        labels: Class name of each frame.
        masks: A tuple ``(base, added, score)`` of boolean frame masks: the
            champion's training frames, the reviewed frames, and the frames
            to score.
        classes: Class names in logit order.
        c: The champion's inverse regularisation strength.
        weight: Weight of each reviewed frame; a base frame counts 1.

    Returns:
        Array of shape ``(score.sum(), len(classes))``.

    Raises:
        ValueError: If a frame to be scored is also trained on.
    """
    base, added, score = masks
    rows = base | added
    if (rows & score).any():
        raise ValueError("an evaluation frame would be trained on")
    weights = np.where(added[rows], weight, 1.0)
    model = probe.refit(features[rows], labels[rows], classes, c, weights)
    return model.decision_function(features[score])


def deployment(frames, features, sequences, split, c, settings, n_jobs=1):
    """Run every cell for the sites of one split.

    Args:
        frames: Every indexed frame with ``seq_id``, ``split`` and
            ``label``, in ``features`` row order.
        features: The frames' embeddings.
        sequences: A pair ``(table, embeddings)``: the replayed sequence
            table (``seq_id``, ``site``, ``split``, ``half``, ``pred``,
            ``msp``) and unit-norm sequence embeddings in its row order.
        split: The split whose sites are new: a fold, or ``holdout``. The
            champion was trained on every other fold.
        c: The champion's C for this split.
        settings: ``SETTINGS`` or a test's version of it.
        n_jobs: Cells to fit at once.

    Returns:
        A tuple ``(queues, labelled, predictions)``: the ranked queues;
        a DataFrame of ``strategy``, ``seed``, ``budget``, ``weight``,
        ``site`` and ``labels`` (sequences reviewed there); and a
        DataFrame with the same cell columns plus ``seq_id`` and ``pred``
        for every late-half sequence.
    """
    table, embeddings = sequences
    classes = sorted(set(frames["label"].to_list()))
    here = (table["split"] == split).to_numpy()
    early = here & (table["half"] == stream.EARLY).to_numpy()
    pool = table.filter(pl.Series(early))
    late_ids = table.filter(pl.Series(here & ~early))["seq_id"]
    queues = build_queues(pool, embeddings[early], settings["random_seeds"])

    frame_split = frames["split"]
    # Fold sites are new to a probe fitted on the other four folds; the
    # hold-out sites are new to one fitted on all five.
    base = (~frame_split.is_in([split, HOLDOUT])).to_numpy()
    score = frames["seq_id"].is_in(late_ids.to_list()).to_numpy()
    labels = frames["label"].to_numpy()
    grid = cells(settings)
    chosen = [reviewed(queues, pool, cell) for cell in grid]

    # Arrays go to the workers as arguments, so joblib shares the large
    # embedding table between them instead of copying it into each task.
    logits = Parallel(n_jobs=n_jobs)(
        delayed(retrain_logits)(
            features,
            labels,
            (base, frames["seq_id"].is_in(sorted(picked)).to_numpy(), score),
            classes,
            c,
            cell.weight,
        )
        for picked, cell in zip(chosen, grid)
    )

    keys = frames.filter(pl.Series(score))["seq_id"].to_list()
    site_of = pool.select("seq_id", "site")
    labelled, predictions = [], []
    for cell, picked, cell_logits in zip(grid, chosen, logits):
        tag = [
            pl.lit(cell.strategy).alias("strategy"),
            pl.lit(cell.seed, dtype=pl.Int64).alias("seed"),
            pl.lit(cell.budget_label).alias("budget"),
            pl.lit(cell.weight).alias("weight"),
        ]
        counts = site_of.filter(pl.col("seq_id").is_in(sorted(picked)))
        labelled.append(
            counts.group_by("site")
            .agg(pl.len().cast(pl.Int64).alias("labels"))
            # A site nobody reviewed still gets a row, with zero labels.
            .join(site_of.select("site").unique(), on="site", how="right")
            .with_columns(pl.col("labels").fill_null(0))
            .select(*tag, "site", "labels")
        )
        scored = scoring.sequence_predictions(keys, cell_logits, classes)
        predictions.append(scored.select(*tag, "seq_id", "pred"))
    order = ["strategy", "weight", "seed", "budget"]
    return (
        queues,
        pl.concat(labelled).sort(*order, "site"),
        pl.concat(predictions).sort(*order, "seq_id"),
    )
