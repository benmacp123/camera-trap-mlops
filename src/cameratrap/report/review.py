"""Review loop: do a few reviewed labels at a new site earn a retrained model?

Simulates the step after the drift study. The champion (the DINOv2 probe)
meets sites it never saw; a reviewer labels a small budget of sequences
from each site's early half, chosen by one of three queues; the probe is
retrained with them; and the promotion gate, with its frozen settings,
decides on each site's late half whether the retrained model replaces the
champion.

The 39 fold sites play the new sites, each fold new to a probe fitted on
the other four, and the gate pools their sites. The ten hold-out sites
repeat the exercise once against the probe fitted on all fold sites. That
hold-out was spent on the first gate decision, so it is a replication here,
never a fresh test.

The rules (sites, the time split, strategies, budgets, weights, the
primary cell) were fixed in the plan on 2026-11-11 before any cell was
scored. The primary cell is the diverse queue at 20 labels per site with
new labels at weight 1. Every other cell is tabled for the learning curves
and is not a headline.

Reads the champion's committed predictions from ``results/drift/`` and the
embedding cache. Writes ``results/review/``: ``queue.csv`` (the first 40 of
every queue), ``cells.csv`` (one gate decision per cell), ``per_site.csv``,
``comparisons.csv`` (each queue against random), ``sequences_primary.csv``
and ``summary.json``; and ``figures/review_learning_curves.png``.

::

    PYTHONPATH=src python -m cameratrap.report.review
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import Parallel, delayed

from cameratrap.arms import scoring
from cameratrap.data import crops
from cameratrap.drift import signals, study
from cameratrap.embed import store
from cameratrap.gate import freeze, rule
from cameratrap.replay import stream
from cameratrap.review import queue, simulate

CHAMPION = "dinov2_probe"
BACKBONE = "dinov2"
FREEZE = Path("config/holdout_freeze.json")
REPLAY = Path("results/drift/sequences.csv")
FOLD_SUMMARY = Path("results/challengers/summary.json")
HOLDOUT_SUMMARY = Path("results/holdout/summary.json")
OUT_DIR = Path("results/review")
FIGURE_DIR = Path("figures")
FOLD = "fold"
CELL = ["strategy", "weight", "seed", "budget"]
QUEUE_ROWS = 40
COLOURS = {
    queue.RANDOM: "#8a8984",
    queue.UNCERTAINTY: "#2a78d6",
    queue.DIVERSE: "#1baf7a",
    simulate.ALL: "#52514e",
}
NAMES = {
    queue.RANDOM: "Random (5 seeds, mean and range)",
    queue.UNCERTAINTY: "Least confident first",
    queue.DIVERSE: "Spread over species and appearance",
    simulate.ALL: "Every early sequence labelled",
}


def chosen_cs(root):
    """Return the champion's C for each split, from the committed results.

    Args:
        root: The repository root, a ``pathlib.Path``.

    Returns:
        A dict of split (each fold, and ``holdout``) to C.
    """
    folds = json.loads((root / FOLD_SUMMARY).read_text())["arms"][CHAMPION]
    held = json.loads((root / HOLDOUT_SUMMARY).read_text())
    return folds["chosen_c_per_fold"] | {simulate.HOLDOUT: held["chosen_c"]}


def sequence_embeddings(index, features, table):
    """Return unit-norm sequence embeddings in the replay table's row order.

    Args:
        index: The crop index, in ``features`` row order.
        features: The frames' embeddings.
        table: The replay table, one row per sequence.

    Returns:
        Array of shape ``(table.height, d)``: the mean of each sequence's
        frames, renormalised, as the drift study defines it.

    Raises:
        ValueError: If the table and the index hold different sequences.
    """
    seq_ids, vectors = scoring.group_mean(
        index["seq_id"].to_list(), features.astype(np.float64)
    )
    if sorted(table["seq_id"].to_list()) != seq_ids:
        raise ValueError("the replay table and the crop index hold other sequences")
    row = {s: i for i, s in enumerate(seq_ids)}
    return signals.normalise(vectors)[[row[s] for s in table["seq_id"].to_list()]]


def eligible_sites(table, min_half):
    """Return the sites with enough sequences in both halves to score.

    Args:
        table: The replay table.
        min_half: Fewest sequences a site needs in each half.

    Returns:
        A sorted list of site names.
    """
    counts = table.group_by("site").agg(
        (pl.col("half") == stream.EARLY).sum().alias("early"),
        (pl.col("half") == stream.LATE).sum().alias("late"),
    )
    keep = counts.filter((pl.col("early") >= min_half) & (pl.col("late") >= min_half))
    return sorted(keep["site"].to_list())


def gate_cell(key, groups, labels, gate):
    """Run the gate for one cell and describe the outcome.

    Args:
        key: The cell's ``strategy``, ``weight``, ``seed`` and ``budget``.
        groups: Mapping of site to ``(labels, champion, challenger)`` as
            integer class codes, one entry per late-half sequence.
        labels: Mapping of site to sequences reviewed there.
        gate: The frozen ``GateConfig``.

    Returns:
        A pair ``(row, site_rows)``: the cell's row for ``cells.csv`` and
        one dict per site for ``per_site.csv``.
    """
    decision = rule.evaluate(groups, scoring.balanced_accuracy, gate)
    tag = dict(zip(CELL, key))
    site_rows, fixed, broken = [], 0, 0
    for site, (truth, before, after) in sorted(groups.items()):
        fixed += int(np.sum((before != truth) & (after == truth)))
        broken += int(np.sum((before == truth) & (after != truth)))
        champion = scoring.balanced_accuracy(truth, before)
        site_rows.append(
            tag
            | {
                "site": site,
                "n_late": len(truth),
                "labels": labels[site],
                "champion": champion,
                "challenger": champion + decision.differences[site],
                "gain": decision.differences[site],
                "p_value": decision.p_values[site],
                "blocks": site in decision.regressions,
            }
        )
    gains = np.array(list(decision.differences.values()))
    row = tag | {
        "sites": len(groups),
        "labels": int(sum(labels[s] for s in groups)),
        "champion_mean": float(np.mean([r["champion"] for r in site_rows])),
        "challenger_mean": float(np.mean([r["challenger"] for r in site_rows])),
        "mean_gain": decision.mean_difference,
        "low": decision.interval[0],
        "high": decision.interval[1],
        "sites_up": int(np.sum(gains > 0)),
        "sites_down": int(np.sum(gains < 0)),
        "sequences_fixed": fixed,
        "sequences_broken": broken,
        "regressions": len(decision.regressions),
        "promote": decision.promote,
        "reasons": "; ".join(decision.reasons),
    }
    return row, site_rows


def score_source(late, predictions, labelled, sites, gate, n_jobs=1):
    """Gate every cell on one set of sites.

    Args:
        late: The late-half sequences of the sites, with ``seq_id``,
            ``site``, ``label`` and the champion's ``pred``.
        predictions: Every cell's late-half predictions, from
            ``simulate.deployment``.
        labelled: Every cell's labels per site, from the same.
        sites: The sites to score.
        gate: The frozen ``GateConfig``.
        n_jobs: Cells to gate at once.

    Returns:
        A pair ``(cells, per_site)`` of DataFrames sorted by cell.
    """
    classes = sorted(set(late["label"].to_list()) | set(late["pred"].to_list()))
    late = late.filter(pl.col("site").is_in(sites)).rename({"pred": "champion"})
    jobs = []
    for key, part in predictions.partition_by(CELL, as_dict=True).items():
        classes = sorted(set(classes) | set(part["pred"].to_list()))
        joined = late.join(part.select("seq_id", "pred"), on="seq_id").sort("seq_id")
        if joined.height != late.height:
            raise ValueError(f"cell {key} does not score every late sequence")
        code = {name: i for i, name in enumerate(classes)}
        groups = {
            site: tuple(
                np.array([code[v] for v in rows[column].to_list()])
                for column in ("label", "champion", "pred")
            )
            for (site,), rows in joined.partition_by(
                "site", as_dict=True, maintain_order=True
            ).items()
        }
        counts = labelled.filter(
            pl.all_horizontal(pl.col(c) == v for c, v in zip(CELL, key))
        )
        labels = dict(zip(counts["site"].to_list(), counts["labels"].to_list()))
        jobs.append(delayed(gate_cell)(key, groups, labels, gate))
    results = Parallel(n_jobs=n_jobs)(jobs)
    cells = pl.DataFrame([row for row, _ in results]).sort(CELL)
    per_site = pl.DataFrame([r for _, rows in results for r in rows])
    return cells, per_site.sort(*CELL, "site")


def compare_with_random(per_site, settings):
    """Compare each queue with the random queue, site by site.

    A queue's gain at a site minus the random queue's (averaged over its
    seeds) is what choosing sequences cleverly bought there. The interval
    resamples sites.

    Args:
        per_site: One source's per-site table from ``score_source``.
        settings: ``simulate.SETTINGS`` or a test's version of it.

    Returns:
        A DataFrame with ``strategy``, ``weight``, ``budget``, ``sites``,
        ``mean_difference``, ``low``, ``high`` and ``sites_ahead``.
    """
    keys = ["weight", "budget", "site"]
    random = (
        per_site.filter(pl.col("strategy") == queue.RANDOM)
        .group_by(keys)
        .agg(pl.col("gain").mean().alias("random_gain"))
    )
    rows = []
    for strategy in (queue.UNCERTAINTY, queue.DIVERSE):
        joined = per_site.filter(pl.col("strategy") == strategy).join(random, on=keys)
        for (weight, budget), part in joined.partition_by(
            "weight", "budget", as_dict=True
        ).items():
            diff = (
                part.sort("site")["gain"] - part.sort("site")["random_gain"]
            ).to_numpy()
            low, high = study.site_bootstrap(np.mean, [diff], settings)
            rows.append(
                {
                    "strategy": strategy,
                    "weight": weight,
                    "budget": budget,
                    "sites": len(diff),
                    "mean_difference": float(diff.mean()),
                    "low": low,
                    "high": high,
                    "sites_ahead": int(np.sum(diff > 0)),
                }
            )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("budget").cast(pl.Int64).alias("order"))
        .sort("strategy", "weight", "order")
        .drop("order")
    )


def cell_row(cells, cell):
    """Return one cell's row from a cells table.

    Args:
        cells: One source's cells table.
        cell: A ``simulate.Cell``.

    Returns:
        The row as a dict.
    """
    return cells.filter(
        (pl.col("strategy") == cell.strategy)
        & (pl.col("weight") == cell.weight)
        & (pl.col("seed") == cell.seed)
        & (pl.col("budget") == cell.budget_label)
    ).row(0, named=True)


def summarise(cells, settings):
    """Pick out one source's headline numbers.

    Args:
        cells: One source's cells table.
        settings: ``simulate.SETTINGS`` or a test's version of it.

    Returns:
        A JSON-ready dict: the primary cell, the two references, and how
        many grid cells the gate promoted and rejected.
    """
    primary = simulate.Cell(seed=0, **settings["primary"])
    grid = cells.filter(~pl.col("strategy").is_in([simulate.NONE, simulate.ALL]))
    return {
        "sites_scored": int(cells["sites"][0]),
        "primary": cell_row(cells, primary),
        "refit_with_no_new_labels": cell_row(
            cells, simulate.Cell(simulate.NONE, 0, 0, 1.0)
        ),
        "every_early_sequence_labelled": {
            str(w): cell_row(cells, simulate.Cell(simulate.ALL, 0, None, w))
            for w in settings["weights"]
        },
        "grid_cells_promoted": int(grid["promote"].sum()),
        "grid_cells_rejected": int((~grid["promote"]).sum()),
    }


def plot_curves(cells, settings, path):
    """Plot mean gain against labels per site, one panel per weight.

    Args:
        cells: The fold sites' cells table.
        settings: ``simulate.SETTINGS`` or a test's version of it.
        path: Where to save the PNG.
    """
    budgets = [str(b) for b in settings["budgets"]]
    x = np.arange(len(budgets))
    primary = settings["primary"]
    fig, axes = plt.subplots(
        1, len(settings["weights"]), figsize=(9.5, 4.8), sharey=True
    )
    for ax, weight in zip(np.atleast_1d(axes), settings["weights"]):
        at_weight = cells.filter(pl.col("weight") == weight)
        ax.axhline(0, color="#52514e", linewidth=1)
        # Refitting alone moves a few predictions; gains near this line are
        # not evidence that the labels helped.
        floor = cell_row(cells, simulate.Cell(simulate.NONE, 0, 0, 1.0))
        ax.axhline(
            100 * floor["mean_gain"],
            color="#52514e",
            linewidth=1,
            linestyle=":",
            label="Refit with no new labels",
        )
        # The random queue: the mean of its seeds, shaded over their range.
        seeds = np.array(
            [
                at_weight.filter(
                    (pl.col("strategy") == queue.RANDOM) & (pl.col("budget") == b)
                )["mean_gain"].to_numpy()
                for b in budgets
            ]
        )
        colour = COLOURS[queue.RANDOM]
        ax.fill_between(
            x,
            100 * seeds.min(axis=1),
            100 * seeds.max(axis=1),
            color=colour,
            alpha=0.25,
        )
        ax.plot(
            x,
            100 * seeds.mean(axis=1),
            color=colour,
            marker="o",
            label=NAMES[queue.RANDOM],
        )
        # The two ranked queues, with the gate's interval; nudged apart so
        # the bars do not hide each other.
        for strategy, nudge in ((queue.UNCERTAINTY, -0.06), (queue.DIVERSE, 0.06)):
            rows = [
                cell_row(at_weight, simulate.Cell(strategy, 0, int(b), weight))
                for b in budgets
            ]
            gain = 100 * np.array([r["mean_gain"] for r in rows])
            low = 100 * np.array([r["low"] for r in rows])
            high = 100 * np.array([r["high"] for r in rows])
            ax.errorbar(
                x + nudge,
                gain,
                yerr=[gain - low, high - gain],
                color=COLOURS[strategy],
                marker="s" if strategy == queue.DIVERSE else "^",
                capsize=3,
                linewidth=1.5,
                label=NAMES[strategy],
            )
            if strategy == primary["strategy"] and weight == primary["weight"]:
                at = budgets.index(str(primary["budget"]))
                ax.scatter(
                    x[at] + nudge,
                    gain[at],
                    s=260,
                    facecolor="none",
                    edgecolor="#0b0b0b",
                    linewidth=1.2,
                    zorder=4,
                    label="Named in advance as the headline",
                )
        ceiling = cell_row(at_weight, simulate.Cell(simulate.ALL, 0, None, weight))
        gain = 100 * ceiling["mean_gain"]
        ax.errorbar(
            len(budgets),
            gain,
            yerr=[[gain - 100 * ceiling["low"]], [100 * ceiling["high"] - gain]],
            color=COLOURS[simulate.ALL],
            marker="D",
            capsize=3,
            linestyle="none",
            label=NAMES[simulate.ALL],
        )
        ax.set_xticks(np.arange(len(budgets) + 1), [*budgets, "all"])
        title = (
            "New labels count the same as old"
            if weight == 1
            else (f"New labels count {weight:g} times")
        )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Sequences reviewed per site", fontsize=9)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Mean gain over the champion (points)", fontsize=9)
    # Below the panels: inside them it would sit on the data.
    handles, names = first.get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    sites = int(cells["sites"][0])
    fig.suptitle(
        f"Gain from retraining with reviewed labels, {sites} fold sites, "
        "scored on each site's later half",
        x=0.02,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(index, features, table, cs, gate, settings=None, n_jobs=1):
    """Run every deployment and gate every cell.

    Args:
        index: The crop index sorted by path, in ``features`` row order.
        features: The frames' embeddings.
        table: The replay table with the champion's ``pred`` and ``msp``.
        cs: Mapping of split to the champion's C.
        gate: The frozen ``GateConfig``.
        settings: Defaults to ``simulate.SETTINGS``.
        n_jobs: Cells to fit or gate at once.

    Returns:
        A dict with ``queues``, ``cells``, ``per_site``, ``comparisons``
        and ``primary`` DataFrames (each with a ``source`` column where it
        applies) and a JSON-ready ``summary``.
    """
    settings = settings or simulate.SETTINGS
    embeddings = sequence_embeddings(index, features, table)
    parts = {FOLD: [], simulate.HOLDOUT: []}
    for split in sorted(set(table["split"].to_list())):
        source = simulate.HOLDOUT if split == simulate.HOLDOUT else FOLD
        parts[source].append(
            simulate.deployment(
                index, features, (table, embeddings), split, cs[split], settings, n_jobs
            )
        )

    out = {k: [] for k in ("queues", "cells", "per_site", "comparisons", "primary")}
    summary = {"champion": CHAMPION, "settings": settings, "chosen_c": cs}
    primary = simulate.Cell(seed=0, **settings["primary"])
    for source, results in parts.items():
        if not results:
            continue
        queues, labelled, predictions = (pl.concat(p) for p in zip(*results))
        where = pl.col("split") == simulate.HOLDOUT
        here = table.filter(where if source == simulate.HOLDOUT else ~where)
        late = here.filter(pl.col("half") == stream.LATE)
        sites = eligible_sites(here, settings["min_half_sequences"])
        cells, per_site = score_source(
            late.select("seq_id", "site", "label", "pred"),
            predictions,
            labelled,
            sites,
            gate,
            n_jobs,
        )
        tag = pl.lit(source).alias("source")
        chosen = predictions.filter(
            (pl.col("strategy") == primary.strategy)
            & (pl.col("weight") == primary.weight)
            & (pl.col("budget") == primary.budget_label)
        ).select("seq_id", pl.col("pred").alias("challenger"))
        out["queues"].append(
            queues.filter(pl.col("rank") <= QUEUE_ROWS).select(tag, pl.all())
        )
        out["cells"].append(cells.select(tag, pl.all()))
        out["per_site"].append(per_site.select(tag, pl.all()))
        out["comparisons"].append(
            compare_with_random(per_site, settings).select(tag, pl.all())
        )
        out["primary"].append(
            late.filter(pl.col("site").is_in(sites))
            .select("seq_id", "site", "label", pl.col("pred").alias("champion"))
            .join(chosen, on="seq_id")
            .sort("site", "seq_id")
            .select(tag, pl.all())
        )
        summary[source] = summarise(cells, settings) | {
            "sites_queued": here["site"].n_unique(),
            "late_sequences_scored": int(late["site"].is_in(sites).sum()),
        }
    return {k: pl.concat(v) for k, v in out.items()} | {"summary": summary}


def main(argv=None):
    """Simulate the review loop and write its tables and figure.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache-root", type=Path, default=store.CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    index = crops.load_index(args.root / crops.INDEX_PATH).sort("path")
    cache = store.read(store.cache_path(BACKBONE, args.cache_root))
    # A cache embedded from another index would score the wrong crops.
    store.check_meta(
        cache["meta"],
        {"index_sha256": store.file_sha256(args.root / crops.INDEX_PATH)},
    )
    features = store.rows_for(cache, index["path"].to_list())
    table = pl.read_csv(args.root / REPLAY)
    config, gate = freeze.load(args.root / FREEZE)
    result = run(index, features, table, chosen_cs(args.root), gate, n_jobs=args.jobs)
    summary = result["summary"] | {
        "gate": config["gate"],
        "freeze_sha256": freeze.config_hash(config),
        "cache": cache["meta"],
    }

    out = args.root / args.out_dir
    figures = args.root / args.figure_dir
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    result["queues"].write_csv(out / "queue.csv")
    result["cells"].write_csv(out / "cells.csv")
    result["per_site"].write_csv(out / "per_site.csv")
    result["comparisons"].write_csv(out / "comparisons.csv")
    result["primary"].write_csv(out / "sequences_primary.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_curves(
        result["cells"].filter(pl.col("source") == FOLD),
        simulate.SETTINGS,
        figures / "review_learning_curves.png",
    )
    print(
        json.dumps(
            {s: summary[s]["primary"] for s in (FOLD, simulate.HOLDOUT)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
