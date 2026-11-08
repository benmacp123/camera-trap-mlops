"""The per-site trust report for the current champion, the DINOv2 probe.

A retrospective, labelled audit: for every site with labels, how accurate
the champion was, how much review a confidence threshold buys, which species
it gets wrong there, and whether switching from BioCLIP 2 zero-shot made the
site worse. A new, unlabelled site gets no number here; predicting one
without labels is the drift study's question.

Fold sites are scored out of fold but are optimistic, because the probe was
selected on them; hold-out sites were scored once by the frozen model, so
the headline numbers come from them. The suggested review threshold is
chosen on fold sites only and then reported as it falls on the hold-out.

Regressions use Holm's step-down over every site (Bonferroni is the gate's
rule; Holm flags at least as many sites, which suits a report). A site with
so few disagreements that no result could clear Holm's strictest cut-off is
labelled "too little data to judge", not passed.

Reads committed sequence predictions only, so it needs no model stack.
Writes ``results/trust/``: ``per_site.csv``, ``per_species.csv``,
``thresholds.csv``, ``summary.json`` and ``trust_report.md``, and
``figures/trust_review_budget.png``.

::

    PYTHONPATH=src python -m cameratrap.report.trust
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import scoring
from cameratrap.gate import freeze
from cameratrap.report import selection

CHAMPION = "dinov2_probe"
PREVIOUS = "bioclip2"
LABELS = {CHAMPION: "DINOv2 probe", PREVIOUS: "BioCLIP 2 zero-shot"}
# Fixed in the plan (2026-11-08) before any trust number was computed.
SETTINGS = {
    "thresholds": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
    "target_residual_error": 0.05,
    "weak_recall": 0.70,
    "min_species_sequences": 5,
    "bootstrap_resamples": 2000,
    "confidence": 0.95,
    "seed": 0,
}
SOURCES = {"fold": Path("results/challengers"), "holdout": Path("results/holdout")}
FREEZE_PATH = Path("config/holdout_freeze.json")
OUT_DIR = Path("results/trust")
FIGURE_DIR = Path("figures")
COLOURS = {"fold": "#8a8984", "holdout": "#1baf7a"}
STATUS_FLAGGED = "worse than BioCLIP 2"
STATUS_SMALL = "too little data to judge"
STATUS_CLEAR = "no regression found"


def load_predictions(root):
    """Join the champion's and previous champion's predictions at every site.

    Args:
        root: Repository root.

    Returns:
        A DataFrame with ``seq_id``, ``site``, ``source`` (``fold`` or
        ``holdout``), ``label``, ``pred``, ``msp`` and ``previous`` (the
        previous champion's prediction), sorted by ``seq_id``.

    Raises:
        ValueError: If the two arms do not cover the same sequences, or a
            site appears under both sources.
    """
    parts = []
    for source, directory in SOURCES.items():
        champion = pl.read_csv(root / directory / f"sequences_{CHAMPION}.csv")
        previous = pl.read_csv(root / directory / f"sequences_{PREVIOUS}.csv")
        joined = champion.join(
            previous.select("seq_id", pl.col("pred").alias("previous")),
            on="seq_id",
            how="inner",
        )
        if not joined.height == champion.height == previous.height:
            raise ValueError(f"{source}: the two arms cover different sequences")
        parts.append(joined.with_columns(pl.lit(source).alias("source")))
    table = pl.concat(parts).select(
        "seq_id", "site", "source", "label", "pred", "msp", "previous"
    )
    if (
        table.group_by("site")
        .agg(pl.col("source").n_unique())
        .filter(pl.col("source") > 1)
        .height
    ):
        raise ValueError("a site appears as both a fold and a hold-out site")
    return table.sort("seq_id")


def holm(p_values, alpha):
    """Return which hypotheses Holm's step-down procedure rejects.

    The smallest p-value is compared with ``alpha / m``, the next with
    ``alpha / (m - 1)``, and so on, stopping at the first that fails.

    Args:
        p_values: Sequence of p-values.
        alpha: Family-wise error rate.

    Returns:
        A boolean array, aligned with ``p_values``; True where rejected.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size
    rejected = np.zeros(m, dtype=bool)
    for rank, i in enumerate(np.argsort(p_values, kind="stable")):
        if p_values[i] > alpha / (m - rank):
            break
        rejected[i] = True
    return rejected


def best_attainable_p(disagreements, draws):
    """Return the smallest p-value the swap null could give a site.

    Only sequences where the two models disagree change when swapped, so
    the most extreme result has chance ``0.5 ** k``; add-one counting caps
    it below at ``1 / (draws + 1)``. Ties in the metric only raise it, so
    this is a best case.

    Args:
        disagreements: Sequences where the two models' predictions differ.
        draws: Swap draws used for the p-value.

    Returns:
        The best attainable p-value.
    """
    return max(0.5**disagreements, 1 / (draws + 1))


def bootstrap_interval(labels, preds, resamples, confidence, rng):
    """Return a percentile bootstrap interval for one site's balanced accuracy.

    Sequences are resampled with replacement, which treats them as
    independent; at one site they are not, so the interval is too narrow.

    Args:
        labels: True class per sequence.
        preds: Predicted class per sequence.
        resamples: Bootstrap resamples.
        confidence: Interval coverage, such as 0.95.
        rng: A ``numpy.random.Generator``.

    Returns:
        A pair ``(low, high)``.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    stats = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, labels.size, labels.size)
        stats[i] = scoring.balanced_accuracy(labels[pick], preds[pick])
    tail = (1 - confidence) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def review_budget(table, thresholds, by):
    """Tabulate what each confidence threshold costs in review and leaves wrong.

    A sequence is auto-accepted when its ``msp`` is at least the threshold;
    the rest go to a reviewer, who is assumed to fix them.

    Args:
        table: Output of ``load_predictions``.
        thresholds: Thresholds to evaluate; 0 accepts everything.
        by: Column to group by (``source`` or ``site``).

    Returns:
        A DataFrame with ``by``, ``threshold``, ``sequences``, ``review_share``,
        ``residual_error`` (wrong and unreviewed over all sequences) and
        ``accepted_error`` (wrong over accepted; null when none accepted).
    """
    rows = []
    for (key,), group in table.group_by(by):
        msp = group["msp"].to_numpy()
        wrong = group["label"].to_numpy() != group["pred"].to_numpy()
        for t in thresholds:
            accepted = msp >= t
            rows.append(
                {
                    by: key,
                    "threshold": float(t),
                    "sequences": group.height,
                    "review_share": float(1 - accepted.mean()),
                    "residual_error": float((wrong & accepted).mean()),
                    "accepted_error": (
                        float(wrong[accepted].mean()) if accepted.any() else None
                    ),
                }
            )
    return pl.DataFrame(rows).sort(by, "threshold")


def choose_threshold(budget, target):
    """Return the lowest threshold whose pooled fold-site residual error meets the target.

    Args:
        budget: ``review_budget`` grouped by ``source``.
        target: Largest acceptable residual error.

    Returns:
        The threshold, or None if none on the grid meets the target.
    """
    ok = budget.filter(
        (pl.col("source") == "fold") & (pl.col("residual_error") <= target)
    )
    return float(ok["threshold"].min()) if ok.height else None


def species_table(table, settings):
    """Score each species at each site and mark the weak ones.

    Args:
        table: Output of ``load_predictions``.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A DataFrame with ``site``, ``source``, ``species``, ``sequences``,
        ``recall``, ``mistaken_for`` (the commonest wrong prediction, if any)
        and ``status`` (``weak``, ``ok`` or ``too few``).
    """
    rows = []
    for (site, source, species), group in table.group_by("site", "source", "label"):
        preds = group["pred"]
        wrong = preds.filter(preds != species)
        recall = 1 - wrong.len() / group.height
        if group.height < settings["min_species_sequences"]:
            status = "too few"
        elif recall < settings["weak_recall"]:
            status = "weak"
        else:
            status = "ok"
        rows.append(
            {
                "site": site,
                "source": source,
                "species": species,
                "sequences": group.height,
                "recall": recall,
                "mistaken_for": (
                    wrong.value_counts(sort=True)["pred"][0] if wrong.len() else None
                ),
                "status": status,
            }
        )
    return pl.DataFrame(rows).sort("site", "species")


def site_table(table, gate, settings):
    """Score each site and test whether the champion regressed there.

    Args:
        table: Output of ``load_predictions``.
        gate: The frozen ``GateConfig``; its alpha, draws and seed are used.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A DataFrame with one row per site: ``site``, ``source``, ``n``,
        ``balanced_accuracy`` with ``ba_low`` and ``ba_high``, ``accuracy``,
        ``previous`` (the previous champion's balanced accuracy), ``gain``,
        ``disagreements``, ``p_value``, ``best_p`` and ``status``.
    """
    tables = {
        PREVIOUS: table.select(
            "seq_id", "site", "label", pl.col("previous").alias("pred")
        ),
        CHAMPION: table.select("seq_id", "site", "label", "pred"),
    }
    groups = selection.site_groups(tables, PREVIOUS, CHAMPION, min_sequences=1)
    tests = selection.site_tests(groups, gate, sites_per_decision=len(groups))
    flagged = holm(tests["p_value"].to_numpy(), gate.family_alpha)
    cutoff = gate.family_alpha / len(groups)
    rng = np.random.default_rng(settings["seed"])
    sources = dict(table.select("site", "source").unique().iter_rows())
    rows = []
    for row, flag in zip(tests.iter_rows(named=True), flagged):
        labels, previous, champion = groups[row["site"]]
        low, high = bootstrap_interval(
            labels,
            champion,
            settings["bootstrap_resamples"],
            settings["confidence"],
            rng,
        )
        k = int(np.sum(previous != champion))
        best_p = best_attainable_p(k, gate.null_draws)
        if flag:
            status = STATUS_FLAGGED
        elif best_p > cutoff:
            status = STATUS_SMALL
        else:
            status = STATUS_CLEAR
        rows.append(
            {
                "site": row["site"],
                "source": sources[row["site"]],
                "n": row["n"],
                "balanced_accuracy": row["challenger"],
                "ba_low": low,
                "ba_high": high,
                "accuracy": float(np.mean(labels == champion)),
                "previous": row["champion"],
                "gain": row["gain"],
                "disagreements": k,
                "p_value": row["p_value"],
                "best_p": best_p,
                "status": status,
            }
        )
    return pl.DataFrame(rows)


def run(table, gate, settings=None):
    """Build every trust table from the joined predictions.

    Args:
        table: Output of ``load_predictions``.
        gate: The frozen ``GateConfig``.
        settings: Defaults to ``SETTINGS``.

    Returns:
        A tuple ``(summary, per_site, species, budget)``: a JSON-ready dict,
        the per-site table with review share and residual error at the
        suggested threshold and its weak species, the per-species table and
        the review budget by source.
    """
    settings = settings or SETTINGS
    budget = review_budget(table, settings["thresholds"], "source")
    chosen = choose_threshold(budget, settings["target_residual_error"])
    species = species_table(table, settings)
    per_site = site_table(table, gate, settings)

    # With no threshold meeting the target, report everything as reviewed.
    applied = chosen if chosen is not None else 1.0
    at_chosen = review_budget(table, [applied], "site").select(
        "site", "review_share", "residual_error"
    )
    weak = (
        species.filter(pl.col("status") == "weak")
        .group_by("site")
        .agg(pl.col("species").sort().str.join(", ").alias("weak_species"))
    )
    per_site = (
        per_site.join(at_chosen, on="site", how="left")
        .join(weak, on="site", how="left")
        .sort("source", "site", descending=[True, False])
    )

    def at(source):
        """Return the budget row for one source at the suggested threshold."""
        rows = budget.filter(
            (pl.col("source") == source) & (pl.col("threshold") == applied)
        )
        return rows.row(0, named=True) if rows.height else None

    held = per_site.filter(pl.col("source") == "holdout")
    summary = {
        "champion": CHAMPION,
        "previous_champion": PREVIOUS,
        "settings": settings,
        "family_alpha": gate.family_alpha,
        "sites": {s: int((per_site["source"] == s).sum()) for s in SOURCES},
        "suggested_threshold": chosen,
        "at_suggested_threshold": {s: at(s) for s in SOURCES},
        "holdout_mean_balanced_accuracy": float(held["balanced_accuracy"].mean()),
        # Sorted, as value_counts orders ties differently from run to run.
        "status_counts": dict(sorted(per_site["status"].value_counts().iter_rows())),
        "flagged_sites": per_site.filter(pl.col("status") == STATUS_FLAGGED)[
            "site"
        ].to_list(),
        "weak_species_cells": int((species["status"] == "weak").sum()),
        "weak_by_species": dict(
            sorted(
                species.filter(pl.col("status") == "weak")["species"]
                .value_counts()
                .iter_rows()
            )
        ),
    }
    return summary, per_site, species, budget


def pct(value, decimals=0):
    """Format a fraction as a percentage, or a dash if missing.

    Args:
        value: A fraction, or None.
        decimals: Decimal places; residual errors need one, as most are
            below 1%.

    Returns:
        A string such as ``"12%"``.
    """
    return "–" if value is None else f"{100 * value:.{decimals}f}%"


def markdown_table(header, rows):
    """Render a list of rows as a GitHub Markdown table.

    Args:
        header: Column titles.
        rows: Lists of cell strings.

    Returns:
        The table as a string.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(summary, per_site, budget, path):
    """Write the trust report as Markdown a field ecologist can read.

    Args:
        summary: From ``run``.
        per_site: From ``run``.
        budget: From ``run``.
        path: Where to write the file.
    """
    t = summary["suggested_threshold"]
    target = summary["settings"]["target_residual_error"]
    held = summary["at_suggested_threshold"]["holdout"]
    lines = [
        f"# Trust report: {LABELS[CHAMPION]}",
        "",
        "Generated by `make trust-report` from committed predictions. A "
        "retrospective audit of sites with labels; it gives no accuracy for a "
        "new, unlabelled site. Fold sites are optimistic (the model was "
        "chosen on them); hold-out sites were scored once by the frozen model.",
        "",
        "## Review budget",
        "",
        f"A sequence is accepted without review when the model's top "
        f"probability is at least the threshold; the rest are reviewed and "
        f"assumed corrected. Residual error is what is left wrong in the "
        f"final data. The suggested threshold is the lowest whose residual "
        f"error on fold sites is at most {pct(target)}, chosen on fold sites "
        f"only.",
        "",
    ]
    if t is None:
        lines.append("No threshold on the grid meets the target on fold sites.")
    else:
        lines.append(
            f"**Suggested threshold {t:g}.** On the hold-out sites it sends "
            f"{pct(held['review_share'])} of sequences to review and leaves "
            f"{pct(held['residual_error'], 1)} wrong."
        )
    rows = []
    for row in budget.sort("threshold", "source", descending=[False, True]).iter_rows(
        named=True
    ):
        rows.append(
            [
                f"{row['threshold']:g}",
                row["source"],
                pct(row["review_share"]),
                pct(row["residual_error"], 1),
                pct(row["accepted_error"], 1),
            ]
        )
    lines += [
        "",
        markdown_table(
            ["Threshold", "Split", "Reviewed", "Residual error", "Error if accepted"],
            rows,
        ),
        "",
        "## Per site",
        "",
        f"Regression: Holm over all {per_site.height} sites at alpha "
        f"{summary['family_alpha']:g}, against {LABELS[PREVIOUS]}. "
        f"'{STATUS_SMALL}' means the two models disagree on too few "
        "sequences for any drop to be detectable: send those sites to "
        "specialist review rather than reading them as safe. Weak species: "
        f"recall below {pct(summary['settings']['weak_recall'])} with at least "
        f"{summary['settings']['min_species_sequences']} sequences. The "
        "accuracy interval treats sequences as independent, so it is too "
        "narrow.",
        "",
    ]
    rows = [
        [
            row["site"],
            row["source"],
            str(row["n"]),
            f"{row['balanced_accuracy']:.2f} ({row['ba_low']:.2f}–{row['ba_high']:.2f})",
            f"{100 * row['gain']:+.1f}",
            row["status"],
            pct(row["review_share"]),
            pct(row["residual_error"], 1),
            (row["weak_species"] or "").replace("_", " "),
        ]
        for row in per_site.iter_rows(named=True)
    ]
    header = [
        "Site",
        "Split",
        "Sequences",
        "Balanced accuracy (95%)",
        "Gain (points)",
        "Regression",
        f"Reviewed at {t:g}" if t is not None else "Reviewed",
        "Residual error",
        "Weak species",
    ]
    lines += [markdown_table(header, rows), ""]
    path.write_text("\n".join(lines))


def plot_budget(budget, chosen, target, path):
    """Plot review share against residual error for fold and hold-out sites.

    Args:
        budget: ``review_budget`` grouped by ``source``.
        chosen: The suggested threshold, or None.
        target: The residual-error target.
        path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(100 * target, color="#52514e", linewidth=1, linestyle="--")
    # x in axes fraction, y in data units, so the label stays inside the plot.
    ax.text(
        0.99,
        100 * target,
        f"target {pct(target)}",
        ha="right",
        va="bottom",
        fontsize=8,
        transform=ax.get_yaxis_transform(),
    )
    names = {"fold": "Fold sites (optimistic)", "holdout": "Hold-out sites"}
    for source in ("fold", "holdout"):
        rows = budget.filter(pl.col("source") == source).sort("threshold")
        x = rows["review_share"].to_numpy() * 100
        y = rows["residual_error"].to_numpy() * 100
        ax.plot(x, y, color=COLOURS[source], linewidth=1.6, zorder=2)
        ax.scatter(
            x,
            y,
            s=26,
            color=COLOURS[source],
            edgecolor="white",
            linewidth=1,
            zorder=3,
            label=names[source],
        )
        if source == "holdout":
            for t, xi, yi in zip(rows["threshold"], x, y):
                ax.annotate(
                    f"{t:g}",
                    (xi, yi),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="#52514e",
                )
        if chosen is not None:
            pick = rows.filter(pl.col("threshold") == chosen)
            ax.scatter(
                pick["review_share"] * 100,
                pick["residual_error"] * 100,
                s=110,
                facecolor="none",
                edgecolor=COLOURS[source],
                linewidth=1.6,
                zorder=4,
            )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sequences sent to review (%)", fontsize=9)
    ax.set_ylabel("Left wrong in the final data (%)", fontsize=9)
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    title = "Review budget for the DINOv2 probe"
    if chosen is not None:
        title += f": circled, threshold {chosen:g}"
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the trust report from committed predictions.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)

    _, gate = freeze.load(args.root / FREEZE_PATH)
    table = load_predictions(args.root)
    summary, per_site, species, budget = run(table, gate)

    out = args.root / args.out_dir
    figures = args.root / args.figure_dir
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    per_site.write_csv(out / "per_site.csv")
    species.write_csv(out / "per_species.csv")
    budget.write_csv(out / "thresholds.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, per_site, budget, out / "trust_report.md")
    plot_budget(
        budget,
        summary["suggested_threshold"],
        SETTINGS["target_residual_error"],
        figures / "trust_review_budget.png",
    )
    print(json.dumps({k: summary[k] for k in list(summary)[3:]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
