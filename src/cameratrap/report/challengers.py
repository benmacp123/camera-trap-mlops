"""Adapted challengers per site from the caches alone: probes and kNN on frozen embeddings.

Each arm is trained out of fold: the model that scores a fold's sites
was fitted on the other folds' frames, with its setting (the probe's C,
kNN's k) chosen by an inner split over those training sites only. Frames are scored at
sequence level exactly as the zero-shot baselines are (same frames, same
sequences, logits averaged before the softmax), and each site's gain over
BioCLIP 2 zero-shot, the headline champion, is reported. Hold-out sites
are neither trained on nor scored: they stay sealed until the freeze.

Writes ``results/challengers/``: ``per_site.csv``, ``sequences_<arm>.csv``
(one prediction per sequence, which the gate and its calibration read
without refitting), ``knn_validation.csv`` (inner balanced accuracy for each candidate k),
``summary.json`` and two figures under ``figures/``.

::

    PYTHONPATH=src python -m cameratrap.report.challengers
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import polars as pl

from cameratrap.arms import knn, probe, scoring
from cameratrap.data import crops
from cameratrap.embed import store
from cameratrap.report import baselines

SETTINGS = {
    # The baselines' thresholds, so the tables compare.
    "min_site_sequences": baselines.SETTINGS["min_site_sequences"],
    "weak_site_threshold": baselines.SETTINGS["weak_site_threshold"],
    "champion": "bioclip2",
    "backbones": ["bioclip2", "dinov2"],
    "probe_cs": probe.CS.tolist(),
    "probe_inner_folds": probe.INNER_FOLDS,
    "knn_ks": list(knn.KS),
    "knn_inner_folds": knn.INNER_FOLDS,
    "knn_pseudo_count": knn.PSEUDO_COUNT,
    # Reported only; the gate's own tolerance is still open.
    "regression_points": 0.05,
}
OUT_DIR = Path("results/challengers")
FIGURE_DIR = Path("figures")
METHODS = {"probe": "Linear probe", "knn": "kNN"}
BACKBONES = {"bioclip2": "BioCLIP 2", "dinov2": "DINOv2"}
# Colour follows the backbone, never the method. Categorical slots 1 and
# 3 of the reference palette, which validate all-pairs; slot 2 (orange)
# stays SpeciesNet's in the baseline figures.
COLOURS = {"bioclip2": "#2a78d6", "dinov2": "#1baf7a"}
MARKERS = {"bioclip2": "o", "dinov2": "s"}


def fit_arms(frames, cache, classes):
    """Return out-of-fold probe and kNN results on one backbone's embeddings.

    Args:
        frames: Frames from ``baselines.fold_frames``, in row order.
        cache: The backbone's cache from ``store.read``.
        classes: Class names in logit order.

    Returns:
        A dict of method (``probe``, ``knn``) to a tuple ``(logits,
        chosen, curves)``: frame logits, the setting each fold chose, and
        kNN's inner validation curves per fold (``None`` for the probe).
    """
    features = store.rows_for(cache, frames["path"].to_list())
    data = (
        features,
        frames["label"].to_numpy(),
        frames["site"].to_numpy(),
        frames["split"].to_numpy(),
        classes,
    )
    return {
        "probe": (*probe.out_of_fold_logits(*data), None),
        "knn": knn.out_of_fold_logits(*data),
    }


def site_summary(sites, column):
    """Summarise one arm's per-site balanced accuracy.

    Args:
        sites: A per-site table.
        column: The balanced-accuracy column to summarise.

    Returns:
        A dict of mean, median, minimum and the count of weak sites.
    """
    ba = sites[column].to_numpy()
    return {
        "mean_site_balanced_accuracy": float(ba.mean()),
        "median_site_balanced_accuracy": float(np.median(ba)),
        "min_site_balanced_accuracy": float(ba.min()),
        "weak_sites": int((ba < SETTINGS["weak_site_threshold"]).sum()),
    }


def run(index, sizes, caches):
    """Score the champion and every probe on the fold sites.

    Args:
        index: The crop index.
        sizes: The crop-size table.
        caches: Mapping of backbone name to its cache from ``store.read``;
            ``bioclip2`` must be present, since it carries the class order
            and the champion's text embeddings.

    Returns:
        A tuple ``(summary, per_site, sequences, knn_curves)``: a
        JSON-serialisable dict, the per-site table, a dict of arm name to
        its per-sequence predictions, and kNN's inner validation scores
        with ``backbone``, ``fold``, ``k`` and ``inner_balanced_accuracy``.
    """
    champion = SETTINGS["champion"]
    classes = caches[champion]["classes"].tolist()
    frames = baselines.fold_frames(index, sizes, baselines.SETTINGS["size_bins"])
    seq_table = baselines.sequence_table(frames)
    paths = frames["path"].to_list()

    logits = {champion: baselines.zero_shot_logits(caches[champion], paths)}
    chosen, curve_rows = {}, []
    for backbone in SETTINGS["backbones"]:
        for method, (arm_logits, arm_chosen, curves) in fit_arms(
            frames, caches[backbone], classes
        ).items():
            arm = f"{backbone}_{method}"
            logits[arm] = arm_logits
            setting = "c" if method == "probe" else "k"
            chosen[arm] = {f"chosen_{setting}_per_fold": arm_chosen}
            curve_rows += [
                {"backbone": backbone, "fold": f, "k": k, "inner_balanced_accuracy": v}
                for f, curve in (curves or {}).items()
                for k, v in curve.items()
            ]

    sequences, per_site = {}, None
    for arm, arm_logits in logits.items():
        sequences[arm] = baselines.score(frames, seq_table, arm_logits, classes)
        sites = scoring.per_site(sequences[arm], SETTINGS["min_site_sequences"])
        renamed = sites.select(
            "site", "n", pl.col("balanced_accuracy").alias(f"{arm}_balanced_accuracy")
        )
        per_site = (
            renamed if per_site is None else per_site.join(renamed, on=["site", "n"])
        )

    base = f"{champion}_balanced_accuracy"
    arms_summary = {champion: site_summary(per_site, base)}
    for arm in chosen:
        per_site = per_site.with_columns(
            (pl.col(f"{arm}_balanced_accuracy") - pl.col(base)).alias(f"{arm}_gain")
        )
        gain = per_site[f"{arm}_gain"].to_numpy()
        arms_summary[arm] = (
            site_summary(per_site, f"{arm}_balanced_accuracy")
            | {
                "mean_gain": float(gain.mean()),
                "median_gain": float(np.median(gain)),
                "sites_improved": int((gain > 0).sum()),
                "sites_regressed_beyond_points": int(
                    (gain < -SETTINGS["regression_points"]).sum()
                ),
                "worst_site_gain": float(gain.min()),
            }
            | chosen[arm]
        )

    summary = {
        "settings": SETTINGS,
        "scope": "fold sites only; hold-out neither trained on nor scored",
        "frames": frames.height,
        "sequences": seq_table.height,
        "sites_scored": per_site.height,
        "arms": arms_summary,
        "caches": {b: caches[b]["meta"] for b in sorted(caches)},
    }
    return summary, per_site, sequences, pl.DataFrame(curve_rows)


def plot_gain(per_site, path):
    """Plot each challenger's per-site gain over BioCLIP 2 zero-shot.

    One panel per method, sharing the site axis (sorted by the BioCLIP 2
    probe's gain), with colour and marker for the backbone.

    Args:
        per_site: The per-site table from ``run``.
        path: Where to save the PNG.
    """
    ordered = per_site.sort("bioclip2_probe_gain")
    y = np.arange(ordered.height)
    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(9, 0.2 * ordered.height + 1.6), sharey=True
    )
    for ax, (method, title) in zip(axes, METHODS.items()):
        ax.axvline(0, color="#52514e", linewidth=1)
        for backbone, name in BACKBONES.items():
            ax.scatter(
                ordered[f"{backbone}_{method}_gain"] * 100,
                y,
                s=28,
                color=COLOURS[backbone],
                marker=MARKERS[backbone],
                edgecolor="white",
                linewidth=1,
                label=name,
                zorder=3,
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, ordered["site"].to_list(), fontsize=7)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.supxlabel(
        "Gain in sequence-level balanced accuracy over BioCLIP 2 zero-shot (points)",
        fontsize=9,
    )
    fig.suptitle("Challenger gain over zero-shot, per site", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_knn_curves(curves, path):
    """Plot kNN's inner validation score against k, one line per outer fold.

    A diagnostic, not the choice itself: k is picked by the maximum inner
    score inside each fold. A flat curve means the choice barely matters.

    Args:
        curves: The kNN validation table from ``run``.
        path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for backbone, name in BACKBONES.items():
        rows = curves.filter(pl.col("backbone") == backbone)
        for i, ((_,), fold) in enumerate(rows.group_by("fold", maintain_order=True)):
            fold = fold.sort("k")
            ax.plot(
                fold["k"],
                fold["inner_balanced_accuracy"],
                color=COLOURS[backbone],
                marker=MARKERS[backbone],
                markersize=5,
                linewidth=1.5,
                alpha=0.8,
                label=name if i == 0 else None,
            )
    ax.set_xscale("log")
    ax.set_xticks(SETTINGS["knn_ks"], [str(k) for k in SETTINGS["knn_ks"]])
    # A log axis labels its minor ticks too ("6x10^0"); only the candidates matter.
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("Inner balanced accuracy (frames)")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("kNN: choosing k inside each fold", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the challenger tables and figures from the embedding caches.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=crops.INDEX_PATH)
    parser.add_argument("--sizes", type=Path, default=crops.SIZES_PATH)
    parser.add_argument("--cache-root", type=Path, default=store.CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)

    index_sha = store.file_sha256(args.index)
    caches = {}
    for backbone in SETTINGS["backbones"]:
        caches[backbone] = store.read(store.cache_path(backbone, args.cache_root))
        # A cache embedded from another index would score the wrong crops.
        store.check_meta(caches[backbone]["meta"], {"index_sha256": index_sha})
    summary, per_site, sequences, curves = run(
        crops.load_index(args.index), crops.load_sizes(args.sizes), caches
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    per_site.write_csv(args.out_dir / "per_site.csv")
    for arm, table in sequences.items():
        table.select("seq_id", "site", "label", "pred", "msp").write_csv(
            args.out_dir / f"sequences_{arm}.csv"
        )
    curves.write_csv(args.out_dir / "knn_validation.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_gain(per_site, args.figure_dir / "challengers_gain.png")
    plot_knn_curves(curves, args.figure_dir / "challengers_knn_k.png")
    print(json.dumps(summary["arms"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
