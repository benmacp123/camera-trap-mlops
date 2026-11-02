"""Adapted challengers per site from the caches alone: linear probes on frozen embeddings.

Each probe is trained out of fold: the model that scores a fold's sites
was fitted on the other folds' frames, with its regularisation chosen by
an inner split over those training sites only. Frames are scored at
sequence level exactly as the zero-shot baselines are (same frames, same
sequences, logits averaged before the softmax), and each site's gain over
BioCLIP 2 zero-shot, the headline champion, is reported. Hold-out sites
are neither trained on nor scored: they stay sealed until the freeze.

Writes ``results/challengers/``: ``per_site.csv``, ``sequences_<arm>.csv``
(one prediction per sequence, which the gate and its calibration read
without refitting), ``summary.json`` and a figure under ``figures/``.

::

    PYTHONPATH=src python -m cameratrap.report.challengers
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import probe, scoring
from cameratrap.data import crops
from cameratrap.embed import store
from cameratrap.report import baselines

SETTINGS = {
    # The baselines' thresholds, so the tables compare.
    "min_site_sequences": baselines.SETTINGS["min_site_sequences"],
    "weak_site_threshold": baselines.SETTINGS["weak_site_threshold"],
    "champion": "bioclip2",
    "probe_backbones": ["bioclip2", "dinov2"],
    "probe_cs": probe.CS.tolist(),
    "probe_inner_folds": probe.INNER_FOLDS,
    # Reported only; the gate's own tolerance is still open.
    "regression_points": 0.05,
}
OUT_DIR = Path("results/challengers")
FIGURE_DIR = Path("figures")
LABELS = {
    "bioclip2_probe": "Probe on BioCLIP 2",
    "dinov2_probe": "Probe on DINOv2",
}
# Categorical slots 1 and 3 of the reference palette, which validate
# all-pairs; slot 2 (orange) stays SpeciesNet's in the baseline figures.
COLOURS = {"bioclip2_probe": "#2a78d6", "dinov2_probe": "#1baf7a"}
MARKERS = {"bioclip2_probe": "o", "dinov2_probe": "s"}


def probe_logits(frames, cache, classes):
    """Return out-of-fold probe logits for the fold frames.

    Args:
        frames: Frames from ``baselines.fold_frames``, in row order.
        cache: The backbone's cache from ``store.read``.
        classes: Class names in logit order.

    Returns:
        A pair ``(logits, chosen_c)`` as from ``probe.out_of_fold_logits``.
    """
    features = store.rows_for(cache, frames["path"].to_list())
    return probe.out_of_fold_logits(
        features,
        frames["label"].to_numpy(),
        frames["site"].to_numpy(),
        frames["split"].to_numpy(),
        classes,
    )


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
        A tuple ``(summary, per_site, sequences)``: a JSON-serialisable
        dict, the per-site table, and a dict of arm name to its
        per-sequence predictions.
    """
    champion = SETTINGS["champion"]
    classes = caches[champion]["classes"].tolist()
    frames = baselines.fold_frames(index, sizes, baselines.SETTINGS["size_bins"])
    seq_table = baselines.sequence_table(frames)
    paths = frames["path"].to_list()

    logits = {champion: baselines.zero_shot_logits(caches[champion], paths)}
    chosen = {}
    for backbone in SETTINGS["probe_backbones"]:
        arm = f"{backbone}_probe"
        logits[arm], chosen[arm] = probe_logits(frames, caches[backbone], classes)

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
        arms_summary[arm] = site_summary(per_site, f"{arm}_balanced_accuracy") | {
            "mean_gain": float(gain.mean()),
            "median_gain": float(np.median(gain)),
            "sites_improved": int((gain > 0).sum()),
            "sites_regressed_beyond_points": int(
                (gain < -SETTINGS["regression_points"]).sum()
            ),
            "worst_site_gain": float(gain.min()),
            "chosen_c_per_fold": chosen[arm],
        }

    summary = {
        "settings": SETTINGS,
        "scope": "fold sites only; hold-out neither trained on nor scored",
        "frames": frames.height,
        "sequences": seq_table.height,
        "sites_scored": per_site.height,
        "arms": arms_summary,
        "caches": {b: caches[b]["meta"] for b in sorted(caches)},
    }
    return summary, per_site, sequences


def plot_gain(per_site, path):
    """Plot each probe's per-site gain over BioCLIP 2 zero-shot.

    Args:
        per_site: The per-site table from ``run``.
        path: Where to save the PNG.
    """
    arms = [a for a in LABELS if f"{a}_gain" in per_site.columns]
    ordered = per_site.sort(f"{arms[0]}_gain")
    y = np.arange(ordered.height)
    fig, ax = plt.subplots(figsize=(6.5, 0.2 * ordered.height + 1.4))
    ax.axvline(0, color="#52514e", linewidth=1)
    for arm in arms:
        ax.scatter(
            ordered[f"{arm}_gain"] * 100,
            y,
            s=28,
            color=COLOURS[arm],
            marker=MARKERS[arm],
            edgecolor="white",
            linewidth=1,
            label=LABELS[arm],
            zorder=3,
        )
    ax.set_yticks(y, ordered["site"].to_list(), fontsize=7)
    ax.set_xlabel("Gain in sequence-level balanced accuracy over BioCLIP 2 (points)")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Probe gain over zero-shot, per site", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the probe tables and figure from the embedding caches.

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
    for backbone in SETTINGS["probe_backbones"]:
        caches[backbone] = store.read(store.cache_path(backbone, args.cache_root))
        # A cache embedded from another index would score the wrong crops.
        store.check_meta(caches[backbone]["meta"], {"index_sha256": index_sha})
    summary, per_site, sequences = run(
        crops.load_index(args.index), crops.load_sizes(args.sizes), caches
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    per_site.write_csv(args.out_dir / "per_site.csv")
    for arm, table in sequences.items():
        table.select("seq_id", "site", "label", "pred", "msp").write_csv(
            args.out_dir / f"sequences_{arm}.csv"
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_gain(per_site, args.figure_dir / "challengers_probe_gain.png")
    print(json.dumps(summary["arms"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
