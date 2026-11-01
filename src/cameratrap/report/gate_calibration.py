"""Calibrate the gate's per-site tolerance on the fold sites.

Runs the swap null (``gate.calibrate``) on a pair of arms scored on the
same fold sequences and reports, for each candidate tolerance, how often
it would block a challenger that is in truth no worse: per site, and for
random sets of ten sites (the hold-out's size). The hold-out is never
read. Until the Phase 3 challengers exist, the pair is BioCLIP 2 and
SpeciesNet, which disagree more than a probe and its own backbone
would, so the rates here are on the pessimistic side; the run is
repeated with the real pair before the freeze.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import scoring, speciesnet_arm
from cameratrap.data import crops, fetch, orinoquia
from cameratrap.embed import store
from cameratrap.gate import calibrate
from cameratrap.report import baselines

SETTINGS = {
    "pair": ["bioclip2", "speciesnet"],
    "min_site_sequences": 20,
    # Smallest hold-out site; sets of fold sites at least this big are the
    # closer match to the hold-out.
    "holdout_like_min_sequences": 49,
    "holdout_sites": 10,
    "tolerances": [round(t, 2) for t in np.arange(0.02, 0.205, 0.01)],
    "draws": 2000,
    "subsets": 5000,
    "target_family_rate": 0.05,
    "seed": 0,
}
OUT_DIR = Path("results/gate")
FIGURE_DIR = Path("figures")


def site_groups(index, sizes, bioclip, speciesnet, min_sequences):
    """Return each fold site's labels and both arms' sequence predictions.

    Args:
        index: The crop index.
        sizes: The crop-size table.
        bioclip: The BioCLIP 2 cache from ``store.read``.
        speciesnet: Output of ``speciesnet_arm.read_predictions``.
        min_sequences: Sites with fewer sequences are left out, as in the
            baselines.

    Returns:
        A dict of site to ``(labels, bioclip2_preds, speciesnet_preds)``.
    """
    classes = bioclip["classes"].tolist()
    frames = baselines.fold_frames(index, sizes, baselines.SETTINGS["size_bins"])
    sequences = baselines.sequence_table(frames)
    paths = frames["path"].to_list()
    sn_logits, _ = baselines.speciesnet_frames(speciesnet, paths)
    logits = {
        "bioclip2": baselines.zero_shot_logits(bioclip, paths),
        "speciesnet": sn_logits,
    }
    scored = {
        arm: baselines.score(frames, sequences, logits[arm], classes).select(
            "seq_id", "site", "label", pl.col("pred").alias(arm)
        )
        for arm in SETTINGS["pair"]
    }
    joined = scored["bioclip2"].join(
        scored["speciesnet"].select("seq_id", "speciesnet"), on="seq_id"
    )
    groups = {}
    for (site,), group in joined.group_by("site"):
        if group.height >= min_sequences:
            groups[site] = tuple(
                group[c].to_numpy() for c in ("label", "bioclip2", "speciesnet")
            )
    return groups


def run(groups):
    """Compute false-block rates for every candidate tolerance.

    Args:
        groups: Output of ``site_groups``.

    Returns:
        A pair ``(summary, tables)``: a JSON-serialisable dict and a dict of
        DataFrames keyed by output file stem.
    """
    tolerances = SETTINGS["tolerances"]
    rates = calibrate.false_block_rates(
        groups,
        scoring.balanced_accuracy,
        tolerances,
        draws=SETTINGS["draws"],
        seed=SETTINGS["seed"],
    )
    sizes = {site: len(groups[site][0]) for site in groups}
    disagreement = {site: float(np.mean(g[1] != g[2])) for site, g in groups.items()}
    big = {
        s: r
        for s, r in rates.items()
        if sizes[s] >= SETTINGS["holdout_like_min_sequences"]
    }

    columns = {"tolerance": tolerances}
    for name, subset in (("all_fold_sites", rates), ("holdout_like_sites", big)):
        columns[f"mean_site_rate_{name}"] = np.mean(list(subset.values()), axis=0)
        columns[f"family_rate_{name}"] = calibrate.family_rate(
            subset, SETTINGS["holdout_sites"], SETTINGS["subsets"], SETTINGS["seed"]
        )
    curve = pl.DataFrame(columns)
    per_site = pl.DataFrame(
        {
            "site": sorted(rates),
            "n": [sizes[s] for s in sorted(rates)],
            "disagreement": [disagreement[s] for s in sorted(rates)],
            **{
                f"rate_at_{t:.2f}": [rates[s][i] for s in sorted(rates)]
                for i, t in enumerate(tolerances)
            },
        }
    )

    def smallest_meeting(column):
        """Return the smallest tolerance whose family rate meets the target."""
        ok = curve.filter(pl.col(column) <= SETTINGS["target_family_rate"])
        return None if ok.is_empty() else float(ok["tolerance"].min())

    summary = {
        "settings": SETTINGS,
        "scope": "fold sites only; hold-out never read",
        "sites": len(rates),
        "holdout_like_sites": len(big),
        "median_disagreement": float(np.median(list(disagreement.values()))),
        "family_rate_at_plan_tolerance_0.05": {
            name: float(
                curve.filter(pl.col("tolerance") == 0.05)[f"family_rate_{name}"][0]
            )
            for name in ("all_fold_sites", "holdout_like_sites")
        },
        "smallest_tolerance_meeting_target": {
            name: smallest_meeting(f"family_rate_{name}")
            for name in ("all_fold_sites", "holdout_like_sites")
        },
        "note": (
            "stand-in pair (BioCLIP 2 vs SpeciesNet); rerun with the Phase 3 "
            "challenger before the freeze"
        ),
    }
    return summary, {"calibration_curve": curve, "calibration_per_site": per_site}


def plot_curve(curve, target, path):
    """Plot the family-wise false-block rate against the tolerance.

    Args:
        curve: The ``calibration_curve`` table.
        target: Target family-wise rate, drawn as a line.
        path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    tol = curve["tolerance"].to_numpy() * 100
    for name, label in (
        ("holdout_like_sites", "10 fold sites with at least 49 sequences"),
        ("all_fold_sites", "10 fold sites, any size (20+ sequences)"),
    ):
        ax.plot(tol, curve[f"family_rate_{name}"].to_numpy(), marker="o", label=label)
    ax.axhline(target, color="grey", linestyle="--", linewidth=1)
    ax.axvline(5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Per-site tolerance (balanced-accuracy points)")
    ax.set_ylabel("Chance an equal challenger is blocked")
    ax.set_title(
        "Gate tolerance under the swap null (BioCLIP 2 vs SpeciesNet, fold sites)"
    )
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the tolerance calibration tables and figure from the caches.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=crops.INDEX_PATH)
    parser.add_argument("--sizes", type=Path, default=crops.SIZES_PATH)
    parser.add_argument("--bioclip", type=Path, default=store.cache_path("bioclip2"))
    parser.add_argument(
        "--speciesnet",
        type=Path,
        default=speciesnet_arm.CROP_CACHE_DIR / "predictions.json",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)

    bioclip = store.read(args.bioclip)
    # A cache embedded from another index would score the wrong crops.
    store.check_meta(bioclip["meta"], {"index_sha256": store.file_sha256(args.index)})
    speciesnet = speciesnet_arm.read_predictions(
        args.speciesnet, fetch.IMAGE_DIR, orinoquia.TAXA
    )
    groups = site_groups(
        crops.load_index(args.index),
        crops.load_sizes(args.sizes),
        bioclip,
        speciesnet,
        SETTINGS["min_site_sequences"],
    )
    summary, tables = run(groups)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for stem, table in tables.items():
        table.write_csv(args.out_dir / f"{stem}.csv")
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot_curve(
        tables["calibration_curve"],
        SETTINGS["target_family_rate"],
        args.figure_dir / "gate_calibration.png",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
