"""First spike on Orinoquía: the measurements that decide the gate dataset.

Reads the cached model outputs (BioCLIP 2 embeddings, SpeciesNet
predictions, the fresh MDv5a run) and the committed manifest, and writes
``results/spike/summary.json`` and ``results/spike/per_site.csv``. Only
fold sites are scored; the hold-out stays unseen.

The criteria in ``CRITERIA`` were fixed in the plan before anything ran.
The settings in ``SETTINGS`` were fixed on 2026-10-28, before any arm was
scored.

Run as a script once the caches are complete::

    PYTHONPATH=src python -m cameratrap.report.spike
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from cameratrap.arms import probe, scoring, speciesnet_arm
from cameratrap.data import fetch, manifest, orinoquia, subset

SETTINGS = {
    "min_site_sequences": 20,
    "detection_threshold": 0.2,
    "power_difference": 0.05,
    "power_alpha": 0.05,
    "power": 0.8,
    "bootstrap_resamples": 2000,
    "seed": 20261028,
}
CRITERIA = {
    "detector_agreement": 0.98,
    "min_classes": 8,
    "min_sequences_per_class": 100,
    "weak_site_threshold": 0.85,
    "min_weak_site_share": 0.25,
}
OUT_DIR = Path("results/spike")
BIOCLIP_CACHE = Path("data/cache/bioclip2/embeddings.npz")


def detector_agreement(precomputed, fresh, threshold):
    """Compare two detector runs on whether each image holds an animal.

    Args:
        precomputed: Mapping of path to LILA's MDv5a detections.
        fresh: Mapping of path to the fresh run's detections.
        threshold: Confidence at which a box counts.

    Returns:
        Dict with ``n``, ``agreement`` and the counts where only one run
        found an animal.
    """

    def has_animal(dets):
        """Return whether any animal box reaches the threshold.

        Args:
            dets: Detections for one image.

        Returns:
            True if an animal box has ``conf >= threshold``.
        """
        return any(d["category"] == "1" and d["conf"] >= threshold for d in dets)

    pairs = [(has_animal(precomputed[p]), has_animal(d)) for p, d in fresh.items()]
    return {
        "n": len(pairs),
        "agreement": float(np.mean([a == b for a, b in pairs])),
        "only_precomputed": sum(a and not b for a, b in pairs),
        "only_fresh": sum(b and not a for a, b in pairs),
    }


def majority_label(labels):
    """Return a sequence's open-set answer from its frames' answers.

    Args:
        labels: Frame answers; ``None`` is an abstention.

    Returns:
        The most common non-abstaining answer (ties go to the first seen),
        or ``None`` if every frame abstained.
    """
    counts = collections.Counter(label for label in labels if label is not None)
    return counts.most_common(1)[0][0] if counts else None


def sites_needed(differences, delta, alpha, power):
    """Return the sites a paired comparison needs to detect a difference.

    Normal approximation to the paired test: ``((z_a + z_b) * sd / delta)^2``.

    Args:
        differences: Per-site paired differences (challenger minus champion).
        delta: Difference to detect, on the same scale.
        alpha: Two-sided significance level.
        power: Desired power.

    Returns:
        A pair ``(sd, n)``: the between-site standard deviation and the
        number of sites, rounded up.
    """
    sd = float(np.std(differences, ddof=1))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return sd, math.ceil((z * sd / delta) ** 2)


def bootstrap_spearman(x, y, resamples, seed):
    """Spearman correlation with a percentile bootstrap over sites.

    Args:
        x: One value per site.
        y: One value per site.
        resamples: Number of bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        Dict with ``rho``, ``low`` and ``high`` (95% interval) and ``n``.
    """
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        index = rng.integers(0, len(x), len(x))
        rho = stats.spearmanr(x[index], y[index]).statistic
        if not np.isnan(rho):
            draws.append(rho)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "rho": float(stats.spearmanr(x, y).statistic),
        "low": float(low),
        "high": float(high),
        "n": len(x),
    }


def score_arm(name, frames, frame_logits, sequences, classes):
    """Score one arm's frame logits at sequence and site level.

    Args:
        name: Arm name, used as a column prefix.
        frames: DataFrame of frames with ``seq_id``, in logit row order.
        frame_logits: Array of shape ``(n_frames, len(classes))``.
        sequences: DataFrame of evaluated sequences with ``seq_id``,
            ``site`` and ``label``.
        classes: Class names in logit order.

    Returns:
        A pair ``(scored, sites)``: per-sequence predictions joined to
        ``sequences``, and the per-site table with columns prefixed by
        ``name``.
    """
    preds = scoring.sequence_predictions(
        frames["seq_id"].to_list(), frame_logits, classes
    )
    scored = sequences.join(preds, on="seq_id")
    sites = scoring.per_site(scored, SETTINGS["min_site_sequences"])
    renamed = sites.rename(
        {c: f"{name}_{c}" for c in ("balanced_accuracy", "accuracy")}
    )
    return scored, renamed


def run(frames, sequences, bioclip, speciesnet, detector, dataset_sites):
    """Compute every spike measurement.

    Args:
        frames: Classify frames (manifest rows) to score, sorted by path.
        sequences: One row per evaluated sequence: ``seq_id``, ``site``,
            ``label``, ``split``.
        bioclip: The BioCLIP 2 cache from ``embed.bioclip.load_cache``.
        speciesnet: Output of ``speciesnet_arm.read_predictions``.
        detector: Dict with ``precomputed`` and ``fresh`` detections for
            the detector check.
        dataset_sites: Number of sites in the whole dataset, which the
            power criterion compares against.

    Returns:
        A pair ``(summary, per_site)``: a JSON-serialisable dict and a
        DataFrame with one row per scored site.
    """
    classes = bioclip["classes"].tolist()
    row = {p: i for i, p in enumerate(bioclip["paths"].tolist())}
    index = [row[p] for p in frames["path"].to_list()]
    image = bioclip["image"][index]

    zero_shot = float(bioclip["logit_scale"]) * image @ bioclip["text"].T
    sn_logits = np.stack([speciesnet[p]["logits"] for p in frames["path"].to_list()])
    probe_logits, chosen_c = probe.out_of_fold_logits(
        image,
        frames["label"].to_list(),
        frames["site"].to_list(),
        frames["split"].to_list(),
        classes,
    )

    scored, per_site = {}, None
    for name, logits in (
        ("bioclip", zero_shot),
        ("speciesnet", sn_logits),
        ("probe", probe_logits),
    ):
        scored[name], sites = score_arm(name, frames, logits, sequences, classes)
        per_site = sites if per_site is None else per_site.join(sites, on=["site", "n"])

    # SpeciesNet as it is used in the field: its own geofenced answer,
    # abstaining when it rolls up above species.
    open_by_seq = collections.defaultdict(list)
    for path, seq_id in frames.select("path", "seq_id").iter_rows():
        open_by_seq[seq_id].append(speciesnet[path]["open_label"])
    open_scored = sequences.with_columns(
        pl.col("seq_id")
        .map_elements(lambda s: majority_label(open_by_seq[s]), return_dtype=pl.String)
        .alias("pred")
    )
    covered = open_scored.filter(pl.col("pred").is_not_null())
    open_sites = open_scored.group_by("site").agg(
        pl.col("pred").is_not_null().mean().alias("speciesnet_open_coverage"),
        (pl.col("pred") == pl.col("label")).mean().alias("speciesnet_open_accuracy"),
    )
    per_site = per_site.join(open_sites, on="site", how="left")

    # The best zero-shot arm by mean per-site balanced accuracy is the
    # comparison the plan's criteria name.
    means = {
        n: float(per_site[f"{n}_balanced_accuracy"].mean())
        for n in ("bioclip", "speciesnet", "probe")
    }
    best = max(("bioclip", "speciesnet"), key=means.get)
    best_ba = per_site[f"{best}_balanced_accuracy"].to_numpy()
    weak_share = float(np.mean(best_ba < CRITERIA["weak_site_threshold"]))

    power = {}
    for champion in ("bioclip", "speciesnet"):
        diff = (
            per_site["probe_balanced_accuracy"]
            - per_site[f"{champion}_balanced_accuracy"]
        ).to_numpy()
        sd, n_sites = sites_needed(
            diff,
            SETTINGS["power_difference"],
            SETTINGS["power_alpha"],
            SETTINGS["power"],
        )
        power[champion] = {
            "mean_difference": float(diff.mean()),
            "sd_difference": sd,
            "sites_needed": n_sites,
        }

    msp = {}
    for name in ("bioclip", "probe"):
        site_msp = scored[name].group_by("site").agg(pl.col("msp").mean())
        joined = per_site.join(site_msp, on="site")
        msp[name] = bootstrap_spearman(
            joined["msp"].to_numpy(),
            joined[f"{name}_accuracy"].to_numpy(),
            SETTINGS["bootstrap_resamples"],
            SETTINGS["seed"],
        )

    evaluated = sequences.filter(pl.col("site").is_in(per_site["site"].to_list()))
    per_class = dict(evaluated.group_by("label").len().sort("label").iter_rows())
    agreement = detector_agreement(
        detector["precomputed"], detector["fresh"], SETTINGS["detection_threshold"]
    )
    decision = {
        "use_precomputed_detections": agreement["agreement"]
        >= CRITERIA["detector_agreement"],
        "classes_with_enough_sequences": sum(
            n >= CRITERIA["min_sequences_per_class"] for n in per_class.values()
        ),
        "weak_site_share": weak_share,
        "sites_needed": power[best]["sites_needed"],
        "dataset_sites": dataset_sites,
    }
    decision["orinoquia_is_gate_dataset"] = bool(
        decision["classes_with_enough_sequences"] >= CRITERIA["min_classes"]
        and weak_share >= CRITERIA["min_weak_site_share"]
        and decision["sites_needed"] <= dataset_sites
    )

    summary = {
        "settings": SETTINGS,
        "criteria": CRITERIA,
        "frames": frames.height,
        "sequences": sequences.height,
        "sites_scored": per_site.height,
        "sequences_per_class": per_class,
        "detector_check": agreement,
        "mean_site_balanced_accuracy": means,
        "pooled_balanced_accuracy": {
            n: scoring.balanced_accuracy(s["label"].to_numpy(), s["pred"].to_numpy())
            for n, s in scored.items()
        },
        "speciesnet_open": {
            "coverage": covered.height / sequences.height,
            "accuracy_on_covered": float((covered["pred"] == covered["label"]).mean()),
        },
        "best_zero_shot": best,
        "probe_c_per_fold": chosen_c,
        "power": power,
        "msp_accuracy_spearman": msp,
        "decision": decision,
    }
    return summary, per_site


def main(argv=None):
    """Load the caches, run the spike and write its results.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    full = manifest.load(manifest.OUT_DIR / f"{orinoquia.NAME}.csv")
    selection = pl.read_csv(subset.OUT_PATH)
    classify = selection.filter("classify").select("path")
    frames = full.join(classify, on="path").sort("path")
    sequences = (
        frames.select("seq_id", "site", "label", "split").unique().sort("seq_id")
    )

    with np.load(BIOCLIP_CACHE) as data:
        bioclip = {k: data[k] for k in data.files}
    speciesnet = speciesnet_arm.read_predictions(
        speciesnet_arm.CACHE_DIR / "predictions.json", fetch.IMAGE_DIR, orinoquia.TAXA
    )
    missing = set(frames["path"].to_list()) - set(speciesnet)
    if missing:
        print(f"SpeciesNet has no classification for {len(missing)} frames; dropped")
        frames = frames.filter(~pl.col("path").is_in(list(missing)))
        sequences = sequences.filter(pl.col("seq_id").is_in(frames["seq_id"].unique()))
    detector = {
        "precomputed": manifest.load_detections(),
        "fresh": speciesnet_arm.read_detections(
            speciesnet_arm.DETECTOR_CHECK_DIR / "predictions.json", fetch.IMAGE_DIR
        ),
    }

    summary, per_site = run(
        frames, sequences, bioclip, speciesnet, detector, full["site"].n_unique()
    )
    summary["frames_without_speciesnet"] = len(missing)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    per_site.write_csv(args.out_dir / "per_site.csv", float_precision=4)
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
