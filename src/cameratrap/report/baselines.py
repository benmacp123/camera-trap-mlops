"""Zero-shot baselines per site from the caches alone: BioCLIP 2 and SpeciesNet.

Scores every fold site's sequences from the Phase 1 crop index (up to three
frames per sequence), with frame logits averaged across a sequence before
the softmax. Hold-out sites are never read: they stay sealed until the
evaluation config is frozen and hashed (decided 2026-10-31).

Inputs are the committed crop index and crop sizes, the BioCLIP 2
embedding cache and SpeciesNet's predictions over the crop index; no model
is loaded. SpeciesNet has probably seen Orinoquía, so its numbers are a
reference, never held-out performance.

Writes ``results/baselines/``: ``per_site.csv``, ``per_class.csv``,
``confusion_<arm>.csv``, ``by_crop_size.csv``, ``summary.json`` and two
figures under ``figures/``.

::

    PYTHONPATH=src python -m cameratrap.report.baselines
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import scoring, speciesnet_arm
from cameratrap.data import crops, fetch, orinoquia
from cameratrap.embed import store
from cameratrap.report.spike import majority_label

SETTINGS = {
    # The spike's threshold, so the tables compare.
    "min_site_sequences": 20,
    "weak_site_threshold": 0.85,
    # Bins of the stored crop's side, in pixels; both backbones take 224.
    "size_bins": [112, 224],
}
ARMS = ("bioclip2", "speciesnet")
LABELS = {"bioclip2": "BioCLIP 2 zero-shot", "speciesnet": "SpeciesNet (restricted)"}
OUT_DIR = Path("results/baselines")
FIGURE_DIR = Path("figures")
# Categorical slots 1 and 2 of the reference palette, validated as a pair.
COLOURS = {"bioclip2": "#2a78d6", "speciesnet": "#eb6834"}
MARKERS = {"bioclip2": "o", "speciesnet": "D"}


def size_bin(side, edges):
    """Return a crop-size bin label for a side length.

    Args:
        side: Crop side in pixels.
        edges: Ascending bin edges, such as ``[112, 224]``.

    Returns:
        A label such as ``"<112"``, ``"112-224"`` or ``">=224"``.
    """
    if side < edges[0]:
        return f"<{edges[0]}"
    for low, high in zip(edges, edges[1:]):
        if side < high:
            return f"{low}-{high}"
    return f">={edges[-1]}"


def fold_frames(index, sizes, edges):
    """Return the fold-site frames with their crop size, hold-out removed.

    Args:
        index: The crop index.
        sizes: The crop-size table.
        edges: Crop-size bin edges.

    Returns:
        A DataFrame of frames sorted by path, with ``side`` and ``size_bin``.
    """
    frames = (
        index.filter(pl.col("split") != "holdout")
        .join(sizes, on="path", how="left")
        .sort("path")
    )
    if frames["side"].null_count():
        raise ValueError("crop-size table is missing indexed crops")
    return frames.with_columns(
        pl.col("side")
        .map_elements(lambda s: size_bin(s, edges), return_dtype=pl.String)
        .alias("size_bin")
    )


def sequence_table(frames):
    """Return one row per sequence with its site and label.

    Args:
        frames: Frames from ``fold_frames``.

    Returns:
        A DataFrame with ``seq_id``, ``site`` and ``label``.

    Raises:
        ValueError: If a sequence's frames disagree on site or label.
    """
    seqs = frames.group_by("seq_id").agg(
        pl.col("site").unique(), pl.col("label").unique()
    )
    if (seqs["site"].list.len() > 1).any() or (seqs["label"].list.len() > 1).any():
        raise ValueError("a sequence's frames disagree on site or label")
    return seqs.with_columns(
        pl.col("site").list.first(), pl.col("label").list.first()
    ).sort("seq_id")


def zero_shot_logits(cache, paths):
    """Return BioCLIP 2 zero-shot logits for some crops.

    Args:
        cache: The BioCLIP 2 cache from ``store.read``.
        paths: Crop paths.

    Returns:
        Array of shape ``(len(paths), n_classes)``, classes in cache order.
    """
    image = store.rows_for(cache, paths)
    return float(cache["logit_scale"]) * image @ cache["text"].T


def speciesnet_frames(predictions, paths):
    """Return SpeciesNet's restricted logits and open-set answers for some frames.

    Args:
        predictions: Output of ``speciesnet_arm.read_predictions``.
        paths: Frame paths.

    Returns:
        A pair ``(logits, open_labels)``.

    Raises:
        ValueError: If SpeciesNet has no answer for a frame; scoring with
            frames missing would quietly change what each arm is scored on.
    """
    missing = [p for p in paths if p not in predictions]
    if missing:
        raise ValueError(f"SpeciesNet has no answer for {len(missing)} frames")
    logits = np.stack([predictions[p]["logits"] for p in paths])
    return logits, [predictions[p]["open_label"] for p in paths]


def score(frames, sequences, logits, classes):
    """Score frame logits at sequence level.

    Args:
        frames: Frames in logit row order, with ``seq_id``.
        sequences: Output of ``sequence_table``.
        logits: Array of shape ``(n_frames, len(classes))``.
        classes: Class names in logit order.

    Returns:
        Per-sequence DataFrame with ``site``, ``label``, ``pred`` and ``msp``.
    """
    preds = scoring.sequence_predictions(frames["seq_id"].to_list(), logits, classes)
    return sequences.join(preds, on="seq_id")


def per_class(scored, classes):
    """Return sequence-level recall per class, pooled over fold sites.

    Args:
        scored: Per-sequence predictions from ``score``.
        classes: Class names.

    Returns:
        A DataFrame with ``label``, ``n`` and ``recall``.
    """
    rows = []
    for name in classes:
        group = scored.filter(pl.col("label") == name)
        recall = float((group["pred"] == name).mean()) if group.height else np.nan
        rows.append({"label": name, "n": group.height, "recall": recall})
    return pl.DataFrame(rows)


def confusion(scored, classes):
    """Return the sequence-level confusion matrix.

    Args:
        scored: Per-sequence predictions from ``score``.
        classes: Class names, the row and column order.

    Returns:
        A DataFrame with a ``label`` column (true class) and one count
        column per predicted class.
    """
    counts = collections.Counter(zip(scored["label"], scored["pred"]))
    return pl.DataFrame(
        {"label": classes} | {p: [counts[(t, p)] for t in classes] for p in classes}
    )


def by_crop_size(frames, logits, classes):
    """Return frame-level accuracy by crop-size bin.

    Size belongs to a frame, not a sequence, so this table is the one place
    frames are scored on their own; it is not comparable with the
    sequence-level tables.

    Args:
        frames: Frames in logit row order, with ``label`` and ``size_bin``.
        logits: Array of shape ``(n_frames, len(classes))``.
        classes: Class names in logit order.

    Returns:
        A DataFrame with ``size_bin``, ``n``, ``accuracy`` and
        ``balanced_accuracy``.
    """
    preds = np.array(classes)[logits.argmax(axis=1)]
    scored = frames.select("label", "size_bin").with_columns(pred=pl.Series(preds))
    rows = []
    for (bin_label,), group in scored.group_by("size_bin"):
        labels, guessed = group["label"].to_numpy(), group["pred"].to_numpy()
        rows.append(
            {
                "size_bin": bin_label,
                "n": group.height,
                "accuracy": float(np.mean(labels == guessed)),
                "balanced_accuracy": scoring.balanced_accuracy(labels, guessed),
            }
        )
    return pl.DataFrame(rows).sort("size_bin")


def open_set_sites(frames, sequences, open_labels):
    """Return SpeciesNet's open-set coverage and accuracy when it answers, per site.

    Args:
        frames: Frames in ``open_labels`` order, with ``seq_id``.
        sequences: Output of ``sequence_table``.
        open_labels: SpeciesNet's open-set answer per frame (``None`` abstains).

    Returns:
        A pair ``(per_site, overall)``: a DataFrame with ``site``,
        ``speciesnet_open_coverage`` and ``speciesnet_open_accuracy``, and a
        dict with the same two measures over every sequence.
    """
    by_seq = collections.defaultdict(list)
    for seq_id, label in zip(frames["seq_id"], open_labels):
        by_seq[seq_id].append(label)
    scored = sequences.with_columns(
        pl.Series("pred", [majority_label(by_seq[s]) for s in sequences["seq_id"]])
    )
    covered = scored.filter(pl.col("pred").is_not_null())
    sites = scored.group_by("site").agg(
        pl.col("pred").is_not_null().mean().alias("speciesnet_open_coverage"),
        (pl.col("pred") == pl.col("label"))
        .filter(pl.col("pred").is_not_null())
        .mean()
        .alias("speciesnet_open_accuracy"),
    )
    overall = {
        "coverage": covered.height / scored.height,
        "accuracy_when_answering": float((covered["pred"] == covered["label"]).mean()),
    }
    return sites, overall


def run(index, sizes, bioclip, speciesnet):
    """Compute every baseline table.

    Args:
        index: The crop index.
        sizes: The crop-size table.
        bioclip: The BioCLIP 2 cache from ``store.read``.
        speciesnet: Output of ``speciesnet_arm.read_predictions``.

    Returns:
        A pair ``(summary, tables)``: a JSON-serialisable dict and a dict of
        DataFrames keyed by output file stem.
    """
    classes = bioclip["classes"].tolist()
    frames = fold_frames(index, sizes, SETTINGS["size_bins"])
    sequences = sequence_table(frames)
    paths = frames["path"].to_list()
    sn_logits, open_labels = speciesnet_frames(speciesnet, paths)
    logits = {"bioclip2": zero_shot_logits(bioclip, paths), "speciesnet": sn_logits}

    tables, per_site, summary_arms = {}, None, {}
    size_tables = []
    for arm in ARMS:
        scored = score(frames, sequences, logits[arm], classes)
        sites = scoring.per_site(scored, SETTINGS["min_site_sequences"])
        ba = sites["balanced_accuracy"].to_numpy()
        summary_arms[arm] = {
            "mean_site_balanced_accuracy": float(ba.mean()),
            "median_site_balanced_accuracy": float(np.median(ba)),
            "min_site_balanced_accuracy": float(ba.min()),
            "weak_sites": int((ba < SETTINGS["weak_site_threshold"]).sum()),
        }
        renamed = sites.rename(
            {c: f"{arm}_{c}" for c in ("balanced_accuracy", "accuracy")}
        )
        per_site = (
            renamed if per_site is None else per_site.join(renamed, on=["site", "n"])
        )
        recall = per_class(scored, classes).rename({"recall": f"{arm}_recall"})
        tables["per_class"] = (
            recall
            if arm == ARMS[0]
            else tables["per_class"].join(recall.drop("n"), on="label")
        )
        tables[f"confusion_{arm}"] = confusion(scored, classes)
        size_tables.append(
            by_crop_size(frames, logits[arm], classes).with_columns(arm=pl.lit(arm))
        )

    open_sites, open_overall = open_set_sites(frames, sequences, open_labels)
    tables["per_site"] = per_site.join(open_sites, on="site", how="left")
    tables["by_crop_size"] = pl.concat(size_tables).select(
        "arm", "size_bin", "n", "accuracy", "balanced_accuracy"
    )
    summary = {
        "settings": SETTINGS,
        "scope": "fold sites only; hold-out sealed until the Phase 4 freeze",
        "frames": frames.height,
        "sequences": sequences.height,
        "sites_scored": per_site.height,
        "arms": summary_arms,
        "speciesnet_open_set": open_overall,
        "bioclip2_cache": bioclip["meta"],
        "note": "SpeciesNet has probably seen Orinoquía; its scores are a reference only",
    }
    return summary, tables


def plot_sites(per_site, path):
    """Plot per-site balanced accuracy for both arms, sites sorted by BioCLIP 2.

    Args:
        per_site: The ``per_site`` table from ``run``.
        path: Where to save the PNG.
    """
    ordered = per_site.sort("bioclip2_balanced_accuracy")
    y = np.arange(ordered.height)
    fig, ax = plt.subplots(figsize=(6.5, 0.2 * ordered.height + 1.4))
    for arm in ARMS:
        ax.scatter(
            ordered[f"{arm}_balanced_accuracy"],
            y,
            s=28,
            color=COLOURS[arm],
            marker=MARKERS[arm],
            edgecolor="white",
            linewidth=1,
            label=LABELS[arm],
            zorder=3,
        )
    ax.axvline(
        SETTINGS["weak_site_threshold"], color="#52514e", linestyle=":", linewidth=1
    )
    ax.set_yticks(y, ordered["site"].to_list(), fontsize=7)
    ax.set_xlabel("Sequence-level balanced accuracy (fold sites)")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Zero-shot accuracy per site", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(table, path, title):
    """Plot a row-normalised sequence confusion matrix.

    Args:
        table: A ``confusion_<arm>`` table from ``run``.
        path: Where to save the PNG.
        title: Figure title.
    """
    classes = table["label"].to_list()
    counts = table.drop("label").to_numpy().astype(float)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    names = [orinoquia.TAXA[c][1] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(shares.shape):
        if counts[i, j]:
            ax.text(
                j,
                i,
                f"{shares[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if shares[i, j] > 0.5 else "#0b0b0b",
            )
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (share of sequences per row)")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the zero-shot baseline tables and figures from the caches.

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
    summary, tables = run(
        crops.load_index(args.index), crops.load_sizes(args.sizes), bioclip, speciesnet
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for stem, table in tables.items():
        table.write_csv(args.out_dir / f"{stem}.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_sites(tables["per_site"], args.figure_dir / "baselines_per_site.png")
    plot_confusion(
        tables["confusion_bioclip2"],
        args.figure_dir / "baselines_confusion_bioclip2.png",
        "BioCLIP 2 zero-shot: sequence confusion (fold sites)",
    )
    print(json.dumps(summary["arms"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
