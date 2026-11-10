"""Drift study: do label-free signals say how accurate the champion is at a site?

Replays every classified sequence through the champion (the DINOv2 probe)
in site and date order, computes six label-free signals per site, and
scores each against the labels held back from it. Fold sites are scored
by probes refitted out of fold; hold-out sites by the frozen head in
``results/holdout/probe_head.npz``. Both are checked against the committed
predictions before anything is reported, so the study describes the same
model the gate judged.

The rules (signals, the primary one, minimum sizes, how the hold-out is
used) were fixed in the plan on 2026-11-10 before any signal was computed.
The primary signal is ATC; the hold-out sites test its predicted accuracy
once. A null result is written up as it falls.

Writes ``results/drift/``: ``sequences.csv`` (the replayed stream with
per-sequence signals), ``per_site.csv``, ``signals.csv``, ``halves.csv``,
``half_signals.csv``, ``summary.json``, and two figures under ``figures/``.

::

    PYTHONPATH=src python -m cameratrap.report.drift
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import probe, scoring
from cameratrap.data import crops, manifest, orinoquia
from cameratrap.drift import signals, study
from cameratrap.embed import store
from cameratrap.replay import stream

CHAMPION = "dinov2_probe"
BACKBONE = "dinov2"
MANIFEST_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}.csv"
HEAD_PATH = Path("results/holdout/probe_head.npz")
COMMITTED = {
    "fold": Path("results/challengers") / f"sequences_{CHAMPION}.csv",
    "holdout": Path("results/holdout") / f"sequences_{CHAMPION}.csv",
}
OUT_DIR = Path("results/drift")
FIGURE_DIR = Path("figures")
# The trust report's pair: grey for fold sites, green for the hold-out.
COLOURS = {"fold": "#8a8984", "holdout": "#1baf7a"}
MARKERS = {"fold": "o", "holdout": "s"}
NAMES = {"fold": "Fold sites (out of fold)", "holdout": "Hold-out sites"}
SIGNAL_LABELS = {
    "atc": "ATC (primary)",
    "msp": "Mean confidence",
    "entropy": "Mean entropy",
    "knn_distance": "kNN distance",
    "mmd": "MMD",
    "class_mix_shift": "Class-mix shift",
}
# How far a refitted confidence may sit from the committed one.
MSP_TOLERANCE = 1e-6


def champion_logits(frames, features, head):
    """Return the champion's frame logits: out of fold, or the frozen head.

    Args:
        frames: Every indexed frame with ``site``, ``split`` and ``label``,
            in ``features`` row order.
        features: The frames' DINOv2 embeddings.
        head: The frozen probe from ``probe_head.npz`` (``coef``,
            ``intercept``, ``classes``).

    Returns:
        Array of shape ``(n_frames, n_classes)`` in the head's class order.
    """
    classes = head["classes"].tolist()
    held = (frames["split"] == study.HOLDOUT).to_numpy()
    logits = np.empty((frames.height, len(classes)))
    fold = frames.filter(~pl.Series(held))
    logits[~held], _ = probe.out_of_fold_logits(
        features[~held],
        fold["label"].to_numpy(),
        fold["site"].to_numpy(),
        fold["split"].to_numpy(),
        classes,
    )
    logits[held] = features[held] @ head["coef"].T + head["intercept"]
    return logits


def replay(index, image_manifest, features, logits, classes):
    """Turn frame logits and embeddings into the per-sequence replay table.

    Args:
        index: The crop index sorted by path, in ``features`` row order.
        image_manifest: The image manifest, for timestamps.
        features: The frames' embeddings.
        logits: The champion's frame logits.
        classes: Class names in logit order.

    Returns:
        A pair ``(table, embeddings)``: one row per sequence in site and
        date order, with ``pred``, ``msp`` and ``entropy`` beside the
        stream's columns, and the unit-norm sequence embeddings (mean of
        a sequence's frames) in the same row order.
    """
    keys = index["seq_id"].to_list()
    # Logits are averaged across a sequence before the softmax, as every
    # arm is scored.
    seq_ids, seq_logits = scoring.group_mean(keys, logits)
    _, seq_vectors = scoring.group_mean(keys, features.astype(np.float64))
    probs = scoring.softmax(seq_logits)
    scored = pl.DataFrame(
        {
            "seq_id": seq_ids,
            "row": np.arange(len(seq_ids)),
            "pred": [classes[i] for i in probs.argmax(axis=1)],
            "msp": probs.max(axis=1),
            "entropy": signals.entropy(probs),
        }
    )
    # The join must keep the stream's site and date order.
    table = stream.sequence_stream(index, image_manifest).join(
        scored, on="seq_id", maintain_order="left"
    )
    embeddings = signals.normalise(seq_vectors)[table["row"].to_numpy()]
    return table.drop("row"), embeddings


def check_committed(table, committed):
    """Check the replayed predictions are the committed ones.

    Args:
        table: The replay table.
        committed: The committed per-sequence predictions for the same
            sequences, with ``seq_id``, ``pred`` and ``msp``.

    Raises:
        ValueError: If the sequences differ, any prediction differs, or a
            confidence is further off than ``MSP_TOLERANCE``.
    """
    joined = table.join(committed, on="seq_id", how="inner", suffix="_committed")
    if not joined.height == table.height == committed.height:
        raise ValueError("the replay and the committed predictions cover other rows")
    changed = int((joined["pred"] != joined["pred_committed"]).sum())
    gap = float((joined["msp"] - joined["msp_committed"]).abs().max())
    if changed or gap > MSP_TOLERANCE:
        raise ValueError(
            f"the replay differs from the committed predictions: {changed} "
            f"predictions changed, largest confidence gap {gap:.2e}"
        )


def run(table, embeddings, classes, settings=None):
    """Compute every signal and score it against the labels.

    Args:
        table: The replay table.
        embeddings: Unit-norm sequence embeddings in ``table`` row order.
        classes: Class names.
        settings: Defaults to ``study.SETTINGS``.

    Returns:
        A tuple ``(summary, table, sites, scores, changes, change_scores)``:
        a JSON-ready dict, the replay table with ``knn_distance`` added,
        the per-site table, each predictor's scores, each site's
        early-to-late changes and each signal's change scores.
    """
    settings = settings or study.SETTINGS
    ctxs = study.contexts(table, embeddings, classes, settings)
    table = table.with_columns(
        pl.Series(
            "knn_distance",
            study.sequence_knn_distance(table, embeddings, ctxs, settings["knn_k"]),
        )
    )
    windows = study.window_table(table, embeddings, ctxs, classes)
    sites = study.site_predictions(windows, settings["min_site_sequences"])
    scores = study.signal_scores(sites, settings)
    changes = study.half_changes(windows, settings["min_half_sequences"])
    change_scores = study.change_scores(changes, settings)

    def primary(source):
        """Return the primary predictor's score row for one source."""
        rows = scores.filter(
            (pl.col("source") == source) & (pl.col("predictor") == "atc_direct")
        )
        return rows.drop("source", "predictor").row(0, named=True)

    sources = sites["source"].unique().sort().to_list()
    summary = {
        "champion": CHAMPION,
        "settings": settings,
        "primary_signal": study.PRIMARY,
        "sequences": table.height,
        "sites_scored": {s: int((sites["source"] == s).sum()) for s in sources},
        "atc_thresholds": {k: v["threshold"] for k, v in sorted(ctxs.items())},
        "primary": {s: primary(s) for s in sources},
        "time_test": {
            s: {"sites": int((changes["source"] == s).sum())}
            | study.noise_check(changes.filter(pl.col("source") == s))
            for s in changes["source"].unique().sort().to_list()
        },
    }
    return summary, table, sites, scores, changes, change_scores


def plot_predicted(sites, summary, path):
    """Plot ATC's predicted accuracy against observed accuracy, per site.

    Args:
        sites: The per-site table from ``run``.
        summary: The summary from ``run``.
        path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    low = 100 * min(sites["accuracy"].min(), sites["predicted_accuracy"].min()) - 2
    ax.plot([low, 100], [low, 100], color="#52514e", linewidth=1, linestyle="--")
    ax.text(
        low + 1, low + 2, "predicted = observed", fontsize=8, rotation=45, va="bottom"
    )
    for source in ("fold", "holdout"):
        rows = sites.filter(pl.col("source") == source)
        if not rows.height:
            continue
        ax.scatter(
            rows["predicted_accuracy"] * 100,
            rows["accuracy"] * 100,
            s=44,
            color=COLOURS[source],
            marker=MARKERS[source],
            edgecolor="white",
            linewidth=1,
            zorder=3 if source == "holdout" else 2,
            label=f"{NAMES[source]} ({rows.height})",
        )
    ax.set_xlim(low, 101)
    ax.set_ylim(low, 101)
    ax.set_aspect("equal")
    ax.set_xlabel("Accuracy predicted without labels (ATC, %)", fontsize=9)
    ax.set_ylabel("Observed sequence accuracy (%)", fontsize=9)
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title = "Predicting a site's accuracy without labels"
    held = summary["primary"].get("holdout")
    if held:
        title += (
            f"\nHold-out sites: off by {held['mae'] * 100:.1f} points on average;"
            f"\nguessing the mean is off by {held['baseline_mae'] * 100:.1f}"
        )
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_signals(scores, path):
    """Plot each signal's rank correlation with per-site accuracy.

    Correlations are flipped where a higher signal should mean lower
    accuracy, so "to the right" always reads as "works as hoped".

    Args:
        scores: The predictor scores from ``run``.
        path: Where to save the PNG.
    """
    names = list(study.SIGNALS)
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axvline(0, color="#52514e", linewidth=1)
    for source, offset in (("fold", 0.14), ("holdout", -0.14)):
        rows = scores.filter(
            (pl.col("source") == source) & pl.col("predictor").str.starts_with("fit_")
        )
        if not rows.height:
            continue
        by_signal = {r["signal"]: r for r in rows.iter_rows(named=True)}
        picked = [by_signal[n] for n in names]
        sign = np.array([r["expected_sign"] for r in picked])
        rho = np.array([r["rho"] for r in picked]) * sign
        ends = np.array([[r["low"], r["high"]] for r in picked]) * sign[:, None]
        ax.hlines(
            y + offset,
            ends.min(axis=1),
            ends.max(axis=1),
            color=COLOURS[source],
            linewidth=2,
        )
        ax.scatter(
            rho,
            y + offset,
            s=44,
            color=COLOURS[source],
            marker=MARKERS[source],
            edgecolor="white",
            linewidth=1,
            zorder=3,
            label=f"{NAMES[source]} ({picked[0]['n']})",
        )
    ax.set_yticks(y, [SIGNAL_LABELS[n] for n in names], fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(
        "Spearman correlation with site accuracy, in the hoped-for direction "
        "(95% interval over sites)",
        fontsize=8,
    )
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Above the plot: every corner inside it holds an interval.
    ax.legend(
        frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1), ncols=2
    )
    ax.set_title("Which label-free signals rank sites by accuracy", loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Replay the champion and write the drift study's tables and figures.

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
    args = parser.parse_args(argv)

    index = crops.load_index(args.root / crops.INDEX_PATH).sort("path")
    cache = store.read(store.cache_path(BACKBONE, args.cache_root))
    # A cache embedded from another index would score the wrong crops.
    store.check_meta(
        cache["meta"],
        {"index_sha256": store.file_sha256(args.root / crops.INDEX_PATH)},
    )
    with np.load(args.root / HEAD_PATH) as data:
        head = {k: data[k] for k in data.files}
    classes = head["classes"].tolist()
    features = store.rows_for(cache, index["path"].to_list())
    table, embeddings = replay(
        index,
        manifest.load(args.root / MANIFEST_PATH),
        features,
        champion_logits(index, features, head),
        classes,
    )
    check_committed(
        table, pl.concat(pl.read_csv(args.root / p) for p in COMMITTED.values())
    )
    summary, table, sites, scores, changes, change_scores = run(
        table, embeddings, classes
    )
    summary["cache"] = cache["meta"]

    out = args.root / args.out_dir
    figures = args.root / args.figure_dir
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "sequences.csv")
    sites.write_csv(out / "per_site.csv")
    scores.write_csv(out / "signals.csv")
    changes.write_csv(out / "halves.csv")
    change_scores.write_csv(out / "half_signals.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_predicted(sites, summary, figures / "drift_predicted_accuracy.png")
    plot_signals(scores, figures / "drift_signals.png")
    print(json.dumps({k: summary[k] for k in ("primary", "time_test")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
