"""Report the CNN control on the fold sites, against what it could have replaced.

The control is reported only. The hold-out is spent and is never read here,
and the reference verdicts below use the frozen gate's settings on fold sites
to show what the gate would have said; nothing reads them to move an alias.

The comparison leans slightly towards the DINOv2 probe, which was selected on
these same sites, so its fold score is a little optimistic. One training seed
also leaves the CNN's training noise unmeasured.

Reads ``results/control/frame_logits.npz`` and the committed challenger
predictions, so it runs without torch or a 16-hour fit. Writes
``results/control/``: ``sequences_cnn_control.csv``, ``per_site.csv``,
``per_class.csv``, ``summary.json``, and ``figures/control_gain.png``.

::

    PYTHONPATH=src python -m cameratrap.report.control
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import scoring
from cameratrap.data import crops
from cameratrap.gate import freeze, rule
from cameratrap.report import baselines, selection

ARM = "cnn_control"
ROLE = "control: reported only, cannot move an alias"
# The current champion, and the one it replaced.
OPPONENTS = {"dinov2_probe": "DINOv2 probe", "bioclip2": "BioCLIP 2 zero-shot"}
LOGITS_NAME = "frame_logits.npz"
OUT_DIR = Path("results/control")
CHALLENGERS = Path("results/challengers")
FREEZE_PATH = Path("config/holdout_freeze.json")
FIGURE_DIR = Path("figures")
# The colour each arm has in the challenger figures.
COLOURS = {"bioclip2": "#2a78d6", "dinov2_probe": "#1baf7a"}
MARKERS = {"bioclip2": "o", "dinov2_probe": "s"}


def load_logits(path, frames):
    """Read the control's frame logits and check they match the frames.

    Args:
        path: The logits file.
        frames: Fold-site frames sorted by path.

    Returns:
        A tuple ``(logits, classes, meta)``.

    Raises:
        ValueError: If the logits are not for exactly these frames, in order.
    """
    with np.load(path) as stored:
        paths = list(map(str, stored["paths"]))
        if paths != frames["path"].to_list():
            raise ValueError("the control's logits are not for the fold-site frames")
        classes = list(map(str, stored["classes"]))
        return stored["logits"], classes, json.loads(str(stored["meta"]))


def reference_verdict(sequences, control, opponent, gate, min_sequences):
    """Return what the frozen gate's rule says of the control, for reference.

    Args:
        sequences: Mapping of arm to per-sequence predictions.
        control: The control's arm name.
        opponent: The arm it is compared with.
        gate: The frozen ``GateConfig``.
        min_sequences: Sites with fewer sequences are left out, as in the
            challenger selection.

    Returns:
        A pair ``(verdict, decision)``: a JSON-serialisable dict and the
        ``GateDecision``.
    """
    groups = selection.site_groups(sequences, opponent, control, min_sequences)
    decision = rule.evaluate(groups, scoring.balanced_accuracy, gate)
    verdict = {
        "would_pass": bool(decision.promote),
        "sites": len(groups),
        "mean_gain": float(decision.mean_difference),
        "interval": [float(v) for v in decision.interval],
        "sites_ahead": int(sum(d > 0 for d in decision.differences.values())),
        "worst_site_gain": float(min(decision.differences.values())),
        "per_site_cutoff": float(decision.cutoff),
        "regressions": list(decision.regressions),
        "reasons": list(decision.reasons),
    }
    return verdict, decision


def run(index, sizes, logits, classes, opponents, gate):
    """Score the control and compare it with the other arms.

    Args:
        index: The crop index.
        sizes: The crop-size table.
        logits: Frame logits in the frames' row order.
        classes: Class names in logit order.
        opponents: Mapping of arm name to its per-sequence predictions.
        gate: The frozen ``GateConfig``.

    Returns:
        A tuple ``(summary, per_site, per_class, scored, gains)``; ``gains``
        has one row per gate-sized site and a gain column per opponent.

    Raises:
        ValueError: If an opponent was scored on other sequences.
    """
    frames = baselines.fold_frames(index, sizes, baselines.SETTINGS["size_bins"])
    table = baselines.sequence_table(frames)
    scored = baselines.score(frames, table, logits, classes).select(
        "seq_id", "site", "label", "pred", "msp"
    )
    for arm, predictions in opponents.items():
        if set(predictions["seq_id"].to_list()) != set(scored["seq_id"].to_list()):
            raise ValueError(f"{arm} was scored on other sequences")

    tables = opponents | {ARM: scored}
    settings = selection.SETTINGS
    verdicts, gains = {}, None
    for arm in opponents:
        verdicts[arm], decision = reference_verdict(
            tables, ARM, arm, gate, settings["holdout_like_min_sequences"]
        )
        column = pl.DataFrame(
            {
                "site": list(decision.differences),
                f"gain_over_{arm}": list(decision.differences.values()),
            }
        )
        gains = column if gains is None else gains.join(column, on="site")

    minimum = baselines.SETTINGS["min_site_sequences"]
    per_site = None
    for arm, predictions in tables.items():
        sites = scoring.per_site(predictions, minimum).select(
            "site", "n", pl.col("balanced_accuracy").alias(f"{arm}_balanced_accuracy")
        )
        per_site = sites if per_site is None else per_site.join(sites, on=["site", "n"])

    recall = None
    for arm, predictions in tables.items():
        rows = baselines.per_class(predictions, classes).rename(
            {"recall": f"{arm}_recall"}
        )
        recall = rows if recall is None else recall.join(rows.drop("n"), on="label")

    summary = {
        "role": ROLE,
        "scope": "fold sites only, out of fold; hold-out neither fitted nor scored",
        "frames": frames.height,
        "sequences": scored.height,
        "sites_scored": per_site.height,
        "mean_site_balanced_accuracy": {
            arm: float(per_site[f"{arm}_balanced_accuracy"].mean()) for arm in tables
        },
        "reference_verdicts": verdicts,
    }
    return summary, per_site, recall, scored, gains


def plot_gain(gains, path):
    """Plot the control's gain over each opponent at each gate-sized site.

    Args:
        gains: The gains table from ``run``.
        path: Where to save the PNG.
    """
    ordered = gains.sort("gain_over_dinov2_probe")
    y = np.arange(ordered.height)
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * ordered.height + 1.6))
    ax.axvline(0, color="#52514e", linewidth=1)
    for arm, name in OPPONENTS.items():
        ax.scatter(
            ordered[f"gain_over_{arm}"] * 100,
            y,
            s=28,
            color=COLOURS[arm],
            marker=MARKERS[arm],
            edgecolor="white",
            linewidth=1,
            label=f"Against {name}",
            zorder=3,
        )
    ax.set_yticks(y, ordered["site"].to_list(), fontsize=7)
    ax.set_xlabel(
        "CNN control minus the other arm, sequence-level balanced accuracy (points)",
        fontsize=9,
    )
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("CNN control at each fold site", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write the control's tables and figure from its committed logits.

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
    index = crops.load_index(args.root / crops.INDEX_PATH)
    sizes = crops.load_sizes(args.root / crops.SIZES_PATH)
    frames = baselines.fold_frames(index, sizes, baselines.SETTINGS["size_bins"])
    logits, classes, meta = load_logits(args.root / OUT_DIR / LOGITS_NAME, frames)
    opponents = {
        arm: pl.read_csv(args.root / CHALLENGERS / f"sequences_{arm}.csv")
        for arm in OPPONENTS
    }
    summary, per_site, recall, scored, gains = run(
        index, sizes, logits, classes, opponents, gate
    )
    summary["made_with"] = meta

    out = args.root / args.out_dir
    figures = args.root / args.figure_dir
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    scored.write_csv(out / f"sequences_{ARM}.csv")
    per_site.write_csv(out / "per_site.csv")
    recall.write_csv(out / "per_class.csv")
    plot_gain(gains, figures / "control_gain.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["reference_verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
