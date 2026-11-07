"""Score the frozen challenger on the hold-out sites and run the gate, once.

Everything comes from the freeze (``config/holdout_freeze.json``), read only
after its hash, its input files, its recipe and the installed libraries are
checked: a decision made under anything else would not be the one
pre-registered. The challenger is fitted on every fold-site frame, then the
champion and challenger score the hold-out at sequence level and
``gate.rule.evaluate`` decides.

The result is reported pass or fail, and no other challenger follows a fail.
The command refuses to overwrite a result; anyone reproducing it passes a
different ``--out-dir`` and compares. ``--check`` runs every check and stops
before any model is fitted or any hold-out crop is scored.

Writes ``results/holdout/``: ``per_site.csv``, ``sequences_<arm>.csv``,
``probe_head.npz``, ``summary.json``, and ``figures/holdout_gate.png``.

::

    PYTHONPATH=src python -m cameratrap.report.holdout --check
    PYTHONPATH=src python -m cameratrap.report.holdout
"""

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import probe, scoring
from cameratrap.data import crops
from cameratrap.embed import store
from cameratrap.gate import freeze, rule
from cameratrap.report import baselines, challengers, holdout_freeze, selection

# Recorded at the freeze; the same value is pinned in the freeze's tests.
FROZEN_SHA256 = "774fff882764979a051792f1a0c1fe4e595f043141095b32f6cd3d28fe203933"
HOLDOUT = "holdout"
METRICS = {"balanced_accuracy": scoring.balanced_accuracy}
OUT_DIR = Path("results/holdout")
FIGURE_DIR = Path("figures")


def load_freeze(root):
    """Read the freeze and check everything it rests on.

    Args:
        root: The repository root, a ``pathlib.Path``.

    Returns:
        A pair ``(config, gate_config)``.

    Raises:
        ValueError: If the freeze, its input files, the code's probe recipe,
            the metric or the installed libraries differ from what was
            frozen.
    """
    config, gate = freeze.load(
        root / holdout_freeze.OUT_PATH, expected_sha256=FROZEN_SHA256
    )
    freeze.check_files(config["inputs"], root)
    challenger = config["challenger"]
    recipe = holdout_freeze.probe_recipe(challenger["backbone"])
    if {k: challenger.get(k) for k in recipe} != recipe:
        raise ValueError("the probe code no longer matches the frozen recipe")
    if config["metric"]["name"] not in METRICS:
        raise ValueError(f"unknown frozen metric {config['metric']['name']}")
    wrong = {
        name: version
        for name, version in config["software"].items()
        if metadata.version(name) != version
    }
    if wrong:
        raise ValueError(f"frozen with other library versions: {wrong}")
    return config, gate


def split_frames(index, sites):
    """Return the training and hold-out frames, checked against the freeze.

    Args:
        index: The crop index.
        sites: The frozen hold-out site names.

    Returns:
        A pair ``(train, test)`` of DataFrames sorted by path.

    Raises:
        ValueError: If the index's hold-out is not the frozen one, or a
            frozen site would be trained on.
    """
    train = index.filter(pl.col("split") != HOLDOUT).sort("path")
    test = index.filter(pl.col("split") == HOLDOUT).sort("path")
    if set(test["site"].to_list()) != set(sites):
        raise ValueError("the index's hold-out sites are not the frozen ones")
    if set(train["site"].to_list()) & set(sites):
        raise ValueError("a frozen hold-out site is among the training frames")
    return train, test


def check_caches(caches, train, test, classes, config):
    """Check the caches are the frozen ones and hold every frame needed.

    Args:
        caches: Mapping of backbone to its cache from ``store.read``.
        train: Training frames.
        test: Hold-out frames.
        classes: Class names in logit order.
        config: The frozen config.

    Raises:
        ValueError: If a cache's metadata differs from the freeze, a frame
            has no embedding, or a class is missing from the training frames
            or the class order.
    """
    arms = {
        config["champion"]["backbone"]: config["champion"],
        config["challenger"]["backbone"]: config["challenger"],
    }
    for backbone, arm in arms.items():
        store.check_meta(caches[backbone]["meta"], arm["embedding"])
    needed = set(test["path"].to_list()) | set(train["path"].to_list())
    for backbone, cache in caches.items():
        if needed - set(cache["paths"]):
            raise ValueError(f"the {backbone} cache lacks frames in the crop index")
    if set(classes) != set(train["label"].to_list()):
        raise ValueError("training frames do not cover exactly the frozen classes")
    if set(test["label"].to_list()) - set(classes):
        raise ValueError("a hold-out label is outside the classes")


def fit_challenger(train, cache, classes, recipe):
    """Fit the challenger on the training frames with the frozen settings.

    Args:
        train: Training frames, in row order.
        cache: The challenger backbone's cache.
        classes: Class names in logit order.
        recipe: The frozen challenger entry.

    Returns:
        The fitted probe.
    """
    return probe.fit(
        store.rows_for(cache, train["path"].to_list()),
        train["label"].to_numpy(),
        train["site"].to_numpy(),
        classes,
        inner_folds=recipe["inner_folds"],
        cs=np.array(recipe["cs"]),
    )


def run(index, caches, config, gate):
    """Fit the challenger, score both arms on the hold-out and decide.

    Args:
        index: The crop index.
        caches: Mapping of backbone to its cache; the champion's carries the
            class order and text embeddings.
        config: The frozen config.
        gate: The frozen ``GateConfig``.

    Returns:
        A tuple ``(summary, per_site, sequences, head)``: a
        JSON-serialisable dict, one row per site, a dict of arm to its
        per-sequence predictions, and the fitted probe's arrays.
    """
    champion, challenger = config["champion"], config["challenger"]
    classes = caches[champion["backbone"]]["classes"].tolist()
    train, test = split_frames(index, config["holdout_sites"])
    check_caches(caches, train, test, classes, config)

    cache = caches[challenger["backbone"]]
    model = fit_challenger(train, cache, classes, challenger)
    paths = test["path"].to_list()
    logits = {
        champion["arm"]: baselines.zero_shot_logits(
            caches[champion["backbone"]], paths
        ),
        challenger["arm"]: model.decision_function(store.rows_for(cache, paths)),
    }
    seq_table = baselines.sequence_table(test)
    sequences = {
        arm: baselines.score(test, seq_table, arm_logits, classes).select(
            "seq_id", "site", "label", "pred", "msp"
        )
        for arm, arm_logits in logits.items()
    }

    groups = selection.site_groups(sequences, champion["arm"], challenger["arm"], 0)
    metric = METRICS[config["metric"]["name"]]
    decision = rule.evaluate(groups, metric, gate)
    per_site = pl.DataFrame(
        [
            {
                "site": site,
                "n": len(labels),
                "champion": metric(labels, a),
                "challenger": metric(labels, b),
                "gain": decision.differences[site],
                "p_value": decision.p_values[site],
                "blocks": site in decision.regressions,
            }
            for site, (labels, a, b) in groups.items()
        ]
    )
    summary = {
        "freeze_sha256": FROZEN_SHA256,
        "champion": champion["arm"],
        "challenger": challenger["arm"],
        "promote": decision.promote,
        "mean_gain": decision.mean_difference,
        "interval": list(decision.interval),
        "per_site_cutoff": decision.cutoff,
        "regressions": decision.regressions,
        "reasons": decision.reasons,
        "sequences": seq_table.height,
        "training_frames": train.height,
        "chosen_c": float(model.C_),
    }
    head = {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "classes": np.array(classes),
    }
    return summary, per_site, sequences, head


def plot_gate(summary, per_site, path):
    """Plot the challenger's gain at each hold-out site.

    Args:
        summary: The summary from ``run``.
        per_site: The per-site table from ``run``.
        path: Where to save the PNG.
    """
    rows = per_site.sort("gain")
    y = np.arange(rows.height)
    colour = challengers.COLOURS[summary["challenger"].split("_")[0]]
    blocks = rows["blocks"].to_numpy()
    gain = rows["gain"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7, 0.32 * rows.height + 1.6))
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.scatter(gain[~blocks], y[~blocks], s=48, color=colour, zorder=3)
    # Shape as well as position, so a block never rests on colour alone.
    ax.scatter(
        gain[blocks],
        y[blocks],
        s=64,
        marker="X",
        facecolors="none",
        edgecolors="#52514e",
        zorder=3,
    )
    for row, g, p in zip(y[blocks], gain[blocks], rows["p_value"].to_numpy()[blocks]):
        ax.annotate(
            f"blocks, p = {p:.4f}", (g, row), xytext=(8, -3), textcoords="offset points"
        )
    ax.set_yticks(y, [f"{s} ({n})" for s, n in zip(rows["site"], rows["n"])])
    ax.set_ylabel("Site (sequences)")
    ax.set_xlabel("Gain in sequence-level balanced accuracy over the champion (points)")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    low, high = (v * 100 for v in summary["interval"])
    verdict = "promoted" if summary["promote"] else "champion kept"
    ax.set_title(
        f"Hold-out gate: {verdict}. Mean gain {summary['mean_gain'] * 100:+.1f} points "
        f"(95% CI {low:+.1f} to {high:+.1f})",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Check the freeze, then score the hold-out and run the gate.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.

    Raises:
        FileExistsError: If the output folder already holds a result.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache-root", type=Path, default=store.CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run every check, then stop before fitting or scoring",
    )
    args = parser.parse_args(argv)

    out = args.root / args.out_dir
    if not args.check and (out / "summary.json").exists():
        raise FileExistsError(
            f"{out} already holds the result; pass another --out-dir to reproduce it"
        )
    config, gate = load_freeze(args.root)
    index = crops.load_index(args.root / holdout_freeze.INDEX)
    backbones = {config[a]["backbone"] for a in ("champion", "challenger")}
    caches = {b: store.read(store.cache_path(b, args.cache_root)) for b in backbones}

    if args.check:
        train, test = split_frames(index, config["holdout_sites"])
        classes = caches[config["champion"]["backbone"]]["classes"].tolist()
        check_caches(caches, train, test, classes, config)
        print(
            f"freeze {FROZEN_SHA256[:12]} ok: {train.height} training frames, "
            f"{test.height} hold-out frames at {test['site'].n_unique()} sites; "
            "nothing fitted or scored"
        )
        return 0

    summary, per_site, sequences, head = run(index, caches, config, gate)
    selected = json.loads((args.root / holdout_freeze.SELECTION).read_text())
    summary["fold_sites_mean_gain"] = next(
        row["mean_gain"]
        for row in selected["challengers"]
        if row["challenger"] == summary["challenger"]
    )

    figures = args.root / args.figure_dir
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    per_site.write_csv(out / "per_site.csv")
    for arm, table in sequences.items():
        table.write_csv(out / f"sequences_{arm}.csv")
    np.savez(out / "probe_head.npz", **head)
    plot_gate(summary, per_site, figures / "holdout_gate.png")
    # Written last: its presence is what marks the hold-out as used.
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
