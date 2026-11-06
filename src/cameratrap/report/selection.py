"""Choose the one challenger that goes to the hold-out, on fold sites only.

Sending every challenger to the hold-out and promoting whichever passes
would be four tries at one exam: the chance that some challenger passes
by luck grows with each try (the multiple comparisons problem). So one
challenger is chosen here, before the freeze, and only it is scored on
the hold-out.

The choice is by simulated gate pass rate. Each challenger first gets
its own per-site tolerance, calibrated against the champion with the
swap null (``gate.calibrate``) so that an equally good challenger is
blocked at most 5% of the time over ten sites. Then many random sets of
ten hold-out-sized fold sites are drawn, and the real gate
(``gate.rule.decide``) is run on each. The pass rate rewards a large
mean gain and few regressions together, at the hold-out's scale, rather
than judging a challenger by its single worst site, which at small sites
is mostly noise. Every challenger sees the same site sets, so the
comparison between them is paired.

Reads the per-sequence predictions written by ``report.challengers``;
the hold-out is never read.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cameratrap.arms import scoring
from cameratrap.gate import calibrate, rule
from cameratrap.report import challengers

SETTINGS = {
    "champion": "bioclip2",
    "challengers": ["bioclip2_probe", "bioclip2_knn", "dinov2_probe", "dinov2_knn"],
    # Smallest hold-out site; only fold sites at least this big are drawn.
    "holdout_like_min_sequences": 49,
    "holdout_sites": 10,
    "tolerances": [round(t, 2) for t in np.arange(0.02, 0.205, 0.01)],
    "draws": 2000,
    "calibration_subsets": 5000,
    "target_family_rate": 0.05,
    "gate_subsets": 2000,
    "seed": 0,
}
OUT_DIR = Path("results/selection")
FIGURE_DIR = Path("figures")
LABELS = {
    f"{b}_{m}": f"{challengers.BACKBONES[b]} {challengers.METHODS[m]}"
    for b in challengers.BACKBONES
    for m in challengers.METHODS
}


def load_sequences(directory, arms):
    """Read each arm's per-sequence predictions.

    Args:
        directory: Folder holding ``sequences_<arm>.csv``.
        arms: Arm names to read.

    Returns:
        A dict of arm to a DataFrame with ``seq_id``, ``site``, ``label``
        and ``pred``.

    Raises:
        ValueError: If the arms were scored on different sequences, which
            would make the pairing meaningless.
    """
    tables = {
        arm: pl.read_csv(directory / f"sequences_{arm}.csv").select(
            "seq_id", "site", "label", "pred"
        )
        for arm in arms
    }
    ids = {arm: set(t["seq_id"].to_list()) for arm, t in tables.items()}
    first = next(iter(ids.values()))
    if any(s != first for s in ids.values()):
        raise ValueError("arms were scored on different sequences")
    return tables


def site_groups(tables, champion, challenger, min_sequences):
    """Return each hold-out-sized site's labels and both arms' predictions.

    Args:
        tables: Output of ``load_sequences``.
        champion: The champion arm's name.
        challenger: The challenger arm's name.
        min_sequences: Sites with fewer sequences are left out.

    Returns:
        A dict of site to ``(labels, champion_preds, challenger_preds)``,
        aligned by sequence.
    """
    joined = tables[champion].join(
        tables[challenger].select("seq_id", pl.col("pred").alias("challenger")),
        on="seq_id",
    )
    groups = {}
    for (site,), group in joined.sort("seq_id").group_by("site"):
        if group.height >= min_sequences:
            groups[site] = tuple(
                group[c].to_numpy() for c in ("label", "pred", "challenger")
            )
    return dict(sorted(groups.items()))


def calibrate_tolerance(groups):
    """Return the smallest tolerance meeting the target false-block rate.

    Args:
        groups: Output of ``site_groups``.

    Returns:
        A pair ``(tolerance, family_rates)``: the tolerance, or None if no
        candidate meets the target, and the family-wise rate at each
        candidate tolerance.
    """
    rates = calibrate.false_block_rates(
        groups,
        scoring.balanced_accuracy,
        SETTINGS["tolerances"],
        draws=SETTINGS["draws"],
        seed=SETTINGS["seed"],
    )
    family = calibrate.family_rate(
        rates,
        SETTINGS["holdout_sites"],
        SETTINGS["calibration_subsets"],
        SETTINGS["seed"],
    )
    meeting = [
        t
        for t, r in zip(SETTINGS["tolerances"], family)
        if r <= SETTINGS["target_family_rate"]
    ]
    return (min(meeting) if meeting else None), family


def site_sets(sites, size, count, seed):
    """Draw random sets of sites, the same for every challenger.

    Args:
        sites: Site names to draw from.
        size: Sites per set.
        count: Number of sets.
        seed: Seed for the draws.

    Returns:
        A list of ``count`` lists of site names.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(sites)
    return [
        [sites[i] for i in rng.choice(len(sites), size=size, replace=False)]
        for _ in range(count)
    ]


def simulate_gate(champion, challenger, sets, config):
    """Run the gate on each set of sites and count passes and failures.

    Args:
        champion: Mapping of site to the champion's balanced accuracy.
        challenger: Mapping of site to the challenger's balanced accuracy.
        sets: Output of ``site_sets``.
        config: The ``GateConfig`` to decide with.

    Returns:
        A dict with the pass rate, its Monte Carlo standard error, and the
        share of sets failing each condition (a set can fail several).
    """
    passed = fail_mean = fail_bound = fail_tolerance = 0
    for chosen in sets:
        decision = rule.decide(
            {s: champion[s] for s in chosen},
            {s: challenger[s] for s in chosen},
            config,
        )
        passed += decision.promote
        fail_mean += decision.mean_difference <= config.min_mean_gain
        fail_bound += decision.interval[0] <= config.margin
        fail_tolerance += bool(decision.regressions)
    n = len(sets)
    rate = passed / n
    return {
        "pass_rate": rate,
        # Simulation noise only: the sets overlap, so this says nothing of
        # the uncertainty from having just these fold sites.
        "pass_rate_se": float(np.sqrt(rate * (1 - rate) / n)),
        "fail_mean_gain": fail_mean / n,
        "fail_margin": fail_bound / n,
        "fail_tolerance": fail_tolerance / n,
    }


def run(tables):
    """Calibrate, simulate and choose a challenger.

    Args:
        tables: Output of ``load_sequences`` for the champion and every
            challenger.

    Returns:
        A tuple ``(summary, results, curves)``: a JSON-serialisable dict,
        one row per challenger, and the family-wise false-block rate per
        challenger and tolerance.
    """
    champion = SETTINGS["champion"]
    rows, curves, sets = [], [], None
    for arm in SETTINGS["challengers"]:
        groups = site_groups(
            tables, champion, arm, SETTINGS["holdout_like_min_sequences"]
        )
        tolerance, family = calibrate_tolerance(groups)
        curves.append(
            pl.DataFrame(
                {
                    "challenger": [arm] * len(family),
                    "tolerance": SETTINGS["tolerances"],
                    "family_rate": family,
                }
            )
        )
        scores = {
            role: {s: scoring.balanced_accuracy(g[0], g[i]) for s, g in groups.items()}
            for role, i in (("champion", 1), ("challenger", 2))
        }
        if sets is None:
            sets = site_sets(
                groups,
                SETTINGS["holdout_sites"],
                SETTINGS["gate_subsets"],
                SETTINGS["seed"],
            )
        gains = [scores["challenger"][s] - scores["champion"][s] for s in groups]
        row = {
            "challenger": arm,
            "sites": len(groups),
            "mean_gain": float(np.mean(gains)),
            "worst_gain": float(np.min(gains)),
            "tolerance": tolerance,
        }
        if tolerance is None:
            # No tolerance keeps false blocks under the target, so the
            # per-site check would mean nothing; the arm is not eligible.
            row |= {"pass_rate": None, "pass_rate_se": None}
        else:
            config = dataclasses.replace(rule.GateConfig(), tolerance=tolerance)
            row |= simulate_gate(scores["champion"], scores["challenger"], sets, config)
        rows.append(row)

    eligible = [r for r in rows if r["pass_rate"] is not None]
    # Ties on pass rate go to the larger mean gain.
    chosen = max(eligible, key=lambda r: (r["pass_rate"], r["mean_gain"]), default=None)
    summary = {
        "settings": SETTINGS,
        "gate": dataclasses.asdict(rule.GateConfig()),
        "scope": "fold sites only; hold-out never read",
        "chosen": None if chosen is None else chosen["challenger"],
        "chosen_tolerance": None if chosen is None else chosen["tolerance"],
        "challengers": rows,
    }
    return summary, pl.DataFrame(rows), pl.concat(curves)


def plot_selection(results, curves, target, path):
    """Plot each challenger's calibration curve and simulated pass rate.

    Args:
        results: The per-challenger table from ``run``.
        curves: The family-rate table from ``run``.
        target: Target family-wise false-block rate, drawn as a line.
        path: Output PNG path.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"probe": "-", "knn": "--"}
    for arm in SETTINGS["challengers"]:
        backbone, method = arm.split("_")
        curve = curves.filter(pl.col("challenger") == arm)
        left.plot(
            curve["tolerance"].to_numpy() * 100,
            curve["family_rate"].to_numpy(),
            color=challengers.COLOURS[backbone],
            linestyle=styles[method],
            marker=challengers.MARKERS[backbone],
            markersize=3,
            label=LABELS[arm],
        )
    left.axhline(target, color="#52514e", linestyle=":", linewidth=1)
    left.set_xlabel("Per-site tolerance (balanced-accuracy points)")
    left.set_ylabel("Chance an equal challenger is blocked")
    left.set_title("Tolerance calibration (swap null, 10 sites)")
    left.set_ylim(0, 1)
    left.legend(frameon=False)

    order = results.sort("pass_rate", nulls_last=True, descending=True)
    names = order["challenger"].to_list()
    y = np.arange(len(names))[::-1]
    right.barh(
        y,
        order["pass_rate"].fill_null(0).to_numpy() * 100,
        xerr=order["pass_rate_se"].fill_null(0).to_numpy() * 100,
        color=[challengers.COLOURS[n.split("_")[0]] for n in names],
        height=0.6,
    )
    right.set_yticks(
        y,
        [
            f"{LABELS[n]}\n(tolerance {t * 100:.0f} points)" if t else LABELS[n]
            for n, t in zip(names, order["tolerance"].to_list())
        ],
    )
    for yi, rate in zip(y, order["pass_rate"].to_list()):
        if rate is None:
            right.text(
                1,
                yi,
                "not eligible: no tolerance up to 20 points\n"
                "keeps false blocks under 5%",
                va="center",
                fontsize=8,
                color="#52514e",
            )
    right.set_xlim(0, 100)
    right.set_xlabel("Simulated gate pass rate (%)")
    right.set_title("Gate on random sets of 10 fold sites")
    right.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    right.set_axisbelow(True)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Choose the hold-out challenger from the fold-site predictions.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--sequences-dir", type=Path, default=challengers.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)

    tables = load_sequences(
        args.sequences_dir, [SETTINGS["champion"], *SETTINGS["challengers"]]
    )
    summary, results, curves = run(tables)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.out_dir / "selection.csv")
    curves.write_csv(args.out_dir / "tolerance_curves.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_selection(
        results,
        curves,
        SETTINGS["target_family_rate"],
        args.figure_dir / "challenger_selection.png",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
