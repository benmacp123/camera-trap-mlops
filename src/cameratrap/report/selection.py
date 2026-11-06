"""Choose the one challenger that goes to the hold-out, on fold sites only.

Sending every challenger to the hold-out and promoting whichever passes
would be four tries at one exam: the chance that some challenger passes
by luck grows with each try (the multiple comparisons problem). So one
challenger is chosen here, before the freeze, and only it is scored on
the hold-out.

The choice is by simulated gate pass rate. Each hold-out-sized fold site
gets the gate's noise-aware check once per challenger: a swap-null
p-value for its drop against the champion, and the drop that p-value
cutoff would allow there (its "ruler"). Then many random sets of ten
such sites are drawn, and the real gate (``gate.rule.decide``) is run on
each. The pass rate rewards a large mean gain and no unexplained drops
together, at the hold-out's scale. Every challenger sees the same site
sets, so the comparison between them is paired. Ties on pass rate go to
the larger mean gain.

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
    "gate_subsets": 2000,
    "seed": 0,
}
# The gate as it will be frozen; tests swap in fewer draws.
GATE = rule.GateConfig()
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


def site_tests(groups, config, sites_per_decision):
    """Score both arms at each site and test the site's drop against chance.

    Args:
        groups: Output of ``site_groups``.
        config: The ``GateConfig``; its seed and draws drive the swap null.
        sites_per_decision: Sites in each gate decision, which sets the
            per-site cutoff ``family_alpha / sites_per_decision``.

    Returns:
        A DataFrame with one row per site: ``site``, ``n``, ``champion``,
        ``challenger``, ``gain``, ``p_value`` and ``ruler`` (the largest
        drop, in metric units, that the cutoff lets through there).
    """
    rng = np.random.default_rng(config.seed)
    cutoff = config.family_alpha / sites_per_decision
    rows = []
    for site, (labels, a, b) in groups.items():
        champ = scoring.balanced_accuracy(labels, a)
        chall = scoring.balanced_accuracy(labels, b)
        null = calibrate.null_differences(
            labels, a, b, scoring.balanced_accuracy, config.null_draws, rng
        )
        rows.append(
            {
                "site": site,
                "n": len(labels),
                "champion": champ,
                "challenger": chall,
                "gain": chall - champ,
                "p_value": calibrate.p_value(null, chall - champ),
                "ruler": float(-np.quantile(null, cutoff)),
            }
        )
    return pl.DataFrame(rows)


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


def simulate_gate(tests, sets, config):
    """Run the gate on each set of sites and count passes and failures.

    Args:
        tests: Output of ``site_tests``.
        sets: Output of ``site_sets``.
        config: The ``GateConfig`` to decide with.

    Returns:
        A dict with the pass rate, its Monte Carlo standard error, and the
        share of sets failing each condition (a set can fail several).
    """
    by_site = {row["site"]: row for row in tests.iter_rows(named=True)}
    passed = fail_mean = fail_bound = fail_drop = 0
    for chosen in sets:
        decision = rule.decide(
            {s: by_site[s]["champion"] for s in chosen},
            {s: by_site[s]["challenger"] for s in chosen},
            {s: by_site[s]["p_value"] for s in chosen},
            config,
        )
        passed += decision.promote
        fail_mean += decision.mean_difference <= config.min_mean_gain
        fail_bound += decision.interval[0] <= config.margin
        fail_drop += bool(decision.regressions)
    n = len(sets)
    rate = passed / n
    return {
        "pass_rate": rate,
        # Simulation noise only: the sets overlap, so this says nothing of
        # the uncertainty from having just these fold sites.
        "pass_rate_se": float(np.sqrt(rate * (1 - rate) / n)),
        "fail_mean_gain": fail_mean / n,
        "fail_margin": fail_bound / n,
        "fail_site_drop": fail_drop / n,
    }


def run(tables, config=None):
    """Test every site, simulate the gate and choose a challenger.

    Args:
        tables: Output of ``load_sequences`` for the champion and every
            challenger.
        config: The ``GateConfig``; defaults to ``GATE``.

    Returns:
        A tuple ``(summary, results, sites)``: a JSON-serialisable dict,
        one row per challenger, and one row per challenger and site.
    """
    config = config or GATE
    size = SETTINGS["holdout_sites"]
    cutoff = config.family_alpha / size
    rows, site_tables, sets = [], [], None
    for arm in SETTINGS["challengers"]:
        groups = site_groups(
            tables, SETTINGS["champion"], arm, SETTINGS["holdout_like_min_sequences"]
        )
        tests = site_tests(groups, config, size)
        site_tables.append(tests.with_columns(challenger_arm=pl.lit(arm)))
        if sets is None:
            sets = site_sets(groups, size, SETTINGS["gate_subsets"], SETTINGS["seed"])
        rows.append(
            {
                "challenger": arm,
                "sites": tests.height,
                "mean_gain": float(tests["gain"].mean()),
                "worst_gain": float(tests["gain"].min()),
                "sites_below_cutoff": int(
                    ((tests["p_value"] < cutoff) & (tests["gain"] < 0)).sum()
                ),
                **simulate_gate(tests, sets, config),
            }
        )

    chosen = max(rows, key=lambda r: (r["pass_rate"], r["mean_gain"]))
    summary = {
        "settings": SETTINGS,
        "gate": dataclasses.asdict(config),
        "per_site_cutoff": cutoff,
        "scope": "fold sites only; hold-out never read",
        "chosen": chosen["challenger"] if chosen["pass_rate"] > 0 else None,
        "challengers": rows,
    }
    return summary, pl.DataFrame(rows), pl.concat(site_tables)


def plot_selection(results, sites, path):
    """Plot each site's noise-aware ruler and each challenger's pass rate.

    Args:
        results: The per-challenger table from ``run``.
        sites: The per-site table from ``run``.
        path: Output PNG path.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    fills = {"probe": None, "knn": "none"}
    for arm in SETTINGS["challengers"]:
        backbone, method = arm.split("_")
        rows = sites.filter(pl.col("challenger_arm") == arm)
        colour = challengers.COLOURS[backbone]
        left.scatter(
            rows["n"].to_numpy(),
            rows["ruler"].to_numpy() * 100,
            s=18,
            marker=challengers.MARKERS[backbone],
            facecolors=fills[method] or colour,
            edgecolors=colour,
            linewidths=1,
            label=LABELS[arm],
        )
    left.set_xscale("log")
    left.set_xlabel("Sequences at the site (log scale)")
    left.set_ylabel("Largest drop allowed (points)")
    left.set_title("Each site's ruler: the drop chance explains there")
    left.set_ylim(bottom=0)
    left.grid(color="#e6e5e1", linewidth=0.6)
    left.set_axisbelow(True)
    left.legend(frameon=False, fontsize=8)

    order = results.sort("pass_rate", descending=True)
    names = order["challenger"].to_list()
    y = np.arange(len(names))[::-1]
    right.barh(
        y,
        order["pass_rate"].to_numpy() * 100,
        xerr=order["pass_rate_se"].to_numpy() * 100,
        color=[challengers.COLOURS[n.split("_")[0]] for n in names],
        height=0.6,
    )
    right.set_yticks(y, [LABELS[n] for n in names])
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
    summary, results, sites = run(tables)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.out_dir / "selection.csv")
    sites.write_csv(args.out_dir / "site_tests.csv")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_selection(results, sites, args.figure_dir / "challenger_selection.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
