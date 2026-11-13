"""Cost report for the measured AWS run, from the committed run files.

Reads ``results/cloud/invocations_<label>.csv`` and ``parity_<label>.json``
(written by ``cloud.batch``), the price snapshot and ``stack.json`` (what
the stack cost to keep standing), and writes ``results/cloud/summary.json``
and ``figures/cloud_cost.png``. Nothing here touches AWS, so a reviewer can
rerun it from the committed files.

Every figure is list price with the free tier ignored. The headline is the
full run's cost per 10,000 crops with cold starts included; the sweep's
figures, which are warm invocations only, are reported beside it.

::

    PYTHONPATH=src python -m cloud.report
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from cloud import batch, cost

RESULTS = Path("results/cloud")
FIGURE = Path("figures/cloud_cost.png")
STACK = RESULTS / "stack.json"
PARTS = ("compute_usd", "requests_usd", "s3_get_usd", "s3_put_usd")
LABELS = ("Lambda compute", "Lambda requests", "S3 reads", "S3 writes")


def read_invocations(path):
    """Read one run's invocation table with typed columns.

    Args:
        path: An ``invocations_<label>.csv``.

    Returns:
        One dict per invocation with ``n``, ``memory_mb``, ``billed_ms``,
        ``max_memory_mb`` and ``wall_seconds`` as numbers and ``cold`` as
        a bool.
    """
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "n": int(row["n"]),
                    "memory_mb": int(row["memory_mb"]),
                    "billed_ms": float(row["billed_ms"]),
                    "max_memory_mb": float(row["max_memory_mb"]),
                    "wall_seconds": float(row["wall_seconds"]),
                    "cold": row["cold"] == "True",
                }
            )
    return rows


def describe_run(prices, rows):
    """Summarise one run: cost, speed, cold starts and memory headroom.

    Args:
        prices: Prices from ``cost.load_prices``.
        rows: Invocations from ``read_invocations``.

    Returns:
        Dict with the priced run (``cost.run_cost``), ``cold_invocations``,
        ``crops_per_billed_second`` over all invocations, the warm-only
        ``warm`` summary from ``batch.summarise``, ``peak_memory_mb`` and
        ``memory_mb``.
    """
    priced = cost.run_cost(prices, rows)
    billed_seconds = sum(row["billed_ms"] for row in rows) / 1000
    return {
        "memory_mb": rows[0]["memory_mb"],
        "cost": priced,
        "cold_invocations": sum(row["cold"] for row in rows),
        "crops_per_billed_second": round(priced["crops"] / billed_seconds, 3),
        "warm": batch.summarise(rows, prices),
        "peak_memory_mb": max(row["max_memory_mb"] for row in rows),
    }


def stack_cost(prices, stack, n_crops):
    """Price what the experiment paid to stand the stack up.

    Args:
        prices: Prices from ``cost.load_prices``.
        stack: Mapping with ``crop_bytes``, ``image_bytes`` and ``days``.
        n_crops: Crops uploaded.

    Returns:
        The dict from ``cost.fixed_cost``.
    """
    return cost.fixed_cost(
        prices, n_crops, stack["crop_bytes"], stack["image_bytes"], stack["days"]
    )


def plot(runs, path=FIGURE):
    """Draw the cost per 10,000 crops of each run, split by charge.

    Args:
        runs: Mapping of run name to a ``describe_run`` result, in the
            order to draw them.
        path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = [0.0] * len(runs)
    for part, label in zip(PARTS, LABELS):
        values = [
            run["cost"][part] / run["cost"]["crops"] * 10_000 for run in runs.values()
        ]
        ax.bar(list(runs), values, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    for index, total in enumerate(bottoms):
        ax.text(index, total, f"${total:.3f}", ha="center", va="bottom")
    ax.set_ylabel("List-price cost per 10,000 crops (US$)")
    ax.set_title("Champion probe on Lambda (eu-west-2, free tier ignored)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Write the summary and the figure.

    Returns:
        Process exit status, 1 if parity failed or ``stack.json`` is missing.
    """
    if not STACK.exists():
        print(f"{STACK} is missing: it records the stack's image size and age")
        return 1
    prices = cost.load_prices()
    stack = json.loads(STACK.read_text(encoding="utf-8"))
    runs = {}
    parity = {}
    for label in sorted(
        path.stem.split("_", 1)[1] for path in RESULTS.glob("invocations_*.csv")
    ):
        runs[label] = describe_run(
            prices, read_invocations(RESULTS / f"invocations_{label}.csv")
        )
        parity[label] = json.loads(
            (RESULTS / f"parity_{label}.json").read_text(encoding="utf-8")
        )
    failed = [label for label, result in parity.items() if not result["passed"]]
    if failed:
        print(f"parity failed for {failed}: not reporting")
        return 1
    full = runs["full"]
    summary = {
        "assumptions": "list price, eu-west-2, free tier ignored; observed run",
        "headline": {
            "usd_per_10k_crops": round(full["cost"]["usd_per_10k_crops"], 4),
            "crops_per_billed_second": full["crops_per_billed_second"],
            "memory_mb": full["memory_mb"],
            "crops": full["cost"]["crops"],
            "cold_invocations": full["cold_invocations"],
        },
        "runs": runs,
        "fixed": stack_cost(prices, stack, full["cost"]["crops"]),
        "stack": stack,
        "parity": parity,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    order = ["sweep-1769", "sweep-3008", "full"]
    plot({name: runs[name] for name in order if name in runs})
    print(json.dumps(summary["headline"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
