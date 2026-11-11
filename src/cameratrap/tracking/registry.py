"""Rebuild the MLflow runs and registry aliases from the committed results.

The tracking store is derived, never a source of truth: a clean clone gets
the same runs and aliases from the committed files, with no server and no
account. Every challenger becomes a run, but only the champion and the one
frozen challenger become model versions, so the registry holds what was
actually put forward.

The aliases follow the gate. Before the hold-out, ``@champion`` is BioCLIP 2
zero-shot and ``@challenger`` the frozen challenger. Once a hold-out result
exists, only its decision moves them; nothing here can promote by hand.

Aliases rather than registry stages, which MLflow has deprecated, and SQLite
rather than the file store.

::

    PYTHONPATH=src python -m cameratrap.tracking.registry
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import mlflow

from cameratrap.gate import freeze

MODEL_NAME = "species-classifier"
EXPERIMENT = "camera-trap-promotion"
FREEZE = Path("config/holdout_freeze.json")
CHALLENGERS = Path("results/challengers")
SELECTION = Path("results/selection/summary.json")
HOLDOUT = Path("results/holdout")
HOLDOUT_FILES = ("summary.json", "per_site.csv")
CONTROL = Path("results/control")
CONTROL_FILES = ("summary.json", "per_site.csv", "per_class.csv")
REVIEW = Path("results/review")
REVIEW_FILES = ("summary.json", "cells.csv", "per_site.csv", "comparisons.csv")
REVIEW_METRICS = ("sites", "labels", "champion_mean", "challenger_mean") + (
    "mean_gain",
    "low",
    "high",
)


def log_run(client, experiment_id, name, params, metrics, tags, files=()):
    """Log one finished run.

    Args:
        client: An ``MlflowClient``.
        experiment_id: The experiment to log under.
        name: The run's name.
        params: A dict of parameters.
        metrics: A dict of numeric metrics.
        tags: A dict of tags.
        files: Paths of files to attach.

    Returns:
        The run's ID.
    """
    run_id = client.create_run(experiment_id, run_name=name, tags=tags).info.run_id
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
    for path in files:
        client.log_artifact(run_id, str(path))
    client.set_terminated(run_id)
    return run_id


def register(client, run_id, card, head=None):
    """Attach a model card to a run and register it as a new version.

    Args:
        client: An ``MlflowClient``.
        run_id: The run that describes the model.
        card: A JSON-serialisable description of the model and how it is
            made; the zero-shot champion has no weights of its own.
        head: Path of a trained classifier head to attach, if there is one.

    Returns:
        The new version's number, as a string.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "card.json"
        path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
        client.log_artifact(run_id, str(path), "model")
    if head is not None:
        client.log_artifact(run_id, str(head), "model")
    source = f"{client.get_run(run_id).info.artifact_uri}/model"
    version = client.create_model_version(
        MODEL_NAME, source=source, run_id=run_id, tags={"arm": card["arm"]}
    )
    return version.version


def apply_decision(client, champion, challenger, promote):
    """Move the aliases as the gate decided, and record it on the versions.

    Args:
        client: An ``MlflowClient``.
        champion: The current champion's version.
        challenger: The challenger's version.
        promote: The gate's decision.
    """
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    outcome = "promoted" if promote else "rejected"
    client.set_model_version_tag(MODEL_NAME, challenger, "gate_decision", outcome)
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger)
        client.set_registered_model_alias(MODEL_NAME, "previous_champion", champion)


def arm_metrics(arm_summary, selection_row):
    """Return an arm's numeric results on the fold sites.

    Args:
        arm_summary: The arm's entry in the challenger summary.
        selection_row: The arm's row in the selection, or ``None``.

    Returns:
        A dict of metric name to number.
    """
    merged = arm_summary | (selection_row or {})
    return {k: v for k, v in merged.items() if isinstance(v, int | float)}


def arm_params(arm, arm_summary):
    """Return an arm's parameters.

    Args:
        arm: The arm's name.
        arm_summary: The arm's entry in the challenger summary.

    Returns:
        A dict with the backbone, the method and each fold's chosen setting.
    """
    backbone, _, method = arm.partition("_")
    chosen = {
        k: json.dumps(v) for k, v in arm_summary.items() if k.startswith("chosen_")
    }
    return {"backbone": backbone, "method": method or "zero_shot"} | chosen


def log_control(client, experiment_id, root):
    """Log the CNN control as a run, if it has been reported.

    It is deliberately only a run: with no model version, no alias can ever
    point at it, whatever its result.

    Args:
        client: An ``MlflowClient``.
        experiment_id: The experiment to log under.
        root: The repository root, a ``pathlib.Path``.

    Returns:
        The run's ID, or ``None`` if there is no control result yet.
    """
    path = root / CONTROL / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    metrics = {
        f"{arm}_mean_site_balanced_accuracy": value
        for arm, value in summary["mean_site_balanced_accuracy"].items()
    }
    for arm, verdict in summary["reference_verdicts"].items():
        metrics[f"mean_gain_over_{arm}"] = verdict["mean_gain"]
    recipe = summary["made_with"]["recipe"]
    return log_run(
        client,
        experiment_id,
        "cnn_control",
        {"backbone": recipe["model_id"], "method": "full_fine_tune"},
        metrics,
        {"evaluated_on": "fold sites", "role": "control"},
        [root / CONTROL / name for name in CONTROL_FILES],
    )


def log_review(client, experiment_id, root):
    """Log the simulated review loop, if it has been run.

    Every cell (a queue strategy, budget, weight and seed on one set of
    sites) is a run tagged with the gate's decision, and one further run
    holds the tables. None gets a model version: the fold-site models are
    five cross-fitted probes, and the hold-out is already spent, so no
    alias may follow these decisions.

    Args:
        client: An ``MlflowClient``.
        experiment_id: The experiment to log under.
        root: The repository root, a ``pathlib.Path``.

    Returns:
        The number of cell runs logged; 0 if there is no review result yet.
    """
    folder = root / REVIEW
    if not (folder / "cells.csv").exists():
        return 0
    primary = json.loads((folder / "summary.json").read_text())["settings"]["primary"]
    with open(folder / "cells.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        params = {k: row[k] for k in ("source", "strategy", "budget", "weight", "seed")}
        is_primary = (
            row["strategy"] == primary["strategy"]
            and row["budget"] == str(primary["budget"])
            and float(row["weight"]) == primary["weight"]
        )
        log_run(
            client,
            experiment_id,
            "review-" + "-".join(params.values()),
            params,
            {k: float(row[k]) for k in REVIEW_METRICS},
            {
                "role": "review-loop",
                "evaluated_on": f"{row['source']} sites, late half",
                "gate_decision": "promoted" if row["promote"] == "true" else "rejected",
                "primary": str(is_primary),
            },
        )
    log_run(
        client,
        experiment_id,
        "review-loop-tables",
        {"primary": json.dumps(primary, sort_keys=True)},
        {"cells": len(rows)},
        {"role": "review-loop-tables"},
        [folder / name for name in REVIEW_FILES],
    )
    return len(rows)


def sync(client, root, artifact_location):
    """Log every arm, register the two put forward, and apply any decision.

    Args:
        client: An ``MlflowClient`` on an empty store.
        root: The repository root, a ``pathlib.Path``.
        artifact_location: URI to keep run artifacts under.

    Returns:
        A dict of alias to version.

    Raises:
        ValueError: If the store already holds the model, or the frozen
            challenger's hold-out result was made under another freeze.
    """
    if client.search_registered_models(f"name = '{MODEL_NAME}'"):
        raise ValueError("the store already holds the model; rebuild it from empty")
    config, _ = freeze.load(root / FREEZE)
    digest = freeze.config_hash(config)
    summary = json.loads((root / CHALLENGERS / "summary.json").read_text())
    selected = json.loads((root / SELECTION).read_text())
    rows = {row["challenger"]: row for row in selected["challengers"]}
    result_path = root / HOLDOUT / "summary.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if result is not None and result["freeze_sha256"] != digest:
        raise ValueError("the hold-out result was made under another freeze")

    experiment = client.create_experiment(
        EXPERIMENT, artifact_location=artifact_location
    )
    run_ids = {
        arm: log_run(
            client,
            experiment,
            arm,
            arm_params(arm, entry),
            arm_metrics(entry, rows.get(arm)),
            {"evaluated_on": "fold sites"},
            [root / CHALLENGERS / f"sequences_{arm}.csv"],
        )
        for arm, entry in summary["arms"].items()
    }
    log_control(client, experiment, root)
    log_review(client, experiment, root)

    client.create_registered_model(MODEL_NAME)
    champion, challenger = config["champion"], config["challenger"]
    head = root / HOLDOUT / "probe_head.npz"
    champion_version = register(client, run_ids[champion["arm"]], champion)
    challenger_version = register(
        client,
        run_ids[challenger["arm"]],
        challenger,
        head if result is not None else None,
    )
    for version in (champion_version, challenger_version):
        client.set_model_version_tag(MODEL_NAME, version, "freeze_sha256", digest)
    client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger_version)

    if result is not None:
        per_site = (root / HOLDOUT / "per_site.csv").read_text().splitlines()
        log_run(
            client,
            experiment,
            "holdout-gate",
            {"freeze_sha256": digest, "challenger": result["challenger"]},
            {
                "mean_gain": result["mean_gain"],
                "interval_low": result["interval"][0],
                "interval_high": result["interval"][1],
                "sites": len(per_site) - 1,
            },
            {"evaluated_on": "hold-out", "promote": str(result["promote"])},
            [root / HOLDOUT / name for name in HOLDOUT_FILES],
        )
        apply_decision(client, champion_version, challenger_version, result["promote"])
    aliases = client.get_registered_model(MODEL_NAME).aliases
    return dict(aliases)


def main(argv=None):
    """Rebuild the tracking store from the committed results.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--uri", default="sqlite:///mlflow.db")
    parser.add_argument("--artifacts", type=Path, default=Path("mlruns"))
    args = parser.parse_args(argv)

    client = mlflow.MlflowClient(tracking_uri=args.uri, registry_uri=args.uri)
    aliases = sync(client, args.root, args.artifacts.resolve().as_uri())
    print(json.dumps(aliases, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
