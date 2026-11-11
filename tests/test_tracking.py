"""Tests that the registry's aliases follow the committed results and the gate."""

import json

import pytest

mlflow = pytest.importorskip("mlflow")

# pylint: disable=wrong-import-position
from cameratrap.gate import freeze
from cameratrap.tracking import registry

ARMS = ("bioclip2", "dinov2_probe", "dinov2_knn")
CONFIG = {
    "gate": {},
    "champion": {"arm": "bioclip2", "backbone": "bioclip2"},
    "challenger": {"arm": "dinov2_probe", "backbone": "dinov2", "cs": [1.0]},
}


@pytest.fixture(name="repo")
def fixture_repo(tmp_path):
    """Write the committed results a tracking rebuild reads.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The repository root.
    """
    root = tmp_path / "repo"
    for folder in (
        registry.FREEZE.parent,
        registry.CHALLENGERS,
        registry.SELECTION.parent,
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
    payload = {"config": CONFIG, "sha256": freeze.config_hash(CONFIG)}
    (root / registry.FREEZE).write_text(json.dumps(payload))
    arms = {
        arm: {"mean_site_balanced_accuracy": 0.9, "chosen_c_per_fold": {"fold0": 1.0}}
        for arm in ARMS
    }
    (root / registry.CHALLENGERS / "summary.json").write_text(
        json.dumps({"arms": arms})
    )
    for arm in ARMS:
        (root / registry.CHALLENGERS / f"sequences_{arm}.csv").write_text("seq_id\nq\n")
    (root / registry.SELECTION).write_text(
        json.dumps({"challengers": [{"challenger": "dinov2_probe", "pass_rate": 1.0}]})
    )
    return root


@pytest.fixture(name="client")
def fixture_client(tmp_path):
    """Return a client on an empty SQLite store."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)


def run_sync(client, root):
    """Rebuild the store under the test's folder and return the aliases."""
    return registry.sync(client, root, (root.parent / "mlruns").as_uri())


def write_result(root, promote, digest=None):
    """Write a hold-out result with the given decision.

    Args:
        root: The repository root.
        promote: The gate's decision.
        digest: The freeze hash it claims; defaults to the real one.
    """
    folder = root / registry.HOLDOUT
    folder.mkdir(parents=True)
    result = {
        "freeze_sha256": digest or freeze.config_hash(CONFIG),
        "challenger": "dinov2_probe",
        "promote": promote,
        "mean_gain": 0.05,
        "interval": [0.01, 0.09],
    }
    (folder / "summary.json").write_text(json.dumps(result))
    (folder / "per_site.csv").write_text("site,gain\nA,0.05\n")
    (folder / "probe_head.npz").write_bytes(b"head")


def test_before_the_hold_out_both_aliases_are_set(client, repo):
    """The champion and the frozen challenger are versions, and nothing else is."""
    aliases = run_sync(client, repo)
    assert set(aliases) == {"champion", "challenger"}
    versions = client.search_model_versions(f"name = '{registry.MODEL_NAME}'")
    assert {v.tags["arm"] for v in versions} == {"bioclip2", "dinov2_probe"}
    champion = client.get_model_version_by_alias(registry.MODEL_NAME, "champion")
    assert champion.tags["arm"] == "bioclip2"


def test_every_arm_gets_a_run(client, repo):
    """Challengers not put forward are still on record, with their metrics."""
    run_sync(client, repo)
    experiment = client.get_experiment_by_name(registry.EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert {r.info.run_name for r in runs} == set(ARMS)
    probe = next(r for r in runs if r.info.run_name == "dinov2_probe")
    assert probe.data.metrics["pass_rate"] == 1.0
    assert probe.data.params["method"] == "probe"
    assert probe.data.tags["evaluated_on"] == "fold sites"


def test_a_pass_moves_champion_to_the_challenger(client, repo):
    """A promotion moves @champion and keeps the old one reachable."""
    write_result(repo, promote=True)
    aliases = run_sync(client, repo)
    assert set(aliases) == {"champion", "previous_champion"}
    champion = client.get_model_version_by_alias(registry.MODEL_NAME, "champion")
    assert champion.tags["arm"] == "dinov2_probe"
    assert champion.tags["gate_decision"] == "promoted"


def test_a_fail_keeps_the_champion(client, repo):
    """A rejection leaves @champion alone and records why the challenger stopped."""
    write_result(repo, promote=False)
    aliases = run_sync(client, repo)
    assert set(aliases) == {"champion"}
    champion = client.get_model_version_by_alias(registry.MODEL_NAME, "champion")
    assert champion.tags["arm"] == "bioclip2"
    rejected = next(
        v
        for v in client.search_model_versions(f"name = '{registry.MODEL_NAME}'")
        if v.tags["arm"] == "dinov2_probe"
    )
    assert rejected.tags["gate_decision"] == "rejected"


def test_a_result_under_another_freeze_is_refused(client, repo):
    """A decision made under other settings cannot move an alias."""
    write_result(repo, promote=True, digest="other")
    with pytest.raises(ValueError, match="another freeze"):
        run_sync(client, repo)


def test_a_store_is_rebuilt_from_empty(client, repo):
    """A second sync would duplicate versions, so it is refused."""
    run_sync(client, repo)
    with pytest.raises(ValueError, match="from empty"):
        run_sync(client, repo)


def test_the_champion_card_can_be_fetched_by_alias(client, repo):
    """A consumer can go from the alias to the model's description."""
    run_sync(client, repo)
    version = client.get_model_version_by_alias(registry.MODEL_NAME, "challenger")
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{version.source}/card.json",
        dst_path=str(repo.parent / "download"),
    )
    assert json.loads(open(local, encoding="utf-8").read())["cs"] == [1.0]


def test_the_head_is_attached_once_the_hold_out_has_run(client, repo):
    """The trained head reaches the registry only with the result."""
    write_result(repo, promote=True)
    run_sync(client, repo)
    version = client.get_model_version_by_alias(registry.MODEL_NAME, "champion")
    files = {f.path for f in client.list_artifacts(version.run_id, "model")}
    assert files == {"model/card.json", "model/probe_head.npz"}


def write_control(root, mean_gain):
    """Write a CNN control report with the given gain.

    Args:
        root: The repository root.
        mean_gain: Its mean gain over the champion.
    """
    folder = root / registry.CONTROL
    folder.mkdir(parents=True)
    verdict = {"would_pass": mean_gain > 0, "mean_gain": mean_gain}
    summary = {
        "mean_site_balanced_accuracy": {"cnn_control": 0.8},
        "reference_verdicts": {"dinov2_probe": verdict},
        "made_with": {"recipe": {"model_id": "timm/efficientnet_b0.ra_in1k"}},
    }
    (folder / "summary.json").write_text(json.dumps(summary))
    for name in registry.CONTROL_FILES[1:]:
        (folder / name).write_text("site\nS0\n")


def test_a_winning_control_is_a_run_and_moves_nothing(client, repo):
    """Even a control that beats the champion gets a run, no version and no alias."""
    write_result(repo, promote=True)
    aliases_without = run_sync(client, repo)
    versions_without = len(client.search_model_versions("name='species-classifier'"))

    uri = f"sqlite:///{repo.parent / 'second.db'}"
    second = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    write_control(repo, mean_gain=0.2)
    aliases_with = registry.sync(second, repo, (repo.parent / "mlruns2").as_uri())

    assert aliases_with == aliases_without
    versions = second.search_model_versions("name='species-classifier'")
    assert len(versions) == versions_without
    experiment = second.get_experiment_by_name(registry.EXPERIMENT).experiment_id
    runs = second.search_runs([experiment], "tags.role = 'control'")
    assert [run.info.run_name for run in runs] == ["cnn_control"]
    assert runs[0].info.run_id not in {version.run_id for version in versions}


def test_no_control_result_logs_no_control_run(client, repo):
    """Before the control has run, the rebuild still works and logs no run for it."""
    run_sync(client, repo)
    experiment = client.get_experiment_by_name(registry.EXPERIMENT).experiment_id
    assert not client.search_runs([experiment], "tags.role = 'control'")


def write_review(root):
    """Write a two-cell review result: the primary rejected, another promoted.

    Args:
        root: The repository root.
    """
    folder = root / registry.REVIEW
    folder.mkdir(parents=True)
    primary = {"strategy": "diverse", "budget": 20, "weight": 1.0}
    (folder / "summary.json").write_text(json.dumps({"settings": {"primary": primary}}))
    header = "source,strategy,weight,seed,budget,sites,labels,champion_mean,"
    header += "challenger_mean,mean_gain,low,high,promote"
    rows = [
        "fold,diverse,1.0,0,20,34,680,0.9,0.9,0.0,-0.01,0.01,false",
        "fold,diverse,10.0,0,20,34,680,0.9,0.92,0.02,0.01,0.03,true",
    ]
    (folder / "cells.csv").write_text("\n".join([header, *rows]) + "\n")
    for name in ("per_site.csv", "comparisons.csv"):
        (folder / name).write_text("site\nS0\n")


def test_review_cells_are_runs_with_their_decision_and_move_nothing(client, repo):
    """Each review cell is logged with the gate's verdict; no alias follows one."""
    write_result(repo, promote=True)
    write_review(repo)
    aliases = run_sync(client, repo)
    assert set(aliases) == {"champion", "previous_champion"}
    experiment = client.get_experiment_by_name(registry.EXPERIMENT).experiment_id
    runs = client.search_runs([experiment], "tags.role = 'review-loop'")
    verdicts = {r.data.params["weight"]: r.data.tags["gate_decision"] for r in runs}
    assert verdicts == {"1.0": "rejected", "10.0": "promoted"}
    assert {r.data.tags["primary"] for r in runs} == {"True", "False"}
    versions = client.search_model_versions(f"name = '{registry.MODEL_NAME}'")
    assert not {r.info.run_id for r in runs} & {v.run_id for v in versions}
    tables = client.search_runs([experiment], "tags.role = 'review-loop-tables'")
    files = {f.path for f in client.list_artifacts(tables[0].info.run_id)}
    assert files == set(registry.REVIEW_FILES)


def test_no_review_result_logs_no_review_run(client, repo):
    """Before the review loop has run, the rebuild logs nothing for it."""
    run_sync(client, repo)
    experiment = client.get_experiment_by_name(registry.EXPERIMENT).experiment_id
    assert not client.search_runs([experiment], "tags.role = 'review-loop'")
