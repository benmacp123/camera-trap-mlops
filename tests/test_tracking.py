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
