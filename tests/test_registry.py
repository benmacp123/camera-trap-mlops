"""Tests that the pinned MLflow supports the registry pattern the gate
relies on: a SQLite backend and mutable aliases instead of the deprecated
stages.
"""

import pytest

mlflow = pytest.importorskip("mlflow")


@pytest.fixture(name="client")
def fixture_client(tmp_path):
    """Return an MLflow client backed by a throwaway SQLite database.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        An ``MlflowClient`` whose tracking and registry URIs point at it.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_alias_moves_between_versions(client, tmp_path):
    """Setting @champion on a new version moves it there; the old one keeps @challenger."""
    name = "species-classifier"
    client.create_registered_model(name)
    first = client.create_model_version(name, source=str(tmp_path / "v1"))
    second = client.create_model_version(name, source=str(tmp_path / "v2"))

    client.set_registered_model_alias(name, "champion", first.version)
    assert client.get_model_version_by_alias(name, "champion").version == first.version

    # Promotion: the gate passes the challenger, the alias moves.
    client.set_registered_model_alias(name, "champion", second.version)
    client.set_registered_model_alias(name, "challenger", first.version)
    assert client.get_model_version_by_alias(name, "champion").version == second.version
    assert (
        client.get_model_version_by_alias(name, "challenger").version == first.version
    )


def test_deleted_alias_no_longer_resolves(client, tmp_path):
    """Deleting an alias makes lookups fail rather than return a stale version."""
    name = "species-classifier"
    client.create_registered_model(name)
    version = client.create_model_version(name, source=str(tmp_path / "v1"))
    client.set_registered_model_alias(name, "challenger", version.version)
    client.delete_registered_model_alias(name, "challenger")
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, "challenger")
