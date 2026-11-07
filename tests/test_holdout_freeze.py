"""Tests that pin the hold-out freeze and protect how it is written."""

import json
from importlib import metadata
from pathlib import Path

import pytest

from cameratrap.arms import probe
from cameratrap.gate import freeze, rule
from cameratrap.report import holdout_freeze

ROOT = Path(__file__).resolve().parent.parent
# Recorded at the freeze. Changing this line is a re-freeze, visible in git.
FROZEN_SHA256 = "774fff882764979a051792f1a0c1fe4e595f043141095b32f6cd3d28fe203933"


def frozen():
    """Return the committed freeze, checked against the pinned hash."""
    return freeze.load(ROOT / holdout_freeze.OUT_PATH, expected_sha256=FROZEN_SHA256)


def write_inputs(root, chosen="dinov2_probe", gate=None, index_sha=None):
    """Write minimal committed inputs for a freeze under a temporary root.

    Args:
        root: The temporary repository root.
        chosen: The challenger the selection chose.
        gate: The gate the selection ran with; defaults to the current one.
        index_sha: The crop index hash the caches record; defaults to the
            real hash of the index written here.
    """
    (root / "data/manifest").mkdir(parents=True)
    (root / "results/selection").mkdir(parents=True)
    (root / "results/challengers").mkdir(parents=True)
    index = root / holdout_freeze.INDEX
    index.write_text("path,split\na.jpg,fold0\n")
    meta = {"index_sha256": index_sha or freeze.file_sha256(index)}
    contents = {
        holdout_freeze.MANIFEST: {"holdout_sites": ["A", "B"], "splits_sha256": "abc"},
        holdout_freeze.CROP_SETTINGS: {"settings": {"frames_per_sequence": 3}},
        holdout_freeze.HUB: {"revision": "r1", "files": {}},
        holdout_freeze.SELECTION: {
            "chosen": chosen,
            "gate": gate or freeze.gate_dict(rule.GateConfig()),
        },
        holdout_freeze.CHALLENGERS: {"caches": {"bioclip2": meta, "dinov2": meta}},
    }
    for name, content in contents.items():
        (root / name).write_text(json.dumps(content))


def test_the_committed_freeze_is_unedited():
    """The frozen file matches both its stored hash and the pinned one."""
    config, gate = frozen()
    assert config["challenger"]["arm"] == "dinov2_probe"
    assert config["champion"]["arm"] == "bioclip2"
    assert len(config["holdout_sites"]) == 10
    assert gate == rule.GateConfig(**config["gate"])


def test_the_files_under_the_freeze_are_unchanged():
    """The crop index, embedding record and selection are the ones frozen."""
    config, _ = frozen()
    freeze.check_files(config["inputs"], ROOT)


def test_the_code_still_matches_the_frozen_recipe():
    """The gate defaults and probe settings have not drifted from the freeze."""
    config, gate = frozen()
    assert gate == rule.GateConfig()
    recipe = holdout_freeze.probe_recipe(config["challenger"]["backbone"])
    assert {k: config["challenger"][k] for k in recipe} == recipe
    assert config["challenger"]["cs"] == probe.CS.tolist()


def test_the_hold_out_sites_are_the_manifests():
    """The frozen sites and split hash are the manifest's."""
    config, _ = frozen()
    manifest = json.loads((ROOT / holdout_freeze.MANIFEST).read_text())
    assert config["holdout_sites"] == manifest["holdout_sites"]
    assert config["splits_sha256"] == manifest["splits_sha256"]


def test_the_frozen_software_is_installed():
    """Results could shift under other versions of the numerical libraries."""
    config, _ = frozen()
    for name, version in config["software"].items():
        assert metadata.version(name) == version


def test_build_config_records_the_decision(tmp_path):
    """A freeze holds the gate, sites, metric, challenger recipe and input hashes."""
    write_inputs(tmp_path)
    config = holdout_freeze.build_config(tmp_path, "2026-01-01")
    assert config["holdout_sites"] == ["A", "B"]
    assert config["gate"] == freeze.gate_dict(rule.GateConfig())
    assert config["metric"]["name"] == "balanced_accuracy"
    assert config["challenger"]["cs"] == probe.CS.tolist()
    assert set(config["inputs"]) == set(holdout_freeze.HASHED_INPUTS)
    freeze.check_files(config["inputs"], tmp_path)


def test_no_chosen_challenger_is_refused(tmp_path):
    """Nothing can be frozen when no challenger passed the selection."""
    write_inputs(tmp_path)
    selection = tmp_path / holdout_freeze.SELECTION
    selection.write_text(json.dumps({"chosen": None, "gate": {}}))
    with pytest.raises(ValueError, match="no challenger"):
        holdout_freeze.build_config(tmp_path, "2026-01-01")


def test_a_selection_under_another_gate_is_refused(tmp_path):
    """The challenger must have been chosen under the gate being frozen."""
    write_inputs(tmp_path, gate={"family_alpha": 0.2})
    with pytest.raises(ValueError, match="different gate"):
        holdout_freeze.build_config(tmp_path, "2026-01-01")


def test_an_arm_without_a_recipe_is_refused(tmp_path):
    """A chosen arm whose training is not recorded cannot be frozen."""
    write_inputs(tmp_path, chosen="dinov2_knn")
    with pytest.raises(ValueError, match="no frozen training recipe"):
        holdout_freeze.build_config(tmp_path, "2026-01-01")


def test_caches_from_another_index_are_refused(tmp_path):
    """Embeddings made from a different crop index would score the wrong crops."""
    write_inputs(tmp_path, index_sha="not-the-index")
    with pytest.raises(ValueError, match="another crop index"):
        holdout_freeze.build_config(tmp_path, "2026-01-01")


def test_main_writes_once_and_refuses_a_second_freeze(tmp_path):
    """A second freeze would be a retune, so it does not overwrite the first."""
    write_inputs(tmp_path)
    args = ["--root", str(tmp_path), "--out", "config/f.json"]
    assert holdout_freeze.main(args) == 0
    first = (tmp_path / "config/f.json").read_text()
    with pytest.raises(FileExistsError):
        holdout_freeze.main(args)
    assert (tmp_path / "config/f.json").read_text() == first


def test_check_files_names_what_changed(tmp_path):
    """An edited or missing input is named, and an unchanged one passes."""
    path = tmp_path / "a.txt"
    path.write_text("one")
    hashes = {"a.txt": freeze.file_sha256(path)}
    freeze.check_files(hashes, tmp_path)
    path.write_text("two")
    with pytest.raises(ValueError, match="a.txt"):
        freeze.check_files(hashes, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        freeze.check_files(hashes, tmp_path)
