"""Tests for scoring the hold-out, on synthetic data only; the real one stays sealed."""

import json

import numpy as np
import polars as pl
import pytest

from cameratrap.data import crops
from cameratrap.embed import store
from cameratrap.gate import freeze
from cameratrap.report import holdout, holdout_freeze
from test_holdout_freeze import FROZEN_SHA256, ROOT

CLASSES = ["agouti", "guan", "tapir"]
TRAIN_SITES = {f"F{i}": f"fold{i % 3}" for i in range(6)}
HOLDOUT_SITES = [f"H{i}" for i in range(6)]


def make_index(per_site=30):
    """Return a crop index over six training and six hold-out sites.

    Args:
        per_site: Two-frame sequences per site, classes in turn.

    Returns:
        A DataFrame with the crop index's columns.
    """
    rows = []
    sites = [*TRAIN_SITES.items(), *((s, "holdout") for s in HOLDOUT_SITES)]
    for site, split in sites:
        for s in range(per_site):
            for f in range(2):
                rows.append(
                    (
                        f"{site}/{s:03d}_{f}.JPG",
                        f"{site}-{s}",
                        site,
                        split,
                        CLASSES[s % 3],
                        f,
                    )
                )
    return (
        pl.DataFrame(
            rows,
            schema=["path", "seq_id", "site", "split", "label", "frame_num"],
            orient="row",
        )
        .with_columns(
            conf=pl.lit(0.9),
            person_conf=pl.lit(0.0),
            bbox_x=pl.lit(0.1),
            bbox_y=pl.lit(0.1),
            bbox_w=pl.lit(0.2),
            bbox_h=pl.lit(0.2),
        )
        .select(list(crops.INDEX_SCHEMA))
    )


def make_vectors(index, holdout_informative=True, seed=0):
    """Return noisy one-hot embeddings for each frame.

    Args:
        index: The crop index.
        holdout_informative: If false, hold-out rows are pure noise, so a
            probe fitted on the training rows does badly there.
        seed: Seed for the noise.

    Returns:
        An array of shape ``(frames, 3)``.
    """
    rng = np.random.default_rng(seed)
    labels = index["label"].to_list()
    vectors = np.eye(3)[[CLASSES.index(c) for c in labels]]
    if not holdout_informative:
        vectors[(index["split"] == "holdout").to_numpy()] = 0
    return (vectors + rng.normal(0, 0.2, vectors.shape)).astype(np.float32)


def make_caches(index, meta, **kwargs):
    """Return caches where zero-shot misses tapir but the probe can learn it.

    Args:
        index: The crop index.
        meta: The embedding metadata both caches carry.
        **kwargs: Passed to ``make_vectors`` for the challenger's backbone.

    Returns:
        A dict of backbone to cache.
    """
    paths = index["path"].to_list()
    return {
        "bioclip2": {
            "paths": paths,
            "vectors": make_vectors(index),
            # Tapir's text embedding sits on agouti's axis.
            "text": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            "classes": np.array(CLASSES),
            "logit_scale": np.float32(10.0),
            "meta": meta,
        },
        "dinov2": {
            "paths": paths,
            "vectors": make_vectors(index, **kwargs),
            "meta": meta,
        },
    }


@pytest.fixture(name="repo")
def fixture_repo(tmp_path, monkeypatch):
    """Build a repository holding a synthetic freeze, index and caches.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's patcher, for the pinned hash.

    Returns:
        A tuple ``(root, cache_root, index, config, gate)``.
    """
    root = tmp_path / "repo"
    (root / "data/manifest").mkdir(parents=True)
    (root / "results/selection").mkdir(parents=True)
    (root / "results/challengers").mkdir(parents=True)
    index = make_index()
    index.write_csv(root / holdout_freeze.INDEX)
    meta = {"index_sha256": freeze.file_sha256(root / holdout_freeze.INDEX)}
    gate = freeze.gate_dict(holdout_freeze.rule.GateConfig())
    contents = {
        holdout_freeze.MANIFEST: {
            "holdout_sites": HOLDOUT_SITES,
            "splits_sha256": "abc",
        },
        holdout_freeze.CROP_SETTINGS: {"settings": {"frames_per_sequence": 2}},
        holdout_freeze.HUB: {"revision": "r1", "files": {}},
        holdout_freeze.SELECTION: {
            "chosen": "dinov2_probe",
            "gate": gate,
            "challengers": [{"challenger": "dinov2_probe", "mean_gain": 0.07}],
        },
        holdout_freeze.CHALLENGERS: {"caches": {"bioclip2": meta, "dinov2": meta}},
    }
    for name, content in contents.items():
        (root / name).write_text(json.dumps(content))

    config = holdout_freeze.build_config(root, "2026-01-01")
    # The real draws are too slow for a unit test.
    config["gate"] |= {"null_draws": 500, "resamples": 999}
    path = root / holdout_freeze.OUT_PATH
    path.parent.mkdir()
    monkeypatch.setattr(holdout, "FROZEN_SHA256", freeze.write(path, config))

    cache_root = tmp_path / "caches"
    for backbone, cache in make_caches(index, meta).items():
        target = store.cache_path(backbone, cache_root)
        target.parent.mkdir(parents=True)
        extras = {k: cache[k] for k in ("text", "classes", "logit_scale") if k in cache}
        store.write(target, cache["paths"], cache["vectors"], meta, extras)
    return root, cache_root, index, config, freeze.load(path)[1]


def arguments(root, cache_root, *extra):
    """Return the command line for a run inside the synthetic repository."""
    return ["--root", str(root), "--cache-root", str(cache_root), *extra]


def test_the_command_pins_the_committed_freeze():
    """The scoring command and the freeze's tests pin the same hash."""
    assert holdout.FROZEN_SHA256 == FROZEN_SHA256


def test_real_index_matches_the_freeze_without_scoring():
    """The committed index's hold-out is the frozen one, and is not trained on."""
    config, _ = freeze.load(ROOT / holdout_freeze.OUT_PATH)
    index = crops.load_index(ROOT / holdout_freeze.INDEX)
    train, test = holdout.split_frames(index, config["holdout_sites"])
    assert test["site"].n_unique() == 10
    assert not set(train["site"].to_list()) & set(config["holdout_sites"])


def test_a_clear_improvement_is_promoted(repo):
    """A probe that learns what zero-shot misses passes the gate."""
    _, _, index, config, gate = repo
    caches = make_caches(index, config["challenger"]["embedding"])
    summary, per_site, sequences, head = holdout.run(index, caches, config, gate)
    assert summary["promote"]
    assert summary["mean_gain"] > 0.2
    assert per_site.height == len(HOLDOUT_SITES)
    assert sequences["dinov2_probe"].height == 6 * 30
    assert head["coef"].shape == (3, 3)


def test_a_challenger_that_fails_on_the_hold_out_is_rejected(repo):
    """A probe that did well in training but not on the hold-out keeps the champion."""
    _, _, index, config, gate = repo
    caches = make_caches(
        index, config["challenger"]["embedding"], holdout_informative=False
    )
    summary, _, _, _ = holdout.run(index, caches, config, gate)
    assert not summary["promote"]
    assert summary["reasons"]


def test_the_probe_never_sees_the_hold_out(repo):
    """Changing every hold-out embedding leaves the fitted head identical."""
    _, _, index, config, gate = repo
    meta = config["challenger"]["embedding"]
    first = holdout.run(index, make_caches(index, meta), config, gate)
    second = holdout.run(
        index, make_caches(index, meta, holdout_informative=False), config, gate
    )
    assert np.array_equal(first[3]["coef"], second[3]["coef"])


def test_the_decision_is_reproducible(repo):
    """The same inputs give the same decision and predictions."""
    _, _, index, config, gate = repo
    caches = make_caches(index, config["challenger"]["embedding"])
    first = holdout.run(index, caches, config, gate)
    second = holdout.run(index, caches, config, gate)
    assert first[0] == second[0]
    assert first[2]["dinov2_probe"].equals(second[2]["dinov2_probe"])


def test_another_hold_out_is_refused():
    """An index whose hold-out differs from the frozen sites cannot be scored."""
    with pytest.raises(ValueError, match="not the frozen"):
        holdout.split_frames(make_index(), ["H0", "H1"])


def test_a_frozen_site_in_training_is_refused():
    """A frozen site among the training frames would be a leak."""
    index = make_index().with_columns(
        split=pl.when(pl.col("seq_id") == "H0-0")
        .then(pl.lit("fold0"))
        .otherwise("split")
    )
    with pytest.raises(ValueError, match="training frames"):
        holdout.split_frames(index, HOLDOUT_SITES)


def test_other_embeddings_are_refused(repo):
    """A cache made another way is not the one the freeze recorded."""
    _, _, index, config, gate = repo
    caches = make_caches(index, {"index_sha256": "other"})
    with pytest.raises(ValueError, match="does not match"):
        holdout.run(index, caches, config, gate)


def test_a_cache_missing_frames_is_refused(repo):
    """Every frame needs an embedding before anything is fitted."""
    _, _, index, config, gate = repo
    caches = make_caches(index, config["challenger"]["embedding"])
    caches["dinov2"]["paths"] = caches["dinov2"]["paths"][:-1]
    with pytest.raises(ValueError, match="lacks frames"):
        holdout.run(index, caches, config, gate)


def test_an_edited_freeze_is_refused(repo):
    """A changed setting stops the run before anything is read."""
    root, cache_root, *_ = repo
    path = root / holdout_freeze.OUT_PATH
    path.write_text(path.read_text().replace('"margin": -0.01', '"margin": -0.1'))
    with pytest.raises(ValueError, match="edited"):
        holdout.main(arguments(root, cache_root))


def test_a_changed_input_is_refused(repo):
    """A selection result changed after the freeze stops the run."""
    root, cache_root, *_ = repo
    (root / holdout_freeze.SELECTION).write_text("{}")
    with pytest.raises(ValueError, match="frozen inputs changed"):
        holdout.main(arguments(root, cache_root))


def test_check_writes_nothing(repo):
    """The preflight passes without producing any result."""
    root, cache_root, *_ = repo
    assert holdout.main(arguments(root, cache_root, "--check")) == 0
    assert not (root / holdout.OUT_DIR).exists()


def test_main_writes_the_result_and_refuses_a_second_run(repo):
    """One shot: the result is written once, and only another folder reruns it."""
    root, cache_root, *_ = repo
    assert holdout.main(arguments(root, cache_root)) == 0
    out = root / holdout.OUT_DIR
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fold_sites_mean_gain"] == 0.07
    for name in ("per_site.csv", "probe_head.npz", "sequences_bioclip2.csv"):
        assert (out / name).exists()
    assert (root / "figures/holdout_gate.png").exists()
    before = (out / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        holdout.main(arguments(root, cache_root))
    assert (out / "summary.json").read_text() == before
    assert holdout.main(arguments(root, cache_root, "--out-dir", "elsewhere")) == 0
    again = json.loads((root / "elsewhere/summary.json").read_text())
    assert again == summary
