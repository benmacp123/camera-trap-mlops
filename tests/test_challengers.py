"""Tests for the challenger report, on synthetic caches where every challenger must win."""

import numpy as np
import polars as pl
import pytest

from cameratrap.data import crops
from cameratrap.report import challengers

CLASSES = ["black_agouti", "spixs_guan", "lowland_tapir"]
SITES = {f"S{i}": f"fold{i // 2}" for i in range(6)}
# Two-frame sequences per site. kNN's largest k (200) needs that many
# training frames in every inner split: two sites of 60 give 240.
SEQUENCES_PER_SITE = 60


def make_inputs(per_site=SEQUENCES_PER_SITE, seed=0):
    """Build a crop index, sizes and two caches over six fold sites and a hold-out.

    Embeddings are noisy one-hot vectors, so a probe separates the classes,
    but BioCLIP 2's text embeddings point tapir at agouti, so zero-shot
    misses every tapir. Hold-out rows get NaN embeddings: training on or
    scoring any of them would fail or poison the results.

    Args:
        per_site: Two-frame sequences per site, classes in turn.
        seed: Seed for the noise.

    Returns:
        A tuple ``(index, sizes, caches)``.
    """
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for site, split in (*SITES.items(), ("Z99", "holdout")):
        for s in range(per_site):
            label = CLASSES[s % 3]
            for f in range(2):
                rows.append(
                    (f"{site}/{s:03d}_{f}.JPG", f"{site}-{s}", site, split, label, f)
                )
                if split == "holdout":
                    vectors.append(np.full(3, np.nan))
                else:
                    vectors.append(
                        np.eye(3)[CLASSES.index(label)] + rng.normal(0, 0.2, 3)
                    )
    index = pl.DataFrame(
        rows,
        schema=["path", "seq_id", "site", "split", "label", "frame_num"],
        orient="row",
    ).with_columns(
        conf=pl.lit(0.9),
        person_conf=pl.lit(0.0),
        bbox_x=pl.lit(0.1),
        bbox_y=pl.lit(0.1),
        bbox_w=pl.lit(0.2),
        bbox_h=pl.lit(0.2),
    )
    sizes = pl.DataFrame({"path": index["path"], "side": [300] * index.height})
    vectors = np.array(vectors, dtype=np.float32)
    paths = index["path"].to_list()
    caches = {
        "bioclip2": {
            "paths": paths,
            "vectors": vectors,
            # Tapir's text embedding sits on agouti's axis.
            "text": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            "classes": np.array(CLASSES),
            "logit_scale": np.float32(10.0),
            "meta": {"backbone": "bioclip2"},
        },
        "dinov2": {"paths": paths, "vectors": vectors, "meta": {"backbone": "dinov2"}},
    }
    return index.select(list(crops.INDEX_SCHEMA)), sizes, caches


@pytest.fixture(name="result")
def fixture_result():
    """Return ``challengers.run`` on the synthetic inputs.

    Returns:
        The tuple ``(summary, per_site, sequences)``.
    """
    return challengers.run(*make_inputs())


def test_holdout_is_never_read(result):
    """No hold-out site or sequence appears in any table, and nothing is NaN."""
    _, per_site, sequences, _ = result
    assert "Z99" not in per_site["site"].to_list()
    for table in sequences.values():
        assert "Z99" not in table["site"].to_list()
        assert not table["msp"].is_nan().any()


def test_challengers_beat_a_flawed_zero_shot(result):
    """Probes and kNN recover the class zero-shot always misses; zero-shot scores 2/3."""
    summary, _, _, _ = result
    arms = summary["arms"]
    assert arms["bioclip2"]["mean_site_balanced_accuracy"] == pytest.approx(2 / 3)
    for backbone in ("bioclip2", "dinov2"):
        for method, setting in (("probe", "c"), ("knn", "k")):
            arm = arms[f"{backbone}_{method}"]
            assert arm["mean_site_balanced_accuracy"] > 0.95
            assert arm["sites_improved"] == len(SITES)
            assert set(arm[f"chosen_{setting}_per_fold"]) == {"fold0", "fold1", "fold2"}


def test_knn_curves_cover_every_fold_and_k(result):
    """The validation table has one row per backbone, outer fold and candidate k."""
    _, _, _, curves = result
    assert curves.height == 2 * 3 * len(challengers.SETTINGS["knn_ks"])


def test_gain_is_paired_per_site(result):
    """Each site's gain is the probe's score minus the champion's at that site."""
    _, per_site, _, _ = result
    expected = (
        per_site["bioclip2_probe_balanced_accuracy"]
        - per_site["bioclip2_balanced_accuracy"]
    )
    assert np.allclose(per_site["bioclip2_probe_gain"], expected)


def test_one_prediction_per_sequence(result):
    """Every arm scores the same sequences, one row each."""
    _, _, sequences, _ = result
    seq_ids = {tuple(t["seq_id"].to_list()) for t in sequences.values()}
    assert len(seq_ids) == 1
    assert len(next(iter(seq_ids))) == len(SITES) * SEQUENCES_PER_SITE


def test_main_writes_outputs(tmp_path, monkeypatch):
    """The command writes the tables and figure from caches on disk."""
    index, sizes, caches = make_inputs()
    index_path, sizes_path = tmp_path / "index.csv", tmp_path / "sizes.csv"
    index.write_csv(index_path)
    sizes.write_csv(sizes_path)
    sha = challengers.store.file_sha256(index_path)
    for backbone, cache in caches.items():
        extras = {
            k: v for k, v in cache.items() if k not in ("paths", "vectors", "meta")
        }
        challengers.store.write(
            challengers.store.cache_path(backbone, tmp_path / "cache"),
            cache["paths"],
            cache["vectors"],
            cache["meta"] | {"index_sha256": sha},
            extras,
        )
    monkeypatch.setattr(crops, "load_index", lambda _: index)
    monkeypatch.setattr(crops, "load_sizes", lambda _: sizes)
    status = challengers.main(
        [
            "--index",
            str(index_path),
            "--sizes",
            str(sizes_path),
            "--cache-root",
            str(tmp_path / "cache"),
            "--out-dir",
            str(tmp_path / "out"),
            "--figure-dir",
            str(tmp_path / "fig"),
        ]
    )
    assert status == 0
    assert (tmp_path / "out" / "sequences_dinov2_knn.csv").exists()
    assert (tmp_path / "out" / "knn_validation.csv").exists()
    assert (tmp_path / "fig" / "challengers_gain.png").exists()
    assert (tmp_path / "fig" / "challengers_knn_k.png").exists()
