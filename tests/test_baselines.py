"""Tests for the zero-shot baseline report, on synthetic caches small enough to check by hand."""

import numpy as np
import polars as pl
import pytest

from cameratrap.data import crops
from cameratrap.report import baselines

CLASSES = ["black_agouti", "spixs_guan"]


def make_inputs(sites=("A01", "B02"), per_site=20):
    """Build a crop index, sizes, BioCLIP 2 cache and SpeciesNet answers.

    Each fold site gets ``per_site`` two-frame sequences, half agouti and
    half guan. BioCLIP 2 gets every sequence right except that the first
    guan sequence at each site has one confidently wrong frame and one
    weakly right one, so averaging logits across the sequence still gets
    it wrong. SpeciesNet gets every frame right and abstains on agouti in
    the open set. One hold-out site is added with no SpeciesNet answers,
    so any read of it fails loudly.

    Args:
        sites: Fold site names.
        per_site: Sequences per fold site.

    Returns:
        A tuple ``(index, sizes, bioclip, speciesnet)``.
    """
    rows, vectors, predictions = [], [], {}
    text = np.eye(2, dtype=np.float32)
    for site in (*sites, "Z99"):
        split = "holdout" if site == "Z99" else "fold0"
        for s in range(per_site):
            label = CLASSES[s % 2]
            for f in range(2):
                path = f"{site}/{s:03d}_{f}.JPG"
                rows.append((path, f"{site}-{s}", site, split, label, f))
                truth = np.eye(2)[CLASSES.index(label)]
                if label == "spixs_guan" and s == 1:
                    # Frame 0 strongly wrong, frame 1 weakly right: the
                    # sequence mean favours agouti.
                    vec = np.array([1.0, 0.0]) if f == 0 else np.array([0.45, 0.55])
                else:
                    vec = truth
                vectors.append(vec)
                if split != "holdout":
                    predictions[path] = {
                        "logits": truth * 5.0,
                        "open_label": None if label == "black_agouti" else label,
                        "source": "classifier",
                    }
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
    sides = [100 if p.endswith("000_0.JPG") else 300 for p in index["path"]]
    sizes = pl.DataFrame({"path": index["path"], "side": sides})
    bioclip = {
        "paths": index["path"].to_list(),
        "vectors": np.array(vectors, dtype=np.float32),
        "text": text,
        "classes": np.array(CLASSES),
        "logit_scale": np.float32(10.0),
        "meta": {"backbone": "bioclip2"},
    }
    return index.select(list(crops.INDEX_SCHEMA)), sizes, bioclip, predictions


@pytest.fixture(name="result")
def fixture_result():
    """Return ``baselines.run`` on the synthetic inputs.

    Returns:
        The pair ``(summary, tables)``.
    """
    return baselines.run(*make_inputs())


def test_size_bins():
    """Crop sides fall into the documented bins, edges going up."""
    edges = [112, 224]
    assert [baselines.size_bin(s, edges) for s in (50, 112, 223, 224, 448)] == [
        "<112",
        "112-224",
        "112-224",
        ">=224",
        ">=224",
    ]


def test_holdout_is_never_scored(result):
    """Hold-out sites are absent from every table, even with no SpeciesNet answers."""
    summary, tables = result
    assert summary["sequences"] == 40
    assert set(tables["per_site"]["site"]) == {"A01", "B02"}


def test_sequence_averaging_and_recall(result):
    """The sequence mean decides: one mixed guan sequence per site is wrong for BioCLIP 2."""
    _, tables = result
    recall = dict(tables["per_class"].select("label", "bioclip2_recall").iter_rows())
    assert recall == {"black_agouti": 1.0, "spixs_guan": pytest.approx(0.9)}
    site = tables["per_site"].row(0, named=True)
    assert site["bioclip2_balanced_accuracy"] == pytest.approx(0.95)
    assert site["speciesnet_balanced_accuracy"] == 1.0
    confusion = tables["confusion_bioclip2"]
    assert confusion.filter(pl.col("label") == "spixs_guan")["black_agouti"][0] == 2


def test_open_set_counts_abstentions(result):
    """SpeciesNet abstaining on agouti halves coverage but not accuracy when answering."""
    summary, tables = result
    assert summary["speciesnet_open_set"] == {
        "coverage": 0.5,
        "accuracy_when_answering": 1.0,
    }
    assert tables["per_site"]["speciesnet_open_coverage"].to_list() == [0.5, 0.5]


def test_by_crop_size_is_frame_level(result):
    """Small crops are counted per frame, separately from large ones."""
    _, tables = result
    small = tables["by_crop_size"].filter(
        (pl.col("arm") == "bioclip2") & (pl.col("size_bin") == "<112")
    )
    assert small["n"].to_list() == [2]


def test_missing_speciesnet_frame_is_refused():
    """A fold frame without a SpeciesNet answer stops the report instead of skewing it."""
    index, sizes, bioclip, predictions = make_inputs()
    predictions.pop("A01/000_0.JPG")
    with pytest.raises(ValueError, match="no answer for 1 frames"):
        baselines.run(index, sizes, bioclip, predictions)


def test_inconsistent_sequence_is_refused():
    """A sequence whose frames disagree on label is an error, not a silent pick."""
    index, sizes, bioclip, predictions = make_inputs()
    index = index.with_columns(
        pl.when(pl.col("path") == "A01/000_1.JPG")
        .then(pl.lit("spixs_guan"))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    with pytest.raises(ValueError, match="disagree"):
        baselines.run(index, sizes, bioclip, predictions)


def test_figures_are_written(result, tmp_path):
    """Both figures render to PNG files from the tables alone."""
    _, tables = result
    baselines.plot_sites(tables["per_site"], tmp_path / "sites.png")
    baselines.plot_confusion(tables["confusion_bioclip2"], tmp_path / "cm.png", "t")
    assert (tmp_path / "sites.png").stat().st_size > 0
    assert (tmp_path / "cm.png").stat().st_size > 0
