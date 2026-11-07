"""Tests for the CNN control's training: splits, resuming and the hold-out guard."""

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

# pylint: disable=wrong-import-position
from PIL import Image

from cameratrap.arms import cnn

CLASSES = ["agouti", "tapir"]
RECIPE = cnn.RECIPE | {"epochs": 2, "batch_size": 4, "seed": 1}


def make_frames(crop_dir):
    """Write a small crop store of two classes over three folds.

    Args:
        crop_dir: Root of the crop store, a ``pathlib.Path``.

    Returns:
        The frames, sorted by path.
    """
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(3):
        for i in range(8):
            label = CLASSES[i % 2]
            path = f"S{fold}/{i}.jpg"
            (crop_dir / f"S{fold}").mkdir(parents=True, exist_ok=True)
            shade = 200 if label == "tapir" else 40
            pixels = rng.integers(shade - 30, shade + 30, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(crop_dir / path)
            rows.append(
                {
                    "path": path,
                    "site": f"S{fold}",
                    "split": f"fold{fold}",
                    "label": label,
                }
            )
    return pl.DataFrame(rows).sort("path")


def tiny_builder(n_classes, _recipe):
    """Build a tiny model and tensor transform, so tests need no download.

    Args:
        n_classes: Number of output classes.
        _recipe: Unused; matches the real builder's signature.

    Returns:
        A tuple ``(model, train_transform, score_transform)``.
    """

    def transform(image):
        """Turn a PIL image into a float tensor.

        Args:
            image: A PIL image.

        Returns:
            A channels-first tensor in [0, 1].
        """
        return torch.as_tensor(np.array(image)).permute(2, 0, 1).float() / 255

    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Dropout(0.1), torch.nn.Linear(192, n_classes)
    )
    return model, transform, transform


def fold_kwargs(crop_dir, cache):
    """Return the keyword arguments every ``run_fold`` call shares.

    Args:
        crop_dir: Root of the crop store.
        cache: The cache folder, created here.

    Returns:
        A dict of keyword arguments.
    """
    cache.mkdir(parents=True, exist_ok=True)
    return {"crop_dir": crop_dir, "cache": cache, "recipe": RECIPE, "workers": 0}


def test_class_weights_match_balanced_formula():
    """Class weights follow n / (k * count), the probe's balancing."""
    weights = cnn.class_weights(["a", "a", "a", "b"], ["a", "b"])
    assert weights == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])


def test_class_weights_refuse_a_missing_class():
    """A class with no training rows stops the fit rather than dividing by zero."""
    with pytest.raises(ValueError, match="no training rows"):
        cnn.class_weights(["a"], ["a", "b"])


def test_fold_plan_drops_holdout_rows_and_refuses_a_frozen_site():
    """Hold-out rows are excluded, and a frozen site relabelled as a fold is refused."""
    index = pl.DataFrame(
        {
            "path": ["b", "a", "c"],
            "site": ["S0", "S1", "H0"],
            "split": ["fold0", "fold1", "holdout"],
        }
    )
    assert cnn.fold_plan(index, ["H0"])["path"].to_list() == ["a", "b"]
    with pytest.raises(ValueError, match="hold-out site"):
        cnn.fold_plan(index, ["S0"])


def test_fold_plan_refuses_a_site_in_two_folds():
    """A site split across folds would be scored by a model that saw it."""
    index = pl.DataFrame(
        {"path": ["a", "b"], "site": ["S0", "S0"], "split": ["fold0", "fold1"]}
    )
    with pytest.raises(ValueError, match="more than one fold"):
        cnn.fold_plan(index, [])


def test_recipe_change_gets_a_new_cache_folder():
    """A changed recipe or crop index never resumes from stale weights."""
    base = cnn.recipe_digest(RECIPE, "index")
    assert base != cnn.recipe_digest(RECIPE | {"epochs": 3}, "index")
    assert base != cnn.recipe_digest(RECIPE, "other")


def test_run_fold_scores_only_its_fold_and_cleans_up(tmp_path):
    """A fold's logits cover its own frames and its finished checkpoint is removed."""
    frames = make_frames(tmp_path / "crops")
    kwargs = fold_kwargs(tmp_path / "crops", tmp_path / "cache")
    out = cnn.run_fold(frames, "fold1", CLASSES, tiny_builder, **kwargs)
    with np.load(out) as stored:
        assert set(stored["paths"].tolist()) == {f"S1/{i}.jpg" for i in range(8)}
        assert stored["logits"].shape == (8, 2)
    assert not (tmp_path / "cache" / "checkpoint_fold1.pt").exists()


def test_run_fold_reuses_finished_logits(tmp_path):
    """A finished fold is not fitted again."""
    frames = make_frames(tmp_path / "crops")
    kwargs = fold_kwargs(tmp_path / "crops", tmp_path / "cache")
    out = cnn.run_fold(frames, "fold0", CLASSES, tiny_builder, **kwargs)

    def refuse(*_):
        """Fail if a builder is called.

        Args:
            *_: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("fitted again")

    assert cnn.run_fold(frames, "fold0", CLASSES, refuse, **kwargs) == out


def test_resumed_fit_matches_an_uninterrupted_one(tmp_path, monkeypatch):
    """A kill after an epoch's checkpoint, then a rerun, gives the logits of no kill."""
    frames = make_frames(tmp_path / "crops")
    whole = cnn.run_fold(
        frames,
        "fold0",
        CLASSES,
        tiny_builder,
        **fold_kwargs(tmp_path / "crops", tmp_path / "whole"),
    )
    interrupted = fold_kwargs(tmp_path / "crops", tmp_path / "parts")
    real_save = cnn.save_atomic

    def save_then_die(save, path):
        """Write the checkpoint, then stop as a kill would.

        Args:
            save: As ``save_atomic``.
            path: As ``save_atomic``.

        Raises:
            KeyboardInterrupt: Always, after saving.
        """
        real_save(save, path)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(cnn, "save_atomic", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            cnn.run_fold(frames, "fold0", CLASSES, tiny_builder, **interrupted)
    assert torch.load(tmp_path / "parts" / "checkpoint_fold0.pt")["epoch"] == 0

    resumed = cnn.run_fold(frames, "fold0", CLASSES, tiny_builder, **interrupted)
    with np.load(whole) as a, np.load(resumed) as b:
        assert a["paths"].tolist() == b["paths"].tolist()
        assert np.array_equal(a["logits"], b["logits"])


def test_merge_refuses_incomplete_folds(tmp_path):
    """Logits missing a fold are refused rather than silently scoring fewer frames."""
    frames = make_frames(tmp_path / "crops")
    kwargs = fold_kwargs(tmp_path / "crops", tmp_path / "cache")
    one = cnn.run_fold(frames, "fold0", CLASSES, tiny_builder, **kwargs)
    with pytest.raises(ValueError, match="do not cover"):
        cnn.merge(frames, [one], CLASSES, {}, tmp_path / "out.npz")


def test_merge_restores_row_order(tmp_path):
    """Merged logits follow the frames' order and carry the classes and metadata."""
    frames = make_frames(tmp_path / "crops")
    kwargs = fold_kwargs(tmp_path / "crops", tmp_path / "cache")
    files = [
        cnn.run_fold(frames, fold, CLASSES, tiny_builder, **kwargs)
        for fold in ("fold2", "fold0", "fold1")
    ]
    out = tmp_path / "out.npz"
    cnn.merge(frames, files, CLASSES, {"note": "x"}, out)
    with np.load(out) as stored:
        assert stored["paths"].tolist() == frames["path"].to_list()
        assert stored["classes"].tolist() == CLASSES
        assert "note" in stored["meta"].item()
