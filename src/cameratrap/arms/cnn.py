"""The CNN control: a full fine-tune of EfficientNet-B0, out of fold by site.

A control is an arm expected to lose, there to show what the cheap
frozen-embedding route is beating. It uses the same stored crops, frames and
site folds as the probes, and the same class balancing, so a loss cannot be
blamed on the imbalance. Its result is reported only and can move no alias.

The recipe was written down before any fold was scored and has no tuning
step: a fixed number of epochs and no early stopping, so nothing in it can
be fitted to the scored sites. Hold-out frames are neither fitted nor scored.

Training takes about 16 hours on an 8-core CPU, so each fold checkpoints
every epoch and resumes. Each epoch seeds itself from its own number, which
makes a resumed run match an uninterrupted one without storing RNG state.

Writes per-fold logits under ``data/cache/cnn_control/`` and, once every fold
is done, ``results/control/frame_logits.npz``, which ``report.control`` reads
without torch.

::

    PYTHONPATH=src python -m cameratrap.arms.cnn
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import timm
import torch

from cameratrap.data import crops, orinoquia
from cameratrap.embed import backbones, store
from cameratrap.gate import freeze

MODEL_ID = "timm/efficientnet_b0.ra_in1k"
# Written before any fold was scored (owner, 2026-11-07); see the module
# docstring. The crops are already tight on the animal, so training crops
# keep at least half the area rather than timm's default 8%, and scoring
# uses the whole square crop.
RECIPE = {
    "model_id": MODEL_ID,
    "pretrained": "ImageNet; every layer trainable",
    "input_size": 224,
    "epochs": 15,
    "batch_size": 32,
    "optimiser": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "schedule": "cosine, stepped per batch",
    "loss": "cross-entropy, class-balanced as the probe",
    "train_crop_scale": [0.5, 1.0],
    "score_crop_pct": 1.0,
    "seed": 20261107,
}
CACHE_DIR = Path("data/cache/cnn_control")
OUT_PATH = Path("results/control/frame_logits.npz")
FREEZE_PATH = Path("config/holdout_freeze.json")


def class_weights(labels, classes):
    """Return class-balanced loss weights, as scikit-learn's ``balanced``.

    Args:
        labels: Class name of each training row.
        classes: Class names in logit order.

    Returns:
        A float array of ``n / (k * count)``, one weight per class.

    Raises:
        ValueError: If a class has no training rows, which would leave its
            logit untrained and its weight undefined.
    """
    labels = np.asarray(labels)
    counts = np.array([(labels == name).sum() for name in classes])
    if (counts == 0).any():
        missing = [name for name, count in zip(classes, counts) if count == 0]
        raise ValueError(f"no training rows for {missing}")
    return labels.size / (len(classes) * counts)


def fold_plan(index, holdout_sites):
    """Return the frames the control may use, checked against the freeze.

    Args:
        index: The crop index.
        holdout_sites: The frozen hold-out site names.

    Returns:
        The fold-site frames sorted by path, the same frames the probes use.

    Raises:
        ValueError: If a frozen hold-out site would be used, or a site
            appears in more than one fold.
    """
    frames = index.filter(pl.col("split") != "holdout").sort("path")
    if set(frames["site"].to_list()) & set(holdout_sites):
        raise ValueError("a frozen hold-out site is among the control's frames")
    folds_per_site = frames.group_by("site").agg(pl.col("split").n_unique())
    if (folds_per_site["split"] > 1).any():
        raise ValueError("a site appears in more than one fold")
    return frames


def recipe_digest(recipe, index_sha256):
    """Return a short hash naming a recipe's cache folder.

    A changed recipe or crop index then never resumes from stale weights.

    Args:
        recipe: The recipe dict.
        index_sha256: The crop index's SHA-256.

    Returns:
        The first 12 hex digits of the hash.
    """
    text = json.dumps({"recipe": recipe, "index": index_sha256}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class LabelledCrops(backbones.StoredCrops):
    """Stored crops with their class index, for training."""

    def __init__(self, paths, targets, crop_dir, preprocess):
        """Store what each item needs.

        Args:
            paths: Crop paths relative to ``crop_dir``.
            targets: Class index of each crop.
            crop_dir: Root of the crop store.
            preprocess: The image transform.
        """
        super().__init__(paths, crop_dir, preprocess)
        self.targets = targets

    def __getitem__(self, index):
        """Return one preprocessed crop and its class index.

        Args:
            index: Position in ``paths``.

        Returns:
            A pair ``(tensor, class_index)``.
        """
        return super().__getitem__(index), self.targets[index]


def build_model(n_classes, recipe):
    """Load the pretrained CNN with a new head and its transforms.

    Args:
        n_classes: Number of output classes.
        recipe: The recipe dict.

    Returns:
        A tuple ``(model, train_transform, score_transform)``.
    """
    model = timm.create_model(
        recipe["model_id"].removeprefix("timm/"),
        pretrained=True,
        num_classes=n_classes,
    )
    size = recipe["input_size"]
    config = timm.data.resolve_data_config({"input_size": (3, size, size)}, model=model)
    train_transform = timm.data.create_transform(
        **config, is_training=True, scale=tuple(recipe["train_crop_scale"])
    )
    score_transform = timm.data.create_transform(
        **(config | {"crop_pct": recipe["score_crop_pct"]})
    )
    return model, train_transform, score_transform


def save_atomic(save, path):
    """Write a file through a temporary name, so a kill never leaves half of it.

    Args:
        save: Callable taking the temporary path and writing to it.
        path: The final ``pathlib.Path``.
    """
    temporary = path.with_name(path.name + ".tmp")
    save(temporary)
    temporary.replace(path)


def train_epoch(model, loader, optimiser, scheduler, loss_fn):
    """Train for one epoch.

    Args:
        model: The model.
        loader: Batches of ``(images, targets)``.
        optimiser: The optimiser.
        scheduler: The per-batch learning-rate schedule.
        loss_fn: The loss.

    Returns:
        The mean batch loss.
    """
    model.train()
    total = 0.0
    for images, targets in loader:
        optimiser.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimiser.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def train(
    frames, classes, transform, model, *, crop_dir, checkpoint, recipe, seed, workers
):
    """Fine-tune a model on some frames, resuming from a checkpoint if present.

    Args:
        frames: Training frames, with ``path`` and ``label``.
        classes: Class names in logit order.
        transform: The training transform.
        model: The model, modified in place.
        crop_dir: Root of the crop store.
        checkpoint: Path of this fit's checkpoint.
        recipe: The recipe dict.
        seed: This fit's seed.
        workers: DataLoader workers.

    Returns:
        The trained model.
    """
    labels = frames["label"].to_list()
    data = LabelledCrops(
        frames["path"].to_list(),
        [classes.index(name) for name in labels],
        crop_dir,
        transform,
    )
    weights = torch.as_tensor(class_weights(labels, classes), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )
    # drop_last: a final batch of one crop would break batch normalisation.
    steps = len(data) // recipe["batch_size"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=steps * recipe["epochs"]
    )
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["epoch"] + 1
        print(f"resuming {checkpoint.name} at epoch {start}", flush=True)

    for epoch in range(start, recipe["epochs"]):
        # Everything random in this epoch follows from its own number.
        torch.manual_seed(seed + epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=recipe["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            generator=generator,
        )
        began = time.time()
        loss = train_epoch(model, loader, optimiser, scheduler, loss_fn)
        print(
            f"{checkpoint.stem} epoch {epoch + 1}/{recipe['epochs']}: "
            f"loss {loss:.4f}, {(time.time() - began) / 60:.1f} min",
            flush=True,
        )
        state = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        save_atomic(lambda p, s=state: torch.save(s, p), checkpoint)
    return model


def predict(model, paths, crop_dir, transform, workers, batch_size=64):
    """Return a model's logits for some crops, in order.

    Args:
        model: The trained model.
        paths: Crop paths relative to ``crop_dir``.
        crop_dir: Root of the crop store.
        transform: The scoring transform.
        workers: DataLoader workers.
        batch_size: Crops per batch.

    Returns:
        A float32 array of shape ``(len(paths), n_classes)``.
    """
    loader = torch.utils.data.DataLoader(
        backbones.StoredCrops(paths, crop_dir, transform),
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(batch).numpy() for batch in loader]).astype(
            np.float32
        )


def run_fold(frames, fold, classes, builder, *, crop_dir, cache, recipe, workers):
    """Fit on every other fold and score this fold; skip it if already done.

    Args:
        frames: All fold-site frames.
        fold: The fold to score.
        classes: Class names in logit order.
        builder: Callable ``(n_classes, recipe)`` returning ``(model,
            train_transform, score_transform)``.
        crop_dir: Root of the crop store.
        cache: This recipe's cache folder.
        recipe: The recipe dict.
        workers: DataLoader workers.

    Returns:
        The path of the fold's logits file.

    Raises:
        ValueError: If a scored site would also be trained on.
    """
    out = cache / f"logits_{fold}.npz"
    if out.exists():
        return out
    fit_on = frames.filter(pl.col("split") != fold)
    scored = frames.filter(pl.col("split") == fold)
    if set(fit_on["site"].to_list()) & set(scored["site"].to_list()):
        raise ValueError(f"fold {fold} shares a site with its training frames")
    checkpoint = cache / f"checkpoint_{fold}.pt"
    # Folds must not share a seed; the sorted position is stable across runs.
    seed = recipe["seed"] + 1000 * sorted(set(frames["split"].to_list())).index(fold)
    torch.manual_seed(seed)
    model, train_transform, score_transform = builder(len(classes), recipe)
    train(
        fit_on,
        classes,
        train_transform,
        model,
        crop_dir=crop_dir,
        checkpoint=checkpoint,
        recipe=recipe,
        seed=seed,
        workers=workers,
    )
    paths = scored["path"].to_list()
    logits = predict(model, paths, crop_dir, score_transform, workers)
    save_atomic(
        # np.savez appends .npz to a bare name, so hand it an open file.
        lambda p: _write_npz(p, paths=np.array(paths), logits=logits),
        out,
    )
    checkpoint.unlink()
    return out


def _write_npz(path, **arrays):
    """Write a compressed npz to exactly this path.

    Args:
        path: The destination.
        **arrays: Arrays to store, by name.
    """
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def merge(frames, fold_files, classes, meta, out):
    """Merge per-fold logits into one file, in the frames' row order.

    Args:
        frames: All fold-site frames, sorted by path.
        fold_files: The per-fold logits files.
        classes: Class names in logit order.
        meta: JSON-serialisable description of how the logits were made.
        out: Destination ``pathlib.Path``.

    Raises:
        ValueError: If the folds do not cover exactly the frames.
    """
    by_path = {}
    for path in fold_files:
        with np.load(path) as stored:
            by_path.update(zip(map(str, stored["paths"]), stored["logits"]))
    expected = frames["path"].to_list()
    if set(by_path) != set(expected):
        raise ValueError("the fold logits do not cover exactly the fold-site frames")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_npz(
        out,
        paths=np.array(expected),
        logits=np.stack([by_path[p] for p in expected]),
        classes=np.array(classes),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )


def main(argv=None):
    """Fit the CNN control on every fold and write its frame logits.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=crops.INDEX_PATH)
    parser.add_argument("--crop-dir", type=Path, default=crops.CROP_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    config, _ = freeze.load(FREEZE_PATH)
    frames = fold_plan(crops.load_index(args.index), config["holdout_sites"])
    index_sha = store.file_sha256(args.index)
    classes = list(orinoquia.CLASSES)
    cache = args.cache_dir / recipe_digest(RECIPE, index_sha)
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    fold_files = [
        run_fold(
            frames,
            fold,
            classes,
            build_model,
            crop_dir=args.crop_dir,
            cache=cache,
            recipe=RECIPE,
            workers=args.workers,
        )
        for fold in sorted(set(frames["split"].to_list()))
    ]
    meta = {
        "recipe": RECIPE,
        "index_sha256": index_sha,
        "revision": backbones.hub_revision(MODEL_ID, "model.safetensors"),
        "library": f"timm {timm.__version__}",
        "torch": torch.__version__,
        "threads": args.threads,
    }
    merge(frames, fold_files, classes, meta, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
