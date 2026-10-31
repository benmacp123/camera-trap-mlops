"""Embed the stored crops with BioCLIP 2 or DINOv2, resumably, into one cache each.

Both backbones read the same JPEG crops from ``data/crops/`` (Phase 1), so
their caches describe identical pixels. The cache records the crop
index's SHA-256 and the weight revision; chunks are kept under a folder
named by the index hash, so a changed index never resumes from stale
chunks.

Run as a script (BioCLIP 2 about 6 hours on CPU, DINOv2 about 1)::

    PYTHONPATH=src python -m cameratrap.embed.backbones bioclip2
    PYTHONPATH=src python -m cameratrap.embed.backbones dinov2
    PYTHONPATH=src python -m cameratrap.embed.backbones bioclip2 --check 50
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
from huggingface_hub import try_to_load_from_cache
from PIL import Image

from cameratrap.data import crops, orinoquia
from cameratrap.embed import bioclip, store

DINOV2_ID = "timm/vit_base_patch14_dinov2.lvd142m"
# DINOv2's default 518 px runs at 1 crop/s on CPU; 224 px runs at 7.5 and
# suits crops whose long side is capped at 448. Fixed before Phase 2.
DINOV2_SIZE = 224
CHUNK = 512
# Re-embedded vectors must match the cache to this cosine; CPU float maths
# differs slightly with thread count, so exact equality is not expected.
CHECK_TOLERANCE = 0.9999
CHECK_DIR = Path("results/embeddings")


@dataclasses.dataclass
class Backbone:
    """A loaded image encoder and what its cache needs to record.

    Attributes:
        name: Short name, used for the cache folder.
        model: The model, in eval mode.
        preprocess: Transform from a PIL image to an input tensor.
        encode: Function from a batch tensor to feature tensor.
        meta: How the vectors are made: model ID, weight revision, input size.
        extras: Further arrays the cache stores alongside the vectors.
    """

    name: str
    model: torch.nn.Module
    preprocess: object
    encode: object
    meta: dict
    extras: dict


def hub_revision(repo_id, filename):
    """Return the commit hash of a Hugging Face repo as cached locally.

    Read from the snapshot folder name, so it needs no network and records
    exactly the weights that were loaded.

    Args:
        repo_id: Repo such as ``imageomics/bioclip-2``.
        filename: A file the loader downloaded from it.

    Returns:
        The commit hash, or ``None`` if the file is not cached.
    """
    path = try_to_load_from_cache(repo_id, filename)
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1]


def load_bioclip2():
    """Load BioCLIP 2 with its class-prompt embeddings for zero-shot.

    Returns:
        A ``Backbone`` whose extras hold the text embeddings, classes,
        prompts and logit scale, so the zero-shot arm needs no torch.
    """
    model, preprocess, tokenizer = bioclip.load_model()
    classes = list(orinoquia.TAXA)
    prompts = [bioclip.prompt(orinoquia.TAXA[c]) for c in classes]
    repo = bioclip.MODEL_ID.removeprefix("hf-hub:")
    return Backbone(
        name="bioclip2",
        model=model,
        preprocess=preprocess,
        encode=model.encode_image,
        meta={
            "model_id": repo,
            "revision": hub_revision(repo, "open_clip_config.json"),
            "library": f"open_clip {open_clip.__version__}",
            "input_size": 224,
        },
        extras={
            "text": bioclip.embed_text(model, tokenizer, prompts),
            "classes": np.array(classes),
            "prompts": np.array(prompts),
            "logit_scale": np.float32(model.logit_scale.exp().item()),
        },
    )


def load_dinov2():
    """Load DINOv2 ViT-B/14 at 224 px with its pooled (CLS) output.

    Returns:
        A ``Backbone`` with no extras.
    """
    model = timm.create_model(
        DINOV2_ID.removeprefix("timm/"),
        pretrained=True,
        num_classes=0,
        img_size=DINOV2_SIZE,
    ).eval()
    # The pretrained config still says 518 px after img_size=224, so the
    # input size is passed explicitly or the transform would resize to 518.
    config = timm.data.resolve_data_config(
        {"input_size": (3, DINOV2_SIZE, DINOV2_SIZE)}, model=model
    )
    return Backbone(
        name="dinov2",
        model=model,
        preprocess=timm.data.create_transform(**config),
        encode=model,
        meta={
            "model_id": DINOV2_ID,
            "revision": hub_revision(DINOV2_ID, "model.safetensors"),
            "library": f"timm {timm.__version__}",
            "input_size": DINOV2_SIZE,
        },
        extras={},
    )


LOADERS = {"bioclip2": load_bioclip2, "dinov2": load_dinov2}


class StoredCrops(torch.utils.data.Dataset):
    """Dataset of stored crops, decoded and preprocessed in DataLoader workers."""

    def __init__(self, paths, crop_dir, preprocess):
        """Store what each item needs.

        Args:
            paths: Crop paths relative to ``crop_dir``.
            crop_dir: Root of the crop store.
            preprocess: The backbone's image transform.
        """
        self.paths = paths
        self.crop_dir = crop_dir
        self.preprocess = preprocess

    def __len__(self):
        """Return the number of crops.

        Returns:
            The number of paths.
        """
        return len(self.paths)

    def __getitem__(self, index):
        """Return one preprocessed crop.

        Args:
            index: Position in ``paths``.

        Returns:
            The crop as a tensor ready for the model.
        """
        with Image.open(self.crop_dir / self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def embed_paths(backbone, paths, crop_dir, batch_size=32, workers=4):
    """Return L2-normalised embeddings for some stored crops.

    Args:
        backbone: A loaded ``Backbone``.
        paths: Crop paths relative to ``crop_dir``.
        crop_dir: Root of the crop store.
        batch_size: Crops per forward pass.
        workers: DataLoader worker processes decoding JPEGs.

    Returns:
        A float32 array with one row per path.
    """
    loader = torch.utils.data.DataLoader(
        StoredCrops(paths, crop_dir, backbone.preprocess),
        batch_size=batch_size,
        num_workers=workers,
    )
    rows = []
    with torch.no_grad():
        for batch in loader:
            features = backbone.encode(batch)
            rows.append(torch.nn.functional.normalize(features, dim=-1).numpy())
    return np.concatenate(rows).astype(np.float32)


def cache_meta(backbone, index_path):
    """Return the full meta a cache records.

    Args:
        backbone: A loaded ``Backbone``.
        index_path: The crop index the paths come from.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        "backbone": backbone.name,
        **backbone.meta,
        "index": str(index_path),
        "index_sha256": store.file_sha256(index_path),
        "crop_max_side": crops.MAX_SIDE,
        "crop_jpeg": crops.JPEG_OPTIONS,
        "normalised": "l2",
        "torch": torch.__version__,
    }


def run(backbone, paths, args):
    """Embed every crop in chunks, resuming, then write the merged cache.

    Args:
        backbone: A loaded ``Backbone``.
        paths: Crop paths in index order.
        args: Parsed command-line arguments.
    """
    meta = cache_meta(backbone, args.index)
    chunk_dir = args.out_root / backbone.name / "chunks" / meta["index_sha256"][:12]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(paths), CHUNK):
        chunk_path = chunk_dir / f"chunk_{start:06d}.npy"
        chunks.append(chunk_path)
        if chunk_path.exists():
            continue
        began = time.time()
        vectors = embed_paths(
            backbone,
            paths[start : start + CHUNK],
            args.crop_dir,
            args.batch_size,
            args.workers,
        )
        # Saved via a temporary name so a stopped run never leaves a
        # partial chunk that a resume would trust.
        partial = chunk_path.with_name(chunk_path.stem + ".part.npy")
        np.save(partial, vectors)
        partial.replace(chunk_path)
        done = min(start + CHUNK, len(paths))
        rate = (done - start) / (time.time() - began)
        print(f"{backbone.name}: {done}/{len(paths)} crops, {rate:.2f}/s", flush=True)
    store.write(
        store.cache_path(backbone.name, args.out_root),
        paths,
        np.concatenate([np.load(c) for c in chunks]),
        meta,
        backbone.extras,
    )


def check(backbone, args):
    """Re-embed a random sample of cached crops and compare with the cache.

    Args:
        backbone: A loaded ``Backbone``.
        args: Parsed command-line arguments; ``args.check`` is the sample size.

    Returns:
        0 if every sampled crop agrees within ``CHECK_TOLERANCE``, else 1.
    """
    cache = store.read(store.cache_path(backbone.name, args.out_root))
    store.check_meta(cache["meta"], cache_meta(backbone, args.index))
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(cache["paths"], size=args.check, replace=False))
    fresh = embed_paths(backbone, sample, args.crop_dir, args.batch_size, workers=0)
    agreement = store.cosine_agreement(store.rows_for(cache, sample), fresh)
    result = {
        "backbone": backbone.name,
        "sample": len(sample),
        "seed": args.seed,
        "min_cosine": float(agreement.min()),
        "tolerance": CHECK_TOLERANCE,
        "passed": bool(agreement.min() >= CHECK_TOLERANCE),
    }
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    out = CHECK_DIR / f"{backbone.name}_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


def main(argv=None):
    """Embed the indexed crops with one backbone, or check a sample of its cache.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("backbone", choices=sorted(LOADERS))
    parser.add_argument("--index", type=Path, default=crops.INDEX_PATH)
    parser.add_argument("--crop-dir", type=Path, default=crops.CROP_DIR)
    parser.add_argument("--out-root", type=Path, default=store.CACHE_ROOT)
    parser.add_argument("--limit", type=int, help="embed only the first N crops")
    parser.add_argument("--check", type=int, help="re-embed N cached crops instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    backbone = LOADERS[args.backbone]()
    if args.check:
        return check(backbone, args)
    paths = crops.load_index(args.index)["path"].to_list()[: args.limit]
    run(backbone, paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
