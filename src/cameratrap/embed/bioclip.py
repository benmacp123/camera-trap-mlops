"""BioCLIP 2 crop embeddings and class-prompt embeddings, cached to disk.

The cache holds everything the zero-shot arm and the probes need, so
nothing downstream loads torch: L2-normalised image embeddings, one
normalised text embedding per class prompt, and the model's logit scale.
Zero-shot logits are then ``scale * image @ text.T``.

Embedding runs in chunks saved as they finish, so a stopped run resumes.

Run as a script to embed the spike's classify images (about 2 hours)::

    PYTHONPATH=src python -m cameratrap.embed.bioclip
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import polars as pl
import torch

from cameratrap.data import fetch, manifest, orinoquia, subset
from cameratrap.data.crops import crop, top_animal

MODEL_ID = "hf-hub:imageomics/bioclip-2"
CACHE_DIR = Path("data/cache/bioclip2")
CHUNK = 512


def prompt(taxon):
    """Return the zero-shot prompt for one class.

    Uses BioCLIP's "taxonomic and common name" text form, the richest of
    the forms it was trained on, with the lineage from the SpeciesNet
    taxonomy. Fixed before the spike (2026-10-28).

    Args:
        taxon: ``(scientific, common, speciesnet_label)`` for the class.

    Returns:
        The prompt string.
    """
    _, common, label = taxon
    _, cls, order, family, genus, species, _ = label.split(";")
    lineage = " ".join(
        ["Animalia", "Chordata"] + [r.capitalize() for r in (cls, order, family, genus)]
    )
    return f"a photo of {lineage} {species} with common name {common}."


class Crops(torch.utils.data.Dataset):
    """Dataset of preprocessed crops, decoded in DataLoader workers."""

    def __init__(self, paths, image_dir, detections, preprocess):
        """Store what each item needs.

        Args:
            paths: Manifest paths, each with at least one animal detection.
            image_dir: Local root the manifest paths are stored under.
            detections: Mapping of manifest path to MDv5a detections.
            preprocess: The model's image transform.
        """
        self.paths = paths
        self.image_dir = image_dir
        self.detections = detections
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
        path = self.paths[index]
        box = top_animal(self.detections[path])
        return self.preprocess(crop(self.image_dir / path, box))


def load_model():
    """Load BioCLIP 2 for inference on CPU.

    Returns:
        A tuple ``(model, preprocess, tokenizer)``.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    return model.eval(), preprocess, open_clip.get_tokenizer(MODEL_ID)


def embed_text(model, tokenizer, prompts):
    """Return normalised text embeddings for prompts.

    Args:
        model: The loaded model.
        tokenizer: Its tokenizer.
        prompts: List of prompt strings.

    Returns:
        A float32 array of shape ``(len(prompts), dim)``.
    """
    with torch.no_grad():
        features = model.encode_text(tokenizer(prompts))
    return torch.nn.functional.normalize(features, dim=-1).numpy().astype(np.float32)


def embed_images(model, loader):
    """Return normalised image embeddings for every batch in a loader.

    Args:
        model: The loaded model.
        loader: DataLoader yielding batches of preprocessed crops.

    Returns:
        A float32 array with one row per crop.
    """
    rows = []
    with torch.no_grad():
        for batch in loader:
            features = model.encode_image(batch)
            rows.append(torch.nn.functional.normalize(features, dim=-1).numpy())
    return np.concatenate(rows).astype(np.float32)


def load_cache(cache_dir=CACHE_DIR):
    """Load the merged cache written by ``main``.

    Args:
        cache_dir: Cache directory.

    Returns:
        Dict with ``paths``, ``image``, ``text``, ``classes``, ``prompts``
        and ``logit_scale``.
    """
    with np.load(cache_dir / "embeddings.npz") as data:
        return {k: data[k] for k in data.files}


def main(argv=None):
    """Embed the spike's classify crops and the class prompts, resumably.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, default=subset.OUT_PATH)
    parser.add_argument("--image-dir", type=Path, default=fetch.IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    paths = pl.read_csv(args.selection).filter("classify")["path"].to_list()
    detections = manifest.load_detections()
    model, preprocess, tokenizer = load_model()
    classes = list(orinoquia.TAXA)
    prompts = [prompt(orinoquia.TAXA[c]) for c in classes]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for start in range(0, len(paths), CHUNK):
        chunk_path = args.out_dir / f"chunk_{start:06d}.npy"
        chunks.append(chunk_path)
        if chunk_path.exists():
            continue
        began = time.time()
        loader = torch.utils.data.DataLoader(
            Crops(paths[start : start + CHUNK], args.image_dir, detections, preprocess),
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        np.save(chunk_path, embed_images(model, loader))
        rate = min(CHUNK, len(paths) - start) / (time.time() - began)
        print(
            f"{min(start + CHUNK, len(paths))}/{len(paths)} crops, {rate:.2f} crops/s",
            flush=True,
        )

    np.savez(
        args.out_dir / "embeddings.npz",
        paths=np.array(paths),
        image=np.concatenate([np.load(c) for c in chunks]),
        text=embed_text(model, tokenizer, prompts),
        classes=np.array(classes),
        prompts=np.array(prompts),
        logit_scale=np.float32(model.logit_scale.exp().item()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
