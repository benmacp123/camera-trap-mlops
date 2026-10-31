"""Publish the Phase 2 caches to the Hugging Face Hub and fetch them back.

What is published: the BioCLIP 2 and DINOv2 embedding caches and
SpeciesNet's answers over the crop index, slimmed to the fields the arms
read and with file paths made relative, so a fresh clone reproduces every
baseline table without images, models or an account.

Three steps:

- **stage** writes the files and a dataset card into a folder to inspect.
- **upload** (the owner, with their own write token) creates the public
  dataset repo, uploads the staged folder and records the repo and commit
  hash in the committed ``data/manifest/hub.json``.
- **download** (anyone, no account) fetches the recorded commit into
  ``data/cache/`` and checks every file's SHA-256 against the record.

::

    PYTHONPATH=src python -m cameratrap.embed.hub stage
    PYTHONPATH=src python -m cameratrap.embed.hub upload --repo-id <user>/<name>
    PYTHONPATH=src python -m cameratrap.embed.hub download
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from cameratrap.data import crops, fetch, manifest, orinoquia
from cameratrap.embed import store

CACHE_DIR = Path("data/cache")
STAGE_DIR = Path("data/hub_stage")
RECORD_PATH = manifest.OUT_DIR / "hub.json"
# Published name -> local source, both relative to ``CACHE_DIR``. Names
# mirror the cache layout so a download lands where every reader looks.
EMBEDDINGS = {
    "embeddings/bioclip2/embeddings.npz": "embeddings/bioclip2/embeddings.npz",
    "embeddings/dinov2/embeddings.npz": "embeddings/dinov2/embeddings.npz",
}
SPECIESNET = "speciesnet_crops/predictions.json"


def slim_speciesnet(predictions_json, image_dir):
    """Return SpeciesNet's predictions with relative paths and only the fields read.

    ``speciesnet_arm.read_predictions`` needs each image's first classified
    detection (its target logits and processed class); the rest of
    SpeciesNet's output is dropped, and the local image root is stripped so
    the file works on any machine.

    Args:
        predictions_json: SpeciesNet's output file.
        image_dir: Local root the image paths start with.

    Returns:
        A dict in SpeciesNet's output format.
    """
    root = str(image_dir.resolve()) + "/"
    raw = json.loads(predictions_json.read_text())
    slim = []
    for prediction in raw["predictions"]:
        classified = [
            d for d in prediction.get("detections") or [] if "raw_classifications" in d
        ]
        if not classified:
            continue
        det = classified[0]
        slim.append(
            {
                "filepath": prediction["filepath"].removeprefix(root),
                "detections": [
                    {
                        "conf": det["conf"],
                        "bbox": det["bbox"],
                        "raw_classifications": {
                            k: det["raw_classifications"][k]
                            for k in ("target_classes", "target_logits")
                        },
                        "processed_classification": det.get(
                            "processed_classification", {}
                        ),
                    }
                ],
            }
        )
    return {"info": raw.get("info", {}), "predictions": slim}


def card(caches):
    """Return the dataset card (README.md) for the published files.

    Args:
        caches: Mapping of backbone name to its cache meta.

    Returns:
        The card as Markdown with YAML front matter.
    """
    models = "\n".join(
        f"| `{name}` | [{m['model_id']}](https://huggingface.co/{m['model_id']}) "
        f"| `{m['revision']}` | {m['input_size']} px |"
        for name, m in caches.items()
    )
    index_sha = next(iter(caches.values()))["index_sha256"]
    return f"""---
license: cdla-permissive-1.0
pretty_name: Orinoquía camera-trap crop embeddings
tags:
- camera-traps
- ecology
- embeddings
---

# Orinoquía camera-trap crop embeddings

Frozen BioCLIP 2 and DINOv2 embeddings of 25,947 animal crops from the
Orinoquía Camera Traps dataset (nine species, 49 sites in Colombia), plus
SpeciesNet's answers for the same frames. Made for
[camera-trap-mlops](https://github.com/benmacp123/camera-trap-mlops), a
site-level promotion gate and trust report for camera-trap classifiers;
its code reproduces every baseline table from these files alone.

## Files

- `embeddings/<backbone>/embeddings.npz`: `paths` (crop path, which is the
  source image path), `vectors` (L2-normalised float32, one row per path)
  and `meta` (JSON: model, weight revision, crop-index SHA-256, crop
  settings). The BioCLIP 2 file also holds `text` (one normalised
  embedding per class prompt), `classes`, `prompts` and `logit_scale`;
  zero-shot logits are `logit_scale * vectors @ text.T`.
- `speciesnet_crops/predictions.json`: SpeciesNet 5.0.6 (model v4.0.3a,
  COL geofence) for every frame, given the same MDv5a box, with relative
  paths and only the target logits and processed answer kept.

| Backbone | Model | Revision | Input |
|---|---|---|---|
{models}

Crops: the most confident MDv5a (LILA's precomputed results) animal box of
up to three frames per sequence, boxes padded 10% and squared, stored as
JPEG (q95, 4:4:4), long side capped at 448 px. Frames with an MDv5a person
box at 0.2 or above are excluded, and no sequence with a human label is
included. The crop index (with site, sequence, label and split per crop)
is `data/manifest/orinoquia_crops.csv` in the GitHub repo, SHA-256
`{index_sha}`.

## Cautions

- **Hold-out sites are included.** The index's `split` column marks them
  (`holdout`); the project scores them once, after its evaluation config
  is frozen. Do not tune against them if you want comparable results.
- **SpeciesNet has probably seen these images** (Orinoquía went through
  Wildlife Insights), so its answers are not held-out performance.
- Embeddings are CPU float32; re-embedding reproduces them to a cosine
  of at least 0.9999.

## Source and licence

{orinoquia.ATTRIBUTION}, released under the Community Data License
Agreement – Permissive 1.0. These derived files are released under the
same licence. Models: BioCLIP 2 (MIT), DINOv2 (Apache-2.0), SpeciesNet
(Apache-2.0).
"""


def stage(out_dir=STAGE_DIR, cache_dir=CACHE_DIR, index_path=crops.INDEX_PATH):
    """Write the files to publish, and their card, into a folder.

    Args:
        out_dir: Folder to stage into; replaced if it exists.
        cache_dir: Root of the local caches.
        index_path: The crop index every cache must have been made from.

    Returns:
        Mapping of published name to SHA-256.

    Raises:
        ValueError: If a cache was made from another crop index.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    index_sha = store.file_sha256(index_path)
    caches = {}
    for name, source in EMBEDDINGS.items():
        meta = store.read(cache_dir / source)["meta"]
        store.check_meta(meta, {"index_sha256": index_sha})
        caches[meta["backbone"]] = meta
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_dir / source, out_dir / name)
    (out_dir / SPECIESNET).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / SPECIESNET).write_text(
        json.dumps(slim_speciesnet(cache_dir / SPECIESNET, fetch.IMAGE_DIR))
    )
    (out_dir / "README.md").write_text(card(caches))
    return {
        name: store.file_sha256(out_dir / name) for name in (*EMBEDDINGS, SPECIESNET)
    }


def upload(repo_id, stage_dir=STAGE_DIR, record_path=RECORD_PATH):
    """Upload the staged folder to a public dataset repo and record the commit.

    Needs a write token (``hf auth login`` or ``HF_TOKEN``); run by the
    owner, never in CI.

    Args:
        repo_id: Dataset repo such as ``user/orinoquia-crop-embeddings``.
        stage_dir: The folder written by ``stage``.
        record_path: Where to write the repo, commit and file hashes.

    Returns:
        The record written.
    """
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=stage_dir,
        commit_message="Phase 2 crop embeddings and SpeciesNet answers",
    )
    record = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": commit.oid,
        "files": {
            name: store.file_sha256(stage_dir / name)
            for name in (*EMBEDDINGS, SPECIESNET)
        },
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def download(record_path=RECORD_PATH, cache_dir=CACHE_DIR, force=False):
    """Fetch the recorded commit's files into the cache and check their hashes.

    Needs no account. A local file whose hash already matches is kept; one
    that differs (such as the owner's full SpeciesNet output) is left alone
    unless ``force`` is set.

    Args:
        record_path: The committed record written by ``upload``.
        cache_dir: Root of the local caches.
        force: Replace local files whose hash differs from the record.

    Returns:
        Mapping of published name to ``"kept"``, ``"downloaded"`` or
        ``"skipped (differs locally)"``.

    Raises:
        ValueError: If a downloaded file's hash differs from the record.
    """
    record = json.loads(record_path.read_text())
    status = {}
    for name, sha in record["files"].items():
        local = cache_dir / name
        if local.exists():
            if store.file_sha256(local) == sha:
                status[name] = "kept"
                continue
            if not force:
                status[name] = "skipped (differs locally)"
                continue
        hf_hub_download(
            record["repo_id"],
            name,
            repo_type=record["repo_type"],
            revision=record["revision"],
            local_dir=cache_dir,
        )
        if store.file_sha256(local) != sha:
            raise ValueError(f"{name}: downloaded file does not match the record")
        status[name] = "downloaded"
    return status


def main(argv=None):
    """Stage, upload or download the published caches.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stage", help="write the files and card to data/hub_stage/")
    up = commands.add_parser("upload", help="upload the staged files (owner only)")
    up.add_argument("--repo-id", required=True)
    down = commands.add_parser("download", help="fetch the recorded files, no account")
    down.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "stage":
        result = stage()
    elif args.command == "upload":
        result = upload(args.repo_id)
    else:
        result = download(force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
