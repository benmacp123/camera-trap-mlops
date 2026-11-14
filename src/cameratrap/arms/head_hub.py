"""Publish the promoted probe head, with a model card, to the Hugging Face Hub.

What is published: ``results/holdout/probe_head.npz`` byte for byte (the
head the gate promoted) and a card written from the committed results, so
every number on the card traces to a file in this repository.

Three steps, mirroring ``embed.hub``:

- **stage** writes the head and its card into a folder to inspect.
- **upload** (the owner, with their own write token) creates the public
  model repo, uploads the staged folder and records the repo and commit
  hash in the committed ``data/manifest/hub_head.json``.
- **check** (anyone, no account) downloads the recorded commit's head and
  confirms it is the committed one.

::

    PYTHONPATH=src python -m cameratrap.arms.head_hub stage
    PYTHONPATH=src python -m cameratrap.arms.head_hub upload --repo-id <user>/<name>
    PYTHONPATH=src python -m cameratrap.arms.head_hub check
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from cameratrap.arms import head as head_module
from cameratrap.data import orinoquia

STAGE_DIR = Path("build/head_stage")
RECORD_PATH = Path("data/manifest/hub_head.json")
FREEZE_PATH = Path("config/holdout_freeze.json")
HOLDOUT_DIR = Path("results/holdout")
TRUST_SUMMARY = Path("results/trust/summary.json")
HEAD_NAME = "probe_head.npz"
GITHUB_URL = "https://github.com/benmacp123/camera-trap-mlops"


def file_sha256(path):
    """Return a file's SHA-256.

    Args:
        path: The file to hash.

    Returns:
        The digest as lowercase hex.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    """Read a CSV into a list of dicts.

    Args:
        path: The CSV file.

    Returns:
        One dict per row, values as strings.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def class_recall(sequences, classes):
    """Return each class's pooled recall over some scored sequences.

    Args:
        sequences: Rows with ``label`` and ``pred``.
        classes: Class names, in the order to report.

    Returns:
        A list of ``(class, sequences, recall)``; classes with no sequence
        are left out.
    """
    table = []
    for name in classes:
        rows = [r for r in sequences if r["label"] == name]
        if rows:
            hits = sum(r["pred"] == name for r in rows)
            table.append((name, len(rows), hits / len(rows)))
    return table


def load_evidence(
    freeze_path=FREEZE_PATH, holdout_dir=HOLDOUT_DIR, trust_summary=TRUST_SUMMARY
):
    """Gather the committed results the card reports.

    Args:
        freeze_path: The hold-out freeze.
        holdout_dir: Folder with the hold-out summary, per-site table,
            sequence predictions and the head.
        trust_summary: The trust report's summary.

    Returns:
        Dict with ``freeze``, ``summary``, ``per_site``, ``recall``,
        ``trust`` (the hold-out row at the suggested threshold, with the
        threshold) and ``head``.

    Raises:
        ValueError: If the hold-out result was not scored under the
            committed freeze, or its challenger is not the head's arm.
    """
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    summary = json.loads(
        (Path(holdout_dir) / "summary.json").read_text(encoding="utf-8")
    )
    # The card's claims hold only for the head scored under this freeze.
    if summary["freeze_sha256"] != freeze["sha256"]:
        raise ValueError(
            "the hold-out result was not scored under the committed freeze"
        )
    if summary["challenger"] != freeze["config"]["challenger"]["arm"]:
        raise ValueError("the hold-out challenger is not the frozen one")
    head = head_module.load(Path(holdout_dir) / HEAD_NAME)
    sequences = read_rows(Path(holdout_dir) / f"sequences_{summary['challenger']}.csv")
    trust = json.loads(Path(trust_summary).read_text(encoding="utf-8"))
    return {
        "freeze": freeze,
        "summary": summary,
        "per_site": read_rows(Path(holdout_dir) / "per_site.csv"),
        "recall": class_recall(sequences, head["classes"]),
        "trust": trust["at_suggested_threshold"]["holdout"],
        "head": head,
    }


def card(evidence, repo_id):
    """Return the model card (README.md) for the published head.

    Args:
        evidence: The results from ``load_evidence``.
        repo_id: The model repo the card will sit in, used in its example.

    Returns:
        The card as Markdown with YAML front matter.
    """
    freeze, summary = evidence["freeze"], evidence["summary"]
    recipe = freeze["config"]["challenger"]
    embedding = recipe["embedding"]
    sites = evidence["per_site"]
    low, high = summary["interval"]
    champion = sum(float(r["champion"]) for r in sites) / len(sites)
    challenger = sum(float(r["challenger"]) for r in sites) / len(sites)
    site_rows = "\n".join(
        f"| {r['site']} | {r['n']} | {float(r['champion']):.3f} "
        f"| {float(r['challenger']):.3f} | {float(r['gain']):+.3f} |"
        for r in sites
    )
    recall_rows = "\n".join(
        f"| {orinoquia.TAXA[name][1]} (*{orinoquia.TAXA[name][0]}*) "
        f"| {count} | {recall:.2f} |"
        for name, count, recall in evidence["recall"]
    )
    trust = evidence["trust"]
    classes = ", ".join(
        f"`{c}` ({orinoquia.TAXA[c][1]})" for c in evidence["head"]["classes"]
    )
    # Sites where the head's edge is under one point of balanced accuracy.
    small = sum(float(r["gain"]) < 0.01 for r in sites)
    width = evidence["head"]["coef"].shape[1]
    timm_name = embedding["model_id"].removeprefix("timm/")
    size = embedding["input_size"]
    return f"""---
license: mit
library_name: numpy
pipeline_tag: image-classification
base_model: {embedding['model_id']}
datasets:
- {freeze['config']['embeddings']['repo_id']}
tags:
- camera-traps
- ecology
- wildlife
- linear-probe
- dinov2
---

# Orinoquía camera-trap species head (DINOv2 linear probe)

A linear classifier for nine mammal and bird species in camera-trap crops
from the Orinoquía region of Colombia. It sits on frozen
[DINOv2 ViT-B/14](https://huggingface.co/{embedding['model_id']})
embeddings: {len(evidence['head']['classes'])} x {width} weights and an
intercept, stored as plain numpy arrays.

It was built for [camera-trap-mlops]({GITHUB_URL}), a project about
deciding, site by site, whether a new classifier should replace the one in
use. This head is the model that project's promotion gate let through. The
repository reproduces every figure below from committed files.

Classes, in logit order: {classes}.

## Intended use

- A worked example of a small adapted classifier with site-level evidence
  behind it, and a baseline for work on the Orinoquía Camera Traps dataset.
- A first-pass sorter for those nine species at sites like the training
  ones, with low-confidence sequences sent to a person.

It is not suitable for:

- **Any species outside the nine.** The head has no "other" or "empty"
  class, so every crop receives one of the nine names.
- **Other regions or habitats.** It was trained and tested on one study
  area and has not been tried anywhere else.
- **Whole images.** It expects a crop of one animal from a detector, and
  images the detector misses never reach it.
- Decisions about individual animals or people. No image of a person was
  used in training or testing.

## How to use

Inputs are animal crops: the most confident
[MegaDetector v5a](https://github.com/agentmorris/MegaDetector) animal box,
padded 10% and squared, long side capped at {embedding['crop_max_side']} px.
Predict per sequence (one burst of frames) by averaging the frames' logits
before the softmax, which is how the scores below were measured.

```python
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

model = timm.create_model(
    "{timm_name}", pretrained=True, num_classes=0, img_size={size}
).eval()
# The pretrained config still reports 518 px, so the size is passed again.
config = timm.data.resolve_data_config({{"input_size": (3, {size}, {size})}}, model=model)
preprocess = timm.data.create_transform(**config)
head = np.load(hf_hub_download("{repo_id}", "{HEAD_NAME}"))

crops = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]  # one sequence
with torch.no_grad():
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in crops])
    vectors = torch.nn.functional.normalize(model(batch), dim=-1).numpy()
logits = (vectors @ head["coef"].T + head["intercept"]).mean(axis=0)
confidence = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
print(head["classes"][logits.argmax()], confidence.max())
```

## Training

- **Data:** {summary['training_frames']:,} crops (up to
  {freeze['config']['metric']['frames_per_sequence']} frames per sequence)
  from 39 camera sites of
  [Orinoquía Camera Traps](https://lila.science/datasets/orinoquia-camera-traps/).
  Only sequences labelled with a single one of the nine species were used,
  and any sequence that ever showed a person was dropped whole.
- **Encoder:** `{embedding['model_id']}` (revision
  `{embedding['revision'][:12]}`), frozen, at {size} px, embeddings
  L2-normalised. The precomputed embeddings are in
  [{freeze['config']['embeddings']['repo_id']}](https://huggingface.co/datasets/{freeze['config']['embeddings']['repo_id']}).
- **Head:** multinomial logistic regression, L2 penalty, balanced class
  weights. The penalty strength was chosen by {recipe['inner_folds']}-fold
  cross-validation grouped by site (chosen C = {summary['chosen_c']:.2f}),
  so no site appeared on both sides of a split.

## Evaluation

The test was fixed and hashed before it was run: ten camera sites the
model had never seen, the metric, the comparison model and this head's
whole training recipe. It was run once.

The metric is balanced accuracy per site at sequence level: recall
averaged over the species present at that site, so a common species cannot
hide a rare one. The comparison is
[BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) used zero-shot
(given the species names, no training).

| | Mean balanced accuracy over the ten sites |
|---|---|
| BioCLIP 2 zero-shot | {champion:.3f} |
| This head | {challenger:.3f} |
| Gain | {summary['mean_gain']:+.3f} (95% interval {low:+.3f} to {high:+.3f}, bootstrap over sites) |

Per site ({summary['sequences']:,} sequences in all):

| Site | Sequences | BioCLIP 2 | This head | Gain |
|---|---|---|---|---|
{site_rows}

Recall per species, pooled over the ten sites:

| Species | Sequences | Recall |
|---|---|---|
{recall_rows}

With answers below {trust['threshold']:.0%} confidence sent to a person,
{trust['review_share']:.1%} of hold-out sequences went to review and
{trust['accepted_error']:.1%} of the accepted answers were wrong.

## Limitations

- **Ten test sites, one study area.** The interval above is wide because
  ten sites is a small sample, and {small} of the ten sites gained under
  one point.
- **Clean sequences only.** Empty frames, multi-species sequences and
  species outside the nine were excluded from training and testing, so
  real-world accuracy on an unfiltered stream will be lower.
- **The detector is upstream.** Animals MegaDetector missed are not in
  these scores.
- **Thin species at some sites.** Several site and species pairs have
  fewer than five test sequences, too few to judge.
- **Confidence is a rough guide.** In the project's drift study the
  model's confidence ranked sites by accuracy but predicted a new site's
  accuracy only slightly better than guessing the average.
- SpeciesNet is not compared here: it has probably seen these images
  (they went through Wildlife Insights), so its scores on them would not
  be a fair test.

## Files

- `{HEAD_NAME}`: `coef` ({len(evidence['head']['classes'])} x {width},
  float64), `intercept` and `classes`. SHA-256 is recorded in the GitHub
  repository (`data/manifest/hub_head.json`), and it is byte for byte the
  file the hold-out run wrote. Hold-out freeze SHA-256:
  `{freeze['sha256']}`.

## Licence and sources

The head is released under the MIT licence. DINOv2 is Apache-2.0.
Training images: {orinoquia.ATTRIBUTION}, released under the Community
Data License Agreement – Permissive 1.0.
"""


def stage(out_dir=STAGE_DIR, repo_id="<user>/<name>", evidence=None):
    """Write the head and its card into a folder.

    Args:
        out_dir: Folder to stage into; replaced if it exists.
        repo_id: The model repo, named in the card's example.
        evidence: Results from ``load_evidence``; read from the committed
            files when ``None``.

    Returns:
        Mapping of published name to SHA-256.
    """
    evidence = evidence or load_evidence()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copyfile(HOLDOUT_DIR / HEAD_NAME, out_dir / HEAD_NAME)
    (out_dir / "README.md").write_text(card(evidence, repo_id), encoding="utf-8")
    return {HEAD_NAME: file_sha256(out_dir / HEAD_NAME)}


def upload(repo_id, stage_dir=STAGE_DIR, record_path=RECORD_PATH):
    """Upload the staged folder to a public model repo and record the commit.

    Needs a write token (``hf auth login`` or ``HF_TOKEN``); run by the
    owner, never in CI. Stages again first so the card names this repo.

    Args:
        repo_id: Model repo such as ``user/orinoquia-dinov2-probe``.
        stage_dir: The folder to stage into and upload.
        record_path: Where to write the repo, commit and file hash.

    Returns:
        The record written.
    """
    files = stage(stage_dir, repo_id)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=stage_dir,
        commit_message="Promoted DINOv2 probe head and model card",
    )
    record = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": commit.oid,
        "files": files,
    }
    Path(record_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check(record_path=RECORD_PATH, head_path=HOLDOUT_DIR / HEAD_NAME):
    """Download the published head and confirm it is the committed one.

    Needs no account.

    Args:
        record_path: The committed record written by ``upload``.
        head_path: The committed head.

    Returns:
        Dict with the repo, revision and the matching SHA-256.

    Raises:
        ValueError: If the published head differs from the record or from
            the committed head.
    """
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    expected = record["files"][HEAD_NAME]
    if file_sha256(head_path) != expected:
        raise ValueError("the committed head does not match the record")
    with tempfile.TemporaryDirectory() as folder:
        hf_hub_download(
            record["repo_id"],
            HEAD_NAME,
            repo_type=record["repo_type"],
            revision=record["revision"],
            local_dir=folder,
        )
        if file_sha256(Path(folder) / HEAD_NAME) != expected:
            raise ValueError("the published head does not match the record")
    return {
        "repo_id": record["repo_id"],
        "revision": record["revision"],
        "sha256": expected,
    }


def main(argv=None):
    """Stage, upload or check the published head.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="write the head and card to a folder")
    staging.add_argument("--repo-id", default="<user>/<name>")
    up = commands.add_parser("upload", help="upload the head and card (owner only)")
    up.add_argument("--repo-id", required=True)
    commands.add_parser("check", help="compare the published head, no account")
    args = parser.parse_args(argv)

    if args.command == "stage":
        result = stage(repo_id=args.repo_id)
    elif args.command == "upload":
        result = upload(args.repo_id)
    else:
        result = check()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
