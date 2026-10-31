"""SpeciesNet as a reference arm: write its inputs, read back its predictions.

SpeciesNet runs as its own command (``python -m speciesnet.scripts.run_model``)
so this module never imports it. It is given LILA's precomputed MDv5a
animal boxes through ``--detections_json`` and classifies only the most
confident one per image, the same crop the other arms see.

Two readings of each image, as the plan requires:

- **open set**: SpeciesNet's own geofenced, rolled-up answer among its
  2,498 labels. A roll-up above species level, or any label outside the
  chosen classes, counts as an abstention.
- **restricted**: its raw logits for the chosen classes only (requested
  with ``--target_species_txt``), so it can be compared like for like
  with the closed-set arms and averaged across a sequence.

SpeciesNet has probably seen Orinoquía during training (its images went
through Wildlife Insights), so these scores are never held-out performance.

Run as a script to classify the spike selection (about 11 hours on CPU),
or every frame in the Phase 1 crop index (resuming from the spike's
answers, so about 8.5 hours more)::

    PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm
    PYTHONPATH=src python -m cameratrap.arms.speciesnet_arm --crop-index
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from cameratrap.data import crops, fetch, manifest, orinoquia, subset

MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
CACHE_DIR = Path("data/cache/speciesnet")
# Phase 2's run over the crop index keeps its own cache so the spike's
# stays as it was scored.
CROP_CACHE_DIR = Path("data/cache/speciesnet_crops")
DETECTOR_CHECK_DIR = Path("data/cache/mdv5a_check")
OTHER_SPECIES = "other_species"


def write_inputs(paths, image_dir, detections, out_dir, taxa, country):
    """Write SpeciesNet's instances, detections and target-species files.

    Args:
        paths: Manifest paths of the images to classify.
        image_dir: Local root the manifest paths are stored under.
        detections: Mapping of manifest path to its MDv5a detections, each
            ``{"category", "conf", "bbox"}`` in LILA's format.
        out_dir: Directory to write the three files into.
        taxa: Mapping of class to ``(scientific, common, speciesnet_label)``.
        country: ISO 3166-1 alpha-3 code for the geofence.

    Returns:
        Mapping of file role (``instances``, ``detections``, ``targets``)
        to the path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "instances": out_dir / "instances.json",
        "detections": out_dir / "detections.json",
        "targets": out_dir / "target_species.txt",
    }
    filepaths = [str((image_dir / p).resolve()) for p in paths]
    files["instances"].write_text(
        json.dumps(
            {"instances": [{"filepath": f, "country": country} for f in filepaths]}
        )
    )
    predictions = []
    for path, filepath in zip(paths, filepaths):
        animals = sorted(
            (d for d in detections[path] if d["category"] == "1"),
            key=lambda d: d["conf"],
            reverse=True,
        )
        predictions.append(
            {
                "filepath": filepath,
                "detections": [
                    {
                        "category": "1",
                        "label": "animal",
                        "conf": d["conf"],
                        "bbox": d["bbox"],
                    }
                    for d in animals
                ],
            }
        )
    files["detections"].write_text(json.dumps({"predictions": predictions}))
    files["targets"].write_text("".join(t[2] + "\n" for t in taxa.values()))
    return files


def command(files, predictions_json, model, batch_size=8):
    """Return the SpeciesNet command line for the written inputs.

    Args:
        files: Mapping returned by ``write_inputs``.
        predictions_json: Output path; SpeciesNet resumes from it if present.
        model: SpeciesNet model identifier, such as
            ``kaggle:google/speciesnet/pyTorch/v4.0.3a/1``.
        batch_size: Classifier batch size.

    Returns:
        The argument list, starting with ``python``.
    """
    return [
        "python",
        "-m",
        "speciesnet.scripts.run_model",
        f"--model={model}",
        f"--instances_json={files['instances']}",
        f"--detections_json={files['detections']}",
        f"--target_species_txt={files['targets']}",
        f"--predictions_json={predictions_json}",
        "--max_classifications_per_image=1",
        f"--batch_size={batch_size}",
        "--noprogress_bars",
        "--bypass_prompts",
    ]


def read_predictions(predictions_json, image_dir, taxa):
    """Read SpeciesNet's predictions back into per-image arrays.

    Args:
        predictions_json: SpeciesNet's output file.
        image_dir: Local root the manifest paths are stored under.
        taxa: Mapping of class to ``(scientific, common, speciesnet_label)``,
            in the class order of the returned logits.

    Returns:
        Mapping of manifest path to a dict with ``logits`` (array over the
        classes, restricted reading), ``open_label`` and ``source``
        (SpeciesNet's own account of how it reached the answer). Images
        SpeciesNet failed on are left out.

        ``open_label`` is a class name; ``OTHER_SPECIES`` for a
        species-level answer outside the classes, which is wrong rather
        than an abstention; or ``None`` for a roll-up above species, blank
        or other non-species answer, which is an abstention.
    """
    root = str(image_dir.resolve()) + "/"
    label_to_class = {t[2]: c for c, t in taxa.items()}
    order = [t[2] for t in taxa.values()]
    out = {}
    for prediction in json.loads(predictions_json.read_text())["predictions"]:
        classified = [
            d for d in prediction.get("detections") or [] if "raw_classifications" in d
        ]
        if not classified:
            continue
        det = classified[0]
        raw = det["raw_classifications"]
        by_label = dict(zip(raw["target_classes"], raw["target_logits"]))
        processed = det.get("processed_classification", {})
        out[prediction["filepath"].removeprefix(root)] = {
            "logits": np.array([by_label[label] for label in order]),
            "open_label": open_label(processed.get("class", ""), label_to_class),
            "source": processed.get("source"),
        }
    return out


def prune_unfinished(predictions_json):
    """Drop entries SpeciesNet has not finished from a partial predictions file.

    SpeciesNet's periodic save lists every image, the unprocessed ones
    flagged with failures, and its resume skips everything listed. Pruning
    the flagged entries first makes a restart pick them up again.

    Args:
        predictions_json: SpeciesNet's output file; nothing happens if it
            does not exist.

    Returns:
        The number of entries kept.
    """
    if not predictions_json.exists():
        return 0
    data = json.loads(predictions_json.read_text())
    data["predictions"] = [p for p in data["predictions"] if not p.get("failures")]
    predictions_json.write_text(json.dumps(data))
    return len(data["predictions"])


def open_label(label, label_to_class):
    """Map one SpeciesNet answer to a class, a wrong species, or an abstention.

    Args:
        label: SpeciesNet's processed label, ``uuid;class;order;family;
            genus;species;common``.
        label_to_class: Mapping of SpeciesNet label to class name.

    Returns:
        The class name, ``OTHER_SPECIES``, or ``None`` for an abstention.
    """
    if label in label_to_class:
        return label_to_class[label]
    parts = label.split(";")
    return OTHER_SPECIES if len(parts) == 7 and parts[5] else None


def detector_command(paths, image_dir, out_dir):
    """Return a SpeciesNet detector-only run for the detector check.

    SpeciesNet ships MDv5a (``md_v5a.0.1``, a repackaging of the weights
    LILA ran as ``md_v5a.0.0``), so this re-runs the detector without
    vendoring MegaDetector.

    Args:
        paths: Manifest paths to detect on.
        image_dir: Local root the manifest paths are stored under.
        out_dir: Directory for the instances and predictions files.

    Returns:
        The argument list, starting with ``python``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = out_dir / "instances.json"
    instances.write_text(
        json.dumps(
            {"instances": [{"filepath": str((image_dir / p).resolve())} for p in paths]}
        )
    )
    return [
        "python",
        "-m",
        "speciesnet.scripts.run_model",
        f"--model={MODEL}",
        "--detector_only",
        f"--instances_json={instances}",
        f"--predictions_json={out_dir / 'predictions.json'}",
        "--noprogress_bars",
        "--bypass_prompts",
    ]


def read_detections(predictions_json, image_dir):
    """Read a detector-only run back into LILA's detection format.

    Args:
        predictions_json: SpeciesNet's output file.
        image_dir: Local root the manifest paths are stored under.

    Returns:
        Mapping of manifest path to its detections, each ``{"category",
        "conf", "bbox"}``.
    """
    root = str(image_dir.resolve()) + "/"
    return {
        p["filepath"].removeprefix(root): [
            {"category": d["category"], "conf": d["conf"], "bbox": d["bbox"]}
            for d in p.get("detections") or []
        ]
        for p in json.loads(predictions_json.read_text())["predictions"]
    }


def seed_from_spike(paths, image_dir, out_dir, spike_dir=CACHE_DIR):
    """Start a new cache from the spike's finished predictions for ``paths``.

    SpeciesNet skips images already in its output file, so seeding it with
    the spike's answers means only the frames it never scored are run.
    Images classified with the same model, detections and geofence give
    the same answer, so reuse changes no result. SpeciesNet refuses to
    resume from an output holding images outside its input list, so the
    spike's other frames are left out.

    Args:
        paths: Manifest paths of the images the new run will classify.
        image_dir: Local root the manifest paths are stored under.
        out_dir: The new cache directory.
        spike_dir: The spike's cache directory.

    Returns:
        The number of predictions carried over (0 if the new cache already
        exists or the spike's is missing).
    """
    target = out_dir / "predictions.json"
    source = spike_dir / "predictions.json"
    if target.exists() or not source.exists():
        return 0
    wanted = {str((image_dir / p).resolve()) for p in paths}
    spike = json.loads(source.read_text())
    kept = [p for p in spike["predictions"] if p["filepath"] in wanted]
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({**spike, "predictions": kept}))
    return len(kept)


def main(argv=None):
    """Run SpeciesNet over the spike's classify images, resuming if stopped.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        SpeciesNet's exit status.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, default=subset.OUT_PATH)
    parser.add_argument("--image-dir", type=Path, default=fetch.IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--crop-index",
        action="store_true",
        help="classify every frame in the crop index instead of the spike's",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--detector-check",
        action="store_true",
        help="re-run MDv5a on the detector-check images instead",
    )
    args = parser.parse_args(argv)

    if args.detector_check:
        paths = pl.read_csv(args.selection).filter("detector_check")["path"].to_list()
        cmd = detector_command(paths, args.image_dir, DETECTOR_CHECK_DIR)
        cmd[0] = sys.executable
        return subprocess.run(cmd, check=False).returncode

    if args.crop_index:
        paths = crops.load_index()["path"].to_list()
        args.out_dir = args.out_dir or CROP_CACHE_DIR
        seeded = seed_from_spike(paths, args.image_dir, args.out_dir)
        print(f"seeded {seeded} predictions from the spike")
    else:
        paths = pl.read_csv(args.selection).filter("classify")["path"].to_list()
        args.out_dir = args.out_dir or CACHE_DIR
    files = write_inputs(
        paths,
        args.image_dir,
        manifest.load_detections(),
        args.out_dir,
        orinoquia.TAXA,
        orinoquia.COUNTRY,
    )
    predictions_json = args.out_dir / "predictions.json"
    print(f"resuming with {prune_unfinished(predictions_json)} finished images")
    cmd = command(files, predictions_json, args.model, args.batch_size)
    cmd[0] = sys.executable
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
