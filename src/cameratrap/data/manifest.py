"""Build and load the image manifest from LILA metadata and MDv5a results.

The manifest has one row per image in a clean, single-species sequence of
one of the chosen classes, with its site, sequence, label, best MDv5a
animal confidence and site split. It is committed, so every later stage
starts from it rather than from the raw metadata.

Run as a script to rebuild it; the two archives (about 5 MB) are
downloaded first if missing::

    PYTHONPATH=src python -m cameratrap.data.manifest
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import polars as pl

from cameratrap.data import orinoquia
from cameratrap.data.splits import site_splits

SCHEMA = {
    "path": pl.String,
    "sha256": pl.String,
    "site": pl.String,
    "timestamp": pl.String,
    "seq_id": pl.String,
    "frame_num": pl.Int64,
    "label": pl.String,
    "image_label": pl.String,
    "md_conf": pl.Float64,
    "split": pl.String,
}

# Split settings, fixed before any arm is scored (provisional judgement,
# 2026-10-27). The hold-out size sits in the plan's 8-10 range; five folds
# keep about 8 sites in each. The minimums stop the draw picking a site
# too small to score or leaving a rare class almost absent.
SPLIT_SETTINGS = {
    "n_holdout": 10,
    "n_folds": 5,
    "seed": 20261027,
    "min_site_sequences": 30,
    "min_sites_per_class": 2,
    "min_sequences_per_class": 15,
}

RAW_DIR = Path("data/raw") / orinoquia.NAME
OUT_DIR = Path("data/manifest")


def sha256_file(path):
    """Return the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        The digest as a lowercase hex string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_labels(metadata):
    """Collect each image's set of labels from COCO Camera Traps metadata.

    Args:
        metadata: Parsed metadata with ``categories`` and ``annotations``.

    Returns:
        Mapping of image ID to a frozenset of category names.
    """
    names = {c["id"]: c["name"] for c in metadata["categories"]}
    labels = collections.defaultdict(set)
    for annotation in metadata["annotations"]:
        labels[annotation["image_id"]].add(names[annotation["category_id"]])
    return {image_id: frozenset(names) for image_id, names in labels.items()}


def sequence_label(labels, classes):
    """Decide a sequence's label from the union of its frames' labels.

    Args:
        labels: Union of the labels of every frame in the sequence,
            including frames whose image file was removed.
        classes: The classes to keep.

    Returns:
        A pair ``(label, reason)``: the class name and ``None`` for a kept
        sequence, or ``None`` and the reason it was excluded.
    """
    if orinoquia.HUMAN in labels:
        return None, "contains human"
    animals = labels - orinoquia.NON_ANIMAL
    if not animals:
        return None, "no animal label"
    if len(animals) > 1:
        return None, "multi-species"
    (animal,) = animals
    if animal not in classes:
        return None, "other class"
    return animal, None


def best_animal_confidence(detections):
    """Return each image's highest MegaDetector animal confidence.

    Args:
        detections: Parsed MegaDetector batch results with ``images``.

    Returns:
        Mapping of file path to the highest ``animal`` confidence, 0.0 when
        the image has no animal detection. Only images present in the
        results appear, so the keys double as the list of files on LILA.
    """
    animal = next(
        k for k, v in detections["detection_categories"].items() if v == "animal"
    )
    return {
        row["file"]: max(
            (d["conf"] for d in row.get("detections") or [] if d["category"] == animal),
            default=0.0,
        )
        for row in detections["images"]
    }


def in_bounds(timestamp, bounds):
    """Return whether an ISO timestamp falls inside a half-open window.

    Args:
        timestamp: ISO 8601 timestamp with a UTC offset.
        bounds: Pair of ISO 8601 timestamps, start inclusive, end exclusive.

    Returns:
        True when ``start <= timestamp < end``.
    """
    start, end = (dt.datetime.fromisoformat(b) for b in bounds)
    return start <= dt.datetime.fromisoformat(timestamp) < end


def build(
    metadata,
    detections,
    classes=orinoquia.CLASSES,
    bounds=orinoquia.TIMESTAMP_BOUNDS,
    split_settings=None,
):
    """Build the manifest and a report of what was dropped and why.

    Human exclusion uses every metadata row, including those whose image
    was removed, so a sequence that ever showed a person is dropped whole.
    Rows with no image file and rows outside the timestamp bounds are then
    dropped one by one.

    Args:
        metadata: Parsed COCO Camera Traps metadata.
        detections: Parsed MegaDetector batch results for the same images.
        classes: The classes to keep.
        bounds: Timestamp window, see ``in_bounds``.
        split_settings: Keyword arguments for ``site_splits``; defaults to
            ``SPLIT_SETTINGS``.

    Returns:
        A pair ``(frame, report)``: the manifest as a polars DataFrame with
        ``SCHEMA`` columns, sorted by site, timestamp and path, and a dict
        of counts.
    """
    split_settings = split_settings or SPLIT_SETTINGS
    labels = image_labels(metadata)
    confidence = best_animal_confidence(detections)

    frames = collections.defaultdict(list)
    for image in metadata["images"]:
        frames[image["seq_id"]].append(image)

    excluded = collections.Counter()
    dropped = collections.Counter()
    rows = []
    for seq_id, images in frames.items():
        union = frozenset().union(*(labels.get(i["id"], ()) for i in images))
        label, reason = sequence_label(union, classes)
        if reason:
            excluded[reason] += 1
            continue
        kept = []
        for image in images:
            if image["file_name"] not in confidence:
                dropped["no image file"] += 1
            elif not in_bounds(image["datetime"], bounds):
                dropped["timestamp out of bounds"] += 1
            else:
                kept.append(image)
        if not kept:
            excluded["no usable image"] += 1
            continue
        rows.extend(
            {
                "path": image["file_name"],
                "sha256": None,
                "site": image["location"],
                "timestamp": image["datetime"],
                "seq_id": seq_id,
                "frame_num": image["frame_num"],
                "label": label,
                "image_label": "+".join(sorted(labels.get(image["id"], ()))),
                "md_conf": confidence[image["file_name"]],
            }
            for image in kept
        )

    site_counts = collections.defaultdict(collections.Counter)
    for _, label, site in {(r["seq_id"], r["label"], r["site"]) for r in rows}:
        site_counts[site][label] += 1
    splits = site_splits(site_counts, **split_settings)
    for row in rows:
        row["split"] = splits[row["site"]]

    frame = pl.DataFrame(rows, schema=SCHEMA, orient="row").sort(
        "site", "timestamp", "path"
    )
    sequences = frame.unique("seq_id")
    report = {
        "images": frame.height,
        "sequences": sequences.height,
        "sites": frame["site"].n_unique(),
        "sequences_excluded": dict(sorted(excluded.items())),
        "images_dropped": dict(sorted(dropped.items())),
        "sequences_per_class": dict(
            sequences.group_by("label").len().sort("label").iter_rows()
        ),
        "sequences_per_split": dict(
            sequences.group_by("split").len().sort("split").iter_rows()
        ),
        "holdout_sites": sorted(s for s, v in splits.items() if v == "holdout"),
        "split_settings": split_settings,
        "splits_sha256": splits_digest(splits),
    }
    return frame, report


def splits_digest(splits):
    """Hash the site-to-split assignment, so the freeze can be checked.

    The manifest file's own hash changes when image checksums are filled
    in; this one changes only if a site moves between splits.

    Args:
        splits: Mapping of site to split name.

    Returns:
        SHA-256 hex digest of the sorted ``site,split`` lines.
    """
    text = "".join(f"{site},{splits[site]}\n" for site in sorted(splits))
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    """Load a manifest CSV with its column types checked.

    Args:
        path: Path to the manifest CSV.

    Returns:
        The manifest as a polars DataFrame with ``SCHEMA`` columns.

    Raises:
        ValueError: If the columns differ from ``SCHEMA``.
    """
    frame = pl.read_csv(path, schema_overrides=SCHEMA)
    if frame.columns != list(SCHEMA):
        raise ValueError(f"{path}: columns {frame.columns}, expected {list(SCHEMA)}")
    return frame


def main(argv=None):
    """Rebuild the committed manifest from LILA's metadata archives.

    Downloads any archive missing from ``--raw-dir``, then checks each
    against its recorded SHA-256, so a changed upstream file stops the
    build instead of silently changing the data.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 on a checksum mismatch.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    parsed = {}
    for key, (url, expected) in orinoquia.ARCHIVES.items():
        path = args.raw_dir / url.rsplit("/", 1)[-1]
        if not path.exists():
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        actual = sha256_file(path)
        if actual != expected:
            print(f"{path}: SHA-256 {actual}, expected {expected}", file=sys.stderr)
            return 1
        parsed[key] = orinoquia.read_zipped_json(path)

    frame, report = build(parsed["metadata"], parsed["detections"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{orinoquia.NAME}.csv"
    frame.write_csv(csv_path)
    provenance = {
        "dataset": orinoquia.NAME,
        "licence": orinoquia.LICENCE,
        "attribution": orinoquia.ATTRIBUTION,
        "image_base_url": orinoquia.IMAGE_BASE_URL,
        "archives": {
            k: {"url": u, "sha256": s} for k, (u, s) in orinoquia.ARCHIVES.items()
        },
        "classes": list(orinoquia.CLASSES),
        "timestamp_bounds": list(orinoquia.TIMESTAMP_BOUNDS),
        **report,
    }
    json_path = args.out_dir / f"{orinoquia.NAME}.json"
    json_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
