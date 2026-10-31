"""Choose, cut and store the crop every arm sees: the most confident MDv5a
animal box.

Boxes are enlarged a little and squared before cropping. Backbones resize
the short side and centre-crop, so a long thin box (a walking tapir, a
guan's tail) would otherwise lose its ends.

Two steps, each run as a script:

- **index**: from the committed manifest and LILA's precomputed MDv5a
  results alone, list the frames to crop (the most confident animal frames
  of every sequence) and the frames dropped because MDv5a also sees a
  person in them. Both lists are committed, so the crop store can be
  rebuilt without the raw detections.
- **write**: cut each indexed crop from its downloaded image and save it
  under ``data/crops/``, mirroring the image path. Resumes by skipping
  crops already on disk.
- **sizes**: record each stored crop's side length, so reports can break
  accuracy down by crop size without the crop store. Kept apart from the
  index, whose hash the embedding caches record.

::

    PYTHONPATH=src python -m cameratrap.data.crops index
    PYTHONPATH=src python -m cameratrap.data.crops write
    PYTHONPATH=src python -m cameratrap.data.crops sizes
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import polars as pl
from PIL import Image

from cameratrap.data import fetch, manifest, orinoquia

# Fixed before the spike (provisional judgement, 2026-10-28).
PAD = 0.1

ANIMAL, PERSON = "1", "2"

# Fixed before Phase 1 wrote any crop (provisional judgement, 2026-10-30).
# Three frames and 0.2 repeat the spike's rule, now over every sequence;
# person frames are dropped before the top three are picked, so a sequence
# with one such frame still gets three crops when it has them. 448 px is
# twice the backbones' 224 px input, leaving room for augmentation; 4:4:4
# chroma (no subsampling) at quality 95 keeps colour detail that tells the
# two peccaries apart.
CROP_SETTINGS = {
    "frames_per_sequence": 3,
    "min_conf": 0.2,
    "person_conf": 0.2,
}
MAX_SIDE = 448
JPEG_OPTIONS = {"quality": 95, "subsampling": 0}

INDEX_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}_crops.csv"
EXCLUDED_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}_crops_excluded.csv"
RECORD_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}_crops.json"
SIZES_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}_crop_sizes.csv"
CROP_DIR = Path("data/crops") / orinoquia.NAME

INDEX_SCHEMA = {
    "path": pl.String,
    "seq_id": pl.String,
    "site": pl.String,
    "split": pl.String,
    "label": pl.String,
    "frame_num": pl.Int64,
    "conf": pl.Float64,
    "person_conf": pl.Float64,
    "bbox_x": pl.Float64,
    "bbox_y": pl.Float64,
    "bbox_w": pl.Float64,
    "bbox_h": pl.Float64,
}


def top_animal(detections, min_conf=0.0):
    """Return the most confident animal detection, or ``None``.

    Args:
        detections: MDv5a detections for one image.
        min_conf: Minimum confidence.

    Returns:
        The detection dict, or ``None`` if no animal box reaches ``min_conf``.
    """
    animals = [
        d for d in detections if d["category"] == ANIMAL and d["conf"] >= min_conf
    ]
    return max(animals, key=lambda d: d["conf"], default=None)


def person_confidence(detections):
    """Return the highest MDv5a person confidence in one image.

    Args:
        detections: MDv5a detections for one image.

    Returns:
        The confidence, 0.0 when there is no person box.
    """
    return max((d["conf"] for d in detections if d["category"] == PERSON), default=0.0)


def square_box(bbox, width, height, pad=PAD):
    """Turn a normalised box into a padded, square pixel box inside the image.

    Args:
        bbox: ``[x, y, w, h]`` normalised to the image size, top-left origin.
        width: Image width in pixels.
        height: Image height in pixels.
        pad: Fraction added to each side of the box before squaring.

    Returns:
        ``(left, top, right, bottom)`` integer pixel coordinates. The square
        is shifted to stay inside the image and only shrinks if it is
        larger than the image's short side.
    """
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(w * width, h * height) * (1 + 2 * pad)
    side = min(side, width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return round(left), round(top), round(left + side), round(top + side)


def crop(path, detection):
    """Open an image and cut its square crop around a detection.

    Args:
        path: Path to the image file.
        detection: The detection to crop, with a normalised ``bbox``.

    Returns:
        The crop as an RGB PIL image.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.crop(square_box(detection["bbox"], *image.size))


def select(frame, detections, *, frames_per_sequence, min_conf, person_conf):
    """List the frames to crop and the frames dropped for a person box.

    Args:
        frame: The manifest, as from ``manifest.load``.
        detections: Mapping of manifest path to MDv5a detections.
        frames_per_sequence: Maximum crops kept from each sequence.
        min_conf: Minimum MDv5a animal confidence for a crop.
        person_conf: A frame with a person box at or above this is dropped.

    Returns:
        A pair ``(index, excluded)`` of DataFrames with ``INDEX_SCHEMA``
        columns. ``index`` holds the kept crops, sorted by path; each
        row's box is the frame's most confident animal box. ``excluded``
        holds the frames that would have been candidates but for a person
        box, sorted by path.
    """
    rows = []
    for path in frame["path"]:
        found = detections.get(path, [])
        box = top_animal(found, min_conf)
        if box is not None:
            rows.append((path, box["conf"], person_confidence(found), *box["bbox"]))
    boxes = pl.DataFrame(
        rows,
        schema=["path", "conf", "person_conf", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        orient="row",
    )
    candidates = frame.join(boxes, on="path").select(list(INDEX_SCHEMA))
    has_person = pl.col("person_conf") >= person_conf
    index = (
        candidates.filter(~has_person)
        .sort(["seq_id", "conf", "frame_num"], descending=[False, True, False])
        .group_by("seq_id", maintain_order=True)
        .head(frames_per_sequence)
        .select(list(INDEX_SCHEMA))
        .sort("path")
    )
    return index, candidates.filter(has_person).sort("path")


def load_index(path=INDEX_PATH):
    """Load a crop index CSV with its column types checked.

    Args:
        path: Path to the index (or excluded-frames) CSV.

    Returns:
        The index as a polars DataFrame with ``INDEX_SCHEMA`` columns.

    Raises:
        ValueError: If the columns differ from ``INDEX_SCHEMA``.
    """
    frame = pl.read_csv(path, schema_overrides=INDEX_SCHEMA)
    if frame.columns != list(INDEX_SCHEMA):
        raise ValueError(
            f"{path}: columns {frame.columns}, expected {list(INDEX_SCHEMA)}"
        )
    return frame


def write_crop(source, dest, bbox, max_side=MAX_SIDE):
    """Cut one crop, shrink it to at most ``max_side`` and save it as JPEG.

    Writes to a temporary name and renames on success, so an interrupted
    run never leaves a partial crop under the real name.

    Args:
        source: Path to the full image.
        dest: Path to write the crop to; parent directories are created.
        bbox: Normalised ``[x, y, w, h]`` animal box.
        max_side: Longest side of the stored crop, in pixels.

    Returns:
        The stored crop's ``(width, height)``.
    """
    image = crop(source, {"bbox": bbox})
    if max(image.size) > max_side:
        # Bicubic matches the resampling the backbones' own resize uses.
        image = image.resize((max_side, max_side), Image.Resampling.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    image.save(partial, format="JPEG", **JPEG_OPTIONS)
    partial.replace(dest)
    return image.size


def write_all(index, image_dir, crop_dir, workers=8, progress_every=2000):
    """Write every indexed crop not already on disk, in parallel.

    Pillow releases the GIL while decoding and encoding JPEGs, so threads
    keep several cores busy without the cost of extra processes.

    Args:
        index: The crop index, as from ``load_index``.
        image_dir: Local root the full images are stored under.
        crop_dir: Local root to store crops under, mirroring the paths.
        workers: Number of parallel crops.
        progress_every: Print a progress line after this many crops.

    Returns:
        A pair ``(written, failures)``: the number of crops written in this
        run, and a mapping of path to error message for crops that could
        not be cut (usually an image not yet downloaded).
    """
    todo = [
        (row["path"], [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]])
        for row in index.iter_rows(named=True)
        if not (crop_dir / row["path"]).exists()
    ]
    written, failures = 0, {}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(write_crop, image_dir / path, crop_dir / path, bbox): path
            for path, bbox in todo
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
                written += 1
            except OSError as error:
                failures[futures[future]] = str(error)
            if done % progress_every == 0:
                print(f"{done}/{len(futures)} ({len(failures)} failed)", flush=True)
    return written, failures


def missing_crops(index, crop_dir):
    """List indexed crops that are not on disk.

    Args:
        index: The crop index.
        crop_dir: Local root the crops are stored under.

    Returns:
        The sorted paths with no crop file.
    """
    return sorted(p for p in index["path"] if not (crop_dir / p).exists())


def crop_sizes(paths, crop_dir):
    """Read each stored crop's side length from its JPEG header.

    Args:
        paths: Crop paths relative to ``crop_dir``.
        crop_dir: Root of the crop store.

    Returns:
        A DataFrame with ``path`` and ``side`` (the longer side in pixels;
        crops are square to within a pixel of rounding).
    """
    sides = []
    for path in paths:
        # Opening reads only the header; the pixels are never decoded.
        with Image.open(crop_dir / path) as image:
            sides.append(max(image.size))
    return pl.DataFrame({"path": paths, "side": sides})


def load_sizes(path=SIZES_PATH):
    """Load the committed crop-size table.

    Args:
        path: Path to the sizes CSV.

    Returns:
        A DataFrame with ``path`` and ``side``.
    """
    return pl.read_csv(path, schema={"path": pl.String, "side": pl.Int64})


def main_sizes(args):
    """Write the crop-size table for every indexed crop.

    Args:
        args: Parsed arguments with ``index``, ``crop_dir`` and ``out``.

    Returns:
        Process exit status, 0 on success.
    """
    sizes = crop_sizes(load_index(args.index)["path"].to_list(), args.crop_dir)
    sizes.write_csv(args.out)
    print(f"{sizes.height} crop sizes -> {args.out}")
    return 0


def main_index(args):
    """Write the crop index, excluded frames and their record.

    Args:
        args: Parsed arguments with ``manifest``, ``raw_dir`` and ``out_dir``.

    Returns:
        Process exit status, 0 on success.
    """
    frame = manifest.load(args.manifest)
    index, excluded = select(
        frame, manifest.load_detections(args.raw_dir), **CROP_SETTINGS
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index.write_csv(args.out_dir / INDEX_PATH.name)
    excluded.write_csv(args.out_dir / EXCLUDED_PATH.name)
    record = {
        "settings": {
            **CROP_SETTINGS,
            "pad": PAD,
            "max_side": MAX_SIDE,
            "jpeg": JPEG_OPTIONS,
        },
        "crops": index.height,
        "sequences": index["seq_id"].n_unique(),
        "sequences_without_crop": frame["seq_id"].n_unique()
        - index["seq_id"].n_unique(),
        "frames_excluded_person": excluded.height,
    }
    (args.out_dir / RECORD_PATH.name).write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{index.height} crops from {record['sequences']} sequences; "
        f"{excluded.height} frames dropped for a person box -> {args.out_dir}"
    )
    return 0


def main_write(args):
    """Write the indexed crops and report any that are still missing.

    Args:
        args: Parsed arguments with ``index``, ``image_dir``, ``crop_dir``
            and ``workers``.

    Returns:
        Process exit status: 0 when every indexed crop is on disk, 1
        otherwise.
    """
    index = load_index(args.index)
    written, failures = write_all(index, args.image_dir, args.crop_dir, args.workers)
    missing = missing_crops(index, args.crop_dir)
    print(
        f"{written} written, {len(failures)} failed; "
        f"{index.height - len(missing)}/{index.height} crops on disk"
    )
    for path, error in sorted(failures.items())[:10]:
        print(f"failed: {path}: {error}", file=sys.stderr)
    return 0 if not missing else 1


def main(argv=None):
    """Build the crop index, write the crops or record their sizes.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="list the crops from the manifest")
    index.add_argument(
        "--manifest", type=Path, default=manifest.OUT_DIR / f"{orinoquia.NAME}.csv"
    )
    index.add_argument("--raw-dir", type=Path, default=manifest.RAW_DIR)
    index.add_argument("--out-dir", type=Path, default=manifest.OUT_DIR)
    write = commands.add_parser("write", help="cut the indexed crops")
    write.add_argument("--index", type=Path, default=INDEX_PATH)
    write.add_argument("--image-dir", type=Path, default=fetch.IMAGE_DIR)
    write.add_argument("--crop-dir", type=Path, default=CROP_DIR)
    write.add_argument("--workers", type=int, default=8)
    sizes = commands.add_parser("sizes", help="record the stored crops' sizes")
    sizes.add_argument("--index", type=Path, default=INDEX_PATH)
    sizes.add_argument("--crop-dir", type=Path, default=CROP_DIR)
    sizes.add_argument("--out", type=Path, default=SIZES_PATH)
    args = parser.parse_args(argv)
    commands_run = {"index": main_index, "write": main_write, "sizes": main_sizes}
    return commands_run[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
