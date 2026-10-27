"""Choose the images the first spike downloads, from the fold sites only.

Two overlapping samples:

- **classify**: up to a fixed number of sequences per class and site, and
  in each the frames with the most confident MDv5a animal box. Frames
  with no box at the threshold give no crop, so they are not fetched;
  the detector's misses are counted from the manifest instead.
- **detector_check**: random fold images of any confidence, empty frames
  included, for comparing the precomputed MDv5a results with a fresh run.

The hold-out sites are never sampled, so nothing about them is seen
before the freeze. Random choices rank by a seeded SHA-256 of the
sequence or path, which does not depend on any library's generator.

Run as a script to write the committed selection::

    PYTHONPATH=src python -m cameratrap.data.subset
"""

import argparse
import hashlib
import sys
from pathlib import Path

import polars as pl

from cameratrap.data import manifest, orinoquia
from cameratrap.data.splits import HOLDOUT

# Fixed before the spike (provisional judgement, 2026-10-27): 100
# sequences of 3 frames is the plan's "about 300 images per class per
# site", about 14,800 images or 11 GB; 0.2 is MDv5a's typical threshold.
SPIKE_SETTINGS = {
    "sequences_per_class_site": 100,
    "frames_per_sequence": 3,
    "min_conf": 0.2,
    "n_detector_check": 500,
    "seed": 20261027,
}

OUT_PATH = manifest.OUT_DIR / f"{orinoquia.NAME}_spike.csv"


def seeded_rank(seed, key):
    """Return a reproducible pseudo-random sort key for a string.

    Args:
        seed: Seed mixed into the hash.
        key: The string to rank, such as a sequence ID or path.

    Returns:
        A hex digest; sorting by it gives a seeded shuffle.
    """
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def rank_expr(column, seed):
    """Return a polars expression ranking a string column by ``seeded_rank``.

    Args:
        column: Name of the string column to rank.
        seed: Seed passed to ``seeded_rank``.

    Returns:
        An expression producing a ``rank`` column of hex digests.
    """
    return (
        pl.col(column)
        .map_elements(lambda key: seeded_rank(seed, key), return_dtype=pl.String)
        .alias("rank")
    )


def select(
    frame,
    *,
    sequences_per_class_site,
    frames_per_sequence,
    min_conf,
    n_detector_check,
    seed,
):
    """Select the spike's images from a manifest.

    Args:
        frame: The manifest, as from ``manifest.load``.
        sequences_per_class_site: Maximum sequences of one class at one site.
        frames_per_sequence: Maximum frames kept from each sequence.
        min_conf: Minimum MDv5a animal confidence for a classify frame.
        n_detector_check: Number of random images for the detector check.
        seed: Seed for both random choices.

    Returns:
        A DataFrame with columns ``path``, ``classify`` and
        ``detector_check`` (booleans), one row per selected image, sorted
        by path.
    """
    folds = frame.filter(pl.col("split") != HOLDOUT)
    detected = folds.filter(pl.col("md_conf") >= min_conf)
    sequences = (
        detected.select("seq_id", "label", "site")
        .unique()
        .with_columns(rank_expr("seq_id", seed))
        .sort("rank")
        .group_by("label", "site", maintain_order=True)
        .head(sequences_per_class_site)
    )
    classify = (
        detected.join(sequences.select("seq_id"), on="seq_id")
        .sort(["seq_id", "md_conf", "frame_num"], descending=[False, True, False])
        .group_by("seq_id", maintain_order=True)
        .head(frames_per_sequence)
        .select("path")
    )
    check = (
        folds.select("path")
        .with_columns(rank_expr("path", seed))
        .sort("rank")
        .head(n_detector_check)
        .select("path")
    )
    in_classify = set(classify["path"].to_list())
    in_check = set(check["path"].to_list())
    paths = sorted(in_classify | in_check)
    return pl.DataFrame(
        {
            "path": paths,
            "classify": [p in in_classify for p in paths],
            "detector_check": [p in in_check for p in paths],
        }
    )


def main(argv=None):
    """Write the spike selection from the committed manifest.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, default=manifest.OUT_DIR / f"{orinoquia.NAME}.csv"
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    selection = select(manifest.load(args.manifest), **SPIKE_SETTINGS)
    selection.write_csv(args.out)
    print(
        f"{selection.height} images: {selection['classify'].sum()} classify, "
        f"{selection['detector_check'].sum()} detector check -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
