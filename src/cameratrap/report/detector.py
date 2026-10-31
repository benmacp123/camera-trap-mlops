"""Detector-miss report: where MDv5a gives the classifiers nothing to see.

Reads only committed files (the manifest's best MDv5a animal confidence
per frame, the crop index and the frames dropped for a person box), so it
reproduces without images. Writes ``results/detector/per_site.csv``,
``per_class.csv`` and ``summary.json``.

A sequence is **missed** when none of its frames has an animal box at the
crop threshold; it reaches no classifier and counts against every arm
alike. A frame is missed when it carries the sequence's animal label but
no box at the threshold, which a sequence-level count hides. Hold-out
sites are included: this measures the upstream detector, not any arm, so
nothing here is tuned against them.

::

    PYTHONPATH=src python -m cameratrap.report.detector
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from cameratrap.data import crops, manifest, orinoquia

OUT_DIR = Path("results/detector")


def sequence_table(frame, index, excluded, min_conf):
    """Return one row per manifest sequence with its detector outcome.

    Args:
        frame: The manifest.
        index: The crop index.
        excluded: Frames dropped for a person box.
        min_conf: The crop threshold on MDv5a animal confidence.

    Returns:
        A DataFrame with ``seq_id``, ``site``, ``split``, ``label``,
        ``frames_labelled``, ``frames_missed``, ``missed`` (no frame with a
        box), ``frames_excluded_person``, ``crops`` and ``lost_to_person``
        (boxes found, but every candidate frame dropped).
    """
    labelled = pl.col("image_label") == pl.col("label")
    per_seq = frame.group_by("seq_id").agg(
        pl.first("site"),
        pl.first("split"),
        pl.first("label"),
        labelled.sum().alias("frames_labelled"),
        (labelled & (pl.col("md_conf") < min_conf)).sum().alias("frames_missed"),
        (pl.col("md_conf").max() < min_conf).alias("missed"),
    )
    counts = [
        (index, "crops"),
        (excluded, "frames_excluded_person"),
    ]
    for table, name in counts:
        per_seq = per_seq.join(
            table.group_by("seq_id").len(name), on="seq_id", how="left"
        ).with_columns(pl.col(name).fill_null(0))
    return per_seq.with_columns(
        ((pl.col("crops") == 0) & ~pl.col("missed")).alias("lost_to_person")
    ).sort("seq_id")


def summarise(sequences, by):
    """Aggregate the sequence table by site or class.

    Args:
        sequences: Output of ``sequence_table``.
        by: Column names to group by, such as ``["site", "split"]``.

    Returns:
        One row per group with sequence and frame counts and the
        sequence and frame miss rates, sorted by the group columns.
    """
    return (
        sequences.group_by(by)
        .agg(
            pl.len().alias("sequences"),
            pl.col("missed").sum().alias("sequences_missed"),
            pl.col("lost_to_person").sum().alias("sequences_lost_to_person"),
            pl.col("frames_labelled").sum(),
            pl.col("frames_missed").sum(),
            pl.col("frames_excluded_person").sum(),
            pl.col("crops").sum(),
        )
        .with_columns(
            (pl.col("sequences_missed") / pl.col("sequences")).alias(
                "sequence_miss_rate"
            ),
            (pl.col("frames_missed") / pl.col("frames_labelled")).alias(
                "frame_miss_rate"
            ),
        )
        .sort(by)
    )


def summary(sequences, per_site, min_conf):
    """Return the headline numbers for the JSON record.

    Args:
        sequences: Output of ``sequence_table``.
        per_site: Output of ``summarise`` by site.
        min_conf: The crop threshold used.

    Returns:
        A JSON-serialisable dict.
    """
    rates = per_site["sequence_miss_rate"]
    worst = per_site.sort("sequence_miss_rate", descending=True).row(0, named=True)
    return {
        "min_conf": min_conf,
        "sequences": sequences.height,
        "sequences_missed": int(sequences["missed"].sum()),
        "sequences_lost_to_person": int(sequences["lost_to_person"].sum()),
        "sequences_with_crop": int((sequences["crops"] > 0).sum()),
        "crops": int(sequences["crops"].sum()),
        "frames_labelled": int(sequences["frames_labelled"].sum()),
        "frames_missed": int(sequences["frames_missed"].sum()),
        "frames_excluded_person": int(sequences["frames_excluded_person"].sum()),
        "sites": per_site.height,
        "sites_with_a_miss": int((per_site["sequences_missed"] > 0).sum()),
        "site_sequence_miss_rate_mean": round(rates.mean(), 4),
        "site_sequence_miss_rate_max": round(rates.max(), 4),
        "worst_site": worst["site"],
    }


def main(argv=None):
    """Write the detector-miss report from the committed manifest and index.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", type=Path, default=manifest.OUT_DIR / f"{orinoquia.NAME}.csv"
    )
    parser.add_argument("--index", type=Path, default=crops.INDEX_PATH)
    parser.add_argument("--excluded", type=Path, default=crops.EXCLUDED_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    min_conf = crops.CROP_SETTINGS["min_conf"]
    sequences = sequence_table(
        manifest.load(args.manifest),
        crops.load_index(args.index),
        crops.load_index(args.excluded),
        min_conf,
    )
    per_site = summarise(sequences, ["site", "split"])
    per_class = summarise(sequences, ["label"])
    record = summary(sequences, per_site, min_conf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_site.write_csv(args.out_dir / "per_site.csv", float_precision=4)
    per_class.write_csv(args.out_dir / "per_class.csv", float_precision=4)
    (args.out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
