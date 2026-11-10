"""Order sequences as a deployment would have seen them: by site, then date.

A camera delivers its sequences in time order, so anything that asks
"did the signal move as the season went on?" needs that order fixed and
reproducible. This module builds it from the committed manifest (which
holds every image's timestamp) and the crop index (which says which
sequences were classified), and cuts each site's stream into an early
and a late half.

Pure polars: no model, no randomness, so the same inputs always give the
same stream.
"""

import polars as pl

EARLY = "early"
LATE = "late"


def sequence_stream(index, manifest):
    """Return one row per classified sequence, in site then date order.

    A sequence's time is its first image's timestamp. Ties are broken by
    sequence ID, so the order never depends on how the inputs were sorted.

    Args:
        index: The crop index, with ``seq_id``, ``site``, ``split`` and
            ``label`` per frame.
        manifest: The image manifest, with ``seq_id`` and ``timestamp``
            per image.

    Returns:
        A DataFrame with ``seq_id``, ``site``, ``split``, ``label``,
        ``timestamp``, ``position`` (0-based place within its site) and
        ``half`` (``early`` for the first half of a site's sequences,
        ``late`` for the rest; an odd count gives the extra one to
        ``late``).

    Raises:
        ValueError: If a sequence's frames disagree on site, split or
            label, or a sequence has no timestamp in the manifest.
    """
    keys = ["site", "split", "label"]
    seqs = index.group_by("seq_id").agg(pl.col(k).unique() for k in keys)
    if any((seqs[k].list.len() > 1).any() for k in keys):
        raise ValueError("a sequence's frames disagree on site, split or label")
    seqs = seqs.with_columns(pl.col(k).list.first() for k in keys)

    times = manifest.group_by("seq_id").agg(pl.col("timestamp").min())
    stream = seqs.join(times, on="seq_id", how="left")
    if stream["timestamp"].null_count():
        raise ValueError("a classified sequence has no timestamp in the manifest")

    stream = stream.sort("site", "timestamp", "seq_id")
    position = pl.int_range(pl.len()).over("site")
    return stream.with_columns(position.alias("position")).with_columns(
        pl.when(pl.col("position") < pl.len().over("site") // 2)
        .then(pl.lit(EARLY))
        .otherwise(pl.lit(LATE))
        .alias("half")
    )
