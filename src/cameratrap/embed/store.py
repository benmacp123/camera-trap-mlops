"""Embedding caches on disk: one ``.npz`` per backbone, keyed by crop path.

Pure numpy, so the arms, the gate and the reports read caches without the
model stack. Each cache records what it was computed from (backbone,
weight revision, crop-index hash, preprocessing) so a cache cannot be
silently paired with a different crop set or model.

Vectors are L2-normalised float32, one row per crop, in ``paths`` order.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_ROOT = Path("data/cache/embeddings")
FILENAME = "embeddings.npz"


def file_sha256(path):
    """Return the SHA-256 of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(backbone, root=CACHE_ROOT):
    """Return where a backbone's merged cache lives.

    Args:
        backbone: Backbone name, such as ``bioclip2``.
        root: Root of the embedding caches.

    Returns:
        Path to the ``.npz`` file.
    """
    return root / backbone / FILENAME


def write(path, paths, vectors, meta, extras=None):
    """Write a merged cache.

    Args:
        path: Output ``.npz`` path; parent directories are created.
        paths: Crop paths, one per row of ``vectors``.
        vectors: Array of shape ``(len(paths), dim)``.
        meta: JSON-serialisable dict describing how the vectors were made.
        extras: Optional mapping of further arrays to store (BioCLIP 2's
            text embeddings and logit scale, for example).

    Raises:
        ValueError: If ``paths`` and ``vectors`` disagree in length or
            ``paths`` has duplicates.
    """
    if len(paths) != len(vectors):
        raise ValueError(f"{len(paths)} paths but {len(vectors)} vectors")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate crop paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name first, so a crash never leaves a
    # truncated cache that looks complete.
    partial = path.with_name(path.name + ".part.npz")
    np.savez(
        partial,
        paths=np.array(paths),
        vectors=np.asarray(vectors, dtype=np.float32),
        meta=np.array(json.dumps(meta, sort_keys=True)),
        **(extras or {}),
    )
    partial.replace(path)


def read(path):
    """Read a merged cache.

    Args:
        path: The ``.npz`` written by ``write``.

    Returns:
        Dict with ``paths`` (list of str), ``vectors``, ``meta`` (dict) and
        any extra arrays stored with the cache.
    """
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    out["paths"] = out["paths"].tolist()
    out["meta"] = json.loads(str(out["meta"]))
    return out


def check_meta(meta, expected):
    """Check that a cache was made from what the caller expects.

    Args:
        meta: The cache's ``meta`` dict.
        expected: Keys and values that must match, such as
            ``{"index_sha256": ...}``.

    Raises:
        ValueError: Naming every key whose value differs.
    """
    wrong = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        detail = "; ".join(
            f"{k}: cache {a!r}, expected {b!r}" for k, (a, b) in wrong.items()
        )
        raise ValueError(f"embedding cache does not match: {detail}")


def rows_for(cache, paths):
    """Return the cache rows for some crop paths, in the order given.

    Args:
        cache: A cache from ``read``.
        paths: Crop paths to look up.

    Returns:
        Array of shape ``(len(paths), dim)``.

    Raises:
        KeyError: If a path is not in the cache.
    """
    row = {p: i for i, p in enumerate(cache["paths"])}
    return cache["vectors"][[row[p] for p in paths]]


def cosine_agreement(cached, fresh):
    """Return the row-wise cosine similarity of two sets of vectors.

    Used to check that re-embedding a sample reproduces the cache: CPU
    float maths is not bit-identical across thread counts, so the check
    is a tolerance on cosine, not equality.

    Args:
        cached: Array of shape ``(n, dim)``.
        fresh: Array of the same shape.

    Returns:
        Array of ``n`` cosine similarities.
    """
    cached = cached / np.linalg.norm(cached, axis=1, keepdims=True)
    fresh = fresh / np.linalg.norm(fresh, axis=1, keepdims=True)
    return np.sum(cached * fresh, axis=1)
