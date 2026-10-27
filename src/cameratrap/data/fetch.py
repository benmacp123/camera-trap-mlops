"""Download images from LILA's public mirror and record their checksums.

Optional and separate from the analysis: every later stage runs from the
committed manifest and cached embeddings. Downloads run in parallel
(single fetches manage about one image a second) and resume, since files
already on disk are hashed rather than fetched again. Each image's SHA-256
is written into the manifest; a file whose hash disagrees with one
already recorded is reported, never silently overwritten.

Run as a script to fetch the spike selection::

    PYTHONPATH=src python -m cameratrap.data.fetch
"""

import argparse
import concurrent.futures
import datetime as dt
import functools
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

from cameratrap.data import manifest, orinoquia, subset

IMAGE_DIR = Path("data/images") / orinoquia.NAME


def download(url, dest, retries=3, timeout=60):
    """Download one file, hashing it as it streams to disk.

    Writes to a temporary name and renames on success, so an interrupted
    download never leaves a partial file under the real name.

    Args:
        url: Source URL.
        dest: Destination path; parent directories are created.
        retries: Attempts before giving up.
        timeout: Seconds to wait on the connection.

    Returns:
        The SHA-256 hex digest of the downloaded file.

    Raises:
        OSError: If every attempt fails (``urllib.error.URLError`` is one).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with open(partial, "wb") as handle:
                    for block in iter(functools.partial(response.read, 1 << 16), b""):
                        digest.update(block)
                        handle.write(block)
            os.replace(partial, dest)
            return digest.hexdigest()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_one(path, base_url, image_dir):
    """Return an image's checksum, downloading it only if it is not on disk.

    Args:
        path: The image's manifest path, relative to ``base_url``.
        base_url: URL prefix ending in ``/``.
        image_dir: Local root the manifest paths are stored under.

    Returns:
        The SHA-256 hex digest of the local file.
    """
    dest = image_dir / path
    if dest.exists():
        return manifest.sha256_file(dest)
    return download(base_url + path, dest)


def fetch(paths, base_url, image_dir, workers=32, progress_every=500):
    """Fetch many images in parallel.

    Args:
        paths: Manifest paths to fetch.
        base_url: URL prefix ending in ``/``.
        image_dir: Local root to store them under.
        workers: Number of parallel downloads.
        progress_every: Print a progress line after this many images.

    Returns:
        A pair ``(hashes, failures)``: mapping of path to SHA-256 for every
        image now on disk, and mapping of path to error message for the
        rest.
    """
    hashes, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_one, p, base_url, image_dir): p for p in paths}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                hashes[path] = future.result()
            except OSError as error:
                failures[path] = str(error)
            if done % progress_every == 0:
                print(f"{done}/{len(futures)} ({len(failures)} failed)", flush=True)
    return hashes, failures


def record_checksums(frame, hashes):
    """Fill the manifest's ``sha256`` column from fresh hashes.

    Args:
        frame: The manifest.
        hashes: Mapping of path to SHA-256.

    Returns:
        A pair ``(frame, mismatches)``: the manifest with empty checksums
        filled, and the sorted paths whose fresh hash disagrees with the
        one already recorded (those keep the recorded value).
    """
    fresh = pl.DataFrame(
        {"path": list(hashes), "fresh": list(hashes.values())},
        schema={"path": pl.String, "fresh": pl.String},
    )
    joined = frame.join(fresh, on="path", how="left", maintain_order="left")
    mismatches = joined.filter(
        pl.col("sha256").is_not_null()
        & pl.col("fresh").is_not_null()
        & (pl.col("sha256") != pl.col("fresh"))
    )["path"].to_list()
    filled = joined.with_columns(pl.coalesce("sha256", "fresh").alias("sha256"))
    return filled.drop("fresh"), sorted(mismatches)


def main(argv=None):
    """Fetch the selected images and record their checksums in the manifest.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 when every image was fetched and matched,
        1 otherwise.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--manifest-dir", type=Path, default=manifest.OUT_DIR)
    parser.add_argument("--selection", type=Path, default=subset.OUT_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--base-url", default=orinoquia.IMAGE_BASE_URL)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)

    csv_path = args.manifest_dir / f"{orinoquia.NAME}.csv"
    json_path = args.manifest_dir / f"{orinoquia.NAME}.json"
    frame = manifest.load(csv_path)
    paths = pl.read_csv(args.selection)["path"].to_list()
    hashes, failures = fetch(paths, args.base_url, args.image_dir, args.workers)
    frame, mismatches = record_checksums(frame, hashes)
    frame.write_csv(csv_path)

    # Keep the date checksums were first recorded, so a re-run that only
    # verifies them leaves the committed record unchanged.
    record = json.loads(json_path.read_text())
    previous = record.get("images_fetched", {})
    record["images_fetched"] = {
        "base_url": args.base_url,
        "first_recorded": previous.get("first_recorded", dt.date.today().isoformat()),
        "checksums_recorded": frame["sha256"].count(),
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    print(
        f"{len(hashes)} on disk, {len(failures)} failed, {len(mismatches)} mismatched"
    )
    for path, error in sorted(failures.items())[:10]:
        print(f"failed: {path}: {error}", file=sys.stderr)
    for path in mismatches[:10]:
        print(f"checksum differs from manifest: {path}", file=sys.stderr)
    return 0 if not failures and not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
