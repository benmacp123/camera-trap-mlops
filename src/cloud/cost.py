"""Price a batch run from its metered usage and a committed price snapshot.

Lambda bills each invocation for its billed duration times the memory it
was configured with (GB-seconds), plus a flat fee per request. S3 bills
per request. Everything here is list price: the monthly free tier is
ignored, because it depends on what else the account ran that month and
would make a small run look free.

Pure arithmetic over plain values, with no AWS SDK, so it is tested and
rerun without an account.
"""

import json
from pathlib import Path

PRICES = Path("data/prices/aws-eu-west-2.json")
SECONDS_PER_MONTH = 30 * 24 * 3600


def load_prices(path=PRICES):
    """Read the price snapshot.

    Args:
        path: The JSON written by ``cloud.prices``.

    Returns:
        Mapping of price name to US dollars per unit.
    """
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: entry["usd"] for name, entry in snapshot["prices"].items()}


def gb_seconds(billed_ms, memory_mb):
    """Return the GB-seconds Lambda bills for some duration at a memory size.

    Args:
        billed_ms: Billed duration in milliseconds.
        memory_mb: Configured memory in MB.

    Returns:
        GB-seconds; Lambda counts 1 GB as 1,024 MB.
    """
    return billed_ms / 1000 * memory_mb / 1024


def compute_usd(prices, billed_ms, memory_mb):
    """Return the duration charge for some billed time.

    Args:
        prices: Prices from ``load_prices``.
        billed_ms: Billed duration in milliseconds.
        memory_mb: Configured memory in MB.

    Returns:
        US dollars.
    """
    return gb_seconds(billed_ms, memory_mb) * prices["lambda_gb_second"]


def request_usd(prices, invocations):
    """Return the per-request charge for some invocations.

    Args:
        prices: Prices from ``load_prices``.
        invocations: Number of invocations.

    Returns:
        US dollars.
    """
    return invocations * prices["lambda_request"]


def run_cost(prices, rows):
    """Price a set of invocations.

    Each invocation reads one object per crop and writes one result
    object. Data transfer between Lambda and S3 in one region is free.

    Args:
        prices: Prices from ``load_prices``.
        rows: One mapping per invocation with ``n`` (crops), ``billed_ms``
            and ``memory_mb``.

    Returns:
        Dict with ``crops``, ``invocations``, ``gb_seconds``, the charge
        for each of ``compute``, ``requests``, ``s3_get`` and ``s3_put``
        in US dollars, their ``total_usd`` and ``usd_per_10k_crops``.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("no invocations to price")
    n_crops = sum(row["n"] for row in rows)
    parts = {
        "compute_usd": sum(
            compute_usd(prices, row["billed_ms"], row["memory_mb"]) for row in rows
        ),
        "requests_usd": request_usd(prices, len(rows)),
        "s3_get_usd": n_crops * prices["s3_get_request"],
        "s3_put_usd": len(rows) * prices["s3_put_request"],
    }
    total = sum(parts.values())
    return {
        "crops": n_crops,
        "invocations": len(rows),
        "gb_seconds": sum(gb_seconds(r["billed_ms"], r["memory_mb"]) for r in rows),
        **parts,
        "total_usd": total,
        "usd_per_10k_crops": total / n_crops * 10_000,
    }


def fixed_cost(prices, n_crops, crop_bytes, image_bytes, days):
    """Price what the experiment pays whether or not anything is classified.

    Args:
        prices: Prices from ``load_prices``.
        n_crops: Crops uploaded, one PUT request each.
        crop_bytes: Total size of the crops in S3.
        image_bytes: Compressed size of the container image in ECR.
        days: How long the stack stood before it was destroyed.

    Returns:
        Dict with the charge for ``upload``, ``s3_storage`` and
        ``ecr_storage`` in US dollars and their ``total_usd``.
    """
    months = days * 24 * 3600 / SECONDS_PER_MONTH
    parts = {
        "upload_usd": n_crops * prices["s3_put_request"],
        "s3_storage_usd": crop_bytes / 1e9 * months * prices["s3_storage_gb_month"],
        "ecr_storage_usd": image_bytes / 1e9 * months * prices["ecr_storage_gb_month"],
    }
    return {**parts, "total_usd": sum(parts.values())}
