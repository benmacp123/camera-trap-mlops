"""Fetch a dated snapshot of the AWS list prices the cost report uses.

Optional: the snapshot is committed (``data/prices/aws-eu-west-2.json``),
so the cost report never needs the network. Reads AWS's public Price List
files (no account or credentials) and keeps only the usage types the
batch route is billed for, each with its offer file and publication date.

Each price is the lowest tier with a non-zero rate: what a small workload
pays at list price. The monthly free tier (400,000 GB-seconds and a
million requests for Lambda) is not in the price list and is not applied
anywhere: the report prices metered usage, not the bill.

::

    PYTHONPATH=src python -m cloud.prices
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

OUT = Path("data/prices/aws-eu-west-2.json")
HOST = "https://pricing.us-east-1.amazonaws.com"
REGION = "eu-west-2"

# Name in the snapshot: (Price List service code, usage type).
WANTED = {
    "lambda_request": ("AWSLambda", "EUW2-Request"),
    "lambda_gb_second": ("AWSLambda", "EUW2-Lambda-GB-Second"),
    "s3_put_request": ("AmazonS3", "EUW2-Requests-Tier1"),
    "s3_get_request": ("AmazonS3", "EUW2-Requests-Tier2"),
    "s3_storage_gb_month": ("AmazonS3", "EUW2-TimedStorage-ByteHrs"),
    "ecr_storage_gb_month": ("AmazonECR", "EUW2-TimedStorage-ByteHrs"),
}


def get_json(path):
    """Fetch one Price List file.

    Args:
        path: Path on the Price List host.

    Returns:
        The parsed file.
    """
    with urlopen(HOST + path, timeout=300) as response:
        return json.load(response)


def pick(offer, usage_type):
    """Return the lowest non-zero tier of a usage type.

    Args:
        offer: One service's regional offer file.
        usage_type: The usage type, such as ``EUW2-Request``.

    Returns:
        Dict with ``sku``, ``usage_type``, ``unit``, ``usd`` and
        ``description``.

    Raises:
        LookupError: If the usage type does not name exactly one product,
            or has no paid on-demand tier.
    """
    skus = [
        sku
        for sku, product in offer["products"].items()
        if product["attributes"].get("usagetype") == usage_type
    ]
    if len(skus) != 1:
        raise LookupError(f"{usage_type}: {len(skus)} products")
    dimensions = [
        dimension
        for term in offer["terms"]["OnDemand"][skus[0]].values()
        for dimension in term["priceDimensions"].values()
    ]
    dimensions.sort(key=lambda d: float(d["beginRange"]))
    paid = [d for d in dimensions if float(d["pricePerUnit"]["USD"]) > 0]
    if not paid:
        raise LookupError(f"{usage_type}: no paid tier")
    return {
        "sku": skus[0],
        "usage_type": usage_type,
        "unit": paid[0]["unit"],
        "usd": float(paid[0]["pricePerUnit"]["USD"]),
        "description": paid[0]["description"],
    }


def main(argv=None):
    """Fetch each service's regional offer file and write the snapshot.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    services = get_json("/offers/v1.0/aws/index.json")["offers"]
    offers = {}
    for code in sorted({code for code, _ in WANTED.values()}):
        regions = get_json(services[code]["currentRegionIndexUrl"])["regions"]
        path = regions[REGION]["currentVersionUrl"]
        offers[code] = (path, get_json(path))
    prices = {}
    for name, (code, usage_type) in WANTED.items():
        path, offer = offers[code]
        prices[name] = {
            "service": code,
            **pick(offer, usage_type),
            "offer": HOST + path,
            "published": offer["publicationDate"],
        }
    snapshot = {
        "region": REGION,
        "currency": "USD",
        "retrieved": date.today().isoformat(),
        "prices": prices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    for name, entry in prices.items():
        print(f"{name}: {entry['usd']} USD per {entry['unit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
