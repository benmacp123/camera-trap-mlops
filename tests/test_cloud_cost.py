"""Tests for the cost model."""

import json

import pytest

from cloud import cost

PRICES = {
    "lambda_gb_second": 0.00002,
    "lambda_request": 0.0000002,
    "s3_get_request": 0.0000004,
    "s3_put_request": 0.000005,
    "s3_storage_gb_month": 0.024,
    "ecr_storage_gb_month": 0.1,
}


def test_gb_seconds_uses_binary_gigabytes():
    """One second at 1,024 MB is one GB-second, as Lambda bills it."""
    assert cost.gb_seconds(1000, 1024) == 1.0
    assert cost.gb_seconds(500, 2048) == 1.0


def test_run_cost_adds_each_charge():
    """A run's total is compute, requests and S3 requests, per 10k crops."""
    rows = [
        {"n": 100, "billed_ms": 10_000, "memory_mb": 1024},
        {"n": 100, "billed_ms": 30_000, "memory_mb": 1024},
    ]
    priced = cost.run_cost(PRICES, rows)
    assert priced["gb_seconds"] == pytest.approx(40)
    assert priced["compute_usd"] == pytest.approx(40 * 0.00002)
    assert priced["requests_usd"] == pytest.approx(2 * 0.0000002)
    assert priced["s3_get_usd"] == pytest.approx(200 * 0.0000004)
    assert priced["s3_put_usd"] == pytest.approx(2 * 0.000005)
    assert priced["total_usd"] == pytest.approx(
        sum(
            priced[k]
            for k in ("compute_usd", "requests_usd", "s3_get_usd", "s3_put_usd")
        )
    )
    assert priced["usd_per_10k_crops"] == pytest.approx(priced["total_usd"] * 50)


def test_doubling_memory_at_the_same_duration_doubles_compute():
    """Memory is a price multiplier: more only pays if it shortens the run."""
    small = cost.compute_usd(PRICES, 60_000, 1024)
    assert cost.compute_usd(PRICES, 60_000, 2048) == pytest.approx(2 * small)


def test_run_cost_refuses_no_invocations():
    """An empty run has no cost per crop to report."""
    with pytest.raises(ValueError):
        cost.run_cost(PRICES, [])


def test_fixed_cost_scales_storage_with_days():
    """Storage is charged by the day the stack stands; the upload once."""
    one = cost.fixed_cost(PRICES, 1000, 2e9, 1e9, days=1)
    three = cost.fixed_cost(PRICES, 1000, 2e9, 1e9, days=3)
    assert three["s3_storage_usd"] == pytest.approx(3 * one["s3_storage_usd"])
    assert three["ecr_storage_usd"] == pytest.approx(3 * one["ecr_storage_usd"])
    assert three["upload_usd"] == one["upload_usd"] == pytest.approx(0.005)
    assert one["s3_storage_usd"] == pytest.approx(2 * 0.024 / 30)


def test_committed_snapshot_has_every_price():
    """The committed price snapshot holds each price the model reads."""
    prices = cost.load_prices()
    assert set(PRICES) <= set(prices)
    assert all(value > 0 for value in prices.values())
    snapshot = json.loads(cost.PRICES.read_text())
    assert snapshot["region"] == "eu-west-2"
