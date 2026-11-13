"""Tests for the cost report."""

import json

import pytest

from cloud import report

HEADER = (
    "label,shard,n,memory_mb,cold,wall_seconds,duration_ms,billed_ms,"
    "max_memory_mb,init_ms,load_seconds,download_seconds,model_seconds,"
    "upload_seconds,cpu_count\n"
)
PRICES = {
    "lambda_gb_second": 0.00002,
    "lambda_request": 0.0000002,
    "s3_get_request": 0.0000004,
    "s3_put_request": 0.000005,
    "s3_storage_gb_month": 0.024,
    "ecr_storage_gb_month": 0.1,
}


def write_invocations(path):
    """Write a two-invocation table, one cold and one warm.

    Args:
        path: Where to write the CSV.
    """
    path.write_text(
        HEADER
        + "x,0,100,1024,True,12,11000,12000,900,,5,1,5,0,2\n"
        + "x,1,100,1024,False,10,9000,10000,950,,0,1,8,0,2\n",
        encoding="utf-8",
    )


def test_read_invocations_types_the_columns(tmp_path):
    """Numbers come back as numbers and the cold flag as a bool."""
    write_invocations(tmp_path / "invocations_x.csv")
    rows = report.read_invocations(tmp_path / "invocations_x.csv")
    assert rows[0]["cold"] is True and rows[1]["cold"] is False
    assert rows[1]["billed_ms"] == 10_000.0 and rows[1]["n"] == 100


def test_describe_run_counts_cold_starts_and_peak_memory(tmp_path):
    """The run summary includes cold starts in cost but not in the warm figure."""
    write_invocations(tmp_path / "invocations_x.csv")
    rows = report.read_invocations(tmp_path / "invocations_x.csv")
    described = report.describe_run(PRICES, rows)
    assert described["cold_invocations"] == 1
    assert described["peak_memory_mb"] == 950.0
    assert described["crops_per_billed_second"] == pytest.approx(200 / 22, abs=1e-3)
    assert described["warm"]["warm_invocations"] == 1
    assert described["cost"]["gb_seconds"] == pytest.approx(22)


def test_stack_cost_uses_the_recorded_size_and_age():
    """Fixed costs follow stack.json: uploads, S3 storage and ECR storage."""
    stack = {"crop_bytes": 2e9, "image_bytes": 1e9, "days": 30}
    fixed = report.stack_cost(PRICES, stack, 1000)
    assert fixed["upload_usd"] == pytest.approx(1000 * 0.000005)
    assert fixed["s3_storage_usd"] == pytest.approx(2 * 0.024)
    assert fixed["ecr_storage_usd"] == pytest.approx(0.1)


def test_plot_writes_a_png(tmp_path):
    """The figure is written to the requested path."""
    write_invocations(tmp_path / "invocations_x.csv")
    rows = report.read_invocations(tmp_path / "invocations_x.csv")
    out = tmp_path / "fig.png"
    report.plot({"x": report.describe_run(PRICES, rows)}, out)
    assert out.stat().st_size > 0 and json.dumps(1)
