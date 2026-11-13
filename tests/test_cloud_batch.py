"""Tests for the batch driver, run against a fake target with no AWS."""

import json

import numpy as np
import polars as pl
import pytest

from cloud import batch, handler

REPORT = (
    "START RequestId: abc Version: $LATEST\n"
    "END RequestId: abc\n"
    "REPORT RequestId: abc\tDuration: 41230.55 ms\tBilled Duration: 44902 ms\t"
    "Memory Size: 3008 MB\tMax Memory Used: 1210 MB\tInit Duration: 3671.20 ms\t\n"
)


class FakeTarget:
    """A target that scores keys in process and keeps results in memory."""

    bucket = "bucket"

    def __init__(self, fail_on=None):
        """Start with nothing stored.

        Args:
            fail_on: Shard output key that raises, to imitate a failed shard.
        """
        self.objects = {}
        self.invoked = []
        self.fail_on = fail_on

    def invoke(self, event):
        """Score a shard: the logit of key ``k7`` is 7 in column 7 mod 2.

        Args:
            event: The handler's event.

        Returns:
            A handler-style payload and a fixed report.

        Raises:
            RuntimeError: For the shard named by ``fail_on``.
        """
        if event["output_key"] == self.fail_on:
            raise RuntimeError("boom")
        self.invoked.append(event["output_key"])
        logits = np.zeros((len(event["keys"]), 2))
        for row, key in enumerate(event["keys"]):
            logits[row, int(key[1:]) % 2] = int(key[1:]) + 1
        self.objects[event["output_key"]] = handler.pack(event["keys"], logits)
        payload = {"n": len(event["keys"]), "cold": not self.invoked[:-1]}
        return payload, batch.parse_report(REPORT)

    def read(self, key):
        """Return a stored result.

        Args:
            key: Object key.

        Returns:
            The stored bytes.
        """
        return self.objects[key]


def test_parse_report_reads_a_cold_start():
    """Billed duration, memory and init time come from Lambda's REPORT line."""
    report = batch.parse_report(REPORT)
    assert report == {
        "duration_ms": 41230.55,
        "billed_ms": 44902,
        "memory_mb": 3008,
        "max_memory_mb": 1210,
        "init_ms": 3671.2,
    }


def test_parse_report_reads_a_warm_start():
    """A warm start has no init phase, and Duration is not Billed Duration."""
    line = (
        "REPORT RequestId: abc\tDuration: 100.50 ms\tBilled Duration: 101 ms\t"
        "Memory Size: 1769 MB\tMax Memory Used: 900 MB\t\n"
    )
    report = batch.parse_report(line)
    assert report["init_ms"] is None
    assert (report["duration_ms"], report["billed_ms"]) == (100.5, 101)


def test_parse_report_refuses_a_log_without_one():
    """A log tail with no REPORT line is an error, not a free invocation."""
    with pytest.raises(ValueError, match="REPORT"):
        batch.parse_report("START RequestId: abc\n")


def test_shards_cover_every_key_once():
    """Sharding neither drops nor repeats a crop."""
    keys = [f"k{i}" for i in range(10)]
    shards = batch.make_shards(keys, 4)
    assert [len(s) for s in shards] == [4, 4, 2]
    assert [k for s in shards for k in s] == keys
    with pytest.raises(ValueError):
        batch.make_shards(keys, 0)


def test_sample_is_seeded_and_sorted():
    """The sweep's sample is the same crops, in the same order, every time."""
    keys = [f"k{i:03d}" for i in range(100)]
    first = batch.sample_keys(keys, 10, seed=0)
    assert first == batch.sample_keys(keys, 10, seed=0) == sorted(first)
    assert len(set(first)) == 10
    assert batch.sample_keys(keys, None, seed=0) == keys


def test_run_shards_resumes_without_reinvoking(tmp_path):
    """After a failed shard, a second attempt invokes only what is missing."""
    shards = batch.make_shards([f"k{i}" for i in range(6)], 2)
    progress = tmp_path / "run.jsonl"
    failing = FakeTarget(fail_on=batch.output_key("t", 2))
    with pytest.raises(RuntimeError):
        batch.run_shards(failing, "t", shards, 1, progress)
    assert len(batch.read_progress(progress)) == 2

    target = FakeTarget()
    target.objects = failing.objects
    rows = batch.run_shards(target, "t", shards, 1, progress)
    assert target.invoked == [batch.output_key("t", 2)]
    assert [row["shard"] for row in rows] == [0, 1, 2]
    assert rows[0]["billed_ms"] == 44902 and rows[0]["n"] == 2
    assert set(rows[0]) == set(batch.FIELDS)


def test_collect_joins_shards_in_order(tmp_path):
    """Collected logits line up with the keys that were sent."""
    shards = batch.make_shards([f"k{i}" for i in range(5)], 2)
    target = FakeTarget()
    batch.run_shards(target, "t", shards, 2, tmp_path / "run.jsonl")
    keys, logits = batch.collect(target, "t", shards)
    assert keys == [f"k{i}" for i in range(5)]
    assert logits.max(axis=1).tolist() == [1, 2, 3, 4, 5]


def test_collect_refuses_a_shard_with_other_keys(tmp_path):
    """A stored result for different crops is never joined in silently."""
    shards = batch.make_shards(["k0", "k1"], 2)
    target = FakeTarget()
    batch.run_shards(target, "t", shards, 1, tmp_path / "run.jsonl")
    target.objects[batch.output_key("t", 0)] = handler.pack(
        ["k0", "k9"], np.zeros((2, 2))
    )
    with pytest.raises(RuntimeError, match="keys differ"):
        batch.collect(target, "t", shards)


@pytest.fixture(name="world")
def fixture_world():
    """A four-crop index, cache and head where local logits are the embeddings.

    Returns:
        Dict with ``index``, ``cache``, ``head`` and the local ``logits``.
    """
    paths = ["a", "b", "c", "d"]
    vectors = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 0.5], [0.2, 0.9]])
    return {
        "index": pl.DataFrame(
            {
                "path": paths,
                "seq_id": ["s1", "s1", "s2", "s3"],
                "split": ["fold0", "fold0", "holdout", "holdout"],
            }
        ),
        "cache": {"paths": paths, "vectors": vectors},
        "head": {"coef": np.eye(2), "intercept": np.zeros(2), "classes": ["x", "y"]},
        "logits": vectors,
    }


def test_parity_passes_on_identical_logits(world):
    """The route's own logits, unchanged, agree with the local route."""
    committed = pl.DataFrame({"seq_id": ["s2", "s3"], "pred": ["x", "y"]})
    check = batch.parity(
        world["cache"]["paths"],
        world["logits"],
        world["index"],
        world["cache"],
        world["head"],
        committed,
    )
    assert check["passed"]
    assert check["sequences"] == 3
    assert check["holdout_sequences_compared"] == 2
    assert check["max_abs_logit_difference"] == 0.0


def test_parity_fails_on_a_flipped_prediction(world):
    """One crop scored differently is counted at crop and sequence level."""
    logits = world["logits"].copy()
    logits[3] = [5.0, 0.0]
    committed = pl.DataFrame({"seq_id": ["s2", "s3"], "pred": ["x", "y"]})
    check = batch.parity(
        world["cache"]["paths"],
        logits,
        world["index"],
        world["cache"],
        world["head"],
        committed,
    )
    assert not check["passed"]
    assert check["crop_predictions_differing"] == 1
    assert check["sequence_predictions_differing"] == 1
    assert check["holdout_sequences_differing"] == 1


def test_parity_skips_partly_scored_holdout_sequences(world):
    """A hold-out sequence missing a crop is not held against the committed table."""
    index = world["index"].with_columns(
        pl.Series("split", ["holdout", "holdout", "holdout", "holdout"])
    )
    committed = pl.DataFrame({"seq_id": ["s1", "s2"], "pred": ["y", "x"]})
    check = batch.parity(
        ["a", "c"],
        world["logits"][[0, 2]],
        index,
        world["cache"],
        world["head"],
        committed,
    )
    assert check["holdout_sequences_compared"] == 1
    assert check["passed"]


def test_write_results_writes_three_files(tmp_path):
    """A finished run leaves its invocations, logits and parity on disk."""
    rows = [{name: 0 for name in batch.FIELDS}]
    batch.write_results("t", rows, ["a"], np.zeros((1, 2)), {"passed": True}, tmp_path)
    assert (tmp_path / "invocations_t.csv").read_text().splitlines()[0] == ",".join(
        batch.FIELDS
    )
    with np.load(tmp_path / "logits_t.npz") as data:
        assert data["paths"].tolist() == ["a"]
    assert json.loads((tmp_path / "parity_t.json").read_text()) == {"passed": True}


def test_load_outputs_needs_deployed_functions(tmp_path):
    """The driver stops with a message before the functions exist."""
    path = tmp_path / "outputs.json"
    with pytest.raises(SystemExit, match="deploy first"):
        batch.load_outputs(path)
    path.write_text(json.dumps({"bucket": {"value": "b"}, "functions": {"value": {}}}))
    with pytest.raises(SystemExit, match="no functions"):
        batch.load_outputs(path)
    assert batch.load_outputs(path, need_functions=False)["bucket"] == "b"
    path.write_text(
        json.dumps({"bucket": {"value": "b"}, "functions": {"value": {"1769": "f"}}})
    )
    assert batch.load_outputs(path)["functions"] == {"1769": "f"}


def test_summarise_uses_warm_invocations_only():
    """The sweep compares sizes on warm invocations, leaving the cold start out."""
    prices = {
        "lambda_gb_second": 0.00002,
        "lambda_request": 0.0,
        "s3_get_request": 0.0,
        "s3_put_request": 0.0,
    }
    rows = [
        {"n": 64, "cold": True, "billed_ms": 90_000, "memory_mb": 1024},
        {"n": 64, "cold": False, "billed_ms": 32_000, "memory_mb": 1024},
        {"n": 64, "cold": False, "billed_ms": 32_000, "memory_mb": 1024},
    ]
    summary = batch.summarise(rows, prices)
    assert summary["warm_invocations"] == 2
    assert summary["crops_per_second"] == 2.0
    assert summary["usd_per_10k_crops"] == pytest.approx(64 * 0.00002 / 128 * 10_000)
    assert batch.summarise(rows[:1], prices) is None
