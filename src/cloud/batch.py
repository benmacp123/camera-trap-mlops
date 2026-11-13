"""Drive the batch route from the laptop: upload crops, invoke shards, collect.

The driver splits the crop index into shards of crop keys and invokes the
Lambda function once per shard, a few at a time. Each invocation writes
its logits to the bucket and returns its own timings; Lambda's ``REPORT``
log line, returned with the response, gives the billed duration and peak
memory. Nothing else is deployed: no queue and no trigger, so concurrency
is simply how many invocations the driver keeps open.

Commands (the AWS ones read ``build/terraform-outputs.json`` and need
credentials; ``emulate`` needs only Docker)::

    PYTHONPATH=src python -m cloud.batch upload
    PYTHONPATH=src python -m cloud.batch sweep
    PYTHONPATH=src python -m cloud.batch run --memory 3008
    PYTHONPATH=src python -m cloud.batch emulate
    PYTHONPATH=src python -m cloud.batch leftovers

A run appends each finished shard to ``build/cloud/<label>.jsonl`` and
resumes from it, so a dropped connection does not repeat paid work. When
every shard is done it writes ``results/cloud/invocations_<label>.csv``,
``logits_<label>.npz`` and ``parity_<label>.json``.
"""

import argparse
import base64
import concurrent.futures
import csv
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import boto3
import botocore.config
import numpy as np
import polars as pl

from cameratrap.arms import head as head_module
from cameratrap.arms import scoring
from cameratrap.data import crops
from cameratrap.embed import store
from cloud import cost

OUTPUTS = Path("build/terraform-outputs.json")
PROGRESS_DIR = Path("build/cloud")
OUT_DIR = Path("results/cloud")
EMULATOR_URL = "http://localhost:9000/2015-03-31/functions/function/invocations"
EMULATOR_ROOT = Path("build/emulator")
EMULATOR_BUCKET = "bucket"
CROP_PREFIX = "crops/"
PROJECT_TAG = "camera-trap-mlops"
HOLDOUT_SEQUENCES = Path("results/holdout/sequences_dinov2_probe.csv")

# Fixed before the measured run (2026-11-12). The sweep reuses one seeded
# sample at every memory size, one invocation at a time, so sizes are
# compared on the same crops and the first invocation of each is cold.
SWEEP_SAMPLE = 512
SWEEP_SHARD = 64
SWEEP_SEED = 0
RUN_SHARD = 256
RUN_CONCURRENCY = 8
# Used only for the estimate printed before a run, never for a result.
ASSUMED_CROPS_PER_SECOND = 2.0

FIELDS = [
    "label",
    "shard",
    "n",
    "memory_mb",
    "cold",
    "wall_seconds",
    "duration_ms",
    "billed_ms",
    "max_memory_mb",
    "init_ms",
    "load_seconds",
    "download_seconds",
    "model_seconds",
    "upload_seconds",
    "cpu_count",
]

REPORT_PATTERNS = {
    "duration_ms": (r"\tDuration: ([\d.]+) ms", float),
    "billed_ms": (r"Billed Duration: (\d+) ms", int),
    "memory_mb": (r"Memory Size: (\d+) MB", int),
    "max_memory_mb": (r"Max Memory Used: (\d+) MB", int),
    "init_ms": (r"Init Duration: ([\d.]+) ms", float),
}


def parse_report(log_text):
    """Read Lambda's ``REPORT`` line from the tail of an invocation's log.

    Args:
        log_text: The log tail Lambda returns with ``LogType="Tail"``.

    Returns:
        Dict with ``duration_ms``, ``billed_ms``, ``memory_mb``,
        ``max_memory_mb`` and ``init_ms``; ``init_ms`` is ``None`` on a
        warm start, which reports no init phase.

    Raises:
        ValueError: If the log holds no complete ``REPORT`` line. The tail
            is 4 KB, so a handler that logs heavily could push it out.
    """
    lines = [line for line in log_text.splitlines() if line.startswith("REPORT")]
    if not lines:
        raise ValueError("no REPORT line in the log tail")
    report = {}
    for name, (pattern, convert) in REPORT_PATTERNS.items():
        match = re.search(pattern, lines[-1])
        report[name] = convert(match.group(1)) if match else None
    if report["billed_ms"] is None or report["memory_mb"] is None:
        raise ValueError(f"incomplete REPORT line: {lines[-1]!r}")
    return report


def make_shards(keys, size):
    """Split crop keys into consecutive shards.

    Args:
        keys: Crop keys in index order.
        size: Most keys per shard.

    Returns:
        A list of lists; every key appears once, order kept.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError("shard size must be at least 1")
    return [keys[i : i + size] for i in range(0, len(keys), size)]


def sample_keys(keys, n, seed):
    """Draw a fixed sample of crop keys.

    Args:
        keys: Every crop key.
        n: Sample size; ``None`` or at least ``len(keys)`` keeps them all.
        seed: Seed for the draw.

    Returns:
        The sampled keys, sorted, so shards are the same on every run.
    """
    if n is None or n >= len(keys):
        return list(keys)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(keys), size=n, replace=False).tolist())


class LambdaTarget:
    """A deployed function and its bucket."""

    def __init__(self, function, bucket, region, connections):
        """Create the clients.

        Args:
            function: Function name.
            bucket: Bucket holding crops and results.
            region: AWS region of both.
            connections: Most invocations held open at once.
        """
        # No retries: a retried invocation would be billed twice and both
        # would be invisible here. The read timeout outlasts the function's.
        config = botocore.config.Config(
            region_name=region,
            read_timeout=930,
            retries={"max_attempts": 0},
            max_pool_connections=connections + 2,
        )
        self.function = function
        self.bucket = bucket
        self.lambda_client = boto3.client("lambda", config=config)
        self.s3 = boto3.client("s3", region_name=region)

    def invoke(self, event):
        """Invoke the function and wait for its answer.

        Args:
            event: The handler's event.

        Returns:
            A pair ``(payload, report)``: the handler's return value and
            the parsed ``REPORT`` line.

        Raises:
            RuntimeError: If the function raised or timed out.
        """
        response = self.lambda_client.invoke(
            FunctionName=self.function,
            LogType="Tail",
            Payload=json.dumps(event).encode(),
        )
        payload = json.loads(response["Payload"].read())
        if response.get("FunctionError"):
            raise RuntimeError(f"{event['output_key']}: {payload}")
        log_text = base64.b64decode(response["LogResult"]).decode()
        return payload, parse_report(log_text)

    def read(self, key):
        """Return one result object's bytes.

        Args:
            key: Object key.

        Returns:
            The object's bytes.
        """
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()


class EmulatorTarget:
    """The container running under the Lambda emulator on this machine."""

    def __init__(self, url=EMULATOR_URL, root=EMULATOR_ROOT, bucket=EMULATOR_BUCKET):
        """Remember where the emulator listens and where its folder store is.

        Args:
            url: The emulator's invocation URL.
            root: Local folder mounted as the container's ``LOCAL_ROOT``.
            bucket: Subfolder standing in for the bucket.
        """
        self.url = url
        self.root = Path(root)
        self.bucket = bucket

    def invoke(self, event):
        """Invoke the emulated function.

        Args:
            event: The handler's event.

        Returns:
            A pair ``(payload, report)``; the report is empty because the
            emulator bills nothing and its timings mean nothing for cost.

        Raises:
            RuntimeError: If the handler raised.
        """
        request = urllib.request.Request(self.url, data=json.dumps(event).encode())
        with urllib.request.urlopen(request, timeout=930) as response:
            payload = json.load(response)
        if "errorMessage" in payload:
            raise RuntimeError(f"{event['output_key']}: {payload}")
        return payload, {name: None for name in REPORT_PATTERNS}

    def read(self, key):
        """Return one result file's bytes.

        Args:
            key: Path inside the stand-in bucket.

        Returns:
            The file's bytes.
        """
        return (self.root / self.bucket / key).read_bytes()


def output_key(label, shard):
    """Return where a shard's logits are stored.

    Args:
        label: Run label.
        shard: Shard number.

    Returns:
        The object key.
    """
    return f"runs/{label}/shard_{shard:05d}.npz"


def read_progress(path):
    """Read the shards a run has already finished.

    Args:
        path: The run's ``.jsonl`` progress file.

    Returns:
        Mapping of shard number to its row; empty if there is no file.
    """
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["shard"]: row for row in rows}


def invoke_shard(target, label, shard, keys):
    """Invoke one shard and build its row.

    Args:
        target: A ``LambdaTarget`` or ``EmulatorTarget``.
        label: Run label.
        shard: Shard number.
        keys: The shard's crop keys.

    Returns:
        The shard's row, with every name in ``FIELDS``.

    Raises:
        RuntimeError: If the function handled a different number of crops.
    """
    event = {
        "bucket": target.bucket,
        "prefix": CROP_PREFIX,
        "keys": keys,
        "output_key": output_key(label, shard),
    }
    began = time.perf_counter()
    payload, report = target.invoke(event)
    wall = time.perf_counter() - began
    if payload["n"] != len(keys):
        raise RuntimeError(f"shard {shard}: sent {len(keys)}, got {payload['n']}")
    row = {"label": label, "shard": shard, "wall_seconds": round(wall, 3)}
    row |= {k: v for k, v in payload.items() if k in FIELDS}
    row |= report
    return {name: row.get(name) for name in FIELDS}


def run_shards(target, label, shards, concurrency, progress_path):
    """Invoke every unfinished shard, a few at a time, recording each as it ends.

    Args:
        target: A ``LambdaTarget`` or ``EmulatorTarget``.
        label: Run label.
        shards: The run's shards, from ``make_shards``.
        concurrency: Invocations held open at once.
        progress_path: The run's ``.jsonl`` progress file.

    Returns:
        One row per shard, sorted by shard number.
    """
    done = read_progress(progress_path)
    todo = [i for i in range(len(shards)) if i not in done]
    if done:
        print(f"{label}: resuming, {len(done)} of {len(shards)} shards already done")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(progress_path, "a", encoding="utf-8") as progress,
        concurrent.futures.ThreadPoolExecutor(concurrency) as pool,
    ):
        jobs = [pool.submit(invoke_shard, target, label, i, shards[i]) for i in todo]
        for job in concurrent.futures.as_completed(jobs):
            # result() re-raises a failed shard; the shards already written
            # stay in the progress file for the next attempt.
            row = job.result()
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            done[row["shard"]] = row
            print(
                f"{label}: shard {row['shard']} done ({len(done)}/{len(shards)}), "
                f"{row['n']} crops in {row['wall_seconds']} s"
                f"{', cold' if row['cold'] else ''}",
                flush=True,
            )
    return [done[i] for i in sorted(done)]


def collect(target, label, shards):
    """Download every shard's logits and join them in key order.

    Args:
        target: A ``LambdaTarget`` or ``EmulatorTarget``.
        label: Run label.
        shards: The run's shards.

    Returns:
        A pair ``(keys, logits)`` covering every shard.

    Raises:
        RuntimeError: If a stored shard holds different keys from the ones
            it was sent.
    """
    keys, logits = [], []
    for i, shard in enumerate(shards):
        with np.load(io.BytesIO(target.read(output_key(label, i)))) as data:
            if np.asarray(data["keys"]).tolist() != shard:
                raise RuntimeError(f"shard {i}: stored keys differ from those sent")
            logits.append(data["logits"])
        keys.extend(shard)
    return keys, np.concatenate(logits)


def parity(keys, logits, index, cache, head, committed=None):
    """Compare the route's logits with the local route's on the same crops.

    The local logits are the promoted head applied to the cached
    embeddings, which is how every committed prediction was made.

    Args:
        keys: Crop keys the route scored.
        logits: Its logits, one row per key.
        index: The crop index (``path``, ``seq_id``, ``split``).
        cache: The DINOv2 embedding cache.
        head: The promoted head.
        committed: The committed hold-out sequence predictions (``seq_id``,
            ``pred``); ``None`` skips that comparison.

    Returns:
        Dict of counts and differences; ``passed`` is true when no crop or
        sequence prediction differs.
    """
    local = head_module.logits(head, store.rows_for(cache, keys))
    classes = head["classes"]
    seq_of = dict(zip(index["path"].to_list(), index["seq_id"].to_list()))
    seq_ids = [seq_of[key] for key in keys]
    # Sequences are averaged over the crops scored here, on both sides, so
    # a sample that holds part of a sequence still compares like with like.
    ours = scoring.sequence_predictions(seq_ids, logits, classes)
    theirs = scoring.sequence_predictions(seq_ids, local, classes)
    result = {
        "crops": len(keys),
        "sequences": ours.height,
        "max_abs_logit_difference": float(np.abs(logits - local).max()),
        "crop_predictions_differing": int(
            (logits.argmax(axis=1) != local.argmax(axis=1)).sum()
        ),
        "sequence_predictions_differing": int((ours["pred"] != theirs["pred"]).sum()),
    }
    if committed is not None:
        # Only sequences whose every indexed crop was scored can be held
        # against the committed table, which averaged all of them.
        scored = set(keys)
        whole = (
            index.filter(pl.col("split") == "holdout")
            .group_by("seq_id")
            .agg(pl.col("path").is_in(scored).all().alias("whole"))
            .filter("whole")["seq_id"]
        )
        joined = committed.join(
            ours.filter(pl.col("seq_id").is_in(whole.to_list())),
            on="seq_id",
            suffix="_cloud",
        )
        result["holdout_sequences_compared"] = joined.height
        result["holdout_sequences_differing"] = int(
            (joined["pred"] != joined["pred_cloud"]).sum()
        )
    result["passed"] = (
        result["crop_predictions_differing"] == 0
        and result["sequence_predictions_differing"] == 0
        and result.get("holdout_sequences_differing", 0) == 0
    )
    return result


def write_results(label, rows, keys, logits, check, out_dir=OUT_DIR):
    """Write a finished run's three files.

    Args:
        label: Run label.
        rows: One row per shard.
        keys: Crop keys scored.
        logits: Their logits.
        check: The dict from ``parity``.
        out_dir: Folder to write into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(
        out_dir / f"invocations_{label}.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / f"logits_{label}.npz",
        paths=np.array(keys),
        logits=logits.astype(np.float32),
    )
    (out_dir / f"parity_{label}.json").write_text(json.dumps(check, indent=2) + "\n")


def run(target, label, keys, shard_size, concurrency):
    """Run one labelled batch end to end and write its results.

    Args:
        target: A ``LambdaTarget`` or ``EmulatorTarget``.
        label: Run label, used in object keys and file names.
        keys: Crop keys to score.
        shard_size: Crops per invocation.
        concurrency: Invocations held open at once.

    Returns:
        The parity dict.
    """
    shards = make_shards(keys, shard_size)
    rows = run_shards(
        target, label, shards, concurrency, PROGRESS_DIR / f"{label}.jsonl"
    )
    print(f"{label}: warm invocations {summarise(rows, cost.load_prices())}")
    scored, logits = collect(target, label, shards)
    check = parity(
        scored,
        logits,
        crops.load_index(),
        store.read(store.cache_path("dinov2")),
        head_module.load(),
        pl.read_csv(HOLDOUT_SEQUENCES),
    )
    write_results(label, rows, scored, logits, check)
    print(f"{label}: parity {json.dumps(check)}")
    return check


def summarise(rows, prices):
    """Summarise a run's warm invocations: speed and list-price cost.

    Cold starts are left out so memory sizes are compared on steady
    running; a long run is nearly all warm.

    Args:
        rows: One row per shard, from ``run_shards``.
        prices: Prices from ``cost.load_prices``.

    Returns:
        Dict with ``warm_invocations``, ``crops_per_second`` (crops over
        billed time) and ``usd_per_10k_crops``, or ``None`` if no
        invocation was warm.
    """
    warm = [row for row in rows if not row["cold"]]
    if not warm:
        return None
    priced = cost.run_cost(prices, warm)
    billed_seconds = sum(row["billed_ms"] for row in warm) / 1000
    return {
        "warm_invocations": len(warm),
        "crops_per_second": round(priced["crops"] / billed_seconds, 3),
        "usd_per_10k_crops": round(priced["usd_per_10k_crops"], 4),
    }


def load_outputs(path=OUTPUTS, need_functions=True):
    """Read what Terraform deployed.

    Args:
        path: The file ``make tf-output`` writes.
        need_functions: Whether the caller invokes the functions. The
            upload needs only the bucket, which exists before they do.

    Returns:
        Dict with ``bucket``, ``region`` and ``functions``, a mapping of
        memory size in MB (as a string) to function name.

    Raises:
        SystemExit: If the file is missing, or the functions are needed
            and not deployed yet.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found: deploy first (make tf-apply)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    outputs = {name: entry["value"] for name, entry in raw.items()}
    if need_functions and not outputs.get("functions"):
        raise SystemExit(
            "no functions deployed: run make tf-apply FUNCTIONS=true after the push"
        )
    return outputs


def print_estimate(n_crops, memory_mb, shard_size):
    """Print a rough cost for a run before it starts.

    Args:
        n_crops: Crops to score.
        memory_mb: The function's memory size.
        shard_size: Crops per invocation.
    """
    prices = cost.load_prices()
    seconds = n_crops / ASSUMED_CROPS_PER_SECOND
    usd = cost.compute_usd(prices, seconds * 1000, memory_mb) + cost.request_usd(
        prices, -(-n_crops // shard_size)
    )
    print(
        f"estimate at an assumed {ASSUMED_CROPS_PER_SECOND} crops/s and "
        f"{memory_mb} MB: {seconds / 3600:.1f} function-hours, about "
        f"${usd:.2f} at list price"
    )


def main_upload(args):
    """Upload every indexed crop that the bucket lacks.

    Only paths in the crop index are sent, so nothing outside it (and no
    frame the index dropped for holding a person) can reach the bucket.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit status.
    """
    outputs = load_outputs(need_functions=False)
    # Uploads are idempotent, so patient retries are safe here (unlike Lambda
    # invokes, which are billed): ten attempts with backoff ride out a flaky link.
    s3 = boto3.client(
        "s3",
        region_name=outputs["region"],
        config=botocore.config.Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        ),
    )
    keys = crops.load_index()["path"].to_list()
    present = set()
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=outputs["bucket"], Prefix=CROP_PREFIX
    )
    for page in pages:
        present.update(
            o["Key"].removeprefix(CROP_PREFIX) for o in page.get("Contents", [])
        )
    missing = [key for key in keys if key not in present]
    print(f"{len(keys)} indexed crops, {len(missing)} to upload")

    def send(key):
        """Upload one crop.

        Args:
            key: Crop path relative to the crop store.

        Returns:
            True if it uploaded, False if it still failed after the client's
            retries (a rerun picks it up, as the bucket is listed first).
        """
        try:
            s3.upload_file(
                str(args.crop_dir / key),
                outputs["bucket"],
                CROP_PREFIX + key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
        except boto3.exceptions.S3UploadFailedError as error:
            print(f"failed {key}: {error}", flush=True)
            return False
        return True

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        for done, ok in enumerate(pool.map(send, missing), start=1):
            failed += not ok
            if done % 2000 == 0 or done == len(missing):
                print(f"processed {done}/{len(missing)}", flush=True)
    if failed:
        print(f"{failed} crops failed; rerun make cloud-upload to send them")
        return 1
    return 0


def main_sweep(_args):
    """Score one fixed sample at every deployed memory size.

    Args:
        _args: Parsed command-line arguments, unused.

    Returns:
        Process exit status, 1 if any size fails parity.
    """
    outputs = load_outputs()
    keys = sample_keys(crops.load_index()["path"].to_list(), SWEEP_SAMPLE, SWEEP_SEED)
    status = 0
    for memory in sorted(outputs["functions"], key=int):
        target = LambdaTarget(
            outputs["functions"][memory], outputs["bucket"], outputs["region"], 1
        )
        check = run(target, f"sweep-{memory}", keys, SWEEP_SHARD, concurrency=1)
        status |= 0 if check["passed"] else 1
    return status


def main_run(args):
    """Score every indexed crop at one memory size: the measured run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit status, 1 if parity fails.
    """
    outputs = load_outputs()
    memory = str(args.memory)
    if memory not in outputs["functions"]:
        raise SystemExit(
            f"no function at {memory} MB; deployed: {list(outputs['functions'])}"
        )
    keys = sample_keys(crops.load_index()["path"].to_list(), args.sample, SWEEP_SEED)
    print_estimate(len(keys), args.memory, args.shard)
    target = LambdaTarget(
        outputs["functions"][memory],
        outputs["bucket"],
        outputs["region"],
        args.concurrency,
    )
    check = run(target, args.label, keys, args.shard, args.concurrency)
    return 0 if check["passed"] else 1


def main_emulate(args):
    """Score a small sample through the local emulator and check parity.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit status, 1 if parity fails.
    """
    keys = sample_keys(crops.load_index()["path"].to_list(), args.sample, SWEEP_SEED)
    # A fresh label each time: the emulator is free, and a stale progress
    # file would skip the very invocations being checked.
    (PROGRESS_DIR / "emulator.jsonl").unlink(missing_ok=True)
    shards = make_shards(keys, args.shard)
    target = EmulatorTarget()
    run_shards(target, "emulator", shards, 1, PROGRESS_DIR / "emulator.jsonl")
    scored, logits = collect(target, "emulator", shards)
    check = parity(
        scored,
        logits,
        crops.load_index(),
        store.read(store.cache_path("dinov2")),
        head_module.load(),
    )
    print(json.dumps(check, indent=2))
    return 0 if check["passed"] else 1


def main_leftovers(args):
    """List anything carrying the project's tag that is still in the account.

    Run after ``terraform destroy``. The tagging API can list a deleted
    resource for a while, so a hit is a prompt to look, not proof.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit status, 1 if anything is listed.
    """
    client = boto3.client("resourcegroupstaggingapi", region_name=args.region)
    pages = client.get_paginator("get_resources").paginate(
        TagFilters=[{"Key": "Project", "Values": [PROJECT_TAG]}]
    )
    arns = [r["ResourceARN"] for page in pages for r in page["ResourceTagMappingList"]]
    for arn in arns:
        # The account ID is the fifth field of an ARN; keep it off the screen.
        print(":".join(arn.split(":")[:4] + ["<account>"] + arn.split(":")[5:]))
    print(f"{len(arns)} tagged resources found in {args.region}")
    return 1 if arns else 0


def main(argv=None):
    """Upload crops, run the sweep or the measured run, or check leftovers.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload", help=main_upload.__doc__.splitlines()[0])
    upload.add_argument("--crop-dir", type=Path, default=crops.CROP_DIR)
    upload.add_argument("--threads", type=int, default=16)
    upload.set_defaults(run=main_upload)

    sweep = commands.add_parser("sweep", help=main_sweep.__doc__.splitlines()[0])
    sweep.set_defaults(run=main_sweep)

    full = commands.add_parser("run", help=main_run.__doc__.splitlines()[0])
    full.add_argument("--memory", type=int, required=True, help="function size in MB")
    full.add_argument("--label", default="full")
    full.add_argument("--shard", type=int, default=RUN_SHARD)
    full.add_argument("--concurrency", type=int, default=RUN_CONCURRENCY)
    full.add_argument("--sample", type=int, help="score only N sampled crops")
    full.set_defaults(run=main_run)

    emulate = commands.add_parser("emulate", help=main_emulate.__doc__.splitlines()[0])
    emulate.add_argument("--sample", type=int, default=96)
    emulate.add_argument("--shard", type=int, default=32)
    emulate.set_defaults(run=main_emulate)

    leftovers = commands.add_parser(
        "leftovers", help=main_leftovers.__doc__.splitlines()[0]
    )
    leftovers.add_argument("--region", default="eu-west-2")
    leftovers.set_defaults(run=main_leftovers)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
