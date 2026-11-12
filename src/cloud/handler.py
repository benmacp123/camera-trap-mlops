"""Lambda handler: classify one shard of crops from S3 with the champion.

An invocation is given a list of crop keys. It downloads them, runs the
same ``champion.predict`` the local route uses, writes one ``.npz`` of
logits back to the bucket and returns its own timings, so the driver can
separate model time from download time and cold starts from warm ones.

The event::

    {"bucket": "...", "prefix": "crops/", "keys": ["A01/0110.JPG", ...],
     "output_key": "runs/full/shard_0000.npz"}

The model loads on the first invocation, not at import. Lambda allows
only 10 seconds for a container's import phase, and loading torch and the
weights from a cold image can take longer; loading inside the handler
also lets the load be timed.
"""

import concurrent.futures
import io
import os
import shutil
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np

# Baked into the image by infra/lambda/Dockerfile.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/model"))
WEIGHTS = "model.safetensors"
HEAD = "probe_head.npz"
DOWNLOAD_THREADS = 16

# Survives between invocations of a warm container, so only the first
# invocation pays for loading the model.
_STATE = {}


class S3Store:
    """Reads and writes objects in S3."""

    def __init__(self):
        """Create the client once; boto3 clients are safe to share across threads."""
        self.client = boto3.client("s3")

    def download(self, bucket, key, dest):
        """Save one object to a local file.

        Args:
            bucket: Bucket name.
            key: Object key.
            dest: Local path to write.
        """
        self.client.download_file(bucket, key, str(dest))

    def upload(self, bucket, key, body):
        """Write bytes to one object.

        Args:
            bucket: Bucket name.
            key: Object key.
            body: The object's bytes.
        """
        self.client.put_object(Bucket=bucket, Key=key, Body=body)


class FolderStore:
    """Stands in for S3 with a local folder, one subfolder per bucket.

    Used when the container runs under the Lambda emulator on the laptop
    (``LOCAL_ROOT`` set), so the image can be checked without an account.
    """

    def __init__(self, root):
        """Remember the folder.

        Args:
            root: Folder holding one subfolder per bucket.
        """
        self.root = Path(root)

    def download(self, bucket, key, dest):
        """Copy one stored file to a local path.

        Args:
            bucket: Subfolder name.
            key: Path inside it.
            dest: Local path to write.
        """
        shutil.copyfile(self.root / bucket / key, dest)

    def upload(self, bucket, key, body):
        """Write bytes to one stored file.

        Args:
            bucket: Subfolder name.
            key: Path inside it.
            body: The file's bytes.
        """
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def default_store():
    """Return the store this environment uses.

    Returns:
        A ``FolderStore`` when ``LOCAL_ROOT`` is set, otherwise S3.
    """
    root = os.environ.get("LOCAL_ROOT")
    return FolderStore(root) if root else S3Store()


def load_model():
    """Load the champion from the image and return its predict function.

    The model stack is imported here, not at the top of the module, so the
    import phase stays short and tests can run without torch.

    Returns:
        A function from a list of crop files to an array of logits.
    """
    from cameratrap.arms import champion  # pylint: disable=import-outside-toplevel

    loaded = champion.load(MODEL_DIR / WEIGHTS, MODEL_DIR / HEAD)
    return lambda files: champion.predict(loaded, files)


def fetch(store, bucket, prefix, keys, folder):
    """Download a shard's crops in parallel.

    Args:
        store: An ``S3Store`` or ``FolderStore``.
        bucket: Bucket holding the crops.
        prefix: Key prefix in front of every crop key.
        keys: Crop keys, relative to ``prefix``.
        folder: Local folder to fill.

    Returns:
        The local file of each key, in ``keys`` order.
    """
    # Numbered names: keys from different folders can share a file name.
    files = [Path(folder) / f"{i:06d}{Path(k).suffix}" for i, k in enumerate(keys)]
    with concurrent.futures.ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        jobs = [
            pool.submit(store.download, bucket, prefix + key, file)
            for key, file in zip(keys, files)
        ]
        # result() re-raises a failed download, failing the whole shard:
        # a partial shard would silently drop crops from a sequence.
        for job in jobs:
            job.result()
    return files


def pack(keys, logits):
    """Serialise a shard's result.

    Args:
        keys: Crop keys, one per row of ``logits``.
        logits: Array of shape ``(len(keys), n_classes)``.

    Returns:
        The bytes of an ``.npz`` holding ``keys`` and ``logits``.
    """
    buffer = io.BytesIO()
    np.savez(buffer, keys=np.array(keys), logits=np.asarray(logits))
    return buffer.getvalue()


def process(event, store, predict, scratch=None):
    """Classify one shard and store its logits.

    Args:
        event: The invocation's event (see the module docstring).
        store: Where crops are read from and the result is written.
        predict: Function from crop files to an array of logits.
        scratch: Folder for the temporary crop files; ``None`` uses the
            system temporary folder (``/tmp``, the only writable path in
            Lambda).

    Returns:
        Dict with ``n``, ``output_key`` and the seconds spent on
        ``download``, ``model`` and ``upload``.

    Raises:
        ValueError: If the event names no crops.
    """
    keys = event["keys"]
    if not keys:
        raise ValueError("the event names no crops")
    # Removed on exit, so a warm container never fills its 512 MB of /tmp.
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        began = time.perf_counter()
        files = fetch(store, event["bucket"], event.get("prefix", ""), keys, folder)
        downloaded = time.perf_counter()
        logits = predict(files)
        predicted = time.perf_counter()
    store.upload(event["bucket"], event["output_key"], pack(keys, logits))
    return {
        "n": len(keys),
        "output_key": event["output_key"],
        "download_seconds": round(downloaded - began, 3),
        "model_seconds": round(predicted - downloaded, 3),
        "upload_seconds": round(time.perf_counter() - predicted, 3),
    }


def handler(event, _context):
    """Lambda entry point.

    Args:
        event: The invocation's event (see the module docstring).
        _context: Lambda's context object, unused.

    Returns:
        The timings from ``process`` plus ``cold`` (whether this invocation
        loaded the model), ``load_seconds`` and ``cpu_count``.
    """
    cold = "predict" not in _STATE
    began = time.perf_counter()
    if cold:
        _STATE["predict"] = load_model()
        _STATE["store"] = default_store()
    load_seconds = round(time.perf_counter() - began, 3)
    result = process(event, _STATE["store"], _STATE["predict"])
    return {
        **result,
        "cold": cold,
        "load_seconds": load_seconds,
        "cpu_count": os.cpu_count(),
    }
