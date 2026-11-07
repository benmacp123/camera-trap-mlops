"""Freeze an evaluation config: write it with its SHA-256 and refuse edits.

The frozen file holds everything a hold-out decision depends on (gate
settings, held-out groups, metric). Its hash is recorded before the
hold-out is scored, so any later change to the file is detectable.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from cameratrap.gate.rule import GateConfig


def canonical(config):
    """Return a config as canonical JSON bytes.

    Sorted keys and fixed separators make the bytes, and so the hash,
    depend only on the content, not on how the dict was built.

    Args:
        config: A JSON-serialisable dict.

    Returns:
        UTF-8 bytes.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def config_hash(config):
    """Return the SHA-256 of a config's canonical JSON.

    Args:
        config: A JSON-serialisable dict.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(canonical(config)).hexdigest()


def write(path, config):
    """Write a config and its hash to a file.

    Args:
        path: A ``pathlib.Path`` to write.
        config: A JSON-serialisable dict; must contain a ``gate`` dict of
            ``GateConfig`` fields.

    Returns:
        The hex digest written.
    """
    digest = config_hash(config)
    payload = {"config": config, "sha256": digest}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest


def load(path, expected_sha256=None):
    """Read a frozen config and check it has not changed.

    Args:
        path: A ``pathlib.Path`` written by ``write``.
        expected_sha256: The digest recorded at the freeze; when given, the
            file must match it as well as its own stored hash.

    Returns:
        A pair ``(config, gate_config)``: the dict and its ``GateConfig``.

    Raises:
        ValueError: If the content no longer matches its stored hash, or
            the stored hash differs from ``expected_sha256``.
    """
    payload = json.loads(path.read_text())
    config, stored = payload["config"], payload["sha256"]
    if config_hash(config) != stored:
        raise ValueError(f"{path} was edited after it was frozen")
    if expected_sha256 is not None and stored != expected_sha256:
        raise ValueError(f"{path} is not the config frozen as {expected_sha256}")
    return config, GateConfig(**config["gate"])


def file_sha256(path):
    """Return the SHA-256 of a file's bytes.

    Args:
        path: A ``pathlib.Path``.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(hashes, root="."):
    """Check that the files a frozen config was built from have not changed.

    The hash of the config only protects the config's own text. A frozen
    recipe is just as void if the data under it moved, so the config also
    records its inputs' hashes and this compares them.

    Args:
        hashes: A dict of file path, relative to ``root``, to SHA-256.
        root: The folder the paths are relative to.

    Raises:
        ValueError: If a file is missing or its content differs.
    """
    changed = []
    for name, expected in sorted(hashes.items()):
        path = Path(root) / name
        if not path.exists():
            changed.append(f"{name} (missing)")
        elif file_sha256(path) != expected:
            changed.append(name)
    if changed:
        raise ValueError(f"frozen inputs changed: {', '.join(changed)}")


def gate_dict(gate_config):
    """Return a ``GateConfig`` as a plain dict for freezing.

    Args:
        gate_config: A ``GateConfig``.

    Returns:
        A dict of its fields.
    """
    return asdict(gate_config)
