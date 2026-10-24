"""Tests for the package layout: each subpackage imports only what its
layer allows.

Three boundaries matter. The model stack (torch and friends) is confined to
``embed`` and ``arms``, so CI and every downstream stage run from cached
embeddings without installing it. The gate imports nothing else from this
project, so vessel-anomaly can reuse it. MLflow and the AWS SDK each live
in one place.

Imports are read from the source with ``ast``, not by importing, so a
forbidden import is caught even in a branch no other test runs, and
without the model stack installed.
"""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

MODEL_LIBRARIES = {"torch", "torchvision", "open_clip", "timm", "speciesnet"}
AWS_LIBRARIES = {"boto3", "botocore", "aiobotocore", "moto", "awslambdaric"}

# Top-level import name -> module prefixes allowed to import it.
CONFINED = {
    **{lib: ("cameratrap.embed", "cameratrap.arms") for lib in MODEL_LIBRARIES},
    **{lib: ("cloud",) for lib in AWS_LIBRARIES},
    "mlflow": ("cameratrap.tracking",),
}

# The gate's whole allowed world beyond the standard library.
GATE_ALLOWED = {"numpy", "scipy"}


def module_name(path, root):
    """Return the dotted module name for a source file.

    Args:
        path: Path to a ``.py`` file under ``root``.
        root: The source root (``src``).

    Returns:
        The dotted name, with ``__init__`` dropped for packages.
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_names(path):
    """Return the absolute module names a source file imports.

    Args:
        path: Path to a ``.py`` file.

    Returns:
        A set of dotted names; relative imports are skipped because they
        stay inside their own package.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def violations(root):
    """List every import in ``root`` that breaks a layout rule.

    Args:
        root: The source root to scan.

    Returns:
        Sorted ``"module imports name"`` strings, empty when the layout holds.
    """
    found = []
    for path in sorted(root.rglob("*.py")):
        module = module_name(path, root)
        for name in imported_names(path):
            top = name.split(".")[0]
            allowed = CONFINED.get(top)
            if allowed and not module.startswith(allowed):
                found.append(f"{module} imports {name}")
            if module.startswith("cameratrap.gate"):
                inside = name.startswith("cameratrap.gate")
                if not inside and (
                    top == "cameratrap"
                    or (top not in GATE_ALLOWED and not is_stdlib(top))
                ):
                    found.append(f"{module} imports {name}")
    return sorted(found)


def is_stdlib(name):
    """Return whether a top-level module name belongs to the standard library.

    Args:
        name: A top-level import name.

    Returns:
        True for standard-library modules.
    """
    return name in sys.stdlib_module_names


def test_layout_holds():
    """The real source tree breaks no layout rule."""
    assert violations(SRC) == []


@pytest.mark.parametrize(
    "module, source",
    [
        ("cameratrap/drift/signals.py", "import torch\n"),
        ("cameratrap/gate/rule.py", "from cameratrap.data import manifest\n"),
        ("cameratrap/gate/rule.py", "import polars\n"),
        ("cameratrap/report/tables.py", "from mlflow import MlflowClient\n"),
        ("cameratrap/data/fetch.py", "import boto3\n"),
    ],
)
def test_planted_bad_import_is_caught(tmp_path, module, source):
    """Each kind of forbidden import is reported, so the rules are not vacuous."""
    path = tmp_path / module
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert len(violations(tmp_path)) == 1


@pytest.mark.parametrize(
    "module, source",
    [
        ("cameratrap/embed/backbones.py", "import torch\n"),
        ("cameratrap/gate/rule.py", "import numpy as np\nimport math\n"),
        ("cameratrap/tracking/registry.py", "from mlflow import MlflowClient\n"),
        ("cloud/handler.py", "import boto3\n"),
    ],
)
def test_allowed_import_passes(tmp_path, module, source):
    """Imports in their permitted layer are not reported."""
    path = tmp_path / module
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert not violations(tmp_path)
