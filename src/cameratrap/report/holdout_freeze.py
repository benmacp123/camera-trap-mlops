"""Write the pre-registration record for the hold-out decision.

Everything the decision depends on is fixed here, before the hold-out is
scored: the gate, the sites, the metric, the one challenger and how it is
trained. Freezing only the challenger's name would leave its training open
to adjustment after seeing the result, which is the retuning the freeze
exists to stop.

The file is written once and hashed; ``gate.freeze.load`` refuses it if
edited, and a test pins the hash. The hold-out is one shot: its result is
reported pass or fail, and a fail does not send another challenger in.

Reads committed files only, and only the hold-out's site names, never its
predictions.

::

    PYTHONPATH=src python -m cameratrap.report.holdout_freeze
"""

import argparse
import datetime
import json
import sys
from importlib import metadata
from pathlib import Path

from cameratrap.arms import probe
from cameratrap.gate import freeze, rule

OUT_PATH = Path("config/holdout_freeze.json")
MANIFEST = "data/manifest/orinoquia.json"
CROP_SETTINGS = "data/manifest/orinoquia_crops.json"
INDEX = "data/manifest/orinoquia_crops.csv"
# Lists the published embedding files' hashes, so hashing it pins them too.
HUB = "data/manifest/hub.json"
SELECTION = "results/selection/summary.json"
CHALLENGERS = "results/challengers/summary.json"
# The manifest is not hashed whole: fetching images adds checksums to it.
# Its split hash and the site names below are what the decision rests on.
HASHED_INPUTS = (CROP_SETTINGS, INDEX, HUB, SELECTION, CHALLENGERS)
# The bootstrap, swap null and probe fit can all shift between versions.
PACKAGES = ("numpy", "scipy", "scikit-learn", "polars")
CHAMPION = {"arm": "bioclip2", "method": "zero_shot", "backbone": "bioclip2"}


def read_json(root, name):
    """Read a JSON file under the repository root.

    Args:
        root: The repository root.
        name: Path relative to the root.

    Returns:
        The parsed content.
    """
    return json.loads((root / name).read_text())


def probe_recipe(backbone):
    """Return how a probe challenger is trained for the hold-out.

    Args:
        backbone: The embedding backbone's name.

    Returns:
        A dict of the training settings; a test checks them against
        ``arms.probe``.
    """
    return {
        "method": "probe",
        "backbone": backbone,
        "train_on": "every frame of every fold site; no hold-out frame",
        "regularisation": "l2",
        "cs": probe.CS.tolist(),
        "inner_folds": probe.INNER_FOLDS,
        "inner_split": "GroupKFold by site, within the training sites",
        "inner_scoring": probe.INNER_SCORING,
        "class_weight": probe.CLASS_WEIGHT,
        "max_iter": probe.MAX_ITER,
    }


def build_config(root, frozen_on):
    """Assemble the frozen config from the committed files.

    Args:
        root: The repository root, a ``pathlib.Path``.
        frozen_on: The freeze date, in ISO format.

    Returns:
        A JSON-serialisable dict for ``gate.freeze.write``.

    Raises:
        ValueError: If no challenger was chosen, the selection used a
            different gate, the chosen arm is not a probe, or the
            embeddings were made from another crop index.
    """
    selection = read_json(root, SELECTION)
    gate = freeze.gate_dict(rule.GateConfig())
    chosen = selection["chosen"]
    if chosen is None:
        raise ValueError("the selection chose no challenger")
    if selection["gate"] != gate:
        raise ValueError("the selection ran with a different gate; rerun it")
    backbone, method = chosen.split("_")
    if method != "probe":
        raise ValueError(f"no frozen training recipe for {chosen}")

    hashes = {name: freeze.file_sha256(root / name) for name in HASHED_INPUTS}
    caches = read_json(root, CHALLENGERS)["caches"]
    if any(c["index_sha256"] != hashes[INDEX] for c in caches.values()):
        raise ValueError("the embedding caches came from another crop index")

    manifest = read_json(root, MANIFEST)
    return {
        "frozen_on": frozen_on,
        "gate": gate,
        "holdout_sites": manifest["holdout_sites"],
        "splits_sha256": manifest["splits_sha256"],
        "metric": {
            "name": "balanced_accuracy",
            "unit": "sequence",
            "aggregation": "frame logits averaged per sequence, then softmax",
            "classes": "those present at the site",
            "frames_per_sequence": read_json(root, CROP_SETTINGS)["settings"][
                "frames_per_sequence"
            ],
        },
        "champion": CHAMPION | {"embedding": caches[CHAMPION["backbone"]]},
        "challenger": {"arm": chosen, "embedding": caches[backbone]}
        | probe_recipe(backbone),
        "embeddings": read_json(root, HUB),
        "inputs": hashes,
        "software": {name: metadata.version(name) for name in PACKAGES},
    }


def main(argv=None):
    """Write the hold-out freeze and print its hash.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status, 0 on success.

    Raises:
        FileExistsError: If a freeze already exists. A second freeze would
            be a retune, so replacing one is a deliberate manual act.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    out = args.root / args.out
    if out.exists():
        raise FileExistsError(f"{out} already exists; the freeze happens once")
    config = build_config(args.root, datetime.date.today().isoformat())
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{out}: {freeze.write(out, config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
