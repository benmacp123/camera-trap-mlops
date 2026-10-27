"""Source record for the Orinoquía Camera Traps dataset on LILA.

Holds the download URLs, the checksums of the two metadata archives the
manifest is built from, the licence and the nine-species class set, so
every number traces back to a recorded source.
"""

import json
import zipfile

NAME = "orinoquia"
LICENCE = "CDLA-Permissive 1.0"
ATTRIBUTION = (
    "Orinoquia Camera Traps, University of Minnesota, via LILA BC "
    "(https://lila.science/datasets/orinoquia-camera-traps/)"
)

IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/orinoquia-camera-traps/public/"
)

# Archive name -> (URL, SHA-256 of the archive as downloaded 2026-10-27).
# MDv5a is the raw result set, not the repeat-detection-filtered one: the
# spike compares it against a fresh MDv5a run, which has no such filter.
ARCHIVES = {
    "metadata": (
        "https://storage.googleapis.com/public-datasets-lila/orinoquia-camera-traps/"
        "orinoquia_camera_traps_metadata.zip",
        "86ed898eb373e01336ad9df989bf486f28ae89f034ef1db410aed9d701612238",
    ),
    "detections": (
        "https://lila.science/public/lila-md-results/"
        "orinoquia-camera-traps_public_mdv5a.0.0_results.json.zip",
        "d2c7d64276b4b8bf1d79339fd7ce16249f54c726341a43a41a7f3f40ad4128b6",
    ),
}

# The nine species chosen by the owner (2026-10-24): each has at least
# 228 clean sequences over 20 or more sites.
CLASSES = (
    "collared_peccary",
    "black_agouti",
    "spotted_paca",
    "south_american_coati",
    "white-lipped_peccary",
    "southern_tamandua",
    "spixs_guan",
    "salvins_curassow",
    "lowland_tapir",
)

# Labels that say nothing about which animal is present. A sequence whose
# other frames carry these keeps its species label.
NON_ANIMAL = frozenset({"empty", "unknown"})
HUMAN = "human"

# Study period with a margin; the metadata spans 2020-01-07 to 2020-07-26.
TIMESTAMP_BOUNDS = ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00")


def read_zipped_json(path):
    """Read the single JSON file inside a zip archive.

    Args:
        path: Path to a zip archive holding exactly one ``.json`` member.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If the archive does not hold exactly one JSON member.
    """
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one JSON member, found {members}")
        with archive.open(members[0]) as handle:
            return json.load(handle)
