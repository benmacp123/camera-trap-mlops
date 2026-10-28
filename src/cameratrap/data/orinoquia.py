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

# Class -> (scientific name, common name, SpeciesNet v4.0.3a model label).
# Names from the SpeciesNet taxonomy release 2026-06-09; every class is a
# species-level model label and allowed in COL by its geofence.
TAXA = {
    "collared_peccary": (
        "Pecari tajacu",
        "collared peccary",
        "5acf63fb-339d-42b1-af7e-95f6151ae5fb;mammalia;artiodactyla;tayassuidae;"
        "pecari;tajacu;collared peccary",
    ),
    "black_agouti": (
        "Dasyprocta fuliginosa",
        "black agouti",
        "d62cd43d-a519-4cd5-aef2-b76ae4bd16c1;mammalia;rodentia;dasyproctidae;"
        "dasyprocta;fuliginosa;black agouti",
    ),
    "spotted_paca": (
        "Cuniculus paca",
        "spotted paca",
        "32c0147f-1967-4644-b107-8133ae1f020a;mammalia;rodentia;cuniculidae;"
        "cuniculus;paca;spotted paca",
    ),
    "south_american_coati": (
        "Nasua nasua",
        "South American coati",
        "1d3bbcaa-f81a-41d9-8904-5cc8a505af32;mammalia;carnivora;procyonidae;"
        "nasua;nasua;south american coati",
    ),
    "white-lipped_peccary": (
        "Tayassu pecari",
        "white-lipped peccary",
        "575008c3-8e3b-4efe-8da6-47631d60d5b7;mammalia;artiodactyla;tayassuidae;"
        "tayassu;pecari;white-lipped peccary",
    ),
    "southern_tamandua": (
        "Tamandua tetradactyla",
        "southern tamandua",
        "89d41ec3-af8a-438a-86dd-4a6cb1819f07;mammalia;pilosa;myrmecophagidae;"
        "tamandua;tetradactyla;southern tamandua",
    ),
    "spixs_guan": (
        "Penelope jacquacu",
        "Spix's guan",
        "6071f8bb-793d-45f5-9013-8668014e86e3;aves;galliformes;cracidae;"
        "penelope;jacquacu;spix's guan",
    ),
    "salvins_curassow": (
        "Mitu salvini",
        "Salvin's curassow",
        "d9880c5a-b487-4ef0-8e26-39b2b40a06b7;aves;galliformes;cracidae;"
        "mitu;salvini;salvin's curassow",
    ),
    "lowland_tapir": (
        "Tapirus terrestris",
        "lowland tapir",
        "39e5ea73-38f1-454a-8054-7c67f20fbced;mammalia;perissodactyla;tapiridae;"
        "tapirus;terrestris;lowland tapir",
    ),
}
COUNTRY = "COL"

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
