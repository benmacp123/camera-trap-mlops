"""Tests for the SpeciesNet arm's file handling: the inputs it writes and
how it reads answers, so roll-ups abstain and wrong species count as wrong.
"""

import json

import pytest

from cameratrap.arms import speciesnet_arm
from cameratrap.data import orinoquia

TAXA = {c: orinoquia.TAXA[c] for c in ("black_agouti", "spixs_guan")}
AGOUTI, GUAN = TAXA["black_agouti"][2], TAXA["spixs_guan"][2]


def test_inputs_keep_only_animal_boxes_sorted(tmp_path):
    """Person boxes are dropped and the most confident animal comes first."""
    dets = {
        "A/1.JPG": [
            {"category": "1", "conf": 0.3, "bbox": [0, 0, 1, 1]},
            {"category": "2", "conf": 0.9, "bbox": [0, 0, 1, 1]},
            {"category": "1", "conf": 0.8, "bbox": [0, 0, 1, 1]},
        ]
    }
    files = speciesnet_arm.write_inputs(
        ["A/1.JPG"], tmp_path / "img", dets, tmp_path / "out", TAXA, "COL"
    )
    written = json.loads(files["detections"].read_text())["predictions"][0]
    assert [d["conf"] for d in written["detections"]] == [0.8, 0.3]
    instance = json.loads(files["instances"].read_text())["instances"][0]
    assert instance["country"] == "COL"
    assert files["targets"].read_text().splitlines() == [AGOUTI, GUAN]


@pytest.mark.parametrize(
    "label, expected",
    [
        (AGOUTI, "black_agouti"),
        (
            "x;mammalia;rodentia;dasyproctidae;dasyprocta;azarae;azara's agouti",
            "other_species",
        ),
        ("x;aves;galliformes;cracidae;penelope;;penelope species", None),
        ("f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank", None),
        ("", None),
    ],
)
def test_open_label(label, expected):
    """A target species maps to its class, another species is wrong, a roll-up abstains."""
    mapping = {t[2]: c for c, t in TAXA.items()}
    assert speciesnet_arm.open_label(label, mapping) == expected


def test_read_predictions_orders_target_logits(tmp_path):
    """Restricted logits follow the class order, whatever order SpeciesNet used."""
    image_dir = tmp_path / "img"
    filepath = str(image_dir.resolve() / "A" / "1.JPG")
    predictions = {
        "predictions": [
            {
                "filepath": filepath,
                "detections": [
                    {
                        "raw_classifications": {
                            "target_classes": [GUAN, AGOUTI],
                            "target_logits": [1.0, 5.0],
                        },
                        "processed_classification": {
                            "class": AGOUTI,
                            "source": "classifier",
                        },
                    }
                ],
            },
            {"filepath": str(image_dir.resolve() / "B" / "2.JPG"), "detections": []},
        ]
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(predictions))
    out = speciesnet_arm.read_predictions(path, image_dir, TAXA)
    assert list(out) == ["A/1.JPG"]
    assert out["A/1.JPG"]["logits"].tolist() == [5.0, 1.0]
    assert out["A/1.JPG"]["open_label"] == "black_agouti"


def test_prune_unfinished_keeps_only_finished(tmp_path):
    """Entries flagged with failures are dropped so a resumed run redoes them."""
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "predictions": [
                    {"filepath": "a", "detections": []},
                    {"filepath": "b", "failures": ["CLASSIFIER"]},
                ]
            }
        )
    )
    assert speciesnet_arm.prune_unfinished(path) == 1
    kept = json.loads(path.read_text())["predictions"]
    assert [p["filepath"] for p in kept] == ["a"]
    assert speciesnet_arm.prune_unfinished(tmp_path / "missing.json") == 0
