"""Tests for the spike analysis: its helpers, and one end-to-end run on
synthetic arms so the decision logic is exercised without any model.
"""

import numpy as np
import polars as pl
import pytest

from cameratrap.report import spike

CLASSES = ["a", "b", "c"]


def test_sites_needed_matches_formula():
    """Sites needed follows ((z_a + z_b) * sd / delta)^2, rounded up."""
    sd, n = spike.sites_needed([0.0, 0.1, -0.1, 0.0], delta=0.05, alpha=0.05, power=0.8)
    assert sd == pytest.approx(np.std([0.0, 0.1, -0.1, 0.0], ddof=1))
    assert n == int(np.ceil((2.8016 * sd / 0.05) ** 2))


def test_majority_label_ignores_abstentions():
    """Abstaining frames do not vote; all abstaining gives an abstention."""
    assert spike.majority_label([None, "b", "a", "b"]) == "b"
    assert spike.majority_label([None, None]) is None


def test_detector_agreement_counts_disagreements():
    """Images are compared on whether any animal box reaches the threshold."""
    box = lambda conf, cat="1": {"category": cat, "conf": conf}  # noqa: E731
    pre = {"x": [box(0.9)], "y": [box(0.1)], "z": [box(0.9, "2")]}
    fresh = {"x": [box(0.5)], "y": [box(0.3)], "z": []}
    result = spike.detector_agreement(pre, fresh, threshold=0.2)
    assert result == {
        "n": 3,
        "agreement": pytest.approx(2 / 3),
        "only_precomputed": 0,
        "only_fresh": 1,
    }


def test_bootstrap_interval_brackets_estimate():
    """A strong monotone relation gives a high rho inside its interval."""
    x = np.arange(20.0)
    y = x + np.random.default_rng(0).normal(scale=2, size=20)
    result = spike.bootstrap_spearman(x, y, resamples=200, seed=1)
    assert result["low"] <= result["rho"] <= result["high"]
    assert result["rho"] > 0.8


def make_inputs():
    """Return synthetic frames, sequences and arm outputs over ten fold sites.

    BioCLIP is made weaker at half the sites, SpeciesNet always right but
    abstaining in the open set on every third sequence.

    Returns:
        The positional arguments for ``spike.run``.
    """
    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        for q in range(24):
            label = CLASSES[q % 3]
            for f in range(2):
                rows.append(
                    {
                        "path": f"S{s}/{q}_{f}.JPG",
                        "seq_id": f"S{s}-{q}",
                        "site": f"S{s}",
                        "label": label,
                        "split": f"fold{s // 2}",
                    }
                )
    frames = pl.DataFrame(rows).sort("path")
    sequences = (
        frames.select("seq_id", "site", "label", "split").unique().sort("seq_id")
    )
    onehot = np.eye(3)[[CLASSES.index(c) for c in frames["label"]]]
    noise = np.where(np.array([int(p[1]) for p in frames["path"]]) < 5, 2.0, 0.2)
    image = onehot * 2 + rng.normal(size=onehot.shape) * noise[:, None]
    bioclip = {
        "paths": frames["path"].to_numpy(),
        "image": image,
        "text": np.eye(3),
        "classes": np.array(CLASSES),
        "logit_scale": np.float32(10.0),
    }
    speciesnet = {
        p: {
            "logits": onehot[i] * 5,
            "open_label": (
                None if int(p.split("/")[1].split("_")[0]) % 3 == 0 else label
            ),
        }
        for i, (p, label) in enumerate(frames.select("path", "label").iter_rows())
    }
    dets = {p: [{"category": "1", "conf": 0.9}] for p in frames["path"][:10]}
    detector = {"precomputed": dets, "fresh": dets}
    return frames, sequences, bioclip, speciesnet, detector, 10


def test_run_scores_every_arm_and_decides():
    """The run scores all arms per site and its decision follows the criteria."""
    summary, per_site = spike.run(*make_inputs())
    assert per_site.height == 10
    assert summary["best_zero_shot"] == "speciesnet"
    assert summary["mean_site_balanced_accuracy"]["speciesnet"] == pytest.approx(1.0)
    assert summary["speciesnet_open"]["coverage"] == pytest.approx(2 / 3)
    assert summary["speciesnet_open"]["accuracy_on_covered"] == pytest.approx(1.0)
    assert summary["detector_check"]["agreement"] == 1.0
    decision = summary["decision"]
    # Three classes cannot meet the eight-class criterion, and SpeciesNet
    # leaves no weak site, so the dataset is rejected.
    assert decision["weak_site_share"] == 0.0
    assert not decision["orinoquia_is_gate_dataset"]
