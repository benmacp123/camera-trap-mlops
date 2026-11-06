"""Tests for choosing the hold-out challenger by simulated gate pass rate."""

import numpy as np
import polars as pl
import pytest

from cameratrap.report import selection

CLASSES = ["a", "b", "c"]
CHAMPION = "bioclip2"
# A clear improvement, a clear regression, and an exact copy of the champion.
GOOD, BAD, COPY = "bioclip2_probe", "dinov2_knn", "bioclip2_knn"


@pytest.fixture(name="small_settings", autouse=True)
def fixture_small_settings(monkeypatch):
    """Shrink the settings so the selection runs on a few synthetic sites."""
    settings = dict(
        selection.SETTINGS,
        challengers=[GOOD, BAD, COPY],
        holdout_like_min_sequences=30,
        holdout_sites=5,
        tolerances=[0.02, 0.05, 0.10, 0.40],
        draws=200,
        calibration_subsets=100,
        gate_subsets=40,
    )
    monkeypatch.setattr(selection, "SETTINGS", settings)
    return settings


def write_sequences(directory, sites=12, per_site=60, seed=0):
    """Write per-sequence predictions for the champion and three challengers.

    One extra site has 20 sequences, below the hold-out-sized minimum.

    Args:
        directory: Folder to write ``sequences_<arm>.csv`` into.
        sites: Hold-out-sized sites.
        per_site: Sequences at each of those sites.
        seed: Seed for which sequences each arm gets right.

    Returns:
        The folder, for convenience.
    """
    rng = np.random.default_rng(seed)
    site = np.repeat([f"S{i:02d}" for i in range(sites)] + ["tiny"], per_site)
    site = site[: sites * per_site + 20]
    n = site.size
    label = np.array(CLASSES)[np.arange(n) % 3]

    def predict(accuracy):
        """Return predictions right with the given probability."""
        wrong = np.array(CLASSES)[(np.arange(n) + 1) % 3]
        return np.where(rng.random(n) < accuracy, label, wrong)

    champion = predict(0.7)
    preds = {CHAMPION: champion, GOOD: predict(0.97), BAD: predict(0.4), COPY: champion}
    for arm, pred in preds.items():
        pl.DataFrame(
            {
                "seq_id": [f"q{i:04d}" for i in range(n)],
                "site": site,
                "label": label,
                "pred": pred,
                "msp": np.full(n, 0.9),
            }
        ).write_csv(directory / f"sequences_{arm}.csv")
    return directory


def load(directory):
    """Read the synthetic predictions for every arm in the settings."""
    return selection.load_sequences(
        directory, [CHAMPION, *selection.SETTINGS["challengers"]]
    )


@pytest.fixture(name="result")
def fixture_result(tmp_path):
    """Return ``selection.run`` on the synthetic predictions."""
    return selection.run(load(write_sequences(tmp_path)))


def by_arm(summary):
    """Return the summary's challenger rows keyed by arm."""
    return {row["challenger"]: row for row in summary["challengers"]}


def test_clear_improvement_is_chosen(result):
    """The challenger that is right far more often passes nearly every set."""
    summary, _, _ = result
    assert summary["chosen"] == GOOD
    assert by_arm(summary)[GOOD]["pass_rate"] > 0.9
    assert summary["chosen_tolerance"] == by_arm(summary)[GOOD]["tolerance"]


def test_regression_never_passes(result):
    """A worse challenger fails the mean-gain condition in every set."""
    summary, _, _ = result
    row = by_arm(summary)[BAD]
    assert row["pass_rate"] == 0
    assert row["fail_mean_gain"] == 1


def test_a_copy_of_the_champion_never_promotes(result):
    """Identical predictions tie at every site, and a tie keeps the champion."""
    summary, _, _ = result
    row = by_arm(summary)[COPY]
    assert row["mean_gain"] == 0
    # The swap null of identical predictions is exactly zero everywhere.
    assert row["tolerance"] == selection.SETTINGS["tolerances"][0]
    assert row["pass_rate"] == 0
    assert row["fail_mean_gain"] == 1


def test_small_sites_are_left_out(tmp_path):
    """Only hold-out-sized sites are calibrated on and drawn."""
    groups = selection.site_groups(load(write_sequences(tmp_path)), CHAMPION, GOOD, 30)
    assert "tiny" not in groups
    assert len(groups) == 12
    labels, champion, challenger = groups["S00"]
    assert len(labels) == len(champion) == len(challenger) == 60


def test_site_sets_are_reproducible_and_distinct():
    """The same seed gives the same sets, each without a repeated site."""
    sites = [f"S{i}" for i in range(12)]
    first = selection.site_sets(sites, 5, 20, seed=3)
    assert first == selection.site_sets(sites, 5, 20, seed=3)
    assert all(len(set(s)) == 5 for s in first)


def test_calibration_curve_covers_every_tolerance(result):
    """One family rate per challenger and candidate tolerance."""
    _, _, curves = result
    assert curves.height == 3 * len(selection.SETTINGS["tolerances"])


def test_no_tolerance_meeting_the_target_makes_an_arm_ineligible(tmp_path, monkeypatch):
    """A per-site check that blocks equal challengers too often rules the arm out."""
    monkeypatch.setitem(selection.SETTINGS, "tolerances", [0.0])
    summary, _, _ = selection.run(load(write_sequences(tmp_path)))
    assert by_arm(summary)[GOOD]["tolerance"] is None
    assert by_arm(summary)[GOOD]["pass_rate"] is None
    assert summary["chosen"] != GOOD


def test_mismatched_sequences_are_refused(tmp_path):
    """Arms scored on different sequences cannot be paired."""
    write_sequences(tmp_path)
    path = tmp_path / f"sequences_{GOOD}.csv"
    pl.read_csv(path).head(100).write_csv(path)
    with pytest.raises(ValueError):
        load(tmp_path)


def test_main_writes_outputs(tmp_path):
    """The command writes the tables, summary and figure."""
    write_sequences(tmp_path)
    status = selection.main(
        [
            "--sequences-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--figure-dir",
            str(tmp_path / "fig"),
        ]
    )
    assert status == 0
    for name in ("selection.csv", "tolerance_curves.csv", "summary.json"):
        assert (tmp_path / "out" / name).exists()
    assert (tmp_path / "fig" / "challenger_selection.png").exists()
