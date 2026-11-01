"""Tests for the tolerance calibration report: its tables line up with the
tolerance grid and the chosen tolerance meets the target.
"""

import numpy as np
import pytest

from cameratrap.report import gate_calibration


@pytest.fixture(name="small_settings")
def fixture_small_settings(monkeypatch):
    """Shrink the calibration settings so the report runs on a few groups."""
    settings = dict(
        gate_calibration.SETTINGS,
        tolerances=[0.02, 0.05, 0.10, 0.40],
        draws=300,
        subsets=50,
        holdout_sites=3,
        holdout_like_min_sequences=60,
    )
    monkeypatch.setattr(gate_calibration, "SETTINGS", settings)
    return settings


def synthetic_groups(sizes, seed=0):
    """Return groups of labels and two noisy, equally good predictors."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, n in enumerate(sizes):
        labels = rng.choice(list("abc"), n)
        groups[f"s{i}"] = (
            labels,
            np.where(rng.random(n) < 0.8, labels, "x"),
            np.where(rng.random(n) < 0.8, labels, "x"),
        )
    return groups


def test_run_tables_follow_the_tolerance_grid(small_settings):
    """One curve row per tolerance, one per-site row per group, rates falling."""
    summary, tables = gate_calibration.run(synthetic_groups([30, 40, 80, 90, 100]))
    curve = tables["calibration_curve"]
    assert curve["tolerance"].to_list() == small_settings["tolerances"]
    assert tables["calibration_per_site"].height == 5
    assert summary["holdout_like_sites"] == 3
    assert np.all(np.diff(curve["family_rate_all_fold_sites"].to_numpy()) <= 0)


def test_run_picks_a_tolerance_that_meets_the_target(small_settings):
    """The chosen tolerance is on the grid and its family rate meets the target."""
    summary, tables = gate_calibration.run(synthetic_groups([30, 40, 80, 90, 100]))
    chosen = summary["smallest_tolerance_meeting_target"]["all_fold_sites"]
    curve = tables["calibration_curve"]
    rate = curve.filter(curve["tolerance"] == chosen)["family_rate_all_fold_sites"][0]
    assert chosen in small_settings["tolerances"]
    assert rate <= small_settings["target_family_rate"]
