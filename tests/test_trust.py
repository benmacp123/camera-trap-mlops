"""Tests for the trust report: Holm flags, unjudgeable sites, review budget and threshold choice."""

import json

import numpy as np
import polars as pl
import pytest

from cameratrap.gate import rule
from cameratrap.report import trust

FAST_GATE = rule.GateConfig(null_draws=500, resamples=999)
FAST_SETTINGS = {**trust.SETTINGS, "bootstrap_resamples": 200}


def make_table(rows):
    """Build a joined prediction table from ``(site, source, label, pred, msp, previous)`` rows.

    Args:
        rows: Tuples, one per sequence.

    Returns:
        A DataFrame shaped like ``trust.load_predictions`` output.
    """
    cols = ["site", "source", "label", "pred", "msp", "previous"]
    table = pl.DataFrame([dict(zip(cols, r)) for r in rows])
    ids = [f"seq{i:05d}" for i in range(table.height)]
    return table.with_columns(pl.Series("seq_id", ids)).select(["seq_id", *cols])


def site_rows(site, source, n, champion_wrong, previous_wrong, msp=0.9):
    """Return rows for one two-class site with chosen error counts.

    Args:
        site: Site name.
        source: ``fold`` or ``holdout``.
        n: Sequences, alternating between two classes.
        champion_wrong: Sequences, from the start, the champion gets wrong.
        previous_wrong: Sequences, from the end, the previous champion gets wrong.
        msp: Champion confidence on every sequence.

    Returns:
        A list of row tuples.
    """
    rows = []
    for i in range(n):
        label = "agouti" if i % 2 else "tapir"
        other = "tapir" if i % 2 else "agouti"
        pred = other if i < champion_wrong else label
        previous = other if i >= n - previous_wrong else label
        rows.append((site, source, label, pred, msp, previous))
    return rows


def test_holm_steps_down_and_stops_at_first_failure():
    """Holm rejects in order of p-value and stops at the first that fails."""
    flags = trust.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # Cut-offs 0.0125, 0.0167, 0.025, 0.05: 0.005 and 0.01 pass, 0.03 fails.
    assert flags.tolist() == [True, False, False, True]


def test_holm_flags_at_least_what_bonferroni_flags():
    """Every Bonferroni rejection is also a Holm rejection."""
    rng = np.random.default_rng(1)
    p = rng.random(30) ** 4
    assert np.all(trust.holm(p, 0.05)[p <= 0.05 / 30])


def test_best_attainable_p_halves_per_disagreement_and_is_floored():
    """Each disagreement halves the best p, but never below one over draws."""
    assert trust.best_attainable_p(3, 20000) == pytest.approx(1 / 8)
    assert trust.best_attainable_p(40, 20000) == pytest.approx(1 / 20001)


def test_review_budget_counts_unreviewed_errors_only():
    """Residual error counts only wrong sequences that were not reviewed."""
    table = make_table(
        [
            ("S", "fold", "a", "a", 0.95, "a"),
            ("S", "fold", "a", "b", 0.95, "a"),
            ("S", "fold", "a", "b", 0.55, "a"),
            ("S", "fold", "a", "a", 0.55, "a"),
        ]
    )
    budget = trust.review_budget(table, [0.0, 0.9], "source")
    zero, high = budget.row(0, named=True), budget.row(1, named=True)
    assert (zero["review_share"], zero["residual_error"]) == (0.0, 0.5)
    assert (high["review_share"], high["residual_error"]) == (0.5, 0.25)
    assert high["accepted_error"] == 0.5


def test_choose_threshold_uses_fold_sites_only():
    """The threshold is chosen on fold sites, whatever the hold-out shows."""
    budget = pl.DataFrame(
        {
            "source": ["fold"] * 3 + ["holdout"] * 3,
            "threshold": [0.0, 0.5, 0.9] * 2,
            "residual_error": [0.08, 0.06, 0.04, 0.01, 0.01, 0.01],
        }
    )
    assert trust.choose_threshold(budget, 0.05) == 0.9
    assert trust.choose_threshold(budget, 0.01) is None


def test_choose_threshold_keeps_zero():
    """A threshold of zero (review nothing) is a valid choice, not a missing one."""
    budget = pl.DataFrame(
        {"source": ["fold"], "threshold": [0.0], "residual_error": [0.02]}
    )
    assert trust.choose_threshold(budget, 0.05) == 0.0


def test_species_table_marks_weak_ok_and_too_few():
    """Weak needs low recall and enough sequences; small cells are too few."""
    rows = [("S", "fold", "tapir", "agouti", 0.9, "tapir")] * 4
    rows += [("S", "fold", "tapir", "tapir", 0.9, "tapir")] * 2
    rows += [("S", "fold", "agouti", "agouti", 0.9, "agouti")] * 6
    rows += [("S", "fold", "paca", "agouti", 0.9, "paca")] * 2
    species = trust.species_table(make_table(rows), FAST_SETTINGS)
    status = dict(zip(species["species"], species["status"]))
    assert status == {"agouti": "ok", "paca": "too few", "tapir": "weak"}
    tapir = species.filter(pl.col("species") == "tapir").row(0, named=True)
    assert tapir["mistaken_for"] == "agouti"


def test_site_table_flags_a_real_regression_and_labels_small_sites():
    """A clear drop is flagged, a tiny site is unjudgeable, an equal site is clear."""
    rows = site_rows("BAD", "fold", 120, champion_wrong=40, previous_wrong=0)
    rows += site_rows("TINY", "holdout", 6, champion_wrong=2, previous_wrong=0)
    rows += site_rows("SAME", "fold", 120, champion_wrong=12, previous_wrong=12)
    per_site = trust.site_table(make_table(rows), FAST_GATE, FAST_SETTINGS)
    status = dict(zip(per_site["site"], per_site["status"]))
    assert status == {
        "BAD": trust.STATUS_FLAGGED,
        "SAME": trust.STATUS_CLEAR,
        "TINY": trust.STATUS_SMALL,
    }
    bad = per_site.filter(pl.col("site") == "BAD").row(0, named=True)
    assert bad["ba_low"] <= bad["balanced_accuracy"] <= bad["ba_high"]


def test_load_predictions_refuses_mismatched_arms(tmp_path):
    """Arms covering different sequences are refused rather than silently joined."""
    for source, directory in trust.SOURCES.items():
        (tmp_path / directory).mkdir(parents=True)
        base = pl.DataFrame(
            {
                "seq_id": [f"{source}1", f"{source}2"],
                "site": [source.upper()] * 2,
                "label": ["a", "b"],
                "pred": ["a", "b"],
                "msp": [0.9, 0.8],
            }
        )
        base.write_csv(tmp_path / directory / f"sequences_{trust.CHAMPION}.csv")
        base.head(1).write_csv(tmp_path / directory / f"sequences_{trust.PREVIOUS}.csv")
    with pytest.raises(ValueError, match="different sequences"):
        trust.load_predictions(tmp_path)


def test_run_writes_a_report_with_the_suggested_threshold(tmp_path):
    """The whole report builds, and the summary's threshold matches the fold budget."""
    rows = []
    for s in range(4):
        rows += site_rows(f"F{s}", "fold", 60, 3, 9, msp=0.95)
        rows += site_rows(f"G{s}", "fold", 20, 10, 0, msp=0.55)
    rows += site_rows("H0", "holdout", 60, 3, 3, msp=0.95)
    table = make_table(rows)
    summary, per_site, _, budget = trust.run(table, FAST_GATE, FAST_SETTINGS)
    # Fold error is 52/320 overall; reviewing the 0.55 sites leaves 12/320 (3.75%).
    assert summary["suggested_threshold"] == 0.6
    fold = summary["at_suggested_threshold"]["fold"]
    assert fold["residual_error"] == pytest.approx(12 / 320)
    assert per_site.height == 9 and per_site["source"][0] == "holdout"
    trust.write_report(summary, per_site, budget, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "Suggested threshold 0.6" in text
    json.dumps(summary)


def test_predicted_accuracy_is_joined_and_its_error_reported(tmp_path):
    """The drift study's estimate lands beside the observed accuracy, gaps and all."""
    rows = site_rows("F0", "fold", 40, 4, 4) + site_rows("H0", "holdout", 40, 2, 2)
    rows += site_rows("H1", "holdout", 40, 0, 0)
    predicted = pl.DataFrame({"site": ["F0", "H0"], "predicted_accuracy": [0.8, 0.9]})
    summary, per_site, _, budget = trust.run(
        make_table(rows), FAST_GATE, FAST_SETTINGS, predicted=predicted
    )
    by_site = dict(per_site.select("site", "predicted_accuracy").iter_rows())
    assert by_site == {"F0": 0.8, "H0": 0.9, "H1": None}
    # F0 scores 0.90 against a predicted 0.80; H0 0.95 against 0.90.
    assert summary["predicted_accuracy_mean_error"] == pytest.approx(
        {"fold": 0.1, "holdout": 0.05}
    )
    trust.write_report(summary, per_site, budget, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "Predicted accuracy" in text and "5.0 points on holdout sites" in text


def test_report_without_a_drift_study_has_no_predicted_column(tmp_path):
    """Before the drift study has run, the report is written without the column."""
    rows = site_rows("F0", "fold", 40, 4, 4) + site_rows("H0", "holdout", 40, 2, 2)
    summary, per_site, _, budget = trust.run(make_table(rows), FAST_GATE, FAST_SETTINGS)
    assert summary["predicted_accuracy_mean_error"] is None
    trust.write_report(summary, per_site, budget, tmp_path / "report.md")
    assert "Predicted accuracy" not in (tmp_path / "report.md").read_text()
