"""Tests for the drift study: no leakage across folds, the size rules, and the scores."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from cameratrap.drift import study
from cameratrap.report import drift

CLASSES = ["agouti", "tapir"]
FAST = {**study.SETTINGS, "bootstrap_resamples": 200, "reference_size": 60, "knn_k": 3}
SPLITS = ["fold0", "fold1", "fold2", "holdout"]


def make_replay(sites_per_split=4, n=60, seed=0, small_site=None):
    """Build a synthetic replay table whose confidence tracks its accuracy.

    Each site has its own error rate; a wrong sequence gets low confidence
    and an embedding scattered away from the training cluster, so every signal
    carries real information about accuracy.

    Args:
        sites_per_split: Sites in each of the three folds and the hold-out.
        n: Sequences per site.
        seed: Seed for the generator.
        small_site: Optional site name cut down to 10 sequences.

    Returns:
        A pair ``(table, embeddings)`` shaped like ``drift.replay`` output.
    """
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for s, split in enumerate(SPLITS):
        for j in range(sites_per_split):
            site = f"{split}_s{j}"
            error_rate = 0.05 + 0.1 * j
            count = 10 if site == small_site else n
            for i in range(count):
                label = CLASSES[i % 2]
                wrong = rng.random() < error_rate
                pred = CLASSES[(i + 1) % 2] if wrong else label
                msp = rng.uniform(0.5, 0.7) if wrong else rng.uniform(0.8, 1.0)
                # Right answers cluster around one direction; wrong ones are
                # scattered, so they sit far from every training neighbour.
                vector = rng.normal(size=6)
                if not wrong:
                    vector = 0.2 * vector + np.eye(6)[0]
                rows.append(
                    {
                        "seq_id": f"{site}_{i:03d}",
                        "site": site,
                        "split": split,
                        "label": label,
                        "timestamp": datetime(2020, 1, 1, tzinfo=timezone.utc)
                        + timedelta(hours=i),
                        "position": i,
                        "half": "early" if i < count // 2 else "late",
                        "pred": pred,
                        "msp": msp,
                        "entropy": 1.0 - msp,
                    }
                )
                vectors.append(vector)
    embeddings = np.array(vectors)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return pl.DataFrame(rows), embeddings


def test_training_mask_excludes_own_fold_and_holdout():
    """A fold is judged by the other folds; the hold-out by every fold."""
    splits = np.array(["fold0", "fold1", "holdout"])
    assert study.training_mask(splits, "fold0").tolist() == [False, True, False]
    assert study.training_mask(splits, "holdout").tolist() == [True, True, False]


def test_a_folds_context_ignores_that_folds_own_labels():
    """Changing fold0's and the hold-out's answers leaves fold0's threshold alone."""
    table, embeddings = make_replay()
    own = pl.col("split").is_in(["fold0", "holdout"])
    altered = table.with_columns(
        pl.when(own).then(pl.lit("tapir")).otherwise(pl.col("pred")).alias("pred"),
        pl.when(own).then(pl.lit(0.5)).otherwise(pl.col("msp")).alias("msp"),
    )
    before = study.contexts(table, embeddings, CLASSES, FAST)["fold0"]
    after = study.contexts(altered, embeddings, CLASSES, FAST)["fold0"]
    assert before["threshold"] == after["threshold"]
    assert np.array_equal(before["label_shares"], after["label_shares"])
    assert np.array_equal(before["reference"], after["reference"])


def test_a_sites_prediction_ignores_that_folds_own_accuracy():
    """Fold0's baseline and fitted lines do not move when fold0's labels change."""
    table, embeddings = make_replay()
    flipped = table.with_columns(
        pl.when(pl.col("split") == "fold0")
        .then(pl.lit("tapir"))
        .otherwise(pl.col("label"))
        .alias("label")
    )
    columns = ["site", "baseline", "predicted_accuracy", "fit_msp", "fit_knn_distance"]
    outputs = []
    for version in (table, flipped):
        sites = drift.run(version, embeddings, CLASSES, FAST)[2]
        outputs.append(sites.filter(pl.col("split") == "fold0").select(columns))
    assert outputs[0].equals(outputs[1])


def test_sites_below_the_minimum_are_left_out():
    """A ten-sequence site is neither scored nor used to fit."""
    table, embeddings = make_replay(small_site="fold1_s2")
    sites = drift.run(table, embeddings, CLASSES, FAST)[2]
    assert "fold1_s2" not in sites["site"].to_list()
    assert sites.height == 15


def test_informative_signals_rank_sites_and_beat_the_baseline():
    """When confidence tracks accuracy, ATC ranks fold sites and beats the mean."""
    table, embeddings = make_replay(sites_per_split=5, n=120)
    scores = drift.run(table, embeddings, CLASSES, FAST)[3]
    atc = scores.filter(
        (pl.col("source") == "fold") & (pl.col("predictor") == "atc_direct")
    ).row(0, named=True)
    assert atc["rho"] > 0.7
    assert atc["improvement"] > 0
    assert atc["n"] == 15
    signs = dict(
        scores.filter(pl.col("source") == "fold").select("signal", "rho").iter_rows()
    )
    # Each secondary signal moves in its expected direction on this data.
    assert signs["entropy"] < 0 and signs["knn_distance"] < 0 and signs["msp"] > 0


def test_holdout_rows_are_scored_separately():
    """The score table has both sources and the hold-out covers only its sites."""
    table, embeddings = make_replay()
    summary, _, sites, scores, _, _ = drift.run(table, embeddings, CLASSES, FAST)
    assert set(scores["source"].to_list()) == {"fold", "holdout"}
    assert summary["sites_scored"] == {"fold": 12, "holdout": 4}
    held = sites.filter(pl.col("source") == "holdout")
    assert held["site"].str.starts_with("holdout").all()


def test_run_is_deterministic():
    """Two runs on the same replay give identical tables: same seed, same output."""
    table, embeddings = make_replay()
    first = drift.run(table, embeddings, CLASSES, FAST)
    second = drift.run(table, embeddings, CLASSES, FAST)
    assert first[0] == second[0]
    for a, b in zip(first[1:], second[1:]):
        assert a.equals(b)


def test_linear_fit_recovers_a_line_and_survives_a_constant_signal():
    """An exact line is recovered; a signal that never varies predicts the mean."""
    slope, intercept = study.linear_fit([0.0, 1.0, 2.0], [0.5, 0.7, 0.9])
    assert (slope, intercept) == pytest.approx((0.2, 0.5))
    assert study.linear_fit([1.0, 1.0, 1.0], [0.2, 0.4, 0.9]) == (0.0, 0.5)


def test_error_summary_by_hand():
    """Errors of 0.1 and 0.3 against a baseline's 0.2 and 0.2 give no net gain."""
    out = study.error_summary([0.9, 0.5], [0.6, 0.6], [0.8, 0.8], FAST)
    assert out["mae"] == pytest.approx(0.2)
    assert out["bias"] == pytest.approx(-0.1)
    assert out["baseline_mae"] == pytest.approx(0.2)
    assert out["improvement"] == pytest.approx(0.0)
    assert out["improvement_low"] <= 0.0 <= out["improvement_high"]


def test_correlation_needs_three_sites_and_handles_a_constant():
    """Two sites give NaN, and a constant signal gives NaN instead of raising."""
    assert np.isnan(study.correlation([1, 2], [1, 2], FAST)["rho"])
    assert np.isnan(study.correlation([1, 1, 1, 1], [1, 2, 3, 4], FAST)["rho"])
    perfect = study.correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], FAST)
    assert perfect["rho"] == pytest.approx(1.0)
    assert perfect["high"] == pytest.approx(1.0)


def make_windows(rows):
    """Build a window table from ``(site, split, window, n, accuracy, msp)`` rows.

    Args:
        rows: One tuple per window; the other signals are set to zero.

    Returns:
        A DataFrame shaped like ``study.window_table`` output.
    """
    out = []
    for site, split, window, n, accuracy, msp in rows:
        signals = {name: 0.0 for name in study.SIGNALS} | {"msp": msp}
        out.append(
            {"site": site, "split": split, "window": window, "n": n}
            | {"accuracy": accuracy, "balanced_accuracy": accuracy}
            | signals
        )
    return pl.DataFrame(out)


def test_half_changes_are_late_minus_early_and_respect_the_minimum():
    """Changes are late minus early; a site with a thin half is dropped."""
    windows = make_windows(
        [
            ("A", "fold0", "early", 30, 0.9, 0.95),
            ("A", "fold0", "late", 30, 0.7, 0.80),
            ("B", "holdout", "early", 30, 0.8, 0.9),
            ("B", "holdout", "late", 19, 0.8, 0.9),
        ]
    )
    changes = study.half_changes(windows, 20)
    assert changes["site"].to_list() == ["A"]
    row = changes.row(0, named=True)
    assert row["source"] == "fold"
    assert row["accuracy_change"] == pytest.approx(-0.2)
    assert row["msp_change"] == pytest.approx(-0.15)


def test_noise_check_by_hand():
    """Two sites at 0.5 accuracy with 50 a half expect an SD of 0.1 from chance."""
    changes = pl.DataFrame(
        {
            "n_early": [50, 50],
            "n_late": [50, 50],
            "accuracy_early": [0.6, 0.4],
            "accuracy_late": [0.4, 0.6],
            "accuracy_change": [-0.2, 0.2],
        }
    )
    out = study.noise_check(changes)
    assert out["expected_sd_no_change"] == pytest.approx(0.1)
    assert out["observed_sd"] == pytest.approx(np.std([-0.2, 0.2], ddof=1))
    assert out["ratio"] == pytest.approx(out["observed_sd"] / 0.1)


def test_change_scores_follow_a_signal_that_tracks_accuracy():
    """A signal whose change mirrors the accuracy change correlates at 1."""
    rows = []
    for i, drop in enumerate([0.0, 0.05, 0.1, 0.15, 0.2]):
        rows.append((f"S{i}", "fold0", "early", 40, 0.9, 0.9))
        rows.append((f"S{i}", "fold0", "late", 40, 0.9 - drop, 0.9 - drop / 2))
    scores = study.change_scores(study.half_changes(make_windows(rows), 20), FAST)
    msp = scores.filter(pl.col("signal") == "msp").row(0, named=True)
    assert msp["rho"] == pytest.approx(1.0)
    assert msp["n"] == 5


def test_replay_keeps_embeddings_aligned_with_the_table():
    """Each row's embedding is its own sequence's mean, after site-date sorting."""
    index = pl.DataFrame(
        {
            "path": ["p1", "p2", "p3", "p4"],
            "seq_id": ["b", "b", "a", "c"],
            "site": ["X", "X", "Y", "X"],
            "split": ["fold0"] * 4,
            "label": ["tapir", "tapir", "agouti", "agouti"],
        }
    )
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    manifest = pl.DataFrame(
        {
            "seq_id": ["a", "b", "c"],
            "timestamp": [start + timedelta(days=d) for d in (0, 5, 1)],
        }
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [3.0, 0.0]])
    logits = np.array([[0.0, 4.0], [0.0, 2.0], [5.0, 0.0], [0.0, 1.0]])
    table, embeddings = drift.replay(index, manifest, features, logits, CLASSES)
    assert table["seq_id"].to_list() == ["c", "b", "a"]
    assert table["pred"].to_list() == ["tapir", "tapir", "agouti"]
    expected = np.array([[1.0, 0.0], [0.5**0.5, 0.5**0.5], [0.0, 1.0]])
    assert embeddings == pytest.approx(expected)
    # Sequence b averages logits (0, 3) before the softmax.
    assert table["msp"][1] == pytest.approx(1 / (1 + np.exp(-3.0)))


def test_champion_logits_use_the_frozen_head_on_the_holdout():
    """Hold-out frames are scored by the saved head, fold frames by a refit."""
    # Three classes: with two, scikit-learn returns one logit column, not two.
    names = [*CLASSES, "paca"]
    rng = np.random.default_rng(0)
    rows = []
    for split in ("fold0", "fold1", "fold2", "fold3", "holdout"):
        for site in range(3):
            for i in range(9):
                rows.append(
                    {"site": f"{split}_{site}", "split": split, "label": names[i % 3]}
                )
    frames = pl.DataFrame(rows)
    y = np.array([names.index(c) for c in frames["label"]])
    features = rng.normal(size=(frames.height, 3)) + 2.0 * np.eye(3)[y]
    head = {
        "coef": np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 0.5], [2.0, 0.0, 1.0]]),
        "intercept": np.array([0.5, -0.5, 0.0]),
        "classes": np.array(names),
    }
    logits = drift.champion_logits(frames, features, head)
    held = (frames["split"] == "holdout").to_numpy()
    assert logits[held] == pytest.approx(
        features[held] @ head["coef"].T + head["intercept"]
    )
    assert not np.allclose(
        logits[~held], features[~held] @ head["coef"].T + head["intercept"]
    )


def test_check_committed_catches_a_changed_prediction():
    """A replay that disagrees with the committed predictions is refused."""
    table, _ = make_replay(sites_per_split=1, n=20)
    committed = table.select("seq_id", "pred", "msp")
    drift.check_committed(table, committed)
    moved = committed.with_columns(pl.col("msp") + 1e-3)
    with pytest.raises(ValueError, match="differs"):
        drift.check_committed(table, moved)
    with pytest.raises(ValueError, match="other rows"):
        drift.check_committed(table, committed.head(5))


def test_figures_are_written(tmp_path):
    """Both figures render from a run's tables without error."""
    table, embeddings = make_replay()
    summary, _, sites, scores, _, _ = drift.run(table, embeddings, CLASSES, FAST)
    drift.plot_predicted(sites, summary, tmp_path / "predicted.png")
    drift.plot_signals(scores, tmp_path / "signals.png")
    assert (tmp_path / "predicted.png").stat().st_size > 0
    assert (tmp_path / "signals.png").stat().st_size > 0
