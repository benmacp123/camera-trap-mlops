"""Score label-free signals against held-back labels, site by site.

The question is whether a number computed without labels says how
accurate the model is at a site. Three things are measured, with the
site as the unit throughout:

* **Ranking.** Spearman correlation between each signal and per-site
  accuracy, with a bootstrap interval over sites.
* **Prediction.** Each signal is turned into a predicted accuracy and its
  mean absolute error is compared with a baseline that gives every site
  the training sites' mean accuracy. A signal that ranks sites well but
  cannot beat that baseline is no use in a trust report.
* **Change over time.** Whether a signal's move between the early and
  late half of a site's stream tracks the move in accuracy.

Nothing a site is judged by was learnt from that site: the model, the
ATC threshold, the reference embeddings and the straight-line fits all
come from the other folds (for a fold site) or from every fold site (for
a hold-out site).
"""

import numpy as np
import polars as pl
from scipy import stats

from cameratrap.arms import scoring
from cameratrap.drift import signals
from cameratrap.replay import stream

HOLDOUT = "holdout"
ALL = "all"
PRIMARY = "atc"
# Signal name -> the direction expected if it works: +1 when a higher
# value should mean higher accuracy.
SIGNALS = {
    "atc": 1,
    "msp": 1,
    "entropy": -1,
    "knn_distance": -1,
    "mmd": -1,
    "class_mix_shift": -1,
}
# Fixed in the plan (2026-11-10) before any signal was computed.
SETTINGS = {
    "knn_k": 10,
    "reference_size": 2000,
    "min_site_sequences": 20,
    "min_half_sequences": 20,
    "bootstrap_resamples": 2000,
    "confidence": 0.95,
    "seed": 0,
}


def training_mask(splits, split):
    """Return which sequences the model scoring ``split`` was trained on.

    Args:
        splits: Split name of each sequence (a fold name or ``holdout``).
        split: The split being scored.

    Returns:
        Boolean array: every fold sequence outside ``split``. For the
        hold-out that is every fold sequence.
    """
    splits = np.asarray(splits)
    return (splits != HOLDOUT) & (splits != split)


def contexts(table, embeddings, classes, settings):
    """Learn, per split, everything its sites are judged against.

    Args:
        table: One row per sequence with ``split``, ``label``, ``pred``
            and ``msp``.
        embeddings: Unit-norm sequence embeddings, in ``table`` row order.
        classes: Class names.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A dict of split name to a dict with ``train`` (boolean mask),
        ``threshold`` (ATC), ``reference`` (a seeded sample of training
        embeddings for MMD), ``bandwidth`` and ``label_shares`` (the
        training sequences' class mix).
    """
    splits = table["split"].to_numpy()
    correct = (table["label"] == table["pred"]).to_numpy()
    msp = table["msp"].to_numpy()
    label_index = np.array([classes.index(c) for c in table["label"].to_list()])
    out = {}
    for i, split in enumerate(sorted(set(splits.tolist()))):
        train = training_mask(splits, split)
        rows = np.flatnonzero(train)
        # Seeded per split, so one split's sample never depends on another's.
        rng = np.random.default_rng([settings["seed"], i])
        size = min(settings["reference_size"], len(rows))
        reference = embeddings[np.sort(rng.choice(rows, size, replace=False))]
        out[split] = {
            "train": train,
            "threshold": signals.atc_threshold(msp[train], correct[train]),
            "reference": reference,
            "bandwidth": signals.median_bandwidth(reference),
            "label_shares": signals.class_shares(label_index[train], len(classes)),
        }
    return out


def sequence_knn_distance(table, embeddings, ctxs, k):
    """Return each sequence's distance to the training set of its scoring model.

    Args:
        table: One row per sequence with ``split``.
        embeddings: Unit-norm sequence embeddings, in ``table`` row order.
        ctxs: From ``contexts``.
        k: Neighbours to average over.

    Returns:
        Array with one mean cosine distance per sequence.
    """
    splits = table["split"].to_numpy()
    out = np.empty(table.height)
    for split, ctx in ctxs.items():
        rows = splits == split
        out[rows] = signals.knn_distance(embeddings[ctx["train"]], embeddings[rows], k)
    return out


def window_row(rows, embeddings, ctx, classes):
    """Score one window of sequences: its accuracy and every signal.

    Args:
        rows: The window's sequences, with ``label``, ``pred``, ``msp``,
            ``entropy`` and ``knn_distance``.
        embeddings: Those sequences' unit-norm embeddings, in row order.
        ctx: The window's split context from ``contexts``.
        classes: Class names.

    Returns:
        A dict with ``n``, ``accuracy``, ``balanced_accuracy`` and one
        entry per name in ``SIGNALS``. MMD is NaN for a single sequence.
    """
    labels, preds = rows["label"].to_numpy(), rows["pred"].to_numpy()
    msp = rows["msp"].to_numpy()
    pred_index = [classes.index(c) for c in preds.tolist()]
    mmd = (
        signals.mmd2(embeddings, ctx["reference"], ctx["bandwidth"])
        if rows.height > 1
        else float("nan")
    )
    return {
        "n": rows.height,
        "accuracy": float(np.mean(labels == preds)),
        "balanced_accuracy": scoring.balanced_accuracy(labels, preds),
        "atc": signals.atc_estimate(msp, ctx["threshold"]),
        "msp": float(msp.mean()),
        "entropy": float(rows["entropy"].mean()),
        "knn_distance": float(rows["knn_distance"].mean()),
        "mmd": mmd,
        "class_mix_shift": signals.total_variation(
            signals.class_shares(pred_index, len(classes)), ctx["label_shares"]
        ),
    }


def window_table(table, embeddings, ctxs, classes):
    """Score every site as a whole and as an early and a late half.

    Args:
        table: One row per sequence with ``site``, ``split``, ``half``,
            ``label``, ``pred``, ``msp``, ``entropy`` and ``knn_distance``.
        embeddings: Unit-norm sequence embeddings, in ``table`` row order.
        ctxs: From ``contexts``.
        classes: Class names.

    Returns:
        A DataFrame with ``site``, ``split``, ``window`` (``all``,
        ``early`` or ``late``) and the ``window_row`` columns, sorted by
        site then window.
    """
    indexed = table.with_row_index("row")
    out = []
    for (site,), group in indexed.group_by("site", maintain_order=True):
        split = group["split"][0]
        parts = {ALL: group}
        for half in (stream.EARLY, stream.LATE):
            parts[half] = group.filter(pl.col("half") == half)
        for window, rows in parts.items():
            if rows.height == 0:
                continue
            scored = window_row(
                rows, embeddings[rows["row"].to_numpy()], ctxs[split], classes
            )
            out.append({"site": site, "split": split, "window": window} | scored)
    return pl.DataFrame(out).sort("site", "window")


def linear_fit(x, y):
    """Fit a straight line by least squares.

    Args:
        x: Signal value per training site.
        y: Accuracy per training site.

    Returns:
        A pair ``(slope, intercept)``; a flat line through the mean when
        the signal does not vary.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(x) == 0:
        return 0.0, float(y.mean())
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def site_predictions(windows, min_sequences):
    """Predict each site's accuracy from signals, never using that site's fold.

    Args:
        windows: From ``window_table``.
        min_sequences: Sites with fewer sequences are left out, both as
            targets and as training sites.

    Returns:
        A DataFrame with one row per kept site: its whole-site
        ``window_row`` columns, ``source`` (``fold`` or ``holdout``),
        ``baseline`` (training sites' mean accuracy), ``predicted_accuracy``
        (the ATC estimate, used as it is) and ``fit_<signal>`` (a line
        fitted on the training sites, clipped to [0, 1]) for every signal.
    """
    sites = windows.filter(
        (pl.col("window") == ALL) & (pl.col("n") >= min_sequences)
    ).drop("window")
    splits = sites["split"].to_numpy()
    accuracy = sites["accuracy"].to_numpy()
    columns = {"baseline": np.empty(sites.height)}
    columns |= {f"fit_{name}": np.empty(sites.height) for name in SIGNALS}
    for split in np.unique(splits):
        train, test = training_mask(splits, split), splits == split
        columns["baseline"][test] = accuracy[train].mean()
        for name in SIGNALS:
            values = sites[name].to_numpy()
            slope, intercept = linear_fit(values[train], accuracy[train])
            columns[f"fit_{name}"][test] = np.clip(
                slope * values[test] + intercept, 0.0, 1.0
            )
    return sites.with_columns(
        pl.when(pl.col("split") == HOLDOUT)
        .then(pl.lit(HOLDOUT))
        .otherwise(pl.lit("fold"))
        .alias("source"),
        pl.col(PRIMARY).alias("predicted_accuracy"),
        *(pl.Series(name, values) for name, values in columns.items()),
    ).sort("source", "site")


def site_bootstrap(statistic, arrays, settings):
    """Return a percentile bootstrap interval for a statistic over sites.

    Sites are resampled whole, with replacement, because the site is the
    independent unit.

    Args:
        statistic: Function of the resampled arrays returning a float.
        arrays: Arrays with one value per site, resampled together.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A pair ``(low, high)``; NaN draws (a resample with no variation,
        for a correlation) are dropped, and both are NaN if none is left.
    """
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(settings["seed"])
    n = len(arrays[0])
    draws = []
    for _ in range(settings["bootstrap_resamples"]):
        index = rng.integers(0, n, n)
        value = statistic(*(a[index] for a in arrays))
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    tail = (1 - settings["confidence"]) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return float(low), float(high)


def spearman(x, y):
    """Return the Spearman rank correlation, NaN when either side is constant.

    Args:
        x: One value per site.
        y: One value per site.

    Returns:
        The correlation as a float.
    """
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def correlation(x, y, settings):
    """Return a signal's rank correlation with accuracy and its interval.

    Args:
        x: Signal value per site.
        y: Accuracy per site.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A dict with ``rho``, ``low``, ``high`` and ``n``; the first three
        are NaN with fewer than three sites.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return {"rho": float("nan"), "low": float("nan"), "high": float("nan")} | {
            "n": len(x)
        }
    low, high = site_bootstrap(spearman, [x, y], settings)
    return {"rho": spearman(x, y), "low": low, "high": high, "n": len(x)}


def error_summary(predicted, baseline, accuracy, settings):
    """Compare a predictor's absolute error with the baseline's, over sites.

    Args:
        predicted: Predicted accuracy per site.
        baseline: Baseline prediction per site.
        accuracy: Observed accuracy per site.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A dict with ``mae``, ``bias`` (mean of predicted minus observed),
        ``baseline_mae``, ``improvement`` (baseline MAE minus this MAE;
        positive is better than the baseline) and its interval
        ``improvement_low`` and ``improvement_high``.
    """
    predicted, baseline = np.asarray(predicted), np.asarray(baseline)
    accuracy = np.asarray(accuracy)
    error = np.abs(predicted - accuracy)
    base_error = np.abs(baseline - accuracy)
    low, high = site_bootstrap(
        lambda a, b: float(np.mean(b - a)), [error, base_error], settings
    )
    return {
        "mae": float(error.mean()),
        "bias": float(np.mean(predicted - accuracy)),
        "baseline_mae": float(base_error.mean()),
        "improvement": float(np.mean(base_error - error)),
        "improvement_low": low,
        "improvement_high": high,
    }


def signal_scores(sites, settings):
    """Score every signal's ranking and prediction, by source.

    Args:
        sites: From ``site_predictions``.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A DataFrame with one row per ``source`` and ``predictor``: each
        signal's fitted line (``fit_<signal>``) plus ``atc_direct`` (the
        ATC estimate used unfitted, the primary). Columns are the
        ``correlation`` and ``error_summary`` fields and ``expected_sign``.
    """
    rows = []
    for (source,), group in sites.group_by("source", maintain_order=True):
        accuracy = group["accuracy"].to_numpy()
        baseline = group["baseline"].to_numpy()
        predictors = {"atc_direct": (PRIMARY, "predicted_accuracy")}
        predictors |= {f"fit_{name}": (name, f"fit_{name}") for name in SIGNALS}
        for predictor, (name, column) in predictors.items():
            rows.append(
                {
                    "source": source,
                    "predictor": predictor,
                    "signal": name,
                    "expected_sign": SIGNALS[name],
                }
                | correlation(group[name].to_numpy(), accuracy, settings)
                | error_summary(group[column].to_numpy(), baseline, accuracy, settings)
            )
    return pl.DataFrame(rows)


def half_changes(windows, min_half):
    """Return each site's change from its early to its late half.

    Args:
        windows: From ``window_table``.
        min_half: Sites with fewer sequences in either half are left out.

    Returns:
        A DataFrame with ``site``, ``source``, ``n_early``, ``n_late``,
        ``accuracy_early``, ``accuracy_late``, ``accuracy_change`` and
        ``<signal>_change`` (late minus early) for every signal, sorted
        by source then site.
    """
    names = ["n", "accuracy", *SIGNALS]
    early, late = (
        windows.filter(pl.col("window") == half).select(
            "site", "split", *(pl.col(c).alias(f"{c}_{half}") for c in names)
        )
        for half in (stream.EARLY, stream.LATE)
    )
    joined = early.join(late, on=["site", "split"]).filter(
        (pl.col("n_early") >= min_half) & (pl.col("n_late") >= min_half)
    )
    return joined.select(
        "site",
        pl.when(pl.col("split") == HOLDOUT)
        .then(pl.lit(HOLDOUT))
        .otherwise(pl.lit("fold"))
        .alias("source"),
        "n_early",
        "n_late",
        "accuracy_early",
        "accuracy_late",
        *(
            (pl.col(f"{c}_late") - pl.col(f"{c}_early")).alias(f"{c}_change")
            for c in names[1:]
        ),
    ).sort("source", "site")


def noise_check(changes):
    """Compare the spread of accuracy changes with what chance alone gives.

    If a site's accuracy did not move at all, its two halves would still
    differ through sampling. The expected spread from that is worked out
    from each site's pooled accuracy and half sizes (a binomial standard
    error); a ratio near 1 means there was little real change to detect.

    Args:
        changes: Rows of ``half_changes`` for one source.

    Returns:
        A dict with ``observed_sd`` of the accuracy change over sites,
        ``expected_sd_no_change`` and their ``ratio``.
    """
    n1, n2 = changes["n_early"].to_numpy(), changes["n_late"].to_numpy()
    pooled = (
        changes["accuracy_early"].to_numpy() * n1
        + changes["accuracy_late"].to_numpy() * n2
    ) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    observed = float(np.std(changes["accuracy_change"].to_numpy(), ddof=1))
    expected = float(np.sqrt(variance.mean()))
    return {
        "observed_sd": observed,
        "expected_sd_no_change": expected,
        "ratio": observed / expected if expected else float("nan"),
    }


def change_scores(changes, settings):
    """Score whether each signal's change tracks the change in accuracy.

    Args:
        changes: From ``half_changes``.
        settings: ``SETTINGS`` or a test's version of it.

    Returns:
        A DataFrame with one row per ``source`` and ``signal``: the
        ``correlation`` fields of signal change against accuracy change,
        and ``expected_sign``.
    """
    rows = []
    for (source,), group in changes.group_by("source", maintain_order=True):
        target = group["accuracy_change"].to_numpy()
        for name, sign in SIGNALS.items():
            rows.append(
                {"source": source, "signal": name, "expected_sign": sign}
                | correlation(group[f"{name}_change"].to_numpy(), target, settings)
            )
    return pl.DataFrame(rows)
