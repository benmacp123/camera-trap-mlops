"""The promotion rule: a group-level paired comparison of two models.

A challenger replaces the champion only if all three hold:

- the mean per-group difference is above a minimum gain (zero by
  default), so a tie keeps the champion: switching models has costs of
  its own;
- the lower end of a two-sided BCa bootstrap interval for the mean
  per-group difference (challenger minus champion) clears the
  non-inferiority margin, so on average it is not worse by more than the
  margin;
- no single group drops by more than chance explains there, so a gain
  at most groups cannot hide a collapse at one.

The per-group check is noise-aware. A fixed tolerance in points has to
be loose enough for the noisiest group (a site with 24 sequences, where
one sequence moves the score 20 points), and is then meaningless at a
group with a thousand. Instead each group's drop gets a p-value from the
swap null (``calibrate``), and a group blocks if its p-value is below
``family_alpha`` divided by the number of groups (Bonferroni). That keeps
the chance of wrongly blocking an equally good challenger at most
``family_alpha`` however many groups there are, and each group's
threshold follows its own size and noise. Holm's method would flag the
same yes/no outcome (its first step is this test), so it is left to
reports that list every group that dropped.

Groups are the unit of resampling (sites here, vessels or regions
elsewhere), never the images inside them. Every group counts equally,
whatever its size.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from cameratrap.gate import calibrate


@dataclass(frozen=True)
class GateConfig:
    """Settings fixed, and hashed, before any challenger is scored.

    Attributes:
        margin: Non-inferiority margin on the metric's scale; the lower
            bound of the mean difference must be above it (``-0.01`` lets
            a challenger be at most one point worse on average).
        min_mean_gain: The observed mean difference must be above this;
            ``0.0`` means a tie does not promote.
        family_alpha: Largest chance, across all groups together, of
            blocking a challenger that is in truth no worse.
        null_draws: Swap-null draws per group; enough to resolve
            ``family_alpha`` divided by the number of groups.
        confidence: Two-sided confidence level of the interval.
        resamples: Bootstrap resamples.
        seed: Seed for the bootstrap, so a decision can be re-run exactly.
        min_groups: Fewest groups the rule will decide on.
    """

    margin: float = -0.01
    min_mean_gain: float = 0.0
    family_alpha: float = 0.05
    null_draws: int = 20000
    confidence: float = 0.95
    resamples: int = 9999
    seed: int = 0
    min_groups: int = 5


@dataclass(frozen=True)
class GateDecision:
    """The outcome of one comparison and the evidence behind it.

    Attributes:
        promote: Whether the challenger should replace the champion.
        mean_difference: Mean per-group difference, challenger minus champion.
        interval: ``(low, high)`` bootstrap interval for that mean.
        differences: Per-group difference, keyed by group.
        p_values: Per-group p-value of the drop, keyed by group.
        cutoff: The per-group p-value cutoff, ``family_alpha / groups``.
        regressions: Groups whose drop chance does not explain.
        reasons: One line per failed condition; empty when promoted.
    """

    promote: bool
    mean_difference: float
    interval: tuple
    differences: dict
    p_values: dict
    cutoff: float
    regressions: list = field(default_factory=list)
    reasons: list = field(default_factory=list)


def paired_differences(champion, challenger):
    """Return the per-group difference, challenger minus champion.

    Args:
        champion: Mapping of group to the champion's metric.
        challenger: Mapping of group to the challenger's metric.

    Returns:
        A dict of group to difference, in sorted group order.

    Raises:
        ValueError: If the two models were scored on different groups, or a
            metric is missing (NaN); comparing on unmatched groups would
            make the pairing meaningless.
    """
    if set(champion) != set(challenger):
        only = sorted(set(champion) ^ set(challenger))
        raise ValueError(f"models scored on different groups: {only}")
    diffs = {g: float(challenger[g]) - float(champion[g]) for g in sorted(champion)}
    missing = [g for g, d in diffs.items() if np.isnan(d)]
    if missing:
        raise ValueError(f"metric missing for groups: {missing}")
    return diffs


def mean_interval(values, config):
    """Return a BCa bootstrap interval for the mean of some values.

    BCa (bias-corrected and accelerated) adjusts the percentile interval
    for skew and bias, which matters with about ten groups, where the
    plain percentile interval tends to be too narrow.

    Args:
        values: One value per group.
        config: A ``GateConfig``.

    Returns:
        A pair ``(low, high)``.
    """
    values = np.asarray(values, dtype=float)
    # Identical values leave nothing to resample, and scipy returns NaN.
    if np.ptp(values) == 0:
        return float(values[0]), float(values[0])
    result = stats.bootstrap(
        (values,),
        np.mean,
        n_resamples=config.resamples,
        confidence_level=config.confidence,
        method="BCa",
        rng=np.random.default_rng(config.seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def decide(champion, challenger, p_values, config=GateConfig()):
    """Decide whether a challenger replaces the champion.

    Args:
        champion: Mapping of group to the champion's metric (higher is better).
        challenger: Mapping of group to the challenger's metric, same groups.
        p_values: Mapping of group to the one-sided p-value of its drop,
            from ``calibrate.p_value``; ``evaluate`` computes them.
        config: The frozen ``GateConfig``.

    Returns:
        A ``GateDecision``.

    Raises:
        ValueError: If the groups differ, a metric is missing, or there are
            fewer than ``config.min_groups`` groups.
    """
    diffs = paired_differences(champion, challenger)
    if len(diffs) < config.min_groups:
        raise ValueError(
            f"{len(diffs)} groups, fewer than the {config.min_groups} required"
        )
    values = np.array(list(diffs.values()))
    mean = float(values.mean())
    low, high = mean_interval(values, config)
    cutoff = config.family_alpha / len(diffs)
    regressions = [g for g, d in diffs.items() if d < 0 and p_values[g] < cutoff]

    reasons = []
    if mean <= config.min_mean_gain:
        reasons.append(
            f"mean gain {mean:+.3f} is not above {config.min_mean_gain:+.3f}"
        )
    if low <= config.margin:
        reasons.append(
            f"lower bound {low:+.3f} does not clear the margin {config.margin:+.3f}"
        )
    if regressions:
        reasons.append(
            f"{len(regressions)} group(s) drop by more than chance explains "
            f"(p < {cutoff:.4f}): {', '.join(regressions)}"
        )
    return GateDecision(
        promote=not reasons,
        mean_difference=mean,
        interval=(low, high),
        differences=diffs,
        p_values={g: float(p_values[g]) for g in diffs},
        cutoff=cutoff,
        regressions=regressions,
        reasons=reasons,
    )


def evaluate(groups, metric, config=GateConfig()):
    """Score both models per group, test each group's drop, and decide.

    Args:
        groups: Mapping of group to ``(labels, champion_preds,
            challenger_preds)``, aligned by item.
        metric: Callable ``metric(labels, preds) -> float``, higher better.
        config: The frozen ``GateConfig``.

    Returns:
        A ``GateDecision``.
    """
    rng = np.random.default_rng(config.seed)
    champion, challenger, p_values = {}, {}, {}
    for group in sorted(groups):
        labels, a, b = groups[group]
        champion[group] = metric(labels, a)
        challenger[group] = metric(labels, b)
        null = calibrate.null_differences(labels, a, b, metric, config.null_draws, rng)
        p_values[group] = calibrate.p_value(null, challenger[group] - champion[group])
    return decide(champion, challenger, p_values, config)
