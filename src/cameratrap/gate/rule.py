"""The promotion rule: a group-level paired comparison of two models.

A challenger replaces the champion only if both hold:

- the lower end of a two-sided BCa bootstrap interval for the mean
  per-group difference (challenger minus champion) clears the
  non-inferiority margin, so on average it is not worse by more than the
  margin;
- no single group regresses by more than the tolerance, so a gain at
  most groups cannot hide a collapse at one.

Groups are the unit of resampling (sites here, vessels or regions
elsewhere), never the images inside them. Every group counts equally,
whatever its size.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GateConfig:
    """Settings fixed, and hashed, before any challenger is scored.

    Attributes:
        margin: Non-inferiority margin on the metric's scale; the lower
            bound of the mean difference must be above it (``-0.01`` lets
            a challenger be at most one point worse on average).
        tolerance: Largest regression allowed at any one group.
        confidence: Two-sided confidence level of the interval.
        resamples: Bootstrap resamples.
        seed: Seed for the bootstrap, so a decision can be re-run exactly.
        min_groups: Fewest groups the rule will decide on.
    """

    margin: float = -0.01
    tolerance: float = 0.05
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
        regressions: Groups that fall more than the tolerance behind.
        reasons: One line per failed condition; empty when promoted.
    """

    promote: bool
    mean_difference: float
    interval: tuple
    differences: dict
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


def decide(champion, challenger, config=GateConfig()):
    """Decide whether a challenger replaces the champion.

    Args:
        champion: Mapping of group to the champion's metric (higher is better).
        challenger: Mapping of group to the challenger's metric, same groups.
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
    low, high = mean_interval(values, config)
    regressions = [g for g, d in diffs.items() if d < -config.tolerance]

    reasons = []
    if low <= config.margin:
        reasons.append(
            f"lower bound {low:+.3f} does not clear the margin {config.margin:+.3f}"
        )
    if regressions:
        reasons.append(
            f"{len(regressions)} group(s) regress by more than "
            f"{config.tolerance:.3f}: {', '.join(regressions)}"
        )
    return GateDecision(
        promote=not reasons,
        mean_difference=float(values.mean()),
        interval=(low, high),
        differences=diffs,
        regressions=regressions,
        reasons=reasons,
    )
