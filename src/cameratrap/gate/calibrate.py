"""How often the per-group tolerance blocks a challenger that is no worse.

The swap null: if two models are equally good, which of them made a
given prediction carries no information, so swapping their predictions
for a random half of the items leaves the joint distribution unchanged.
Repeating the swap gives the spread of per-group differences that chance
alone produces, kept paired on the same items. The share of draws in
which some group regresses beyond the tolerance is the rule's
false-block rate for that pair of models.

Generic over groups and metrics, so it stays within the gate's imports.
"""

import numpy as np


def null_differences(labels, champion, challenger, metric, draws, rng):
    """Simulate per-group differences between two equally good models.

    Args:
        labels: True label of each item in one group.
        champion: The champion's prediction for each item.
        challenger: The challenger's prediction for each item.
        metric: Callable ``metric(labels, preds) -> float``, higher better.
        draws: Number of swap draws.
        rng: A ``numpy.random.Generator``.

    Returns:
        Array of ``draws`` differences (swapped challenger minus swapped
        champion).
    """
    labels = np.asarray(labels)
    champion, challenger = np.asarray(champion), np.asarray(challenger)
    out = np.empty(draws)
    for i in range(draws):
        swap = rng.random(labels.size) < 0.5
        a = np.where(swap, challenger, champion)
        b = np.where(swap, champion, challenger)
        out[i] = metric(labels, b) - metric(labels, a)
    return out


def false_block_rates(groups, metric, tolerances, draws=2000, seed=0):
    """Return each group's chance of a regression beyond each tolerance.

    Args:
        groups: Mapping of group to a tuple ``(labels, champion, challenger)``.
        metric: Callable ``metric(labels, preds) -> float``.
        tolerances: Tolerances to evaluate.
        draws: Swap draws per group.
        seed: Seed for the draws.

    Returns:
        A dict of group to an array of rates, one per tolerance.
    """
    rng = np.random.default_rng(seed)
    tolerances = np.asarray(tolerances, dtype=float)
    rates = {}
    for group in sorted(groups):
        diffs = null_differences(*groups[group], metric, draws, rng)
        rates[group] = (diffs[:, None] < -tolerances[None, :]).mean(axis=0)
    return rates


def family_rate(rates, size, subsets=5000, seed=0):
    """Return the chance that any of ``size`` random groups is blocked.

    Swaps are independent between groups, so for one set of groups the
    chance that none is blocked is the product of each group's chance.
    Averaging over random sets of ``size`` groups estimates the rate for
    a hold-out of that size drawn from the same population.

    Args:
        rates: Output of ``false_block_rates``.
        size: Groups in each set (the hold-out's size).
        subsets: Random sets to average over.
        seed: Seed for drawing the sets.

    Returns:
        Array of family-wise rates, one per tolerance.
    """
    table = np.stack([rates[g] for g in sorted(rates)])
    rng = np.random.default_rng(seed)
    total = np.zeros(table.shape[1])
    for _ in range(subsets):
        chosen = rng.choice(len(table), size=size, replace=False)
        total += 1 - np.prod(1 - table[chosen], axis=0)
    return total / subsets
