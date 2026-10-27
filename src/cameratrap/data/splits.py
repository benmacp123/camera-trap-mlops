"""Site-level splits: a frozen hold-out and grouped folds over the rest.

Every split is by site, never by image or sequence, so a model is always
scored on sites it never saw. Both steps use only label counts per site,
never model scores, so choosing them cannot leak a result.
"""

import random

HOLDOUT = "holdout"


def choose_holdout(
    site_counts,
    n_sites,
    min_site_sequences,
    min_sites_per_class,
    min_sequences_per_class,
    seed,
    max_draws=10_000,
):
    """Draw the hold-out sites at random, rejecting draws that starve a class.

    A purely random draw of nine or ten sites can pick a site with a
    handful of sequences, which gives the gate no per-site signal, or
    leave a rare class (lowland tapir has 228 sequences in all) with
    almost no hold-out sequences, so a regression on it would go unseen.
    Draws are repeated from one seeded generator until the constraints
    hold, so the result is fixed by the seed.

    Args:
        site_counts: Mapping of site to a mapping of class to sequence count.
        n_sites: Number of hold-out sites.
        min_site_sequences: Sites with fewer sequences than this are never
            held out; they stay in the folds.
        min_sites_per_class: Minimum number of hold-out sites at which each
            class must have at least one sequence.
        min_sequences_per_class: Minimum hold-out sequences for each class.
        seed: Seed for the random generator.
        max_draws: Number of draws to try before giving up.

    Returns:
        The hold-out sites, sorted.

    Raises:
        ValueError: If no draw within ``max_draws`` satisfies the constraints.
    """
    eligible = sorted(
        s
        for s, counts in site_counts.items()
        if sum(counts.values()) >= min_site_sequences
    )
    classes = sorted({c for counts in site_counts.values() for c in counts})
    rng = random.Random(seed)
    for _ in range(max_draws):
        draw = rng.sample(eligible, n_sites)
        if all(
            sum(1 for s in draw if site_counts[s].get(c, 0) > 0) >= min_sites_per_class
            and sum(site_counts[s].get(c, 0) for s in draw) >= min_sequences_per_class
            for c in classes
        ):
            return sorted(draw)
    raise ValueError(
        f"no draw of {n_sites} sites meets the hold-out constraints "
        f"after {max_draws} draws"
    )


def assign_folds(site_counts, n_folds):
    """Assign sites to grouped folds with roughly equal sequence totals.

    Largest sites first, each to the fold with the fewest sequences so far
    (ties to the lowest fold number), which keeps fold sizes close without
    any randomness.

    Args:
        site_counts: Mapping of site to a mapping of class to sequence count.
        n_folds: Number of folds.

    Returns:
        Mapping of site to fold number, from 0 to ``n_folds - 1``.

    Raises:
        ValueError: If there are fewer sites than folds.
    """
    if len(site_counts) < n_folds:
        raise ValueError(f"{len(site_counts)} sites cannot fill {n_folds} folds")
    totals = [0] * n_folds
    folds = {}
    by_size = sorted(site_counts, key=lambda s: (-sum(site_counts[s].values()), s))
    for site in by_size:
        fold = min(range(n_folds), key=lambda f: (totals[f], f))
        folds[site] = fold
        totals[fold] += sum(site_counts[site].values())
    return folds


def site_splits(site_counts, n_holdout, n_folds, seed, **constraints):
    """Split sites into the hold-out and grouped folds.

    Args:
        site_counts: Mapping of site to a mapping of class to sequence count.
        n_holdout: Number of hold-out sites.
        n_folds: Number of folds over the remaining sites.
        seed: Seed for the hold-out draw.
        **constraints: ``min_site_sequences``, ``min_sites_per_class`` and
            ``min_sequences_per_class``; see ``choose_holdout``.

    Returns:
        Mapping of site to its split name: ``"holdout"`` or ``"fold<k>"``.
    """
    holdout = set(choose_holdout(site_counts, n_holdout, seed=seed, **constraints))
    rest = {s: c for s, c in site_counts.items() if s not in holdout}
    splits = {s: HOLDOUT for s in holdout}
    splits.update({s: f"fold{k}" for s, k in assign_folds(rest, n_folds).items()})
    return splits
