"""Tests for site-level splits: the hold-out draw honours its constraints
and is fixed by its seed, and folds cover every remaining site once.
"""

import pytest

from cameratrap.data import splits

# Twelve sites; "rare" lives at only three of them and "tiny" sites are small.
SITE_COUNTS = {
    **{f"S{i:02d}": {"common": 40 + i} for i in range(8)},
    "R1": {"common": 30, "rare": 10},
    "R2": {"common": 30, "rare": 8},
    "R3": {"common": 30, "rare": 1},
    "T1": {"common": 2, "rare": 20},
}

STRICT = {
    "min_site_sequences": 30,
    "min_sites_per_class": 2,
    "min_sequences_per_class": 15,
}


def test_holdout_meets_constraints():
    """Every class reaches its site and sequence minimums; small sites are never held out."""
    held = splits.choose_holdout(SITE_COUNTS, 4, seed=1, **STRICT)
    assert "T1" not in held
    assert sum(1 for s in held if "rare" in SITE_COUNTS[s]) >= 2
    assert sum(SITE_COUNTS[s].get("rare", 0) for s in held) >= 15


def test_holdout_fixed_by_seed():
    """The same seed gives the same hold-out; the freeze depends on it."""
    first = splits.choose_holdout(SITE_COUNTS, 4, seed=7, **STRICT)
    assert splits.choose_holdout(SITE_COUNTS, 4, seed=7, **STRICT) == first


def test_holdout_raises_when_impossible():
    """An unsatisfiable constraint fails loudly instead of returning a weak draw."""
    impossible = {**STRICT, "min_sequences_per_class": 100}
    with pytest.raises(ValueError):
        splits.choose_holdout(SITE_COUNTS, 4, seed=1, max_draws=50, **impossible)


def test_folds_balance_sequence_totals():
    """Greedy assignment keeps fold totals within one site's size of each other."""
    counts = {f"S{i}": {"a": n} for i, n in enumerate([50, 40, 30, 30, 20, 10])}
    folds = splits.assign_folds(counts, 3)
    totals = [sum(counts[s]["a"] for s, f in folds.items() if f == k) for k in range(3)]
    assert sorted(totals) == [60, 60, 60]


def test_folds_need_enough_sites():
    """More folds than sites would leave a fold empty, so it is refused."""
    with pytest.raises(ValueError):
        splits.assign_folds({"A": {"a": 1}}, 2)


def test_site_splits_partition_sites():
    """Each site lands in exactly one split, hold-out or one of the folds."""
    result = splits.site_splits(SITE_COUNTS, n_holdout=4, n_folds=3, seed=1, **STRICT)
    assert set(result) == set(SITE_COUNTS)
    assert sum(v == splits.HOLDOUT for v in result.values()) == 4
    assert {v for v in result.values() if v != splits.HOLDOUT} == {
        "fold0",
        "fold1",
        "fold2",
    }
