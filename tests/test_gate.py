"""Tests for the promotion gate on synthetic groups: clear win, clear loss,
one-group regression judged against its own noise, all-equal, and the
freeze that pins its settings.
"""

import numpy as np
import pytest

from cameratrap.arms import scoring
from cameratrap.gate import calibrate, freeze, rule

GROUPS = [f"g{i}" for i in range(10)]
CHAMPION = dict(zip(GROUPS, np.linspace(0.70, 0.90, len(GROUPS))))
# No group's drop is surprising: p-values for the rule's other conditions.
UNSURPRISING = {g: 1.0 for g in GROUPS}


def shifted(by):
    """Return the champion's scores moved by a per-group amount."""
    by = np.broadcast_to(by, len(GROUPS))
    return {g: CHAMPION[g] + d for g, d in zip(GROUPS, by)}


def test_clear_win_promotes():
    """A challenger better at every group by varying amounts is promoted."""
    decision = rule.decide(CHAMPION, shifted(np.linspace(0.03, 0.07, 10)), UNSURPRISING)
    assert decision.promote
    assert decision.reasons == []
    assert decision.interval[0] > 0


def test_clear_loss_rejects_on_margin():
    """A challenger worse everywhere fails the margin, even with no surprising drop."""
    decision = rule.decide(
        CHAMPION, shifted(np.linspace(-0.04, -0.02, 10)), UNSURPRISING
    )
    assert not decision.promote
    assert any("margin" in r for r in decision.reasons)
    assert decision.regressions == []


def test_one_group_regression_blocks_a_mean_gain():
    """A big mean gain cannot hide one group dropping beyond chance."""
    by = np.linspace(0.08, 0.14, 10)
    by[3] = -0.06
    decision = rule.decide(CHAMPION, shifted(by), UNSURPRISING | {"g3": 0.001})
    assert decision.interval[0] > rule.GateConfig().margin
    assert not decision.promote
    assert decision.regressions == ["g3"]
    assert len(decision.reasons) == 1


def test_same_drop_is_judged_against_each_groups_noise():
    """A drop chance explains (a noisy group) passes; the same drop at a
    well-measured group blocks.
    """
    by = np.full(10, 0.05)
    by[3] = -0.06
    noisy = rule.decide(CHAMPION, shifted(by), UNSURPRISING | {"g3": 0.2})
    precise = rule.decide(CHAMPION, shifted(by), UNSURPRISING | {"g3": 0.0001})
    assert noisy.promote
    assert precise.regressions == ["g3"]


def test_cutoff_tightens_with_more_groups():
    """Bonferroni: the per-group cutoff is the family alpha over the groups."""
    decision = rule.decide(CHAMPION, shifted(0.02), UNSURPRISING)
    assert decision.cutoff == pytest.approx(0.05 / 10)
    # A p-value between the 5-group and 10-group cutoffs blocks only with 5.
    five = {g: CHAMPION[g] for g in GROUPS[:5]}
    worse = {g: v - 0.02 for g, v in five.items()}
    p_values = {g: 0.007 for g in five}
    assert rule.decide(five, worse, p_values).regressions == list(five)
    assert (
        rule.decide(CHAMPION, shifted(-0.02), {g: 0.007 for g in GROUPS}).regressions
        == []
    )


def test_evaluate_blocks_a_real_drop_and_passes_an_equal_model():
    """From raw predictions: a large well-measured drop blocks; a copy never does."""
    rng = np.random.default_rng(4)
    config = rule.GateConfig(null_draws=2000, resamples=999)
    groups, copies = {}, {}
    for g in GROUPS:
        labels = rng.choice(list("abc"), 300)
        champ = np.where(rng.random(300) < 0.8, labels, "x")
        chall = np.where(rng.random(300) < 0.95, labels, "x")
        if g == "g3":
            chall = np.where(rng.random(300) < 0.5, labels, "x")
        groups[g] = (labels, champ, chall)
        copies[g] = (labels, champ, champ)
    blocked = rule.evaluate(groups, scoring.balanced_accuracy, config)
    assert blocked.regressions == ["g3"]
    assert not blocked.promote
    equal = rule.evaluate(copies, scoring.balanced_accuracy, config)
    assert equal.regressions == []
    assert set(equal.p_values.values()) == {1.0}


def test_all_equal_keeps_the_champion():
    """A tie clears the -1 point margin but not the gain rule, so nothing changes."""
    decision = rule.decide(CHAMPION, dict(CHAMPION), UNSURPRISING)
    assert decision.interval == (0.0, 0.0)
    assert decision.differences == {g: 0.0 for g in GROUPS}
    assert not decision.promote
    assert len(decision.reasons) == 1
    assert "mean gain" in decision.reasons[0]


def test_small_consistent_gain_promotes():
    """A gain too small to beat the champion confidently still passes non-inferiority."""
    decision = rule.decide(CHAMPION, shifted(np.linspace(0.0, 0.01, 10)), UNSURPRISING)
    assert decision.interval[0] < 0.01
    assert decision.promote


def test_gain_rule_can_be_relaxed():
    """With the gain rule off, a tie is promoted on non-inferiority alone."""
    config = rule.GateConfig(min_mean_gain=-1.0)
    assert rule.decide(CHAMPION, dict(CHAMPION), UNSURPRISING, config).promote


def test_decision_is_reproducible():
    """The same inputs and seed give the same interval."""
    challenger = shifted(np.linspace(-0.02, 0.05, 10))
    assert (
        rule.decide(CHAMPION, challenger, UNSURPRISING).interval
        == rule.decide(CHAMPION, challenger, UNSURPRISING).interval
    )


def test_unmatched_groups_raise():
    """Models scored on different groups cannot be paired."""
    challenger = shifted(0.01)
    del challenger["g0"]
    with pytest.raises(ValueError, match="different groups"):
        rule.decide(CHAMPION, challenger, UNSURPRISING)


def test_missing_metric_raises():
    """A NaN metric is an error, not a silent drop of the group."""
    challenger = shifted(0.01)
    challenger["g2"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        rule.decide(CHAMPION, challenger, UNSURPRISING)


def test_too_few_groups_raise():
    """The rule refuses to decide on fewer groups than configured."""
    few = {g: CHAMPION[g] for g in GROUPS[:3]}
    with pytest.raises(ValueError, match="fewer than"):
        rule.decide(few, few, UNSURPRISING)


def test_bca_interval_contains_the_mean():
    """The interval brackets the observed mean difference."""
    decision = rule.decide(
        CHAMPION, shifted(np.linspace(-0.03, 0.06, 10)), UNSURPRISING
    )
    low, high = decision.interval
    assert low < decision.mean_difference < high


def test_freeze_round_trip(tmp_path):
    """A frozen config reads back with the same settings and hash."""
    config = {"gate": freeze.gate_dict(rule.GateConfig()), "holdout": ["a", "b"]}
    digest = freeze.write(tmp_path / "f.json", config)
    loaded, gate_config = freeze.load(tmp_path / "f.json", expected_sha256=digest)
    assert loaded == config
    assert gate_config == rule.GateConfig()


def test_freeze_hash_ignores_key_order():
    """The hash depends on content, not on the order keys were added."""
    assert freeze.config_hash({"a": 1, "b": 2}) == freeze.config_hash({"b": 2, "a": 1})


def test_freeze_detects_an_edit(tmp_path):
    """Changing a frozen value without re-freezing is caught."""
    path = tmp_path / "f.json"
    freeze.write(path, {"gate": freeze.gate_dict(rule.GateConfig())})
    path.write_text(
        path.read_text().replace('"family_alpha": 0.05', '"family_alpha": 0.2')
    )
    with pytest.raises(ValueError, match="edited"):
        freeze.load(path)


def test_freeze_detects_a_refreeze(tmp_path):
    """Re-freezing with new values changes the hash the tests pin."""
    path = tmp_path / "f.json"
    digest = freeze.write(path, {"gate": freeze.gate_dict(rule.GateConfig())})
    freeze.write(path, {"gate": freeze.gate_dict(rule.GateConfig(family_alpha=0.2))})
    with pytest.raises(ValueError, match="not the config"):
        freeze.load(path, expected_sha256=digest)


def test_swap_null_of_identical_models_is_zero():
    """Two models with the same predictions differ by nothing under any swap."""
    labels = np.array(list("aabbc"))
    diffs = calibrate.null_differences(
        labels, labels, labels, scoring.balanced_accuracy, 50, np.random.default_rng(0)
    )
    assert np.all(diffs == 0)


def test_swap_null_is_centred_and_noisier_on_small_groups():
    """The null is symmetric about zero, and wider with fewer items."""
    rng = np.random.default_rng(1)

    def spread(n):
        """Return the null's mean and spread for a group of n items."""
        labels = rng.choice(list("abc"), n)
        champ = np.where(rng.random(n) < 0.8, labels, "x")
        chall = np.where(rng.random(n) < 0.8, labels, "x")
        diffs = calibrate.null_differences(
            labels, champ, chall, scoring.balanced_accuracy, 2000, rng
        )
        return diffs.mean(), diffs.std()

    small_mean, small_sd = spread(40)
    _, large_sd = spread(400)
    assert abs(small_mean) < 0.01
    assert small_sd > 2 * large_sd


def test_false_block_rate_falls_with_tolerance():
    """A looser fixed tolerance never blocks more often, and the family rate exceeds
    any single group's.
    """
    rng = np.random.default_rng(2)
    groups = {}
    for g in range(6):
        labels = rng.choice(list("abc"), 50)
        groups[g] = (
            labels,
            np.where(rng.random(50) < 0.8, labels, "x"),
            np.where(rng.random(50) < 0.8, labels, "x"),
        )
    rates = calibrate.false_block_rates(
        groups, scoring.balanced_accuracy, [0.02, 0.05, 0.10], draws=500
    )
    for r in rates.values():
        assert np.all(np.diff(r) <= 0)
    family = calibrate.family_rate(rates, size=4, subsets=200)
    assert np.all(family >= np.mean([r for r in rates.values()], axis=0))


def test_p_value_counts_the_observed_draw():
    """A drop beyond every null draw gets 1/(draws+1), never zero."""
    null = np.zeros(99)
    assert calibrate.p_value(null, -0.1) == pytest.approx(1 / 100)
    assert calibrate.p_value(null, 0.0) == 1.0
