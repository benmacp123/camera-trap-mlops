"""Tests for the review queues: each ranking's order, ties and determinism."""

import numpy as np
import pytest

from cameratrap.review import queue


def unit(rows):
    """Return rows scaled to unit length.

    Args:
        rows: A list of vectors.

    Returns:
        Array with unit-norm rows.
    """
    rows = np.array(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_uncertainty_puts_least_confident_first_and_breaks_ties_by_id():
    """Lowest confidence leads; equal confidences fall back to sequence ID."""
    order = queue.uncertainty_order(["b", "a", "c"], [0.5, 0.5, 0.2])
    assert order.tolist() == [2, 1, 0]


def test_random_order_ignores_input_order():
    """The same sequences shuffled on the way in give the same queue."""
    ids = [f"s{i}" for i in range(20)]
    first = [ids[i] for i in queue.random_order(ids, "A", seed=3)]
    flipped = ids[::-1]
    second = [flipped[i] for i in queue.random_order(flipped, "A", seed=3)]
    assert first == second
    assert sorted(first) == sorted(ids)


def test_random_order_changes_with_seed_and_site():
    """Another seed or another site gives another shuffle."""
    ids = [f"s{i}" for i in range(20)]
    base = queue.random_order(ids, "A", seed=0).tolist()
    assert base != queue.random_order(ids, "A", seed=1).tolist()
    assert base != queue.random_order(ids, "B", seed=0).tolist()


def test_diverse_takes_turns_rarest_predicted_class_first():
    """A class predicted once is reviewed before a common class's second pick."""
    ids = ["a", "b", "c", "d"]
    preds = ["paca", "paca", "paca", "tapir"]
    msp = [0.1, 0.2, 0.3, 0.99]
    vectors = unit([[1, 0], [0, 1], [1, 1], [1, 2]])
    order = queue.diverse_order(ids, msp, preds, vectors)
    assert [ids[i] for i in order][:2] == ["d", "a"]
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_diverse_skips_a_near_copy_of_what_is_already_picked():
    """Of two uncertain sequences, the one unlike the first pick goes next."""
    ids = ["a", "b", "c"]
    vectors = unit([[1, 0], [1, 0.01], [0, 1]])
    order = queue.diverse_order(ids, [0.1, 0.2, 0.3], ["paca"] * 3, vectors)
    assert [ids[i] for i in order] == ["a", "c", "b"]


def test_diverse_window_limits_how_far_down_confidence_it_reaches():
    """With a window of one the queue is plain uncertainty order."""
    ids = ["a", "b", "c"]
    vectors = unit([[1, 0], [1, 0.01], [0, 1]])
    order = queue.diverse_order(ids, [0.1, 0.2, 0.3], ["paca"] * 3, vectors, window=1)
    assert [ids[i] for i in order] == ["a", "b", "c"]


def test_order_dispatches_and_refuses_unknown_strategy():
    """Each named strategy returns a full ranking; a typo raises."""
    ids = ["a", "b", "c"]
    args = (ids, [0.3, 0.1, 0.2], ["x", "y", "x"], unit([[1, 0], [0, 1], [1, 1]]), "A")
    for strategy in queue.STRATEGIES:
        assert sorted(queue.order(strategy, *args).tolist()) == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown strategy"):
        queue.order("clever", *args)
