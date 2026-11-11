"""Review queues: which sequences a reviewer should label first.

Each strategy turns one site's unlabelled sequences into a single ranking,
and a budget of ``b`` labels takes the first ``b``. Because the ranking
does not depend on the budget, a larger budget always contains a smaller
one, so a learning curve compares like with like.

Nothing here sees a label. The inputs are what a deployment has before any
review: the champion's prediction and confidence for each sequence, and
the sequence's embedding.

Three strategies:

- ``random``: a seeded shuffle; the baseline any cleverer queue must beat.
- ``uncertainty``: least confident first.
- ``diverse``: predicted classes take turns, so a rarely predicted species
  is not crowded out, and within a turn the queue prefers an uncertain
  sequence unlike anything already chosen, so a burst of near-identical
  frames does not use up the budget.

Pure numpy.
"""

import zlib

import numpy as np

RANDOM = "random"
UNCERTAINTY = "uncertainty"
DIVERSE = "diverse"
STRATEGIES = (RANDOM, UNCERTAINTY, DIVERSE)
# How many of a class's least confident sequences compete at its turn.
# Fixed in the plan before any queue was scored.
WINDOW = 5


def random_order(seq_ids, site, seed):
    """Return a seeded shuffle of one site's sequences.

    The shuffle depends only on the seed, the site's name and the set of
    sequence IDs, never on the order they arrive in.

    Args:
        seq_ids: The site's sequence IDs.
        site: The site's name, mixed into the seed so sites differ.
        seed: The repetition's seed.

    Returns:
        Array of positions into ``seq_ids``, first to review first.
    """
    by_id = np.argsort(np.asarray(seq_ids), kind="stable")
    rng = np.random.default_rng([seed, zlib.crc32(site.encode())])
    return by_id[rng.permutation(len(by_id))]


def uncertainty_order(seq_ids, msp):
    """Return one site's sequences, least confident first.

    Args:
        seq_ids: The site's sequence IDs, which break ties.
        msp: The champion's confidence (max softmax) for each sequence.

    Returns:
        Array of positions into ``seq_ids``, first to review first.
    """
    return np.lexsort((np.asarray(seq_ids), np.asarray(msp, dtype=float)))


def diverse_order(seq_ids, msp, preds, embeddings, window=WINDOW):
    """Return one site's sequences, spread over classes and embedding space.

    Predicted classes take turns, the one predicted least often at the site
    first (ties by name). At its turn a class offers its ``window`` least
    confident sequences not yet picked, and the one farthest, in cosine
    distance, from everything already picked is taken. The first pick has
    nothing to be far from, so it is that class's least confident. A class
    with nothing left is skipped.

    Args:
        seq_ids: The site's sequence IDs, which break ties.
        msp: The champion's confidence for each sequence.
        preds: The champion's predicted class for each sequence.
        embeddings: Unit-norm array of shape ``(n, d)``, one row per
            sequence.
        window: Sequences a class offers at each turn.

    Returns:
        Array of positions into ``seq_ids``, first to review first.
    """
    preds = np.asarray(preds)
    by_confidence = uncertainty_order(seq_ids, msp)
    names, counts = np.unique(preds, return_counts=True)
    turn_order = [names[i] for i in np.lexsort((names, counts))]
    waiting = {c: [i for i in by_confidence if preds[i] == c] for c in turn_order}

    # Each sequence's cosine distance to the nearest one already picked.
    nearest = np.full(len(preds), np.inf)
    picked = []
    while len(picked) < len(preds):
        for name in turn_order:
            offered = waiting[name][:window]
            if not offered:
                continue
            # argmax takes the first of equals, which is the less confident.
            choice = offered[int(np.argmax(nearest[offered]))]
            waiting[name].remove(choice)
            picked.append(choice)
            nearest = np.minimum(nearest, 1.0 - embeddings @ embeddings[choice])
    return np.array(picked, dtype=int)


def order(strategy, seq_ids, msp, preds, embeddings, site, seed=0):
    """Rank one site's sequences by the named strategy.

    Args:
        strategy: One of ``STRATEGIES``.
        seq_ids: The site's sequence IDs.
        msp: The champion's confidence for each sequence.
        preds: The champion's predicted class for each sequence.
        embeddings: Unit-norm sequence embeddings.
        site: The site's name (used by ``random``).
        seed: The repetition's seed (used by ``random``).

    Returns:
        Array of positions into ``seq_ids``, first to review first.

    Raises:
        ValueError: If the strategy is unknown.
    """
    if strategy == RANDOM:
        return random_order(seq_ids, site, seed)
    if strategy == UNCERTAINTY:
        return uncertainty_order(seq_ids, msp)
    if strategy == DIVERSE:
        return diverse_order(seq_ids, msp, preds, embeddings)
    raise ValueError(f"unknown strategy {strategy!r}")
