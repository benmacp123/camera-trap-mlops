"""k-nearest-neighbour classifier on frozen embeddings, scored out of fold by site.

The labelled training crops are the model: a crop's class scores are the
class-weighted votes of its k most cosine-similar training crops. Votes
are weighted by inverse class frequency in the training rows, as the
probe's ``class_weight="balanced"`` is, so a common class (2,710 collared
peccary sequences against 292 white-lipped) cannot win on numbers alone.

Vote shares become logits by taking their log, after a pseudo-count on
every class: without it a class with no votes gets minus infinity, which
vetoes it for the whole sequence and saturates the confidence at 1.0
that the drift study and review queue read. k is chosen inside the
training sites by an inner grouped split, never on the scored fold.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from cameratrap.arms import scoring

# First fixed at 5-50 before any kNN arm was scored (2026-11-05); 100 and
# 200 added after 50, the grid's edge, won most inner splits. The best k
# must fall inside the grid, or it is only known to be at least the edge.
KS = (5, 10, 20, 50, 100, 200)
INNER_FOLDS = 3
# In units of neighbours: half a vote added to every class.
PSEUDO_COUNT = 0.5
# Test rows per similarity block; bounds memory at about 70 MB of float32
# against 17k training rows.
CHUNK = 1024


def class_weights(y, n_classes):
    """Return inverse-frequency weights, scaled as scikit-learn's "balanced".

    Args:
        y: Class index of each training row.
        n_classes: Number of classes.

    Returns:
        Array of shape ``(n_classes,)``: ``n / (n_classes * count)``.

    Raises:
        ValueError: If a class has no training rows.
    """
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("every class needs at least one training row")
    return len(y) / (n_classes * counts)


def neighbours(train, test, k):
    """Return the indices of each test row's k most similar training rows.

    Rows are L2-normalised, so the dot product is the cosine similarity.

    Args:
        train: Array of shape ``(n_train, d)``.
        test: Array of shape ``(n_test, d)``.
        k: Number of neighbours; at most ``n_train``.

    Returns:
        Integer array of shape ``(n_test, k)``, most similar first.
    """
    out = np.empty((len(test), k), dtype=np.intp)
    for start in range(0, len(test), CHUNK):
        sims = test[start : start + CHUNK] @ train.T
        # argpartition finds the top k without sorting all n_train; only
        # those k are then sorted.
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out[start : start + CHUNK] = np.take_along_axis(top, order, axis=1)
    return out


def vote_logits(neighbour_y, weights, k, pseudo_count=PSEUDO_COUNT):
    """Turn the first k neighbours' classes into log vote shares.

    Weighted votes are rescaled to total k, so the pseudo-count is in
    units of neighbours whatever the weights.

    Args:
        neighbour_y: Class index of each row's neighbours, most similar
            first, shape ``(n, >= k)``.
        weights: Per-class vote weights from ``class_weights``.
        k: Number of neighbours to use.
        pseudo_count: Votes added to every class before taking shares.

    Returns:
        Array of shape ``(n, n_classes)`` of log shares.
    """
    n_classes = len(weights)
    nearest = neighbour_y[:, :k]
    votes = np.zeros((len(nearest), n_classes))
    np.add.at(votes, (np.arange(len(nearest))[:, None], nearest), weights[nearest])
    votes *= k / votes.sum(axis=1, keepdims=True)
    return np.log((votes + pseudo_count) / (k + pseudo_count * n_classes))


def choose_k(features, y, groups, n_classes, inner_folds=INNER_FOLDS, ks=KS):
    """Choose k by frame-level balanced accuracy over inner site folds.

    Args:
        features: Training rows, shape ``(n, d)``.
        y: Class index of each row.
        groups: Site of each row, for the inner split.
        n_classes: Number of classes.
        inner_folds: Number of inner grouped folds.
        ks: Candidate numbers of neighbours.

    Returns:
        A pair ``(k, curve)``: the chosen k (the smallest on a tie) and a
        dict of each candidate k to its mean inner balanced accuracy.
    """
    scores = {k: [] for k in ks}
    for train, test in GroupKFold(inner_folds).split(features, y, groups):
        weights = class_weights(y[train], n_classes)
        # One neighbour search at the largest k serves every candidate.
        near = y[train][neighbours(features[train], features[test], max(ks))]
        for k in ks:
            preds = vote_logits(near, weights, k).argmax(axis=1)
            scores[k].append(scoring.balanced_accuracy(y[test], preds))
    curve = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(ks, key=lambda k: (curve[k], -k)), curve


def out_of_fold_logits(features, labels, groups, folds, classes, ks=KS):
    """Return kNN logits for every row from training rows outside its fold.

    Args:
        features: Array of shape ``(n, d)``, rows L2-normalised.
        labels: Class name of each row.
        groups: Site of each row.
        folds: Fold name of each row; folds are unions of whole sites.
        classes: Class names in logit order.
        ks: Candidate numbers of neighbours; each must be at most the
            training rows in every inner split.

    Returns:
        A tuple ``(logits, chosen_k, curves)``: an array of shape
        ``(n, len(classes))``, a mapping of fold to the k it used, and a
        mapping of fold to its inner validation curve.

    Raises:
        ValueError: If a training fold lacks one of the classes.
    """
    labels, groups, folds = np.asarray(labels), np.asarray(groups), np.asarray(folds)
    y = np.array([classes.index(c) for c in labels])
    logits = np.zeros((len(labels), len(classes)))
    chosen, curves = {}, {}
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        missing = set(classes) - set(labels[train])
        if missing:
            raise ValueError(f"training folds for {fold} lack {sorted(missing)}")
        k, curves[str(fold)] = choose_k(
            features[train], y[train], groups[train], len(classes), ks=ks
        )
        near = y[train][neighbours(features[train], features[test], k)]
        logits[test] = vote_logits(near, class_weights(y[train], len(classes)), k)
        chosen[str(fold)] = k
    return logits, chosen, curves
