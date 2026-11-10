"""Label-free signals: numbers a deployment can compute without any labels.

Each one is a candidate predictor of accuracy at a new site, built either
from the model's own probabilities (confidence, entropy, ATC, the mix of
predicted classes) or from how far the site's embeddings sit from the
training crops (kNN distance, MMD). None is assumed to work: the drift
report scores each against held-back labels.

Pure numpy, so the study runs from cached embeddings without the model
stack.
"""

import numpy as np

# Rows scored per block when comparing against the reference set, which
# bounds memory at CHUNK x reference size.
CHUNK = 1024


def entropy(probs):
    """Return the Shannon entropy of each row of probabilities.

    Args:
        probs: Array of shape ``(n, k)`` whose rows sum to one.

    Returns:
        Array of ``n`` entropies in nats; 0 for a one-hot row.
    """
    # 0 * log(0) is taken as 0, its limit; the clip only avoids the warning.
    return -np.sum(probs * np.log(np.clip(probs, 1e-300, None)), axis=1)


def atc_threshold(scores, correct):
    """Learn the ATC threshold from labelled sequences.

    Average thresholded confidence (Garg et al., 2022) picks the score
    ``t`` at which the share of labelled sequences scoring above ``t``
    equals their accuracy. The share of unlabelled sequences above the
    same ``t`` is then the estimate of accuracy there.

    Args:
        scores: Confidence of each labelled sequence (max softmax here).
        correct: Whether each of those sequences was classified correctly.

    Returns:
        The threshold: midway between the two scores either side of the
        cut, 0 when every sequence is correct and 1 when none is.

    Raises:
        ValueError: If there are no sequences.
    """
    scores = np.sort(np.asarray(scores, dtype=float))[::-1]
    if scores.size == 0:
        raise ValueError("no sequences to learn a threshold from")
    k = int(np.sum(correct))
    if k == 0:
        return 1.0
    if k == scores.size:
        return 0.0
    return float((scores[k - 1] + scores[k]) / 2)


def atc_estimate(scores, threshold):
    """Estimate accuracy as the share of sequences scoring above a threshold.

    Args:
        scores: Confidence of each unlabelled sequence.
        threshold: From ``atc_threshold``.

    Returns:
        The estimated accuracy in [0, 1].
    """
    return float(np.mean(np.asarray(scores) > threshold))


def normalise(vectors):
    """Scale each row to unit length.

    Args:
        vectors: Array of shape ``(n, d)``.

    Returns:
        Array of the same shape with unit-norm rows.
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def knn_distance(reference, query, k):
    """Return each query row's mean cosine distance to its k nearest references.

    Args:
        reference: Unit-norm array of shape ``(m, d)``; the training set.
        query: Unit-norm array of shape ``(n, d)``.
        k: Number of neighbours to average over.

    Returns:
        Array of ``n`` distances; 0 means the neighbours are identical.

    Raises:
        ValueError: If the reference holds fewer than ``k`` rows.
    """
    if len(reference) < k:
        raise ValueError(f"{len(reference)} reference rows, fewer than k = {k}")
    out = np.empty(len(query))
    for start in range(0, len(query), CHUNK):
        similarity = query[start : start + CHUNK] @ reference.T
        # The k largest similarities, unordered: the mean needs no sort.
        nearest = np.partition(similarity, -k, axis=1)[:, -k:]
        out[start : start + CHUNK] = 1.0 - nearest.mean(axis=1)
    return out


def squared_distances(a, b):
    """Return the squared Euclidean distance between every pair of rows.

    Args:
        a: Array of shape ``(n, d)``.
        b: Array of shape ``(m, d)``.

    Returns:
        Array of shape ``(n, m)``, clipped at 0 against rounding.
    """
    cross = a @ b.T
    return np.clip(
        np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * cross,
        0.0,
        None,
    )


def median_bandwidth(reference):
    """Return the median pairwise distance in a sample: the usual RBF bandwidth.

    Args:
        reference: Array of shape ``(m, d)``, ``m`` at least 2.

    Returns:
        The median Euclidean distance over distinct pairs.
    """
    distances = np.sqrt(squared_distances(reference, reference))
    return float(np.median(distances[np.triu_indices(len(reference), k=1)]))


def off_diagonal_mean(kernel):
    """Return the mean of a square matrix leaving out its diagonal.

    Args:
        kernel: Array of shape ``(n, n)``, ``n`` at least 2.

    Returns:
        The mean over the ``n * (n - 1)`` off-diagonal entries.
    """
    n = len(kernel)
    return float((kernel.sum() - np.trace(kernel)) / (n * (n - 1)))


def mmd2(sample, reference, bandwidth):
    """Return the unbiased squared maximum mean discrepancy of two samples.

    MMD (Gretton et al., 2012) compares two sets of vectors through a
    kernel: it is near 0 when both come from the same distribution and
    grows as they differ. The unbiased estimate leaves each point's
    similarity to itself out, so its expected value does not depend on
    sample size (and it can come out slightly negative).

    Args:
        sample: Array of shape ``(n, d)``, ``n`` at least 2; a site's rows.
        reference: Array of shape ``(m, d)``, ``m`` at least 2.
        bandwidth: RBF kernel width, from ``median_bandwidth``.

    Returns:
        The estimate of MMD squared.

    Raises:
        ValueError: If either sample has fewer than two rows.
    """
    if len(sample) < 2 or len(reference) < 2:
        raise ValueError("MMD needs at least two rows on each side")
    gamma = 1.0 / (2.0 * bandwidth**2)
    within_sample = np.exp(-gamma * squared_distances(sample, sample))
    within_reference = np.exp(-gamma * squared_distances(reference, reference))
    between = np.exp(-gamma * squared_distances(sample, reference))
    return (
        off_diagonal_mean(within_sample)
        + off_diagonal_mean(within_reference)
        - 2.0 * float(between.mean())
    )


def class_shares(indices, n_classes):
    """Return the share of rows in each class.

    Args:
        indices: Class index of each row.
        n_classes: Number of classes.

    Returns:
        Array of ``n_classes`` shares summing to one.
    """
    counts = np.bincount(np.asarray(indices), minlength=n_classes)
    return counts / counts.sum()


def total_variation(p, q):
    """Return the total variation distance between two class distributions.

    Args:
        p: Shares summing to one.
        q: Shares summing to one, same classes.

    Returns:
        A distance in [0, 1]: the share of mass that would have to move
        to turn one distribution into the other.
    """
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())
