"""Sequence-level scoring shared by every arm.

Frame logits are averaged across a sequence before the softmax (Dussert
et al.), and accuracy is measured per site over sequences, never pooled
over images. Pure numpy and polars, so it runs without the model stack.
"""

import numpy as np
import polars as pl


def group_mean(keys, values):
    """Average rows of an array that share a key.

    Args:
        keys: Sequence of hashable keys, one per row.
        values: Array of shape ``(n, k)``.

    Returns:
        A pair ``(unique_keys, means)``: the sorted distinct keys as a list
        and an array of shape ``(len(unique_keys), k)``.
    """
    unique, inverse = np.unique(np.asarray(keys), return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(unique))[:, None]
    return unique.tolist(), sums / counts


def softmax(logits):
    """Return row-wise softmax probabilities.

    Args:
        logits: Array of shape ``(n, k)``.

    Returns:
        Array of the same shape whose rows sum to one.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sequence_predictions(frame_keys, frame_logits, classes):
    """Turn frame logits into one prediction per sequence.

    Args:
        frame_keys: Sequence ID of each frame.
        frame_logits: Array of shape ``(n_frames, len(classes))``.
        classes: Class names in logit order.

    Returns:
        A DataFrame with ``seq_id``, ``pred`` (class name) and ``msp`` (the
        maximum softmax probability of the averaged logits).
    """
    seq_ids, logits = group_mean(frame_keys, frame_logits)
    probs = softmax(logits)
    return pl.DataFrame(
        {
            "seq_id": seq_ids,
            "pred": [classes[i] for i in probs.argmax(axis=1)],
            "msp": probs.max(axis=1),
        }
    )


def balanced_accuracy(labels, preds):
    """Return the mean recall over the classes present in ``labels``.

    A class absent from a site does not count, so a site with two classes
    is scored on those two.

    Args:
        labels: True class of each sequence.
        preds: Predicted class of each sequence; may include classes or
            values (such as an abstention) that never appear in ``labels``.

    Returns:
        Balanced accuracy in [0, 1], or NaN when ``labels`` is empty.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    if labels.size == 0:
        return float("nan")
    recalls = [np.mean(preds[labels == c] == c) for c in np.unique(labels)]
    return float(np.mean(recalls))


def per_site(scored, min_sequences):
    """Score each site from sequence-level predictions.

    Args:
        scored: DataFrame with ``site``, ``label`` and ``pred`` per sequence.
        min_sequences: Sites with fewer sequences are left out; their
            balanced accuracy is too noisy to compare.

    Returns:
        A DataFrame with ``site``, ``n`` (sequences), ``balanced_accuracy``
        and ``accuracy``, sorted by site.
    """
    rows = []
    for (site,), group in scored.group_by("site"):
        if group.height < min_sequences:
            continue
        labels, preds = group["label"].to_numpy(), group["pred"].to_numpy()
        rows.append(
            {
                "site": site,
                "n": group.height,
                "balanced_accuracy": balanced_accuracy(labels, preds),
                "accuracy": float(np.mean(labels == preds)),
            }
        )
    return pl.DataFrame(rows).sort("site")
