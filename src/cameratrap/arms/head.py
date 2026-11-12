"""The probe's linear head as plain arrays, applied without scikit-learn.

A fitted multinomial logistic probe is a weight matrix and an intercept,
so serving it needs only numpy: ``logits = vectors @ coef.T + intercept``,
which is what scikit-learn's ``decision_function`` computes. The hold-out
run saves the promoted head in this form (``results/holdout/probe_head.npz``).
"""

from pathlib import Path

import numpy as np

CHAMPION_HEAD = Path("results/holdout/probe_head.npz")


def load(path=CHAMPION_HEAD):
    """Read a saved head.

    Args:
        path: The ``.npz`` holding ``coef``, ``intercept`` and ``classes``.

    Returns:
        Dict with ``coef`` of shape ``(k, d)``, ``intercept`` of shape
        ``(k,)`` and ``classes``, a list of ``k`` names in logit order.
    """
    with np.load(path, allow_pickle=False) as data:
        return {
            "coef": data["coef"],
            "intercept": data["intercept"],
            "classes": np.asarray(data["classes"]).tolist(),
        }


def logits(head, vectors):
    """Return the head's logits for some embeddings.

    Args:
        head: A head from ``load``.
        vectors: Array of shape ``(n, d)``, L2-normalised as the probe's
            training embeddings were.

    Returns:
        Array of shape ``(n, k)``, columns in ``head["classes"]`` order.

    Raises:
        ValueError: If the embedding width does not match the head.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[1] != head["coef"].shape[1]:
        raise ValueError(
            f"embeddings have {vectors.shape[1]} dimensions, "
            f"the head expects {head['coef'].shape[1]}"
        )
    return vectors @ head["coef"].T + head["intercept"]
