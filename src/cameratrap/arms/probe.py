"""Linear probe on frozen embeddings, scored out of fold by site.

Each outer fold is a set of whole sites; the probe for it is trained on
every other fold's frames, so no site is ever scored by a model that saw
it. The regularisation strength is chosen inside the training sites by
an inner grouped split, never on the scored fold.
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

# Fixed before the spike (provisional judgement, 2026-10-28).
CS = np.logspace(-2, 2, 9)
INNER_FOLDS = 3


def fit(features, labels, groups, classes, inner_folds=INNER_FOLDS, cs=CS):
    """Fit a class-balanced logistic probe, choosing C over inner site folds.

    Args:
        features: Array of shape ``(n, d)``.
        labels: Class name of each row.
        groups: Site of each row, for the inner split.
        classes: Class names; the returned model's logits follow this order.
        inner_folds: Number of inner grouped folds.
        cs: Candidate inverse regularisation strengths.

    Returns:
        A fitted ``LogisticRegressionCV``; ``C_`` is the chosen C.
    """
    y = np.array([classes.index(c) for c in labels])
    splits = list(GroupKFold(inner_folds).split(features, y, groups))
    # Refit on all the training rows at the chosen C; L2 only, and the
    # attribute layout that becomes the default in scikit-learn 1.10.
    return LogisticRegressionCV(
        Cs=cs,
        cv=splits,
        l1_ratios=(0.0,),
        class_weight="balanced",
        max_iter=5000,
        scoring="balanced_accuracy",
        use_legacy_attributes=False,
    ).fit(features, y)


def out_of_fold_logits(features, labels, groups, folds, classes):
    """Return probe logits for every row from a model that never saw its fold.

    Args:
        features: Array of shape ``(n, d)``.
        labels: Class name of each row.
        groups: Site of each row.
        folds: Fold name of each row; folds are unions of whole sites.
        classes: Class names in logit order.

    Returns:
        A pair ``(logits, chosen_c)``: an array of shape
        ``(n, len(classes))`` and a mapping of fold to the C it used.

    Raises:
        ValueError: If a training fold lacks one of the classes, which
            would leave a logit column undefined.
    """
    labels, groups, folds = np.asarray(labels), np.asarray(groups), np.asarray(folds)
    logits = np.zeros((len(labels), len(classes)))
    chosen = {}
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        missing = set(classes) - set(labels[train])
        if missing:
            raise ValueError(f"training folds for {fold} lack {sorted(missing)}")
        model = fit(features[train], labels[train], groups[train], classes)
        logits[test] = model.decision_function(features[test])
        chosen[str(fold)] = float(model.C_)
    return logits, chosen
