"""One benchmark-only native forest, serialized numerically without pickle."""

import numpy as np

from super_deli.benchmarking.observed_feasibility_config import THRESHOLDS, TREE_PARAMETERS


def native_matrix(x):
    with np.errstate(over="raise", invalid="raise"):
        z = np.asarray(x, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 12 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Finite twelve-feature float32 matrix required")
    return z


def fit_detector(x, y):
    from sklearn.ensemble import GradientBoostingClassifier

    x = native_matrix(x)
    y = np.asarray(y)
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or set(y.tolist()) != {0, 1}:
        raise ValueError("Binary diagnostic training labels required")
    counts = np.bincount(y, minlength=2)
    if counts[0] != counts[1] or min(counts) < 10:
        raise ValueError("Balanced known/unfamiliar diagnostic specimens required")
    model = GradientBoostingClassifier(**TREE_PARAMETERS).fit(x, y)
    if model.n_estimators_ != 100 or model.estimators_.shape != (100, 1):
        raise ValueError("Fixed hundred-tree fit required")
    offsets = [0]
    columns = {
        k: []
        for k in (
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "n_node_samples",
            "weighted_n_node_samples",
            "impurity",
        )
    }
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        offsets.append(offsets[-1] + tree.node_count)
        for key in columns:
            value = getattr(tree, key)
            columns[key].append(value[:, 0, 0] if key == "value" else value)
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    arrays.update(
        offsets=np.asarray(offsets, dtype=np.int64),
        classes=model.classes_.copy(),
        initial_log_odds=np.array(
            [np.log(model.init_.class_prior_[1] / model.init_.class_prior_[0])]
        ),
        training_counts=counts,
        learning_rate=np.array([0.1]),
        native_train_score=model.train_score_.copy(),
    )
    return model, arrays


def native_stages(model, x):
    from scipy.special import expit

    z = native_matrix(x)
    stages = np.stack([np.asarray(v).reshape(-1) for v in model.staged_decision_function(z)])
    score = model.predict_proba(z)[:, 1]
    if (
        stages.shape != (100, len(z))
        or not np.isfinite(stages).all()
        or not np.isfinite(score).all()
    ):
        raise ValueError("Finite complete native tree stages required")
    np.testing.assert_allclose(score, expit(stages[-1]), rtol=1e-12, atol=1e-12)
    return stages, score


def threshold_counts(score, correct=None):
    score = np.asarray(score, dtype=float)
    if (
        score.ndim != 1
        or not len(score)
        or not np.isfinite(score).all()
        or np.any((score < 0) | (score > 1))
    ):
        raise ValueError("Finite diagnostic scores between zero and one required")
    if correct is not None:
        correct = np.asarray(correct)
        if correct.shape != score.shape or correct.dtype.kind != "b":
            raise ValueError("Aligned Boolean correctness required")
    accepted = []
    right = []
    for t in THRESHOLDS:
        chosen = score >= t
        accepted.append(int(chosen.sum()))
        if correct is not None:
            right.append(int(np.count_nonzero(chosen & correct)))
    result = dict(n=len(score), accepted=accepted)
    if correct is not None:
        result["correct"] = right
    return result


def choose_threshold(known, near, far):
    """Exact integer target comparisons; no external evaluation inputs are accepted."""
    qualified = []
    for i in range(len(THRESHOLDS)):
        k = known["accepted"][i]
        c = known["correct"][i]
        if (
            5 * k >= 4 * known["n"]
            and k > 0
            and 10 * c >= 9 * k
            and 10 * near["accepted"][i] <= near["n"]
            and 10 * far["accepted"][i] <= far["n"]
        ):
            qualified.append(i)
    if not qualified:
        return dict(status="available_no_operating_point", index=None, threshold=None)
    i = max(qualified, key=lambda j: (known["accepted"][j], j))
    return dict(status="selected", index=i, threshold=THRESHOLDS[i])


def at_threshold(counts, index):
    k = counts["accepted"][index]
    v = dict(n=counts["n"], accepted=k, accepted_fraction=k / counts["n"])
    if "correct" in counts:
        c = counts["correct"][index]
        v.update(
            correct_accepted_fraction=c / counts["n"],
            incorrect_accepted_fraction=(k - c) / counts["n"],
            accepted_accuracy=c / k if k else None,
        )
    return v
