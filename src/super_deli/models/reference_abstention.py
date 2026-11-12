"""Frozen raw-feature support, classwise held-out ranks and explicit abstention."""

import math

import numpy as np

from super_deli.models.reference_abstention_config import DECISIONS, POLICIES

PREPROCESSING = ("means", "scales", "kept_indices", "column_weights")


def probabilities(value, *, n=None, classes=None):
    p = np.asarray(value, dtype=float)
    if (
        p.ndim != 2
        or not len(p)
        or (n is not None and len(p) != n)
        or (classes is not None and p.shape[1] != classes)
        or p.shape[1] < 2
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(1), 1, rtol=1e-10, atol=1e-10)
    ):
        raise ValueError("Finite normalized reference probabilities required")
    return p


def labels(value, n, classes):
    y = np.asarray(value)
    if y.shape != (n,) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Known reference indices required for every calibration/training specimen")
    return y


def transform(x, support):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(support["means"]) or not np.isfinite(x).all():
        raise ValueError("Complete finite fitted reference features required")
    kept = support["kept_indices"]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        z = (x[:, kept] - support["means"][kept]) / support["scales"][kept]
        z *= support["column_weights"]
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite reference-support transform")
    return z


def fit_support(x, y, preprocessing, classes):
    """Estimate raw support only from the original classifier training specimens."""
    support = {name: np.array(preprocessing[name], copy=True) for name in PREPROCESSING}
    z = transform(x, support)
    y = labels(y, len(z), classes)
    counts = np.bincount(y, minlength=classes)
    if np.any(counts < 2):
        raise ValueError("At least two original training specimens per reference required")
    # Anchoring avoids inventing within-class scatter when repeated rows are equal.
    centers = np.stack([z[y == c][0] + (z[y == c] - z[y == c][0]).mean(0) for c in range(classes)])
    residual = z - centers[y]
    covariance = residual.T @ residual / len(z)
    scale = np.trace(covariance) / z.shape[1]
    if not np.isfinite(covariance).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Unresolved within-reference covariance; no variance-floor repair")
    shrunk = 0.9 * covariance + 0.1 * scale * np.eye(z.shape[1])
    factor = np.linalg.cholesky(shrunk)
    if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
        raise ValueError("Unresolved reference-support Cholesky factor")
    support.update(
        centers=centers,
        covariance=shrunk,
        unshrunk_covariance=covariance,
        cholesky=factor,
        training_counts=counts,
    )
    return support


def distances(x, support):
    z = transform(x, support)
    # Keep scratch proportional to rows * traits, rather than rows * classes * traits.
    result = np.empty((len(z), len(support["centers"])))
    for c, center in enumerate(support["centers"]):
        delta = z - center
        whitened = np.linalg.solve(support["cholesky"], delta.T).T
        result[:, c] = np.einsum("ij,ij->i", whitened, whitened)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reference distances")
    return result


def rank_thresholds(scores, y):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not 2 <= scores.shape[1] <= 32 or not np.isfinite(scores).all():
        raise ValueError("Finite classwise calibration scores required")
    y = labels(y, len(scores), scores.shape[1])
    thresholds, counts, ranks = [], [], []
    for c in range(scores.shape[1]):
        values = np.sort(scores[y == c, c])
        m = len(values)
        if m < 20:
            raise ValueError("At least twenty independent calibration specimens per class required")
        k = math.ceil((m + 1) * 0.95)
        thresholds.append(values[k - 1])
        counts.append(m)
        ranks.append(k)
    return np.asarray(thresholds), np.asarray(counts), np.asarray(ranks)


def calibrate_distance(support, x, y):
    d = distances(x, support)
    result = {name: np.array(value, copy=True) for name, value in support.items()}
    threshold, counts, ranks = rank_thresholds(d, y)
    result.update(
        distance_thresholds=threshold,
        calibration_counts=counts,
        calibration_ranks=ranks,
        calibration_distances=d,
        calibration_labels=np.asarray(y, dtype=int),
    )
    return result


def calibrate_probability(distance_calibration, probability):
    result = {name: np.array(value, copy=True) for name, value in distance_calibration.items()}
    p = probabilities(
        probability, n=len(result["calibration_labels"]), classes=len(result["centers"])
    )
    threshold, counts, ranks = rank_thresholds(1 - p, result["calibration_labels"])
    np.testing.assert_array_equal(counts, result["calibration_counts"])
    np.testing.assert_array_equal(ranks, result["calibration_ranks"])
    result.update(probability_thresholds=threshold, calibration_probabilities=p.copy())
    return result


def calibrate(support, x, y, probability):
    return calibrate_probability(calibrate_distance(support, x, y), probability)


def decide(probability, distance, thresholds, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown reference-support policy")
    p = probabilities(probability)
    score = np.asarray(distance, dtype=float) if policy == "distance" else 1 - p
    q = np.asarray(thresholds, dtype=float)
    if (
        score.shape != p.shape
        or q.shape != (p.shape[1],)
        or not np.isfinite(score).all()
        or not np.isfinite(q).all()
        or np.any(score < 0)
        or np.any(q < 0)
    ):
        raise ValueError("Finite nonnegative support scores and fitted class thresholds required")
    compatible = score <= q
    count = compatible.sum(1)
    predicted = p.argmax(1)
    consistent = compatible[np.arange(len(p)), predicted]
    status = np.full(len(p), DECISIONS[3], dtype="U48")
    status[count == 0] = DECISIONS[1]
    status[count > 1] = DECISIONS[2]
    accepted = (count == 1) & consistent
    status[accepted] = DECISIONS[0]
    return dict(
        probabilities=p,
        predicted_indices=predicted,
        compatible=compatible,
        decisions=status,
        accepted=accepted,
        accepted_indices=np.where(accepted, predicted, -1),
    )
