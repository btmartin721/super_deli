"""Fixed fifth-neighbor support in the unchanged training-only Mahalanobis metric."""

import numpy as np

from super_deli.models.reference_abstention import labels, rank_thresholds, transform

NEIGHBORS = 5
DEFAULT_BYTES = 64 * 1024**2


def whiten(x, metric):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = np.linalg.solve(metric["cholesky"], transform(x, metric).T).T
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite class-local whitening")
    return result


def fit_local_support(x, y, metric):
    """Only original training specimens become neighbors; callers bind their IDs."""
    z = whiten(x, metric)
    y = labels(y, len(z), len(metric["centers"]))
    counts = np.bincount(y, minlength=len(metric["centers"]))
    if np.any(counts < NEIGHBORS):
        raise ValueError("At least five original training specimens per reference required")
    np.testing.assert_array_equal(counts, metric["training_counts"])
    return dict(whitened_training=z.copy(), training_labels=y.copy())


def score_whitened(query, support, classes, *, max_working_bytes=DEFAULT_BYTES, provenance=False):
    """Direct squared differences avoid cancellation and query x library x trait scratch.

    Only one query chunk and one class's distance matrix are materialized. Stable
    sorting breaks equal-distance ties by original training-row order. No score
    depends on an evaluation identity, class label, or other query specimens.
    """
    z, library = np.asarray(query, dtype=float), np.asarray(support["whitened_training"])
    y = np.asarray(support["training_labels"])
    if (
        type(classes) is not int
        or not 2 <= classes <= 32
        or z.ndim != 2
        or not len(z)
        or library.ndim != 2
        or library.shape[1] != z.shape[1]
        or not np.isfinite(z).all()
        or not np.isfinite(library).all()
    ):
        raise ValueError("Finite aligned class-local coordinates required")
    y = labels(y, len(library), classes)
    counts = np.bincount(y, minlength=classes)
    if np.any(counts < NEIGHBORS):
        raise ValueError("At least five original training specimens per reference required")
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("Positive integer class-local working-memory budget required")
    output_bytes = len(z) * classes * (8 + (8 * NEIGHBORS + 8 if provenance else 0))
    fixed = 2 * z.nbytes + library.nbytes + y.nbytes + output_bytes
    row_bytes = 64 * (z.shape[1] + int(counts.max()) + NEIGHBORS)
    chunk = min(512, (max_working_bytes - fixed) // row_bytes)
    if chunk < 1:
        raise ValueError("Class-local working-memory budget exceeded")
    scores = np.empty((len(z), classes))
    neighbors = np.empty((len(z), classes, NEIGHBORS), dtype=np.int64) if provenance else None
    ties = np.empty((len(z), classes), dtype=np.int64) if provenance else None
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(z), chunk):
            end = min(start + chunk, len(z))
            block = z[start:end]
            for c in range(classes):
                indices = np.flatnonzero(y == c)
                d = np.empty((len(block), len(indices)))
                for j, i in enumerate(indices):
                    delta = block - library[i]
                    d[:, j] = np.einsum("ij,ij->i", delta, delta)
                order = np.argsort(d, axis=1, kind="stable")[:, :NEIGHBORS]
                q = np.take_along_axis(d, order[:, -1:], axis=1)[:, 0]
                if not np.isfinite(q).all():
                    raise ValueError("Nonfinite class-local distances")
                scores[start:end, c] = q
                if provenance:
                    neighbors[start:end, c] = indices[order]
                    ties[start:end, c] = np.sum(d == q[:, None], axis=1)
    return (scores, neighbors, ties) if provenance else scores


def local_distances(x, metric, support, *, max_working_bytes=DEFAULT_BYTES, provenance=False):
    return score_whitened(
        whiten(x, metric),
        support,
        len(metric["centers"]),
        max_working_bytes=max_working_bytes,
        provenance=provenance,
    )


def calibrate_local_support(support, metric, x, y, *, max_working_bytes=DEFAULT_BYTES):
    result = {k: np.array(v, copy=True) for k, v in support.items()}
    z = whiten(x, metric)
    scores = score_whitened(z, result, len(metric["centers"]), max_working_bytes=max_working_bytes)
    q, counts, ranks = rank_thresholds(scores, y)
    result.update(
        whitened_calibration=z.copy(),
        calibration_labels=np.asarray(y, dtype=int).copy(),
        calibration_scores=scores,
        thresholds=q,
        calibration_counts=counts,
        calibration_ranks=ranks,
    )
    return result
