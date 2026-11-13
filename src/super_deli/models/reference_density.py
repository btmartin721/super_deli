"""Known-only class-specific Gaussian level sets and independent rank calibration."""

import numpy as np

from super_deli.models.reference_abstention import PREPROCESSING, labels, rank_thresholds, transform

DEFAULT_BYTES = 64 * 1024**2


def fit_density_support(x, y, preprocessing, classes):
    if type(classes) is not int or not 2 <= classes <= 32:
        raise ValueError("Between two and thirty-two reference classes required")
    support = {k: np.array(preprocessing[k], copy=True) for k in PREPROCESSING}
    z = transform(x, support)
    y = labels(y, len(z), classes)
    counts = np.bincount(y, minlength=classes)
    if np.any(counts < 2):
        raise ValueError("At least two original training specimens per reference required")
    centers, covariances, raw, factors = [], [], [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for c in range(classes):
            rows = z[y == c]
            center = rows[0] + (rows - rows[0]).mean(0)
            delta = rows - center
            covariance = np.einsum("ni,nj->ij", delta, delta) / len(rows)
            scale = np.trace(covariance) / z.shape[1]
            if not np.isfinite(covariance).all() or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Unresolved within-class covariance; no variance-floor repair")
            shrunk = 0.9 * covariance + 0.1 * scale * np.eye(z.shape[1])
            factor = np.linalg.cholesky(shrunk)
            if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
                raise ValueError("Unresolved class-specific Cholesky factor")
            centers.append(center)
            raw.append(covariance)
            covariances.append(shrunk)
            factors.append(factor)
    factors = np.stack(factors)
    support.update(
        centers=np.stack(centers),
        unshrunk_covariances=np.stack(raw),
        covariances=np.stack(covariances),
        cholesky=factors,
        log_determinants=2 * np.log(np.diagonal(factors, axis1=1, axis2=2)).sum(1),
        training_counts=counts,
    )
    return support


def density_scores(x, support, *, max_working_bytes=DEFAULT_BYTES):
    x = np.asarray(x, dtype=float)
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("Positive integer density working-memory budget required")
    if x.ndim != 2 or not len(x):
        raise ValueError("Nonempty complete density feature matrix required")
    c, d = np.asarray(support["centers"]).shape
    fixed = x.nbytes + len(x) * c * 8 + sum(np.asarray(v).nbytes for v in support.values())
    chunk = min(512, (max_working_bytes - fixed) // max(1, 64 * (x.shape[1] + d)))
    if chunk < 1:
        raise ValueError("Density working-memory budget exceeded")
    output = np.empty((len(x), c))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, len(x), chunk):
            end = min(start + chunk, len(x))
            z = transform(x[start:end], support)
            for j in range(c):
                whitened = np.linalg.solve(support["cholesky"][j], (z - support["centers"][j]).T).T
                output[start:end, j] = np.einsum("ij,ij->i", whitened, whitened)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite class-specific scores")
    return output


def calibrate_density_support(support, x, y, *, max_working_bytes=DEFAULT_BYTES):
    result = {k: np.array(v, copy=True) for k, v in support.items()}
    score = density_scores(x, support, max_working_bytes=max_working_bytes)
    q, counts, ranks = rank_thresholds(score, y)
    result.update(
        calibration_features=np.asarray(x, dtype=float).copy(),
        calibration_labels=np.asarray(y, dtype=int).copy(),
        calibration_scores=score,
        thresholds=q,
        calibration_counts=counts,
        calibration_ranks=ranks,
    )
    return result
