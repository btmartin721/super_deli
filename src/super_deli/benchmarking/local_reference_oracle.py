"""True-latent likelihood-ratio diagnostic; never a deployable rule or formal bound."""

import math

import numpy as np
from scipy.special import logsumexp

from super_deli.benchmarking.abstention_metrics import metrics, unscreened

LEVELS = np.arange(1, 20) / 20


def parameters(task):
    if task["case"] not in ("linear_weak", "linear_standard", "combined", "confounded_size"):
        raise ValueError("Unsupported true-latent reference law")
    a = 1.5 if task["case"] in ("linear_weak", "confounded_size") else 3.0
    means = np.array(
        [[-a, 0, 0, -a, 0], [-a, 0, 0, a, 0], [a, 0, 0, -a, 0], [a, 0, 0, a, 0]], dtype=float
    )
    unknown = np.array([[a, 0, 0, 0, 0], [6, 0, 0, 6, 0]], dtype=float)
    if task["case"] == "confounded_size":
        means[:2, 4] = -0.8
        means[2:, 4] = 0.8
        unknown[:, 4] = 0.8
    return means, unknown, np.array([1.0, 1.0, 1.0, 1.0, 0.25])


def coordinates(data):
    return np.column_stack((data["latent_morphology"], data["latent_ecology"], data["size"]))


def likelihoods(x, means, variance):
    delta = x[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.sum(delta**2 / variance, axis=2)
        + len(variance) * np.log(2 * np.pi)
        + np.log(variance).sum()
    )


def score(data, task):
    means, unknown, variance = parameters(task)
    x = coordinates(data)
    known = likelihoods(x, means, variance)
    other = likelihoods(x, unknown, variance)
    probability = np.exp(known - logsumexp(known, axis=1)[:, None])
    ratio = logsumexp(other, axis=1) - np.log(2) - logsumexp(known, axis=1) + np.log(4)
    if not np.isfinite(ratio).all() or not np.isfinite(probability).all():
        raise ValueError("Nonfinite true-latent diagnostic")
    return ratio, probability


def evaluate(cohorts, task):
    calibration, _ = score(cohorts["calibration"], task)
    ranks = np.array([math.ceil((len(calibration) + 1) * (1 - float(v))) for v in LEVELS])
    if np.any(ranks > len(calibration)):
        raise ValueError("Insufficient known calibration for diagnostic quantiles")
    thresholds = np.sort(calibration)[ranks - 1]
    retained = dict(
        calibration_score=calibration,
        thresholds=thresholds,
        nominal_withholding=LEVELS.copy(),
        ranks=ranks,
    )
    rows = [
        dict(nominal_withholding=float(v), threshold=float(q), cohorts={})
        for v, q in zip(LEVELS, thresholds, strict=True)
    ]
    for name, data in cohorts.items():
        if name in ("train", "calibration"):
            continue
        s, p = score(data, task)
        retained[name + "_score"], retained[name + "_probabilities"] = s, p
        for row, q in zip(rows, thresholds, strict=True):
            result = unscreened(p)
            result["accepted"] = s <= q
            result["decisions"] = np.where(
                result["accepted"], "assigned", "withheld_outside_reference_support"
            )
            row["cohorts"][name] = metrics(data["truth"], result, known=name in ("iid", "shifted"))
    return retained, dict(
        scope="True-latent known/unseen-law diagnostic; no deployment or formal bound",
        thresholds=rows,
    )
