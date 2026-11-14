"""Frozen complete-genotype laws and independent point/subsample audits."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.genomic_neural_steps import observe_training
from super_deli.io.config import write_json
from super_deli.models.genomic_neural import load_genomic_model

LAWS = ("common_hwe", "rare_complete_hwe", "four_groups", "cline", "admixture", "perfect_ld")


def simulate(law, n, seed):
    if law not in LAWS or n not in (64, 128) or type(seed) is not int:
        raise ValueError("Use a frozen genomic law, N=64/128 and an integer seed")
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.15, 0.5, 128)
    labels = np.zeros(n, dtype=int)
    reference = {}
    if law == "rare_complete_hwe":
        p = rng.uniform(0.01, 0.1, 128)
    probability = np.broadcast_to(p, (n, 128)).copy()
    if law == "four_groups":
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        axes = np.column_stack((2 * (labels % 2) - 1, 2 * (labels // 2) - 1))
        probability = expit(logit(p)[None] + 2.5 * axes[:, np.arange(128) // 64])
    elif law == "cline":
        position = rng.uniform(-1, 1, n)
        probability = expit(
            logit(p)[None] + 3 * position[:, None] * np.where(np.arange(128) < 64, 1, -1)
        )
        reference["position"] = position
    elif law == "admixture":
        ancestry = rng.beta(0.8, 0.8, n)
        pa = expit(logit(p) + 1.5 * np.where(np.arange(128) < 64, 1, -1))
        pb = expit(logit(p) - 1.5 * np.where(np.arange(128) < 64, 1, -1))
        probability = ancestry[:, None] * pa + (1 - ancestry[:, None]) * pb
        reference.update(ancestry=ancestry, ancestral_frequencies=np.stack((pa, pb)))
    if law == "perfect_ld":
        block = rng.binomial(2, p[:16], size=(n, 16))
        values = np.repeat(block, 8, axis=1)
        probability = np.broadcast_to(np.repeat(p[:16], 8), (n, 128)).copy()
    else:
        values = rng.binomial(2, probability)
    return values.astype(float), labels, dict(probability=probability, **reference)


def independent_preprocessing(values):
    means = values.mean(0)
    keep = np.flatnonzero(np.var(values, axis=0) > 0)
    frequency = means[keep] / 2
    scales = np.sqrt(2 * frequency * (1 - frequency))
    return (values[:, keep] - means[keep]) / scales, keep, means[keep], scales


@contextmanager
def audited_points(directory):
    """Read-only checks attached to each actual fit; no training replay."""
    from super_deli.workflows import genomic_neural as workflow

    original = workflow._point
    out = Path(directory)
    out.mkdir()
    counter = [0]

    def point(family, values, model, keys):
        index = counter[0]
        counter[0] += 1
        if family != "categorical_ae":
            return original(family, values, model, keys)
        x, keep, _, _ = independent_preprocessing(values)
        checked = None
        try:
            with observe_training(x, values[:, keep].astype(int)) as checked:
                return original(family, values, model, keys)
        finally:
            if checked is not None:
                np.savez_compressed(
                    out / f"training-{index:04d}.npz",
                    **checked.initial,
                    errors=np.asarray(checked.errors),
                    step_sha256=np.asarray(checked.hashes),
                )

    workflow._point = point
    try:
        yield
    finally:
        workflow._point = original


def audit_model(directory, values):
    model = load_genomic_model(directory)
    a = model.arrays
    transformed, keep, means, scales = independent_preprocessing(values)
    np.testing.assert_array_equal(keep, a["kept_indices"])
    np.testing.assert_array_equal(means, a["means"])
    np.testing.assert_array_equal(scales, a["scales"])
    result = model.predict(values, locus_keys=model.locus_keys)
    np.testing.assert_allclose(result["scores"], a["scores"], rtol=1e-11, atol=1e-11)
    if model.family != "categorical_ae":
        _, _, vectors = np.linalg.svd(transformed - transformed.mean(0), full_matrices=False)
        np.testing.assert_allclose(
            a["pca_components"].T @ a["pca_components"],
            vectors[:2].T @ vectors[:2],
            rtol=1e-9,
            atol=1e-10,
        )
    else:
        probability = model.reconstruct_probabilities(values, locus_keys=model.locus_keys)
        np.testing.assert_allclose(probability.sum(2), 1, rtol=1e-12, atol=1e-12)
        selected = np.take_along_axis(probability, values[:, keep, None].astype(int), axis=2)
        loss = float(-np.log(selected).mean())
        np.testing.assert_allclose(loss, a["losses"][-1], rtol=1e-10, atol=1e-12)
    logp = []
    for weight, mean, covariance in zip(
        a["mixture_weights"],
        a["mixture_means"],
        a["mixture_covariance"]
        if bool(a["diagonal"])
        else [a["mixture_covariance"]] * len(a["mixture_weights"]),
        strict=True,
    ):
        density = (
            norm.logpdf(a["scores"], mean, np.sqrt(covariance)).sum(1)
            if bool(a["diagonal"])
            else multivariate_normal.logpdf(a["scores"], mean=mean, cov=covariance)
        )
        logp.append(np.log(weight) + density)
    logp = np.column_stack(logp)
    membership = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    np.testing.assert_allclose(membership, a["membership"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(membership.argmax(1), a["labels"])
    k = len(a["mixture_weights"])
    parameter_count = (k - 1) + 2 * k + (2 * k if bool(a["diagonal"]) else 3)
    bic = float(-2 * logsumexp(logp, axis=1).sum() + parameter_count * np.log(len(values)))
    meta = json.loads((Path(directory) / "model.json").read_text())
    eligible = [r for r in meta["candidates"] if r["status"] == "eligible"]
    best = min(eligible, key=lambda r: (r["bic"], r["k"]))
    assert best["k"] == k
    np.testing.assert_allclose(bic, best["bic"], rtol=1e-10, atol=1e-9)
    return dict(
        passed=True,
        selected_k=k,
        selected_bic=bic,
        n_parameters=parameter_count,
        preprocessing=True,
        frozen_projection=True,
        selected_density=True,
        candidate_ranking=True,
    )


def audit_run(directory, truth):
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text())
    raw = np.load(out / "inputs/dosages.npy", allow_pickle=False)
    cohort = json.loads((out / "inputs/cohort.json").read_text())
    assert len(raw) == len(summary["unit_ids"]) and cohort["unit_ids"] == summary["unit_ids"]
    assert summary["inference"] == dict(status="not_computed", p_value=None)
    rows = []
    for row in summary["models"]:
        if row["status"] == "complete":
            check = audit_model(out / "arms" / row["name"], raw)
            labels = np.load(out / "arms" / row["name"] / "model.npz")["labels"]
            check.update(name=row["name"], ari=float(adjusted_rand_score(truth, labels)))
        else:
            check = dict(name=row["name"], passed=True, retained_failure=row["error"])
        rows.append(check)
    draws = []
    from super_deli.models.subsample_stability import subsample_indices

    manifest = json.loads((out / "manifest.json").read_text())
    model = manifest["model"]
    expected = subsample_indices(len(raw), model["n_subsamples"], 0.8, model["subsampling_seed"])
    for d, indices in enumerate(expected):
        folder = out / "subsamples" / f"draw-{d:04d}"
        records = json.loads((folder / "records.json").read_text())
        for name, record in records.items():
            assert record["training_indices"] == indices.tolist()
            assert record["unit_ids"] == [cohort["unit_ids"][i] for i in indices]
            if record["status"] == "complete":
                draws.append(dict(draw=d, name=name, **audit_model(folder / name, raw[indices])))
    return dict(passed=True, points=rows, subsamples=draws)


def audit_steps(directory):
    errors = []
    for path in sorted(Path(directory).glob("training-*.npz")):
        with np.load(path, allow_pickle=False) as a:
            assert a["errors"].shape == (500, 4) and a["step_sha256"].shape == (500,)
            assert np.isfinite(a["errors"]).all()
            errors.append(a["errors"].max(0))
    return dict(
        passed=True,
        trainings=len(errors),
        optimizer_updates=500 * len(errors),
        maximum_errors=np.max(errors, axis=0).tolist() if errors else [],
    )


def write_audit(directory, truth):
    result = audit_run(directory, truth)
    write_json(result, Path(directory) / "independent-audit.json")
    return result
