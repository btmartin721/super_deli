"""Independent raw-genotype, binomial-likelihood, PCA and selected-GMM audits."""

import hashlib
import io
import json
from pathlib import Path

from super_deli.benchmarking.empirical_analysis import audit_fitted_study
from super_deli.benchmarking.genomic_study import case_policy, simulate_genomic, summarize_genomic
from super_deli.io.config import read_json
from super_deli.models.genomic_config import CASES, GenomicStudyConfig
from super_deli.workflows.genomic_calibration import NAMESPACE, TAIL_OPTIONS, TARGET


def _equivalent(expected, actual):
    import numpy as np

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    observed = ~np.isnan(expected)
    same = np.all(~observed | (expected == actual), axis=0)
    reversed_alleles = np.all(~observed | (2 - expected == actual), axis=0)
    if not np.all(same | reversed_alleles):
        raise ValueError("Encoded genotypes differ beyond a whole-locus allele reversal")


def verify_model(base, output, values, layout, policy):
    import numpy as np
    from scipy.stats import binom

    fitted = read_json(output / "layout.json")
    for key in ("sample_ids", "unit_ids", "locus_indices", "modality"):
        if fitted[key] != layout[key]:
            raise ValueError("Genomic source identity layout mismatch")
    with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
        x = arrays["values"].copy()
        _equivalent(values, x)
        mask = np.isnan(x)
        np.testing.assert_array_equal(arrays["missing_mask"], mask)
        counts = np.sum(~mask, axis=0)
        frequencies = np.nansum(x, axis=0) / (2 * counts)
        np.testing.assert_array_equal(arrays["called_counts"], counts)
        np.testing.assert_array_equal(arrays["frequencies"], frequencies)
    observed = np.where(mask, 0, x)
    nll = -np.where(mask, 0, binom.logpmf(observed, 2, frequencies)).sum()
    if not np.isclose(nll, fitted["fit"]["negative_log_likelihood"], rtol=1e-12, atol=1e-9):
        raise ValueError("Fitted binomial likelihood disagrees with independent PMF evaluation")
    if (
        fitted["fit"]["boundary_loci"]
        != np.flatnonzero((frequencies == 0) | (frequencies == 1)).tolist()
    ):
        raise ValueError("Boundary-frequency diagnostics disagree")
    if fitted["fit"]["missing_calls"] != int(mask.sum()):
        raise ValueError("Missing-call diagnostics disagree")

    def draw(seed):
        generated = np.random.default_rng(seed).binomial(2, frequencies, size=x.shape).astype(float)
        generated[mask] = np.nan
        return generated

    return fitted, draw, *x.shape


def point(archive, record, policy, values, layout):
    import numpy as np
    from scipy.special import logsumexp

    prefix = record["archive_prefix"] + "/run/"
    if record["status"] != "complete":
        failure = json.loads(archive.read(prefix + "failure.json"))
        if failure["error"] != record["error"]:
            raise ValueError("Failed genomic pipeline diagnostics disagree")
        return
    run = json.loads(archive.read(prefix + "summary.json"))

    def load(name):
        return np.load(io.BytesIO(archive.read(prefix + name)), allow_pickle=False)

    for name, key in (
        ("dosages.npy", "dosages_sha256"),
        ("mixture.npz", "mixture_sha256"),
        ("preprocessing.npz", "preprocessing_sha256"),
    ):
        if hashlib.sha256(archive.read(prefix + name)).hexdigest() != record[key]:
            raise ValueError("Genomic numerical artifact hash mismatch")
    if run["sample_ids"] != layout["sample_ids"] or run["specimen_ids"] != layout["unit_ids"]:
        raise ValueError("Pipeline source/specimen order mismatch")
    if run["provenance"]["configuration"]["model"] != policy.model.to_dict():
        raise ValueError("Pipeline analysis settings disagree")
    x = load("dosages.npy")
    _equivalent(values, x)
    counts = np.sum(~np.isnan(x), axis=0)
    means = np.divide(np.nansum(x, axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
    centered = np.where(np.isnan(x), 0, x - means)
    keep = (counts > 0) & (np.sum(centered**2, axis=0) > 0)
    indices = np.flatnonzero(keep)
    scales = np.sqrt(2 * (means[keep] / 2) * (1 - means[keep] / 2))
    transformed = centered[:, keep] / scales
    with load("preprocessing.npz") as pre:
        np.testing.assert_array_equal(pre["retained_locus_indices"], indices)
        np.testing.assert_allclose(pre["means"], means[keep], atol=1e-14)
        np.testing.assert_allclose(pre["scales"], scales, atol=1e-14)
        np.testing.assert_allclose(pre["pca_mean"], transformed.mean(axis=0), atol=1e-14)
        components = pre["pca_components"]
        centered = transformed - pre["pca_mean"]
        scores = centered @ components.T
        np.testing.assert_allclose(load("pca_scores.npy"), scores, atol=1e-10)
        np.testing.assert_allclose(
            components @ components.T, np.eye(policy.model.n_pcs), atol=1e-10
        )
        # Direct eigensystem check, without refitting PCA or any clustering model.
        eigenvalues = np.linalg.eigvalsh(centered @ centered.T)[::-1][: policy.model.n_pcs]
        np.testing.assert_allclose(np.sum(scores**2, axis=0), eigenvalues, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            (scores.T @ centered), eigenvalues[:, None] * components, rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            pre["explained_variance_ratio"], eigenvalues / np.sum(centered**2), atol=1e-10
        )
        np.testing.assert_array_equal(pre["sample_missing_fraction"], np.isnan(x).mean(axis=1))
    if run["preprocessing"]["imputed_call_count"] != int(np.isnan(x[:, keep]).sum()):
        raise ValueError("Pipeline imputation count mismatch")
    with load("mixture.npz") as mix:
        weights, centers, variances = mix["weights"], mix["means"], mix["variances"]
    if not np.all(weights > 0) or not np.isclose(weights.sum(), 1) or not np.all(variances > 0):
        raise ValueError("Invalid selected mixture parameters")
    log_components = np.log(weights)[None, :] - 0.5 * np.sum(
        np.log(2 * np.pi * variances)[None, :, :]
        + (scores[:, None, :] - centers[None, :, :]) ** 2 / variances[None, :, :],
        axis=2,
    )
    likelihoods = logsumexp(log_components, axis=1)
    memberships = np.exp(log_components - likelihoods[:, None])
    np.testing.assert_allclose(load("membership.npy"), memberships, atol=1e-8)
    np.testing.assert_array_equal(load("labels.npy"), memberships.argmax(axis=1))
    k, n, d = len(weights), len(x), scores.shape[1]
    candidates = run["candidates"]
    eligible = [c for c in candidates if c["status"] == "eligible"]
    selected = min(eligible, key=lambda c: (c["bic"], c["k"]))
    bic = -2 * likelihoods.sum() + (2 * k * d + k - 1) * np.log(n)
    single = next(c for c in eligible if c["k"] == 1)
    expected_seed = int(
        np.random.SeedSequence(policy.model.random_seed).spawn(1)[0].generate_state(1)[0]
    )
    if not np.isclose(bic, selected["bic"], rtol=1e-9, atol=1e-7) or selected["k"] != k:
        raise ValueError("Selected diagonal-mixture likelihood/BIC mismatch")
    if (
        json.loads(json.dumps(record["candidates"])) != candidates
        or record["selected_k"] != k
        or run["selected_k"] != k
        or record["n_retained"] != n
        or record["point_seed"] != expected_seed
    ):
        raise ValueError("Selected model, candidate ledger or seed mismatch")
    if not np.isclose(record["statistic"], max(0, single["bic"] - selected["bic"]), atol=1e-9):
        raise ValueError("Global BIC-improvement statistic mismatch")


def audit_genomic_study(root, *, output_dir):
    from super_deli.models.genomic_null import write_raw_genotypes
    from super_deli.workflows.genomic_calibration import _dependencies

    _dependencies()
    backend = {
        "cases": CASES,
        "study_type": GenomicStudyConfig,
        "policy_for": case_policy,
        "summarize": summarize_genomic,
        "simulator": simulate_genomic,
        "point_auditor": point,
        "tail_options": TAIL_OPTIONS,
        "namespace": NAMESPACE,
        "decision_key": "reject_pooled_hwe",
        "target": TARGET,
        "writer": write_raw_genotypes,
        "verify_model": verify_model,
        "audit_scope": (
            "Regenerated all raw genotype inputs/draws with fixed masks and identities; "
            "independently checked pooled frequency MLEs/binomial likelihood, all archive hashes, "
            "Patterson filtering/scaling/imputation, PCA eigensystem/projection, selected "
            "diagonal-mixture likelihood/BIC/membership, candidate selection, seeds, tails and "
            "failure denominators. No SNPio or clustering refits in audit; nonselected mixture "
            "parameters are not exported and their likelihoods are not independently re-evaluated. "
            "Outer failures retain counts without complete numerical re-audit."
        ),
    }
    return audit_fitted_study(Path(root), output_dir=output_dir, backend=backend)
