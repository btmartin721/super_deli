"""Independent joint likelihood, raw-draw and integrative numerical audits."""

import io
import json
from pathlib import Path

from super_deli.benchmarking.empirical_analysis import audit_fitted_study
from super_deli.benchmarking.genomic_analysis import _equivalent
from super_deli.benchmarking.joint_study import case_policy, simulate_joint, summarize_joint
from super_deli.benchmarking.linked_analysis import verify_profiles
from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
from super_deli.io.config import read_json
from super_deli.models.joint_config import CASES, JointStudyConfig
from super_deli.workflows.joint_calibration import NAMESPACE, TAIL_OPTIONS, TARGET


def verify_model(base, output, values, layout, policy):
    import numpy as np
    from scipy.stats import multivariate_normal

    fitted = read_json(output / "layout.json")
    if (
        fitted["unit_ids"] != layout["unit_ids"]
        or fitted["views"] != layout["views"]
        or fitted["block_ids"] != list(policy.block_ids)
        or fitted["predictor_blocks"] != list(policy.predictor_blocks)
    ):
        raise ValueError("Joint identities or prespecified predictors differ")
    for key in ("sample_ids", "unit_ids", "modality", "locus_indices"):
        if fitted["genomic"][key] != layout["genomic"][key]:
            raise ValueError("Genomic source identity mismatch")
    with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
        raw = arrays["values"].copy()
        width = values["genotypes"].shape[1]
        g, y = raw[:, :width], raw[:, width:]
        _equivalent(values["genotypes"], g)
        np.testing.assert_array_equal(values["phenotypes"], y)
        mapping = verify_profiles(g, arrays, fitted["genomic"], policy.block_ids)
        x = np.column_stack(
            [
                np.ones(len(g)),
                *[
                    np.sum(g[:, mapping[b]], axis=1) / len(mapping[b])
                    for b in policy.predictor_blocks
                ],
            ]
        )
        np.testing.assert_array_equal(arrays["design"], x)
        # QR-based independent estimator; production uses SVD least squares.
        q, r = np.linalg.qr(x, mode="reduced")
        beta = np.linalg.solve(r, q.T @ y)
        np.testing.assert_allclose(arrays["coefficients"], beta, rtol=1e-9, atol=1e-10)
        residuals = y - x @ beta
        covariance = sum(np.outer(row, row) for row in residuals) / len(y)
        np.testing.assert_allclose(arrays["residual_covariance"], covariance, rtol=1e-9, atol=1e-10)
        nll = -sum(
            multivariate_normal.logpdf(y[i], mean=x[i] @ beta, cov=covariance)
            for i in range(len(y))
        )
        if not np.isclose(
            nll, fitted["fit"]["gaussian_negative_log_likelihood"], atol=1e-8, rtol=1e-10
        ):
            raise ValueError("Independent conditional Gaussian likelihood mismatch")
        if (
            fitted["fit"]["design_rank"] != x.shape[1]
            or fitted["fit"]["residual_df"] != len(x) - x.shape[1]
            or fitted["fit"]["residual_covariance_divisor"] != len(x)
            or not np.isclose(
                nll + fitted["genomic"]["fit"]["negative_log_likelihood"],
                fitted["fit"]["negative_log_likelihood"],
                atol=1e-8,
            )
        ):
            raise ValueError("Joint likelihood or fit dimensions disagree")
        # Use the checked archived parameters for exact byte regeneration.
        beta = arrays["coefficients"].copy()
        covariance = arrays["residual_covariance"].copy()
    scales = np.sqrt(np.diag(covariance))
    factor = scales[:, None] * np.linalg.cholesky(covariance / scales[:, None] / scales[None, :])

    def draw(seed):
        gs, ys = np.random.SeedSequence(seed).spawn(2)
        donors = np.random.default_rng(gs).integers(len(g), size=(len(g), len(mapping)))
        generated = np.empty_like(g)
        for b, indices in enumerate(mapping.values()):
            generated[:, indices] = g[donors[:, b]][:, indices]
        design = np.column_stack(
            [
                np.ones(len(g)),
                *[
                    np.sum(generated[:, mapping[b]], axis=1) / len(mapping[b])
                    for b in policy.predictor_blocks
                ],
            ]
        )
        z = np.random.default_rng(ys).standard_normal(y.shape)
        return {
            "genotypes": generated,
            "phenotypes": design @ beta + z @ factor.T,
            "donor_indices": donors,
        }

    return fitted, draw, *raw.shape


def point(archive, record, policy, values, layout):
    import numpy as np

    audit_pipeline_point(archive, record, policy)
    prefix = record["archive_prefix"]
    if "donor_indices" in values:
        with io.BytesIO(archive.read(prefix + "/raw/genomic/donor_indices.npy")) as handle:
            np.testing.assert_array_equal(
                np.load(handle, allow_pickle=False), values["donor_indices"]
            )
    if record["status"] != "complete":
        return
    prefix += "/run/"
    run = json.loads(archive.read(prefix + "summary.json"))
    if run["unit_ids"] != layout["unit_ids"] or len(run["views"]) != len(policy.model.views):
        raise ValueError("Joint pipeline unit/view identities disagree")
    offset = 0
    for i, _view in enumerate(policy.model.views):
        rows = json.loads(archive.read(prefix + f"view_{i:03d}/source_rows.json"))
        with io.BytesIO(archive.read(prefix + f"view_{i:03d}/source_values.npy")) as handle:
            source = np.load(handle, allow_pickle=False)
        if i == 0:
            if rows["unit_ids"] != layout["genomic"]["unit_ids"]:
                raise ValueError("Genomic pipeline row identity mismatch")
            _equivalent(values["genotypes"], source)
        else:
            spec = layout["views"][i - 1]
            order = {u: j for j, u in enumerate(layout["unit_ids"])}
            if (
                rows["unit_ids"] != spec["unit_ids"]
                or rows["feature_names"] != spec["feature_names"]
            ):
                raise ValueError("Phenotypic pipeline schema mismatch")
            width = len(spec["feature_names"])
            np.testing.assert_array_equal(
                source,
                values["phenotypes"][[order[u] for u in spec["unit_ids"]], offset : offset + width],
            )
            offset += width


def audit_joint_study(root, *, output_dir):
    from super_deli.models.joint_null import write_raw_joint

    JointStudyConfig.from_dict(read_json(Path(root) / CASES[0] / "study.json"))
    return audit_fitted_study(
        root,
        output_dir=output_dir,
        backend={
            "cases": CASES,
            "study_type": JointStudyConfig,
            "policy_for": case_policy,
            "summarize": summarize_joint,
            "simulator": simulate_joint,
            "point_auditor": point,
            "tail_options": TAIL_OPTIONS,
            "namespace": NAMESPACE,
            "decision_key": "reject_joint_genomic_phenotypic",
            "target": TARGET,
            "writer": write_raw_joint,
            "verify_model": verify_model,
            "audit_scope": (
                "Regenerated raw observations, every joint draw and block donors; "
                "independent categorical counts/likelihood and QR Gaussian-regression "
                "MLE/dense likelihood checks; archive hashes, raw identities, "
                "fusion/projection, selected full-GMM likelihood and K "
                "search/tails/denominators. No pipeline refits. View PCA and nonselected "
                "likelihoods preserved/checksummed. Outer failures retain counts without "
                "complete numerical audit."
            ),
        },
    )
