"""Independent dense-likelihood and identity checks for grouped raw nulls."""

from __future__ import annotations

import io
import json


def verify_model(base, output, values, layout, policy):
    """Check the fitted ML score using dense individual covariance, without refitting."""
    import numpy as np

    from super_deli.io.config import read_json

    fitted = read_json(output / "layout.json")
    for key in ("unit_ids", "views", "target_level", "counts", "specimens"):
        if fitted[key] != json.loads(json.dumps(layout[key])):
            raise ValueError("Grouped fitted layout/counts differ from the observed design")
    order = {u: i for i, u in enumerate(layout["unit_ids"])}
    mview = next(v for v in layout["views"] if v["kind"] == "morphological")
    eview = next(v for v in layout["views"] if v["kind"] == "ecological")
    targets = np.array([order[u] for u in mview["target_ids"]])
    ecolookup = {u: i for i, u in enumerate(eview["target_ids"])}
    ecology = values["ecological"][[ecolookup[u] for u in layout["unit_ids"]]]
    morphology = values["morphological"]
    g, q = ecology.shape
    p = morphology.shape[1]
    with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
        for key, expected in (
            ("morphology", morphology),
            ("ecology", ecology),
            ("targets", targets),
            ("counts", np.bincount(targets)),
        ):
            np.testing.assert_array_equal(arrays[key], expected)
        z = np.column_stack(([morphology[targets == i].mean(axis=0) for i in range(g)], ecology))
        np.testing.assert_array_equal(arrays["values"], z)
        mean, between, within = (arrays[k].copy() for k in ("mean", "covariance", "within"))
        factor = (
            arrays["sampling_factor"].copy()
            if policy.schema_version == 2
            else np.linalg.cholesky(between)
        )
        residual_factor = (
            arrays["within_sampling_factor"].copy()
            if policy.schema_version == 2
            else np.linalg.cholesky(within)
        )
    diagnostics = fitted["diagnostics"]
    scales = np.asarray(diagnostics["standardization_scale"])
    gb, gr, gm = np.zeros_like(between), np.zeros_like(within), np.zeros_like(mean)
    likelihood = 0.0
    for i in range(g):
        rows = morphology[targets == i]
        n = len(rows)
        design = np.zeros((n * p + q, p + q))
        design[: n * p, :p] = np.tile(np.eye(p), (n, 1))
        design[n * p :, p:] = np.eye(q)
        covariance = design @ between @ design.T
        covariance[: n * p, : n * p] += np.kron(np.eye(n), within)
        residual = np.r_[rows.ravel(), ecology[i]] - design @ mean
        inverse = np.linalg.inv(covariance)
        projected = inverse @ residual
        sign, determinant = np.linalg.slogdet(covariance)
        if sign != 1:
            raise ValueError("Nonpositive dense grouped covariance")
        likelihood += 0.5 * (determinant + residual @ projected + len(residual) * np.log(2 * np.pi))
        score = 0.5 * (inverse - np.outer(projected, projected))
        gb += design.T @ score @ design
        gr += sum(score[j * p : (j + 1) * p, j * p : (j + 1) * p] for j in range(n))
        gm -= design.T @ projected
    np.testing.assert_allclose(
        likelihood, diagnostics["negative_log_likelihood"], rtol=1e-11, atol=1e-8
    )
    if policy.schema_version == 2:
        from super_deli.benchmarking.boundary_audit import verify_boundary

        verify_boundary(
            diagnostics,
            between,
            within,
            mean,
            factor,
            residual_factor,
            gb,
            gr,
            gm,
            ecology,
            g,
            p,
            policy,
        )
    else:
        gradients = []
        for covariance, score, scale in ((between, gb, scales), (within, gr, scales[:p])):
            standardized = covariance / scale[:, None] / scale[None, :]
            eigenvalues = np.linalg.eigvalsh(standardized)
            if eigenvalues[0] <= policy.rank_tol * max(eigenvalues[-1], 1):
                raise ValueError(
                    "Null covariance violates its prespecified interior rank constraint"
                )
            score_factor = np.linalg.cholesky(standardized)
            transformed = 2 * (score * scale[:, None] * scale[None, :]) @ score_factor
            transformed[np.diag_indices(len(score_factor))] *= np.diag(score_factor)
            gradients.extend(transformed[np.tril_indices(len(score_factor))] / g)
        gradient_max = max(np.max(np.abs(gradients)), np.max(np.abs(gm * scales / g)))
        if gradient_max > policy.gradient_tol:
            raise ValueError("Dense raw-data ML score fails the convergence tolerance")
        accepted = [a for a in diagnostics["attempts"] if a["accepted"]]
        selected = next(a for a in accepted if a["start"] == diagnostics["selected_start"])
        if selected != min(accepted, key=lambda a: (a["objective_per_group"], a["start"])):
            raise ValueError(
                "Selected null initialization does not maximize the accepted likelihoods"
            )
        np.testing.assert_allclose(
            selected["gradient_max"], np.max(np.abs(gradients)), atol=2e-8, rtol=1e-3
        )

    def simulate(seed):
        rng = np.random.default_rng(seed)
        latent = mean + rng.standard_normal((g, p + q)) @ factor.T
        draw_morphology = (
            latent[targets, :p] + rng.standard_normal(morphology.shape) @ residual_factor.T
        )
        return {
            "morphological": draw_morphology,
            "ecological": latent[[order[u] for u in eview["target_ids"]], p:],
        }

    return fitted, simulate, g, p + q


def point(archive, record, policy, values, layout):
    import numpy as np

    from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point

    audit_pipeline_point(archive, record, policy)
    if record["status"] != "complete":
        return
    prefix = record["archive_prefix"] + "/run/"
    run = json.loads(archive.read(prefix + "summary.json"))
    if run["unit_ids"] != layout["unit_ids"] or record["n_retained"] != len(layout["unit_ids"]):
        raise ValueError("Grouped retained unit identities disagree")
    for i, view in enumerate(layout["views"]):
        rows = json.loads(archive.read(prefix + f"view_{i:03d}/source_rows.json"))
        array = np.load(
            io.BytesIO(archive.read(prefix + f"view_{i:03d}/source_values.npy")), allow_pickle=False
        )
        for key in ("unit_ids", "observation_ids", "target_ids", "feature_names"):
            if rows[key] != list(view[key]):
                raise ValueError("Grouped source identities/features/counts disagree")
        np.testing.assert_array_equal(array, values[view["kind"]])
    if any(v.pca_fit == "target_means" for v in policy.model.views):
        from super_deli.benchmarking.tabular_pca_audit import audit_tabular_views

        audit_tabular_views(archive, prefix, policy.model, values, layout)


def audit_grouped_study(root, *, output_dir):
    from super_deli.benchmarking.empirical_analysis import audit_fitted_study

    return audit_fitted_study(root, output_dir=output_dir, grouped=True)
