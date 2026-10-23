"""Approximate full-pipeline parametric bootstrap for grouped continuous views."""

from super_deli.models.grouped_config import GroupedCalibrationConfig
from super_deli.workflows.empirical_calibration import run_fitted_calibration

SCOPE = (
    "Approximate plug-in hierarchical Gaussian null fitted before PCA: independent groups, "
    "joint between-group morphology/ecology covariance, shared iid within-morphology covariance, "
    "fixed noninformative unequal counts, one ecology row/group. Every draw regenerates raw "
    "individuals/group ecology and refits the complete integrative point pipeline. "
    "No species-boundary or cluster-specific significance; no informative-size, related/spatial, "
    "missingness, genomic/count/shape or separately identified ecology measurement-error model."
)
TAIL_OPTIONS = {
    "p_value_kind": (
        "Approximate fitted hierarchical-Gaussian parametric bootstrap with plus-one rank"
    ),
    "decision_key": "reject_grouped_raw_gaussian",
    "interval_scope": (
        "Binomial fitted-null tail interval conditional on observed "
        "T, fitted parameters and counts; "
        "excludes null-estimation/misspecification uncertainty, not species confidence; "
        "failed-outcome bounds are not confidence intervals"
    ),
}


BOUNDARY_TAIL_OPTIONS = {
    **TAIL_OPTIONS,
    "p_value_kind": (
        "Approximate fitted conditional-PSD Gaussian bootstrap; boundary fits are nonregular"
    ),
    "interval_scope": TAIL_OPTIONS["interval_scope"] + "; no uniform boundary size guarantee",
}


def _dependencies():
    try:
        import numpy as np
        import scipy.optimize  # noqa: F401
        import sklearn  # noqa: F401

        from super_deli.models.grouped_null import fit_null, simulate_null, write_raw_views
        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.integrative import run_integrative
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for grouped calibration") from exc
    return np, fit_null, simulate_null, write_raw_views, summarize_upper_tail, run_integrative


def run_grouped_calibration(dataset, *, config, base_dir, output_dir):
    config = GroupedCalibrationConfig.from_dict(config.to_dict())
    # Keep the fitted raw law and seed namespace independent of the GMM family.
    shared = config.model.covariance_type == "tied"
    scope = SCOPE + (
        " Shared covariance is fixed before analysis for every observed/null K search; "
        "no selection across covariance families is calibrated."
        if shared
        else ""
    )
    evidence_options = (
        {
            "evidence_method": (
                "Approximate full-pipeline conditional-PSD grouped Gaussian BIC bootstrap "
                "(prespecified shared covariance)"
            )
        }
        if shared
        else {}
    )
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(),
        scope=scope,
        tail_options=BOUNDARY_TAIL_OPTIONS if config.schema_version == 2 else TAIL_OPTIONS,
        seed_namespace="grouped_raw_gaussian_v2"
        if config.schema_version == 2
        else "grouped_raw_gaussian_v1",
        target="grouped_gaussian_raw_continuous_views",
        null_hypothesis=(
            "Independent joint Gaussian group effects and iid Gaussian morphology residuals; "
            "shared unknown between/within covariance; fixed noninformative counts"
        ),
        **evidence_options,
    )
