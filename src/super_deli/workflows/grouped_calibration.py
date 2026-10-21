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
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(),
        scope=SCOPE,
        tail_options=TAIL_OPTIONS,
        seed_namespace="grouped_raw_gaussian_v1",
        target="grouped_gaussian_raw_continuous_views",
        null_hypothesis=(
            "Independent joint Gaussian group effects and iid Gaussian morphology residuals; "
            "shared unknown between/within covariance; fixed noninformative counts"
        ),
    )
