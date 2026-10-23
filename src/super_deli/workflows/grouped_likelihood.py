"""Full raw-data bootstrap of likelihood gain, independent of AIC component choice."""

from super_deli.models.grouped_likelihood_config import GroupedLikelihoodConfig
from super_deli.models.likelihood_selection import candidate_statistics
from super_deli.workflows.calibration_artifacts import fit_pipeline
from super_deli.workflows.empirical_calibration import run_fitted_calibration
from super_deli.workflows.grouped_calibration import BOUNDARY_TAIL_OPTIONS, SCOPE, _dependencies

TAIL_OPTIONS = {
    **BOUNDARY_TAIL_OPTIONS,
    "p_value_kind": (
        "Approximate full-pipeline fitted conditional-PSD Gaussian likelihood-gain bootstrap; "
        "nonregular boundaries"
    ),
}


def point_fitter(staging, declaration, directory, prefix, *, model, runner):
    def record_for(run, output, model):
        from super_deli.io.config import read_json

        details = candidate_statistics(read_json(output / "candidates.json"))
        if run["selected_k"] != details["selected_k"]:
            raise ValueError("AIC point selection differs from candidate log")
        return {
            **details,
            "point_seed": run["point_seed"],
            "n_retained": len(run["unit_ids"]),
            **{
                key: run["provenance"][key]
                for key in ("mixture_sha256", "fusion_sha256", "candidate_sha256")
            },
        }

    return fit_pipeline(
        staging, declaration, directory, prefix, model=model, runner=runner, record_for=record_for
    )


def run_grouped_likelihood(dataset, *, config, base_dir, output_dir):
    config = GroupedLikelihoodConfig.from_dict(config.to_dict())
    return run_fitted_calibration(
        dataset,
        config=config,
        base_dir=base_dir,
        output_dir=output_dir,
        dependencies=_dependencies(),
        scope=SCOPE
        + (
            " Full covariance, group-mean morphology PCA and AIC component selection are "
            "prespecified. The global statistic is maximum eligible twice-log-likelihood "
            "gain over K=1, independently of AIC selection. It includes the full K search, "
            "reg_covar and minimum hard component-size policy; no chi-squared reference. "
            "No selection between statistics or families is calibrated."
        ),
        tail_options=TAIL_OPTIONS,
        seed_namespace="grouped_raw_gaussian_v2",
        target="grouped_gaussian_raw_continuous_views",
        null_hypothesis=(
            "Independent joint Gaussian group effects and iid Gaussian morphology residuals; "
            "shared unknown between/within covariance; fixed noninformative counts"
        ),
        point_fitter=point_fitter,
        evidence_method=(
            "Approximate full-pipeline conditional-PSD Gaussian likelihood-gain bootstrap"
        ),
    )
