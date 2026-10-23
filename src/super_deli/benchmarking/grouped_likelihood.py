"""Independent fresh validation; both criteria/statistics share one candidate pool."""

from dataclasses import replace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import run_fitted_study
from super_deli.benchmarking.grouped_pca_study import simulate as simulate_stress
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.models.grouped_likelihood_config import CASES, GroupedLikelihoodStudyConfig


def simulator(study, case, n, seed):
    if case in {"unequal_variance", "cline"}:
        return simulate_stress(case, n, seed, target_level=study.calibration.model.target_level)
    return simulate_grouped(
        "complementary" if case == "complementary_256" else case,
        study.large_n if case == "complementary_256" else n,
        seed,
        target_level=study.calibration.model.target_level,
    )


def case_policy(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown likelihood validation case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize(records, study, case):
    interpretation = (
        "Declared hierarchical Gaussian null"
        if case.endswith("gaussian")
        else "Constructed four-group alternative"
        if case in {"complementary", "complementary_256", "unequal_variance"}
        else "Misspecification stress; not species/type-I error"
    )
    result = summarize_records(records, study, case, interpretation=interpretation)
    rate = result["rejection_rate"]
    rate["withheld_outcome_bounds"] = [
        rate["events"] / study.n_datasets,
        (rate["events"] + rate["n_missing"]) / study.n_datasets,
    ]
    return result


def run_grouped_likelihood_study(study, *, case, output_dir):
    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.grouped_likelihood import run_grouped_likelihood

    study = GroupedLikelihoodStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown likelihood validation case")
    return run_fitted_study(
        study,
        case=case,
        output_dir=output_dir,
        simulator=lambda c, n, s: simulator(study, c, n, s),
        writer=write_raw_views,
        runner=run_grouped_likelihood,
        policy_for=case_policy,
        summarizer=summarize,
        decision_key="reject_grouped_raw_gaussian",
    )
