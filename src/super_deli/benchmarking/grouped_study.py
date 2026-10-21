"""Independent validation datasets with known grouped continuous-data laws."""

from __future__ import annotations

from dataclasses import replace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import run_fitted_study
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.models.grouped_config import BOUNDARY_CASES, CASES, GroupedStudyConfig, study_cases


def simulate_grouped(case, n, seed, *, target_level="locality"):
    import numpy as np

    if case in BOUNDARY_CASES:
        from super_deli.benchmarking.boundary_simulation import simulate_boundary

        return simulate_boundary(case, n, seed, target_level=target_level)
    if (
        case not in CASES
        or type(n) is not int
        or n < 20
        or n % 4
        or target_level not in {"population", "locality"}
    ):
        raise ValueError("Use a supported case, target level and >=20 groups divisible by four")
    rng = np.random.default_rng(seed)
    # Draw noninformative counts independently, then hold them fixed within each test.
    counts = rng.integers(1, 5, size=n)
    targets = np.repeat(np.arange(n), counts)
    z, errors = rng.standard_normal((n, 8)), rng.standard_normal((int(counts.sum()), 4))
    labels = np.zeros(n, dtype=int)
    if case == "independent_gaussian":
        between, within = np.eye(8), 0.75 * np.eye(4)
        mean = np.zeros(8)
    else:
        correlation = 0.6 ** np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
        scales = np.linspace(0.5, 2, 8)
        between = correlation * scales[:, None] * scales[None, :]
        within = 1.5 * 0.35 ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
        mean = np.linspace(-2, 2, 8)
    latent = mean + z @ np.linalg.cholesky(between).T
    residuals = errors @ np.linalg.cholesky(within).T
    if case == "heavy_tailed":
        latent = mean + (latent - mean) / np.sqrt(rng.chisquare(3, size=(n, 1)) / 3)
        residuals /= np.sqrt(rng.chisquare(3, size=(len(errors), 1)) / 3)
    elif case == "complementary":
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        centers = np.column_stack((2 * (labels // 2) - 1, 2 * (labels % 2) - 1))
        effects = np.concatenate(
            [centers[:, j, None] * np.array([1, 0.7, 0.5, 0.3]) for j in range(2)], axis=1
        )
        latent += 3 * effects
    layout = grouped_layout(counts, target_level)
    return (
        {"morphological": latent[targets, :4] + residuals, "ecological": latent[:, 4:]},
        layout,
        labels,
    )


def grouped_layout(counts, target_level):
    import numpy as np

    n = len(counts)
    targets = np.repeat(np.arange(n), counts)
    units = tuple(f"group_{i:04d}" for i in range(n))
    specimens = tuple(f"specimen_{i:05d}" for i in range(len(targets)))
    layout = {
        "schema_version": 1,
        "unit_ids": units,
        "target_level": target_level,
        "counts": counts.tolist(),
        "specimens": [
            {"specimen_id": s, f"{target_level}_id": units[g]}
            for s, g in zip(specimens, targets, strict=True)
        ],
        "views": [],
    }
    for kind in ("morphological", "ecological"):
        ids = specimens if kind == "morphological" else units
        layout["views"].append(
            {
                "modality": kind,
                "kind": kind,
                "level": "specimen" if kind == "morphological" else target_level,
                "feature_names": [f"feature_{j}" for j in range(4)],
                "feature_units": [None] * 4,
                "unit_ids": ids,
                "observation_ids": [f"{kind}_{u}" for u in ids],
                "target_ids": [units[g] for g in targets] if kind == "morphological" else units,
            }
        )
    return layout


def case_policy(study, case, repetition):
    if case not in study_cases(study) or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown grouped validation case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize_grouped(records, study, case):
    interpretation = (
        "Type-I error under the specified hierarchical Gaussian with noninformative unequal counts"
        if case.endswith("gaussian")
        else "Power for the complementary four-group alternative"
        if case == "complementary"
        else "Gaussian misspecification sensitivity; NOT species/clustering type-I error"
    )
    summary = summarize_records(records, study, case, interpretation=interpretation)
    rate = summary["rejection_rate"]
    rate["withheld_outcome_bounds"] = [
        rate["events"] / study.n_datasets,
        (rate["events"] + rate["n_missing"]) / study.n_datasets,
    ]
    rate["rate_scope"] = (
        "Conditional on test availability; withheld bounds are not confidence intervals"
    )
    return summary


def run_grouped_study(study, *, case, output_dir):
    from super_deli.workflows.grouped_calibration import _dependencies, run_grouped_calibration

    study = GroupedStudyConfig.from_dict(study.to_dict())
    if case not in study_cases(study):
        raise ValueError("Unknown grouped validation case")
    _, _, _, writer, _, _ = _dependencies()

    def simulator(case, n, seed):
        return simulate_grouped(case, n, seed, target_level=study.calibration.model.target_level)

    return run_fitted_study(
        study,
        case=case,
        output_dir=output_dir,
        simulator=simulator,
        writer=writer,
        runner=run_grouped_calibration,
        policy_for=case_policy,
        summarizer=summarize_grouped,
        decision_key="reject_grouped_raw_gaussian",
    )
