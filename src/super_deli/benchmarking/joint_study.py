"""Independent observed datasets under declared joint laws and stresses."""

from dataclasses import replace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import run_fitted_study
from super_deli.benchmarking.linked_study import simulate_linked
from super_deli.benchmarking.pipeline_study import summarize_records
from super_deli.models.joint_config import CASES, JointStudyConfig


def simulate_joint(case, n, seed):
    import numpy as np

    if case not in CASES or type(n) is not int or n < 20 or n % 4:
        raise ValueError("Use a supported joint case and >=20 specimens divisible by four")
    rng = np.random.default_rng(seed)
    g, genomic, labels = simulate_linked(
        "moderate_ld", n, derived_seed(seed, "joint", 0, "genotypes")
    )
    if case == "cline":
        t = rng.uniform(-1, 1, n)
        intercepts = np.log(np.linspace(0.2, 0.5, 128) / (1 - np.linspace(0.2, 0.5, 128)))
        probabilities = 1 / (1 + np.exp(-(intercepts + t[:, None] * np.tile([-3.0, 3.0], 64))))
        g = rng.binomial(2, probabilities).astype(float)
    predictors = np.column_stack([g[:, j : j + 8].mean(axis=1) for j in range(0, 32, 8)])
    effects = np.array(
        [
            [1.0, 0.7, 0.5, 0.3, 1.0, 0.7, 0.5, 0.3],
            [0.3, -0.5, 0.7, -1.0, 0.3, -0.5, 0.7, -1.0],
            [0.6, 0.4, -0.4, -0.6, -0.6, -0.4, 0.4, 0.6],
            [-0.2, 0.5, -0.5, 0.2, 0.2, -0.5, 0.5, -0.2],
        ]
    )
    mean = np.linspace(-2, 2, 8) + (
        0 if case == "zero_association" else (predictors - 0.7) @ effects
    )
    correlation = 0.5 ** np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
    noise = (rng.standard_normal((n, 8)) @ np.linalg.cholesky(correlation).T) * np.linspace(
        0.5, 1.5, 8
    )
    if case == "heavy_tailed":
        noise /= np.sqrt(rng.chisquare(3, (n, 1)) / 3)
    y = mean + noise
    if case == "complementary":
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        # One binary group axis in genomics/morphology, the other in ecology.
        a, b = 2 * (labels // 2) - 1, 2 * (labels % 2) - 1
        prob = 1 / (1 + np.exp(-(np.log(0.3 / 0.7) + 3 * a[:, None] * np.tile([-1.0, 1.0], 64))))
        g = rng.binomial(2, prob).astype(float)
        y = noise + 3 * np.column_stack(
            [a[:, None] * np.array([1, 0.7, 0.5, 0.3]), b[:, None] * np.array([1, 0.7, 0.5, 0.3])]
        )
    units = genomic["unit_ids"]
    layout = {
        "schema_version": 1,
        "unit_ids": units,
        "genomic": genomic,
        "views": [
            {
                "modality": kind,
                "kind": kind,
                "feature_names": [f"feature_{j}" for j in range(4)],
                "feature_units": [None] * 4,
                "unit_ids": units,
                "observation_ids": [f"{kind}_{u}" for u in units],
            }
            for kind in ("morphological", "ecological")
        ],
    }
    return {"genotypes": g, "phenotypes": y}, layout, labels


def case_policy(study, case, repetition):
    if case not in CASES or not 0 <= repetition < study.n_datasets:
        raise ValueError("Unknown joint case/repetition")
    return replace(
        study.calibration,
        random_seed=derived_seed(study.random_seed, case, repetition, "null_stream"),
    )


def summarize_joint(records, study, case):
    interpretation = (
        "Model-law rejection rate under the specified joint null"
        if case in CASES[:2]
        else "Power for the constructed complementary four-group alternative"
        if case == "complementary"
        else "Misspecification sensitivity; NOT species or discrete-cluster type-I error"
    )
    return summarize_records(records, study, case, interpretation=interpretation)


def run_joint_study(study, *, case, output_dir):
    from super_deli.models.joint_null import write_raw_joint
    from super_deli.workflows.joint_calibration import run_joint_calibration

    study = JointStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown joint validation case")
    return run_fitted_study(
        study,
        case=case,
        output_dir=output_dir,
        simulator=simulate_joint,
        writer=write_raw_joint,
        runner=run_joint_calibration,
        policy_for=case_policy,
        summarizer=summarize_joint,
        decision_key="reject_joint_genomic_phenotypic",
    )
