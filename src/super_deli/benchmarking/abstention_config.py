"""Approved independent cohorts, classifier budgets and immutable support settings."""

from super_deli.benchmarking.residual_config import ResidualStudy, model_for

AbstentionStudy = ResidualStudy
CASES = ("linear_weak", "linear_standard", "combined", "confounded_size")
SEEDS = (0, 101, 202)
ARMS = ("logistic", "residual_0", "residual_101", "residual_202")
POLICIES = ("unscreened", "probability", "distance")
COHORTS = ("train", "calibration", "iid", "near", "far", "overlap", "shifted")


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown abstention phase")
    return [
        dict(
            key=f"{case}_n{n}_{replicate:04d}",
            case=case,
            n=n,
            replicate=replicate,
            fixture=config.fixture,
            seed_path=[20261112, 730, int(phase == "main"), CASES.index(case), n, replicate],
        )
        for case in (("linear_standard",) if config.fixture else CASES)
        for n in ((32,) if config.fixture else (64, 256))
        for replicate in range(1 if config.fixture or phase == "preflight" else 10)
    ]


def cohort_names(task):
    return list(COHORTS if task["case"] == "confounded_size" else COHORTS[:-1])


def counts(config, phase):
    grid = tasks(config, phase)
    replay = 8 if phase == "main" and not config.fixture else 0
    return dict(
        observations=len(grid),
        legacy_replays=replay,
        workflows=4 * len(grid) + replay,
        classifier_trainings=4 * len(grid),
        neural_fits=3 * len(grid),
        verified_updates=1500 * len(grid),
        logistic_fits=len(grid),
        support_fits=len(grid),
        calibration_bindings=4 * len(grid),
        external_cohorts=sum(len(cohort_names(t)) - 2 for t in grid) * 4,
        policy_evaluations=sum(len(cohort_names(t)) - 2 for t in grid) * 12,
    )


def settings():
    return dict(
        cases=CASES,
        seeds=SEEDS,
        primary_seed=0,
        arms=ARMS,
        policies=POLICIES,
        exclusion_level=0.05,
        shrinkage=0.1,
        calibration_n=160,
        known_evaluation_n=2048,
        unfamiliar_n=512,
        primary_rates=32,
        target=dict(
            known_acceptance=0.8,
            accepted_known_accuracy=0.9,
            near_acceptance=0.1,
            far_acceptance=0.1,
        ),
        significance="disabled",
        stability="disabled",
        reference_model=model_for("residual", 0).to_dict(),
    )
