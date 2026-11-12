"""Frozen one-candidate design, explicit archived/fresh counts and independent seeds."""

from dataclasses import dataclass

from super_deli.benchmarking.abstention_config import ARMS, CASES, SEEDS, cohort_names
from super_deli.benchmarking.abstention_config import tasks as old_tasks
from super_deli.benchmarking.residual_config import ResidualStudy, model_for

POLICIES = ("unscreened", "probability", "distance", "local")


@dataclass(frozen=True)
class LocalReferenceStudy(ResidualStudy):
    artifact_cap_bytes: int = 3 * 1024**3

    def __post_init__(self):
        if (
            type(self.artifact_cap_bytes) is not int
            or not 0 < self.artifact_cap_bytes <= 3 * 1024**3
        ):
            raise ValueError("Three-GiB class-local evidence ceiling exceeded")
        # Reuse all existing limits without changing the predecessor's two-GiB contract.
        values = self.to_dict() | {"artifact_cap_bytes": min(self.artifact_cap_bytes, 2 * 1024**3)}
        ResidualStudy(**values)


def tasks(config, phase):
    return [
        {**t, "key": "local_" + t["key"], "seed_path": [20261112, 740, *t["seed_path"][2:]]}
        for t in old_tasks(config, phase)
    ]


def counts(config, phase):
    fresh = tasks(config, phase)
    archive = (
        1
        if config.fixture and phase == "preflight"
        else 0
        if config.fixture
        else 8
        if phase == "preflight"
        else 72
    )
    replays = 8 if not config.fixture and phase == "main" else 0
    external = sum(len(cohort_names(t)) - 2 for t in fresh) + (
        4 * archive if config.fixture else 34 if phase == "preflight" else 306
    )
    return dict(
        observations=len(fresh),
        archived_observations=archive,
        legacy_replays=replays,
        workflows=4 * (len(fresh) + archive) + replays,
        classifier_trainings=4 * len(fresh),
        neural_fits=3 * len(fresh),
        verified_updates=1500 * len(fresh),
        logistic_fits=len(fresh),
        support_fits=len(fresh) + archive,
        external_cohorts=external * 4,
        policy_evaluations=external * 4 * 4,
        oracle_evaluations=external * 19,
    )


def settings():
    return dict(
        cases=CASES,
        seeds=SEEDS,
        primary_seed=0,
        arms=ARMS,
        policies=POLICIES,
        namespace=[20261112, 740],
        neighbors=5,
        exclusion_level=0.05,
        shrinkage=0.1,
        calibration_n=160,
        known_evaluation_n=2048,
        unfamiliar_n=512,
        primary_rates=64,
        target=dict(
            known_acceptance=0.8,
            accepted_known_accuracy=0.9,
            near_acceptance=0.1,
            far_acceptance=0.1,
        ),
        benefit=dict(
            min_coverage_gain=0.1,
            min_accuracy=0.9,
            max_accuracy_loss=0.01,
            max_unfamiliar_increase=0.02,
        ),
        oracle_withholding=[i / 20 for i in range(1, 20)],
        significance="disabled",
        stability="disabled",
        reference_model=model_for("residual", 0).to_dict(),
    )
