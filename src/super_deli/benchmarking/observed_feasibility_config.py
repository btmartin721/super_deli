"""A fixed favourable diagnostic; production reference policies stay unchanged."""

from dataclasses import dataclass

from super_deli.benchmarking.abstention_config import ARMS, CASES
from super_deli.benchmarking.local_reference_config import LocalReferenceStudy

TREE_PARAMETERS = dict(
    loss="log_loss",
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=75101,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0,
)
THRESHOLDS = tuple(j / 100 for j in range(101))


@dataclass(frozen=True)
class ObservedFeasibilityStudy(LocalReferenceStudy):
    max_projected_seconds: int = 1800
    preflight_max_seconds: int = 600

    def __post_init__(self):
        super().__post_init__()
        if self.max_projected_seconds > 1800 or self.preflight_max_seconds > 600:
            raise ValueError("Observed-feature ten/thirty-minute ceilings exceeded")


class TreeSpecification:
    def to_dict(self):
        return dict(TREE_PARAMETERS)


def model_for(arm, seed):
    if arm != "diagnostic" or type(seed) is not int or seed != 0:
        raise ValueError("One frozen observed-feature detector")
    return TreeSpecification()


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Known phase required")
    return [
        dict(
            key=f"observed_{case}_n{n}_{r:04d}",
            case=case,
            n=n,
            replicate=r,
            fixture=config.fixture,
            seed_path=[20261112, 750, int(phase == "main"), CASES.index(case), n, r],
        )
        for case in (("linear_standard",) if config.fixture else CASES)
        for n in ((32,) if config.fixture else (64, 256))
        for r in range(1 if config.fixture or phase == "preflight" else 10)
    ]


def roles(task):
    known = 128 if task["fixture"] else 1024
    unknown = 64 if task["fixture"] else 512
    result = [
        dict(
            name=f"{stage}_{group}",
            stage=stage,
            group=group,
            generator_cohort="train" if group == "known" else group,
            n=known if group == "known" else unknown,
        )
        for stage in ("train", "calibration")
        for group in ("known", "near", "far")
    ]
    result += [
        dict(
            name="test_" + c,
            stage="test",
            group=c,
            generator_cohort=c,
            n=(64 if task["fixture"] else 2048) if c in ("iid", "shifted") else unknown,
        )
        for c in (
            ("iid", "near", "far", "overlap", "shifted")
            if task["case"] == "confounded_size"
            else ("iid", "near", "far", "overlap")
        )
    ]
    return result


def detector_replay_tasks(config, phase):
    grid = tasks(config, phase)
    if config.fixture:
        return grid
    wanted = (
        {("linear_weak", 64), ("combined", 256)}
        if phase == "preflight"
        else {
            ("linear_weak", 64),
            ("linear_standard", 256),
            ("combined", 64),
            ("confounded_size", 256),
        }
    )
    return [t for t in grid if t["replicate"] == 0 and (t["case"], t["n"]) in wanted]


def counts(config, phase):
    grid = tasks(config, phase)
    repeats = len(detector_replay_tasks(config, phase))
    legacy = (
        (1 if phase == "preflight" else 2) if config.fixture else (2 if phase == "preflight" else 8)
    )
    return dict(
        observations=len(grid),
        legacy_replays=2 if config.fixture else 8,
        scheduled_frozen_replays=legacy,
        detector_replays=repeats,
        workflows=len(grid) + repeats + legacy,
        detector_fits=len(grid) + repeats,
        trees=100 * (len(grid) + repeats),
        neural_fits=0,
        verified_updates=0,
        cohorts=sum(len(roles(t)) for t in grid),
        rows=sum(r["n"] for t in grid for r in roles(t)),
    )


def settings():
    return dict(
        cases=CASES,
        reference_arms=ARMS,
        primary_arms=["logistic", "residual_0"],
        namespace=[20261112, 750],
        tree_parameters=TREE_PARAMETERS,
        thresholds=THRESHOLDS,
        predictors="four measured morphology + eight measured ecology; raw values only",
        training_exposure="specified nearby/distant examples plus extra labelled specimens",
        target=dict(
            known_acceptance=0.8, accepted_accuracy=0.9, near_acceptance=0.1, far_acceptance=0.1
        ),
        selection="maximum qualifying calibration coverage; ties largest threshold",
        focal=dict(
            case="combined",
            n=256,
            arm="residual_0",
            min_qualified=8,
            min_joint=5,
            min_diagnostic_only=3,
            max_reverse=1,
        ),
        primary_rates=64,
        paired_contrasts=48,
        interpretation="conditional archived-head feasibility, not unseen-class generalization",
        significance="disabled",
        stability="disabled",
    )
