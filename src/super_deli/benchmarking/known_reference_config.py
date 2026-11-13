"""Frozen known-only three-reference folds and parent-level validation budgets."""

from dataclasses import asdict, dataclass, fields

from super_deli.benchmarking.abstention_config import CASES
from super_deli.benchmarking.residual_config import model_for

ARMS = ("logistic", "residual_0")
POLICIES = ("unscreened", "probability", "distance", "local", "density")


@dataclass(frozen=True)
class KnownReferenceStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 6 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 5400
    preflight_max_seconds: int = 1200
    max_workers: int = 2

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixture) is not bool
        ):
            raise ValueError("Unsupported known-reference study schema/fixture")
        for name, cap in [
            ("artifact_cap_bytes", 6 * 1024**3),
            ("max_projected_seconds", 5400),
            ("preflight_max_seconds", 1200),
            ("max_workers", 2),
        ]:
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) <= cap:
                raise ValueError("Known-reference study ceiling exceeded")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB free reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown known-reference study fields")
        return cls(**value)


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Known phase required")
    return [
        dict(
            key=f"known_{case}_n{n}_{r:04d}",
            case=case,
            n=n,
            replicate=r,
            fixture=config.fixture,
            seed_path=[20261112, 760, int(phase == "main"), CASES.index(case), n, r],
        )
        for case in (("linear_standard",) if config.fixture else CASES)
        for n in ((32,) if config.fixture else (64, 256))
        for r in range(1 if config.fixture or phase == "preflight" else 10)
    ]


def replays(config, phase):
    grid = tasks(config, phase)
    if config.fixture:
        return [dict(parent=grid[0], heldout=0)]
    wanted = (
        {("linear_weak", 64): 0, ("combined", 256): 3}
        if phase == "preflight"
        else {
            ("linear_weak", 64): 0,
            ("linear_standard", 256): 1,
            ("combined", 64): 2,
            ("confounded_size", 256): 3,
        }
    )
    return [
        dict(parent=t, heldout=wanted[(t["case"], t["n"])])
        for t in grid
        if t["replicate"] == 0 and (t["case"], t["n"]) in wanted
    ]


def roles(task):
    result = [
        ("train", task["n"]),
        ("calibration", 80 if task["fixture"] else 160),
        ("iid", 64 if task["fixture"] else 2048),
        ("near", 64 if task["fixture"] else 512),
        ("far", 64 if task["fixture"] else 512),
        ("overlap_pool", 64 if task["fixture"] else 2048),
    ]
    if task["case"] == "confounded_size":
        result.append(("shifted", 64 if task["fixture"] else 2048))
    return result


def counts(config, phase):
    grid = tasks(config, phase)
    folds = 4 * len(grid) + len(replays(config, phase))
    return dict(
        observations=len(grid),
        work_units=len(grid) + len(replays(config, phase)),
        legacy_replays=0,
        fold_replays=len(replays(config, phase)),
        fold_fits=folds,
        workflows=2 * folds,
        neural_fits=folds,
        logistic_fits=folds,
        verified_updates=500 * folds,
        cohorts=sum(len(roles(t)) for t in grid),
        rows=sum(n for t in grid for _, n in roles(t)),
    )


def settings():
    return dict(
        cases=CASES,
        arms=ARMS,
        policies=POLICIES,
        namespace=[20261112, 760],
        holdouts=[0, 1, 2, 3],
        original_panel_n=[64, 256],
        known_training_n=[48, 192],
        calibration_per_known_class=40,
        covariance_shrinkage=0.1,
        exclusion_level=0.05,
        primary_rates=64,
        paired_contrasts=56,
        parent_success="At least three of four heldout folds meet the joint target",
        target=dict(
            known_coverage=0.8,
            accepted_accuracy=0.9,
            heldout_acceptance=0.1,
            near_acceptance=0.1,
            far_acceptance=0.1,
        ),
        focal=dict(
            case="combined", n=256, arm="residual_0", min_success=5, min_new_only=3, max_reverse=1
        ),
        unknown_training_examples=0,
        threshold_search="disabled",
        significance="disabled",
        stability="disabled",
        reference_model=model_for("residual", 0).to_dict(),
    )
