"""Frozen nonlinear laws, fit budgets and separate supervised/discovery arms."""

from dataclasses import asdict, dataclass, fields

CASES = (
    "linear_weak",
    "linear_standard",
    "allometric",
    "saturating",
    "combined",
    "confounded_size",
    "gaussian",
    "gradient",
)
SEEDS = (0, 101, 202)
NAMESPACE = (20261112, 720)


@dataclass(frozen=True)
class ResidualStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 2 * 1024**3
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
            raise ValueError("Unsupported residual study schema/fixture")
        for key, cap in (
            ("artifact_cap_bytes", 2 * 1024**3),
            ("max_projected_seconds", 5400),
            ("preflight_max_seconds", 1200),
            ("max_workers", 2),
        ):
            if type(getattr(self, key)) is not int or not 0 < getattr(self, key) <= cap:
                raise ValueError("Residual study ceiling exceeded")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB free reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown residual study fields")
        return cls(**value)


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Known residual phase required")
    grid = []
    for case in ("linear_standard",) if config.fixture else CASES:
        for n in (32,) if config.fixture else (64, 256):
            for replicate in range(1 if config.fixture or phase == "preflight" else 5):
                path = [*NAMESPACE, int(phase == "main"), CASES.index(case), n, replicate]
                grid.append(
                    dict(
                        key=f"{case}_n{n}_{replicate:04d}",
                        case=case,
                        n=n,
                        replicate=replicate,
                        seed_path=path,
                        fixture=config.fixture,
                        labeled=case not in ("gaussian", "gradient"),
                        negative=case == "linear_standard" and replicate == 0,
                    )
                )
    return grid


def arms(task):
    result = ["pca4", *[f"neural_{s}" for s in SEEDS]]
    if task["labeled"]:
        result += ["logistic", *[f"residual_{s}" for s in SEEDS]]
    if task["negative"]:
        result += ["permuted_logistic", *[f"permuted_residual_{s}" for s in SEEDS]]
    return result


def counts(config, phase):
    grid = tasks(config, phase)
    replay = 12 if phase == "main" and not config.fixture else 0
    workflows = sum(len(arms(t)) for t in grid) + replay
    neural = (
        sum(
            sum(2 if a.startswith("neural_") else int("residual_" in a) for a in arms(t))
            for t in grid
        )
        + replay
    )
    searches = 4 * len(grid) + replay
    return dict(
        observations=len(grid),
        legacy_replays=replay,
        workflows=workflows,
        neural_fits=neural,
        verified_updates=500 * neural,
        candidate_searches=searches,
        candidate_component_fits=4 * searches,
        logistic_fits=sum(sum("logistic" in a for a in arms(t)) for t in grid),
    )


def model_for(arm, seed):
    from super_deli.benchmarking.matched_config import ECO, MORPH
    from super_deli.models.residual_config import ResidualConfig
    from super_deli.models.separate_gmm_config import SeparateGMMConfig

    options = dict(
        morphology="morphology",
        ecology="ecology",
        morphology_features=MORPH,
        ecology_features=ECO,
        assume_independent_units=True,
        ecology_measurement_scope="individual",
        random_seed=seed,
    )
    return SeparateGMMConfig(**options) if arm in ("neural", "pca4") else ResidualConfig(**options)
