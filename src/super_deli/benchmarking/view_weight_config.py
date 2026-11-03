"""Frozen, dependency-free design for one prespecified weighting sensitivity study."""

from dataclasses import asdict, dataclass, fields, replace

CASES = (
    "morphology_only",
    "ecology_only",
    "four_complementary",
    "four_redundant",
    "gaussian_independent",
    "gaussian_correlated",
)
FRACTIONS = {"equal": (0.5, 0.5), "morphology": (0.8, 0.2), "ecology": (0.2, 0.8)}
SEEDS = (0, 101, 202)
JOINT = tuple(f"joint_{w}_{s}" for w in FRACTIONS for s in SEEDS)
ARMS = (
    JOINT
    + tuple(f"{v}_{s}" for v in ("morphology", "ecology") for s in SEEDS)
    + ("integrative_gmm",)
)
ROOTS = {"main": [98611, 98617, 98621, 98627], "preflight": [98639, 98641, 98647, 98653]}


@dataclass(frozen=True)
class ViewWeightStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 2 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 3600
    max_workers: int = 2

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixture) is not bool
        ):
            raise ValueError("Invalid view-weight study schema/fixture flag")
        for key, cap in (
            ("artifact_cap_bytes", 2 * 1024**3),
            ("max_projected_seconds", 3600),
            ("max_workers", 2),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= cap:
                raise ValueError("Study resource limit exceeds approval")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown view-weight study fields")
        return cls(**value)


def tasks(config, phase):
    if phase not in ROOTS:
        raise ValueError("Unknown study phase")
    cases, sizes = (("morphology_only",), (32,)) if config.fixture else (CASES, (64, 256))
    r = ROOTS[phase]
    return [
        {
            "case": c,
            "n": n,
            "replicate": i,
            "key": f"{c}_n{n}_{i:04d}",
            "data_seed": [r[0], CASES.index(c), n, i],
            "noise_seed": [r[1], CASES.index(c), n, i],
            "permutation_seed": [r[2], CASES.index(c), n, i],
            "ecology_order_seed": [r[2], CASES.index(c), n, i, 1],
            "subsampling_path": [r[3], CASES.index(c), n, i],
            "draws": (1 if config.fixture else 3)
            if i == 0 and c in ("morphology_only", "four_complementary")
            else 0,
        }
        for c in cases
        for n in sizes
        for i in range(1 if config.fixture or phase == "preflight" else 5)
    ]


def counts(config, phase):
    grid = tasks(config, phase)
    points, draws = 16 * len(grid), 9 * sum(t["draws"] for t in grid)
    replays = (1 if config.fixture else 12) if phase == "main" else 0
    return {
        "observations": len(grid),
        "point_workflows": points,
        "subsample_refits": draws,
        "legacy_replays": replays,
        "workflows": points + draws + replays,
        "neural_fits": 15 * len(grid) + draws + replays,
        "gmm_searches": len(grid),
    }


def model_for(arm, subsampling_seed, draws=0):
    from super_deli.benchmarking.matched_data import configuration

    if arm.startswith("joint_"):
        _, name, seed = arm.split("_")
        return replace(
            configuration("paired_" + seed, subsampling_seed, draws),
            schema_version=2,
            view_loss_weights=FRACTIONS[name],
        )
    return configuration(arm, subsampling_seed, 0)
