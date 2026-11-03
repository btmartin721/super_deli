"""Frozen paired-view study grid and independent engineering resource limits."""

from dataclasses import asdict, dataclass, fields

CASES = (
    "gaussian_independent",
    "gaussian_correlated",
    "four_redundant",
    "four_complementary",
    "morphology_only",
    "discordant",
    "heavy_tail",
    "moons",
)
ARMS = tuple(
    f"{view}_{seed}" for view in ("paired", "morphology", "ecology") for seed in (0, 101, 202)
) + ("integrative_gmm", "morphology_gmm", "ecology_gmm", "morphology_density", "ecology_density")
MORPH = tuple(f"m{i + 1}" for i in range(4))
ECO = tuple(f"e{i + 1}" for i in range(8))
SETTINGS = {
    "schema_version": 1,
    "neural_seeds": [0, 101, 202],
    "noise_sd": 0.15,
    "main_roots": [96211, 96217, 96223, 96231],
    "preflight_roots": [96241, 96247, 96253, 96259],
    "primary_rates": 224,
    "model_seed": 71323,
    "loading_morphology": [[1, 0], [0, 1], [1, 1], [1, -1]],
    "loading_ecology": [[1, 0], [0, 1], [1, 1], [1, -1], [1, 2], [2, 1], [1, -2], [-2, 1]],
}


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 2147483648
    min_free_bytes: int = 8589934592
    max_projected_seconds: int = 7200
    max_workers: int = 2

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixture) is not bool
        ):
            raise ValueError("Invalid matched study schema or fixture flag")
        for name, cap in (
            ("artifact_cap_bytes", 2**31),
            ("max_projected_seconds", 7200),
            ("max_workers", 2),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError("Study scope/resource limit exceeds approval")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 2**33:
            raise ValueError("At least eight GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown study fields")
        return cls(**value)


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown phase")
    roots = SETTINGS[phase + "_roots"]
    cases = ("four_complementary",) if config.fixture else CASES
    sizes = (32,) if config.fixture else (64, 256)
    count = 1 if config.fixture or phase == "preflight" else 10
    pre_sub = {
        ("gaussian_independent", 64),
        ("four_complementary", 64),
        ("four_redundant", 256),
        ("morphology_only", 256),
    }
    main_sub = {
        (c, n)
        for c in ("gaussian_independent", "four_redundant", "four_complementary", "morphology_only")
        for n in sizes
    }
    replay = {("gaussian_independent", sizes[0]), ("four_complementary", sizes[-1])}
    return [
        {
            "key": f"{case}_n{n}_{i:04d}",
            "case": case,
            "n": n,
            "replicate": i,
            "data_seed": [roots[0], CASES.index(case), n, i],
            "noise_seed": [roots[1], CASES.index(case), n, i],
            "permutation_seed": [roots[2], CASES.index(case), n, i],
            "ecology_order_seed": [roots[2], CASES.index(case), n, i, 1],
            "subsampling_path": [roots[3], CASES.index(case), n, i],
            "draws": 5
            if i == 0
            and ((case, n) in (main_sub if phase == "main" else pre_sub) or config.fixture)
            else 0,
            "replay": i == 0 and (case, n) in replay,
        }
        for case in cases
        for n in sizes
        for i in range(count)
    ]


def counts(config, phase):
    grid = tasks(config, phase)
    points = 14 * len(grid)
    draws = sum(t["draws"] * 11 for t in grid)
    replay = sum(t["replay"] * 14 for t in grid)
    neural = 9 * len(grid) + sum(t["draws"] * 9 + t["replay"] * 9 for t in grid)
    return {
        "observations": len(grid),
        "point_fits": points,
        "subsample_refits": draws,
        "point_replays": replay,
        "workflows": points + draws + replay,
        "neural_fits": neural,
        "classical_fits": points + draws + replay - neural,
    }
