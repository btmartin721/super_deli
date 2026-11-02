"""Exact frozen scientific grid and separate engineering budgets."""

from dataclasses import asdict, dataclass, fields

CASES = (
    "gaussian",
    "gaussian_8d",
    "four_spherical",
    "moons",
    "four_unequal",
    "heavy_tail",
    "cline",
    "eight_spherical",
)
ARMS = ("spectral_5", "spectral_15", "gmm", "density")
CONDITIONS = ("clean", "noisy")
SETTINGS = {
    "schema_version": 1,
    "max_k": 6,
    "source_seed": 93411,
    "noise_seed": 93427,
    "main_namespace": 19,
    "preflight_namespace": 17,
    "probe_seed": 93431,
    "model_seed": 71323,
    "noise_sd": 0.15,
    "gap_tolerance": 1e-8,
    "main_workflows": 2592,
    "preflight_evaluations": 154,
    "primary_rates": 128,
}


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 2147483648
    min_free_bytes: int = 8589934592
    max_projected_seconds: int = 3600
    max_workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported study schema")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be Boolean")
        limits = {"artifact_cap_bytes": 2147483648, "max_projected_seconds": 3600, "max_workers": 2}
        for name, upper in limits.items():
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= upper:
                raise ValueError("Study scope/resource limit exceeds the approved plan")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8589934592:
            raise ValueError("At least eight GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown study fields")
        return cls(**value)


def tasks(config, phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown study phase")
    sizes = (32, 48) if config.fixture else (64, 256)
    cases = ("gaussian", "four_spherical", "moons") if config.fixture else CASES
    count = 1 if config.fixture or phase == "preflight" else 20
    namespace = 17 if phase == "preflight" else 19
    replay = {("gaussian", sizes[0]), ("moons", sizes[1])}
    if phase == "main" and not config.fixture:
        replay |= {("gaussian_8d", 256), ("four_spherical", 64)}
    return [
        {
            "key": f"{case}_n{n}_{i:04d}",
            "case": case,
            "n": n,
            "replicate": i,
            "data_seed": [93411, namespace, CASES.index(case), n, i],
            "noise_seed": [93427, namespace, CASES.index(case), n, i],
            "permutation_seed": [93411, namespace, CASES.index(case), n, i, 1],
            "replay": i == 0 and (case, n) in replay,
        }
        for case in cases
        for n in sizes
        for i in range(count)
    ]


def counts(config, phase):
    grid = tasks(config, phase)
    workflows = sum(8 + 8 * t["replay"] for t in grid)
    return {
        "observations": len(grid),
        "workflows": workflows,
        "probes": 10 if phase == "preflight" else 0,
        "evaluations": workflows + (10 if phase == "preflight" else 0),
    }
