"""Frozen power/recovery design; resource fields cannot alter scientific settings."""

from dataclasses import asdict, dataclass, fields

ARMS = ("neural", "pca4")
NAMESPACE = (20261109, 700)


@dataclass(frozen=True)
class NeuralPowerStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 8 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 10 * 3600
    preflight_max_seconds: int = 2 * 3600
    max_workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use power study schema 1")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be boolean")
        for name, cap in (
            ("artifact_cap_bytes", 8 * 1024**3),
            ("max_projected_seconds", 36000),
            ("preflight_max_seconds", 7200),
            ("max_workers", 2),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError(name + " exceeds approved ceiling")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown power study fields")
        return cls(**value)


def cells(config):
    if config.fixture:
        return [("four_complementary", 32, 1.0, 1)]
    return [
        *(("four_complementary", n, s, 10) for n in (64, 256) for s in (0.5, 1.0, 1.5)),
        ("gaussian_independent", 64, 1.0, 10),
        ("gaussian_independent", 256, 1.0, 10),
        ("unequal_complementary", 64, 1.0, 10),
        ("heavy_tail", 64, 1.0, 5),
        ("continuous_gradient", 64, 1.0, 5),
    ]


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown phase")
    out = []
    for i in range(1 if config.fixture or phase == "preflight" else 10):
        for index, (case, n, strength, count) in enumerate(cells(config)):
            if i >= count:
                continue
            key = f"{case}_n{n}_s{int(10 * strength):02d}_{i:04d}"
            names = (
                "data_seed",
                "noise_seed",
                "permutation_seed",
                "ecology_order_seed",
                "null_seed_path",
            )
            out.append(
                dict(
                    case=case,
                    n=n,
                    signal_multiplier=strength,
                    cell=index,
                    replicate=i,
                    key=key,
                    n_simulations=2 if config.fixture else 99,
                    **{
                        name: [*NAMESPACE, int(phase == "main"), index, int(10 * strength), n, i, j]
                        for j, name in enumerate(names)
                    },
                )
            )
    return out


def counts(config, phase):
    grid = tasks(config, phase)
    replay = (1 if config.fixture else 2) if phase == "main" else 0
    raw = sum(1 + t["n_simulations"] for t in grid) + replay * (3 if config.fixture else 100)
    return dict(
        observations=len(grid),
        legacy_replays=replay,
        raw_inputs=raw,
        workflows=2 * raw,
        candidate_searches=2 * raw,
        neural_fits=2 * raw,
        verified_updates=1000 * raw,
    )


def archived_jobs(binding):
    return [
        dict(kind="replay", task={**e["task"], "key": e["key"]}, source_path=e["path"])
        for e in binding["replays"]
    ]
