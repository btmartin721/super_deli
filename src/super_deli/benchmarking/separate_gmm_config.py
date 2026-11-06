"""Frozen fresh validation design; data replicates, encoders and heads are distinct."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.separate_gmm_config import FAMILY, SeparateGMMConfig

CASES = (
    "gaussian_independent",
    "gaussian_correlated",
    "morphology_only",
    "ecology_only",
    "four_complementary",
    "four_redundant",
    "unequal_complementary",
    "moons",
)
PRIMARY = CASES[:6]
SEEDS = (0, 101, 202)
NATIVE = tuple(f"separate_{s}" for s in SEEDS)
ARMS = NATIVE + ("pca4", "raw_shared")
HEADS = ("hdbscan", "gmm_full", "gmm_shared")
ROOTS = {"main": [109001, 109003, 109013, 109037], "preflight": [109049, 109063, 109069, 109073]}


@dataclass(frozen=True)
class SharedEncoderStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 4 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 7200
    max_workers: int = 2

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixture) is not bool
        ):
            raise ValueError("Invalid shared-encoder study schema")
        for name, cap in (
            ("artifact_cap_bytes", 4 * 1024**3),
            ("max_projected_seconds", 7200),
            ("max_workers", 2),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError("Study resource ceiling exceeded")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown shared-encoder study configuration")
        return cls(**value)


def tasks(config, phase):
    if phase not in ROOTS:
        raise ValueError("Unknown phase")
    cases, sizes = (("four_complementary",), (32,)) if config.fixture else (CASES, (64, 256))
    r = ROOTS[phase]
    out = []
    subs = (
        ("gaussian_independent", "four_complementary")
        if phase == "preflight"
        else (
            "gaussian_independent",
            "four_complementary",
            "morphology_only",
            "unequal_complementary",
        )
    )
    for case in cases:
        for n in sizes:
            count = (
                1
                if config.fixture or phase == "preflight"
                else 100
                if case.startswith("gaussian_")
                else 10
            )
            for i in range(count):
                out.append(
                    dict(
                        case=case,
                        n=n,
                        replicate=i,
                        key=f"{case}_n{n}_{i:04d}",
                        data_seed=[r[0], CASES.index(case), n, i],
                        noise_seed=[r[1], CASES.index(case), n, i],
                        permutation_seed=[r[2], CASES.index(case), n, i],
                        ecology_order_seed=[r[2], CASES.index(case), n, i, 1],
                        subsampling_path=[r[3], CASES.index(case), n, i],
                        draws=(1 if config.fixture else 10)
                        if i == 0 and (config.fixture or case in subs)
                        else 0,
                    )
                )
    return out


def counts(config, phase):
    grid = tasks(config, phase)
    n = len(grid)
    draw = sum(t["draws"] for t in grid)
    replay = (1 if config.fixture else 12) if phase == "main" else 0
    return dict(
        observations=n,
        point_workflows=13 * n,
        subsample_refits=9 * draw,
        legacy_replays=replay,
        legacy_replay_evaluations=3 * replay,
        workflows=13 * n + 9 * draw + 3 * replay,
        neural_fits=6 * n + 6 * draw + 2 * replay,
        density_fits=4 * n + 3 * draw + replay,
        gmm_searches=9 * n + 6 * draw + 2 * replay,
    )


def model_for(arm, subsampling_seed, draws=0):
    from super_deli.benchmarking.coordinate_config import MixtureSettings
    from super_deli.benchmarking.matched_data import configuration

    if arm not in ARMS:
        raise ValueError("Unknown representation")
    if arm not in NATIVE:
        return MixtureSettings("tied")
    seed = int(arm.rsplit("_", 1)[1])
    base = configuration("paired_" + str(seed), subsampling_seed, draws)
    return SeparateGMMConfig.from_dict({**base.to_dict(), "model_family": FAMILY})
