"""Fixed initial calibration pilot; simulation replicates are independent datasets."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.neural_calibration_config import NeuralCalibrationConfig
from super_deli.models.separate_gmm_config import FAMILY, SeparateGMMConfig

CASES = ("gaussian_independent", "gaussian_correlated", "four_complementary", "heavy_tail")
ARMS = ("neural", "pca4")
ROOTS = {"main": (110017, 110023, 110039, 110051), "preflight": (110059, 110063, 110069, 110083)}


@dataclass(frozen=True)
class NeuralCalibrationStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 4 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 14400
    max_workers: int = 2

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixture) is not bool
        ):
            raise ValueError("Use study schema 1 and boolean fixture")
        for name, cap in (
            ("artifact_cap_bytes", 4 * 1024**3),
            ("max_projected_seconds", 14400),
            ("max_workers", 2),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError("Study resource ceiling exceeded")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB free reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, v):
        if not isinstance(v, dict) or v.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown study fields")
        return cls(**v)


def model_for(arm, seed):
    from super_deli.benchmarking.matched_data import configuration

    if arm not in ARMS:
        raise ValueError("Unknown representation")
    m = SeparateGMMConfig.from_dict(
        {**configuration("paired_0", 0).to_dict(), "model_family": FAMILY}
    )
    return NeuralCalibrationConfig(True, m, random_seed=seed)


def tasks(config, phase):
    if phase not in ROOTS:
        raise ValueError("Unknown phase")
    roots = ROOTS[phase]
    out = []
    for index, case in enumerate(CASES[:1] if config.fixture else CASES):
        for i in range(1 if config.fixture or phase == "preflight" else 10):
            out.append(
                dict(
                    case=case,
                    n=32 if config.fixture else 64,
                    replicate=i,
                    key=f"{case}_{i:04d}",
                    data_seed=[roots[0], index, i],
                    noise_seed=[roots[1], index, i],
                    permutation_seed=[roots[2], index, i],
                    ecology_order_seed=[roots[2], index, i, 1],
                    null_seed_path=[roots[3], index, i],
                    n_simulations=2 if config.fixture else 99,
                )
            )
    return out


def counts(config, phase):
    grid = tasks(config, phase)
    raw = sum(1 + t["n_simulations"] for t in grid)
    replay = (1 if config.fixture else 4) if phase == "main" else 0
    return dict(
        observations=len(grid),
        raw_inputs=raw,
        legacy_replays=replay,
        workflows=2 * (raw + replay),
        neural_fits=2 * (raw + replay),
        verified_updates=1000 * (raw + replay),
        candidate_searches=2 * (raw + replay),
    )
