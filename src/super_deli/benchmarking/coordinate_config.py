"""Pure contract for a retrospective, fixed-coordinate clustering diagnosis."""

from dataclasses import asdict, dataclass

from super_deli.benchmarking.separate_config import CASES, PRIMARY, SEEDS
from super_deli.benchmarking.view_weight_config import ViewWeightStudy

SPACES = tuple(
    f"{family}_{seed}" for family in ("separate", "joint2", "joint4") for seed in SEEDS
) + ("pca4",)
CLUSTERERS = ("hdbscan", "gmm_full", "gmm_shared")
PRIMARY_SPACES = tuple(f"separate_{seed}" for seed in SEEDS)
SOURCE_HEAD = "c4dde4cb9c1524e8c9c182987f3edbfbfc57dd25"


@dataclass(frozen=True)
class CoordinateStudy(ViewWeightStudy):
    artifact_cap_bytes: int = 1024**3

    def __post_init__(self):
        super().__post_init__()
        if self.artifact_cap_bytes > 1024**3:
            raise ValueError("Coordinate evidence cap is at most one GiB")


@dataclass(frozen=True)
class MixtureSettings:
    covariance_type: str
    max_k: int = 4
    min_cluster_size: int = 5
    n_init: int = 10
    max_iter: int = 500
    tol: float = 0.001
    reg_covar: float = 1e-6
    selection_criterion: str = "bic"
    random_seed: int = 71323
    max_working_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if self.covariance_type not in {"full", "tied"}:
            raise ValueError("Prespecified full/shared covariance required")
        reference = {**asdict(self), "covariance_type": self.covariance_type}
        expected = dict(
            covariance_type=self.covariance_type,
            max_k=4,
            min_cluster_size=5,
            n_init=10,
            max_iter=500,
            tol=0.001,
            reg_covar=1e-6,
            selection_criterion="bic",
            random_seed=71323,
            max_working_bytes=256 * 1024**2,
        )
        if reference != expected:
            raise ValueError("Diagnostic mixture settings are frozen")

    def to_dict(self):
        return asdict(self)


def model_for(space, seed=0):
    from super_deli.models.hdbscan_config import HDBSCANConfig

    if space not in SPACES:
        raise ValueError("Unknown frozen coordinate space")
    d = 2 if space.startswith("joint2_") else 4
    return HDBSCANConfig(True, feature_names=tuple(f"coordinate_{i + 1}" for i in range(d)))


def tasks(config, phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown coordinate-diagnostic phase")
    cases, sizes = (("four_complementary",), (32,)) if config.fixture else (CASES, (64, 256))
    return [
        dict(case=case, n=n, replicate=rep, key=f"{case}_n{n}_{rep:04d}")
        for case in cases
        for n in sizes
        for rep in range(1 if config.fixture or phase == "preflight" else 5)
    ]


def counts(config, phase):
    n = len(tasks(config, phase))
    return dict(
        observations=n,
        point_workflows=30 * n,
        workflows=30 * n,
        density_fits=10 * n,
        gmm_searches=20 * n,
        neural_fits=0,
        subsample_refits=0,
        legacy_replays=0,
        maximum_gmm_candidates=80 * n,
        maximum_em_starts=800 * n,
        derived_hierarchy_counterfactuals=20 * n,
    )


SETTINGS = dict(
    spaces=SPACES,
    clusterers=CLUSTERERS,
    primary_spaces=PRIMARY_SPACES,
    primary_cases=PRIMARY,
    seeds=SEEDS,
    primary_rates=108,
    paired_comparisons=72,
    geometry_neighbors=[5, 15],
    policies=["baseline", "no_root", "leaf"],
    rank_tolerance=1e-12,
    source_head=SOURCE_HEAD,
    neural_training=0,
    new_datasets=0,
    resampling=0,
    projection="5x audited serial preflight plus 25%; no parallel speed credit",
)
