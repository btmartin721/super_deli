"""Prespecified ablation of PCA fit level and retained joint dimensions."""

from dataclasses import dataclass, replace

from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig, configuration_dict

COHORTS = (
    "archived",
    "fresh_64",
    "fresh_256",
    "independent_gaussian",
    "correlated_gaussian",
    "unequal_variance",
    "cline",
)
ARMS = ("source_2", "source_4", "target_2", "target_4")
PINS = {
    "source_inventory_sha256": "study/files.json",
    "source_summary_sha256": "study/summary.json",
    "source_study_sha256": "study/source_study.json",
    "source_modules_sha256": "runtime_sources.json",
}


@dataclass(frozen=True)
class GroupedPCAStudyConfig:
    model: IntegrativeConfig
    source_inventory_sha256: str
    source_summary_sha256: str
    source_study_sha256: str
    source_modules_sha256: str
    n_datasets: int = 25
    small_n: int = 64
    large_n: int = 256
    n_stability: int = 5
    n_bootstrap: int = 25
    random_seed: int = 20261106
    schema_version: int = 1

    def __post_init__(self):
        for field in PINS:
            v = getattr(self, field)
            if (
                not isinstance(v, str)
                or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v)
            ):
                raise ValueError("Source pins must be lowercase SHA-256 digests")
        for key in ("n_datasets", "small_n", "large_n", "n_bootstrap", "schema_version"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            self.schema_version != 1
            or self.n_datasets > 1000
            or self.small_n < 24
            or self.small_n % 4
            or self.large_n % 4
            or self.large_n <= self.small_n
        ):
            raise ValueError("Use schema 1, 1..1000 datasets and sizes >=24 divisible by four")
        if type(self.n_stability) is not int or not 0 <= self.n_stability <= self.n_datasets:
            raise ValueError("n_stability must be in [0,n_datasets]")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be a uint32 integer")
        m = self.model
        if (
            not isinstance(m, IntegrativeConfig)
            or m.covariance_type != "tied"
            or m.n_pcs != 2
            or m.n_init != 10
            or m.max_k != 4
            or m.min_cluster_size != 5
            or m.n_bootstrap
            or m.n_threads != 1
            or m.target_level not in {"population", "locality"}
            or tuple(v.kind for v in m.views) != ("morphological", "ecological")
            or tuple(v.aggregation for v in m.views) != ("mean", "none")
            or any(v.n_pcs != 2 or v.weight != 1 or v.pca_fit != "source_rows" for v in m.views)
        ):
            raise ValueError("Use the unchanged shared-covariance, two-PC grouped baseline")

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "model", *PINS})
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        return configuration_dict(self)


def study_model(config, cohort, repetition, arm):
    if cohort not in COHORTS or arm not in ARMS or not 0 <= repetition < config.n_datasets:
        raise ValueError("Unknown cohort, arm or repetition")
    m = config.model
    return replace(
        m,
        n_pcs=int(arm[-1]),
        views=(
            replace(
                m.views[0], pca_fit="target_means" if arm.startswith("target") else "source_rows"
            ),
            m.views[1],
        ),
        n_bootstrap=config.n_bootstrap
        if cohort in {"fresh_64", "fresh_256"} and repetition < config.n_stability
        else 0,
    )
