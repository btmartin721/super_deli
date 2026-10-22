"""Frozen paired validation of full and shared-covariance integrative models."""

from dataclasses import dataclass

from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig, configuration_dict

COHORTS = ("archived", "fresh_64", "fresh_256", "zero_association", "unequal_variance", "cline")
FAMILIES = ("full", "tied")


@dataclass(frozen=True)
class SharedStudyConfig:
    model: IntegrativeConfig
    source_inventory_sha256: str
    source_summary_sha256: str
    source_study_sha256: str
    n_datasets: int = 25
    small_n: int = 64
    large_n: int = 256
    stability_datasets: int = 5
    stability_draws: int = 25
    random_seed: int = 20261030
    schema_version: int = 1

    def __post_init__(self):
        for name in ("source_inventory_sha256", "source_summary_sha256", "source_study_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Source pins must be lowercase SHA-256 digests")
        for name in ("n_datasets", "small_n", "large_n", "stability_datasets", "stability_draws"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.small_n < 24
            or self.small_n % 4
            or self.large_n <= self.small_n
            or self.large_n % 4
            or self.stability_datasets > self.n_datasets
        ):
            raise ValueError(
                "Use >=24 units divisible by four, a larger second cohort "
                "and a bounded stability subset"
            )
        if (
            type(self.random_seed) is not int
            or not 0 <= self.random_seed < 2**32
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("Use schema one and an unsigned 32-bit seed")
        m = self.model
        if (
            not isinstance(m, IntegrativeConfig)
            or m.covariance_type != "full"
            or m.n_bootstrap
            or m.n_pcs != 2
            or m.max_k != 4
            or m.min_cluster_size != 5
            or m.target_level != "specimen"
            or m.missing_views != "error"
            or tuple(v.kind for v in m.views) != ("genomic", "morphological", "ecological")
            or any(v.n_pcs != 2 or v.weight != 1 for v in m.views)
        ):
            raise ValueError(
                "Use the frozen matched, equal-weight, two-PC full-covariance baseline"
            )

    @classmethod
    def from_dict(cls, value):
        options = _options(
            cls,
            value,
            {
                "schema_version",
                "model",
                "source_inventory_sha256",
                "source_summary_sha256",
                "source_study_sha256",
            },
        )
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        return cls(**options)

    def to_dict(self):
        return configuration_dict(self)
