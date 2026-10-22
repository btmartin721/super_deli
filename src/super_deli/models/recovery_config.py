"""Frozen diagnostic grid; no clustering defaults or calibration policies change."""

from dataclasses import asdict, dataclass, replace

from super_deli.models.empirical_config import _options

VARIANTS = ("baseline", "pcs_3", "pcs_4", "ecology_0.5", "ecology_2", "ecology_4", "starts_100")
COHORTS = ("archived", "fresh_64", "fresh_256")


@dataclass(frozen=True)
class RecoveryConfig:
    source_study_sha256: str
    source_inventory_sha256: str
    source_modules_sha256: str
    n_datasets: int = 25
    small_n: int = 64
    large_n: int = 256
    random_seed: int = 20261029
    schema_version: int = 1

    def __post_init__(self):
        for name in ("source_study_sha256", "source_inventory_sha256", "source_modules_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Source fingerprints must be lowercase SHA-256 digests")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported recovery diagnostic schema")
        if type(self.n_datasets) is not int or not 1 <= self.n_datasets <= 1000:
            raise ValueError("n_datasets must be in [1,1000]")
        for n in (self.small_n, self.large_n):
            if type(n) is not int or n < 24 or n % 4:
                raise ValueError("Cohort sizes must be >=24 and divisible by four")
        if self.large_n <= self.small_n:
            raise ValueError("large_n must exceed small_n")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an unsigned 32-bit integer")

    @classmethod
    def from_dict(cls, value):
        return cls(
            **_options(
                cls,
                value,
                {
                    "schema_version",
                    "source_study_sha256",
                    "source_inventory_sha256",
                    "source_modules_sha256",
                },
            )
        )

    def to_dict(self):
        return asdict(self)


def variants(model):
    if (
        model.n_bootstrap
        or model.n_pcs != 2
        or model.max_k != 4
        or model.min_cluster_size != 5
        or tuple(v.kind for v in model.views) != ("genomic", "morphological", "ecological")
        or any(v.n_pcs != 2 or v.weight != 1 for v in model.views)
    ):
        raise ValueError(
            "Diagnostic source requires the declared three-view, two-PC, K=1..4 baseline"
        )
    result = {"baseline": model, "pcs_3": replace(model, n_pcs=3), "pcs_4": replace(model, n_pcs=4)}
    for weight in (0.5, 2, 4):
        result[f"ecology_{weight}"] = replace(
            model,
            views=tuple(
                replace(v, weight=weight) if v.kind == "ecological" else v for v in model.views
            ),
        )
    result["starts_100"] = replace(model, n_init=100)
    return result
