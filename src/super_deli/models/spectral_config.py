"""Dependency-free contract for point-only, experimental group-count selection."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SpectralConfig:
    assume_independent_units: bool
    feature_names: tuple[str, ...]
    n_neighbors: int
    schema_version: int = 1
    model_family: str = "tabular_spectral_eigengap"
    max_k: int = 6
    min_cluster_size: int = 5
    random_seed: int = 71323
    max_working_bytes: int = 268435456

    def __post_init__(self):
        if type(self.assume_independent_units) is not bool or not self.assume_independent_units:
            raise ValueError("Explicitly assume independent biological units")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported spectral schema version")
        if self.model_family != "tabular_spectral_eigengap":
            raise ValueError("Unsupported spectral model family")
        if (
            not isinstance(self.feature_names, tuple)
            or len(self.feature_names) < 2
            or any(not isinstance(s, str) or not s or s != s.strip() for s in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Explicitly select at least two unique analytical features")
        if type(self.n_neighbors) is not int or self.n_neighbors not in (5, 15):
            raise ValueError("Explicitly choose five or fifteen other neighbors")
        if type(self.max_k) is not int or self.max_k < 2:
            raise ValueError("max_k must be an integer at least two")
        if type(self.min_cluster_size) is not int or self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be an integer at least two")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if type(self.max_working_bytes) is not int or self.max_working_bytes < 1:
            raise ValueError("Positive max_working_bytes required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "assume_independent_units", "feature_names", "n_neighbors"}
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown spectral configuration fields")
        value = dict(value)
        if not isinstance(value["feature_names"], (tuple, list)):
            raise ValueError("feature_names must be an array")
        value["feature_names"] = tuple(value["feature_names"])
        return cls(**value)
