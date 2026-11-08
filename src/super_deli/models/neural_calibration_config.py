"""Explicit raw-Gaussian assumptions for one prespecified separate-encoder model."""

import math
from dataclasses import asdict, dataclass, fields

from super_deli.models.separate_gmm_config import SeparateGMMConfig


@dataclass(frozen=True)
class NeuralCalibrationConfig:
    assume_iid_gaussian_raw_views: bool
    model: SeparateGMMConfig
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 110017
    rank_tol: float = 1e-12
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        if self.assume_iid_gaussian_raw_views is not True or not isinstance(
            self.model, SeparateGMMConfig
        ):
            raise ValueError("Explicit iid joint raw-Gaussian assumption and neural model required")
        if self.model.n_subsamples or self.model.max_k < 2:
            raise ValueError("Calibrate one prespecified K search; keep subsampling separate")
        for name in ("n_simulations", "max_archive_bytes", "schema_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if (
            self.schema_version != 1
            or type(self.random_seed) is not int
            or not (0 <= self.random_seed < 2**64)
        ):
            raise ValueError("Use schema 1 and a uint64 simulation seed")
        for name in ("alpha", "rank_tol"):
            v = getattr(self, name)
            try:
                valid = type(v) in (int, float) and math.isfinite(v) and 0 < v < 1
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(name + " must be finite and strictly between zero and one")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "model", "assume_iid_gaussian_raw_views"}
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown neural calibration fields")
        options = dict(value)
        options["model"] = SeparateGMMConfig.from_dict(options["model"])
        return cls(**options)


def raw_null_policy(config):
    """Adapt feature declarations to the existing raw collector; no PCA is fitted."""
    from types import SimpleNamespace

    from super_deli.models.integrative_config import ViewConfig

    m = config.model
    return SimpleNamespace(
        rank_tol=config.rank_tol,
        model=SimpleNamespace(
            max_working_bytes=m.max_working_bytes,
            min_cluster_size=m.min_cluster_size,
            views=(
                ViewConfig(m.morphology, "morphological", feature_names=m.morphology_features),
                ViewConfig(m.ecology, "ecological", feature_names=m.ecology_features),
            ),
        ),
    )
