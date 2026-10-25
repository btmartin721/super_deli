"""Frozen raw-data validation of locality-resampling stability, not coverage calibration."""

from dataclasses import asdict, dataclass, replace

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_study_config import PredictiveCell, PredictiveStudyConfig


@dataclass(frozen=True)
class PredictiveBootstrapStudy:
    model: IntegrativeConfig
    screening: PredictiveScreenConfig
    cells: tuple[PredictiveCell, ...]
    n_bootstrap: int = 99
    random_seed: int = 20261212
    max_artifact_bytes: int = 4 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    schema_version: int = 1

    def __post_init__(self):
        PredictiveBootstrapConfig(self.n_bootstrap, self.random_seed, self.schema_version)
        PredictiveBootstrapConfig.validate_model(self.model, self.screening)
        # Reuse the fixed point-study contracts; no external data are generated here.
        PredictiveStudyConfig(
            self.model,
            self.screening,
            self.cells,
            n_test=20,
            random_seed=self.random_seed,
            max_artifact_bytes=self.max_artifact_bytes,
            min_free_bytes=self.min_free_bytes,
        )

    def to_dict(self):
        result = asdict(self)
        result.update(
            model=self.model.to_dict(),
            screening=self.screening.to_dict(),
            cells=[c.to_dict() for c in self.cells],
        )
        return result

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "model", "screening", "cells"})
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        options["screening"] = PredictiveScreenConfig.from_dict(options["screening"])
        options["cells"] = tuple(PredictiveCell.from_dict(c) for c in options["cells"])
        return cls(**options)

    def plan(self):
        return [
            {
                "cell": c.name,
                "cell_config": c.to_dict(),
                "repetition": i,
                "data_seed": derived_seed(self.random_seed, c.name, i, "training_data"),
                "split_seed": derived_seed(self.random_seed, c.name, i, "fold_split"),
                "bootstrap_seed": derived_seed(self.random_seed, c.name, i, "bootstrap"),
                "planned_fits": (1 + self.n_bootstrap) * 6,
            }
            for c in self.cells
            for i in range(c.n_datasets)
        ]

    def probe(self, seed=20261213):
        return replace(
            self, cells=tuple(replace(c, n_datasets=1) for c in self.cells), random_seed=seed
        )
