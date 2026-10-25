"""Prespecified mechanistic comparisons, separate from production bootstrap settings."""

from dataclasses import asdict, dataclass, replace

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_bootstrap_study import PredictiveBootstrapStudy
from super_deli.models.predictive_config import PredictiveScreenConfig
from super_deli.models.predictive_study_config import PredictiveCell


@dataclass(frozen=True)
class BootstrapDiagnosticConfig:
    model: IntegrativeConfig
    screening: PredictiveScreenConfig
    cells: tuple[PredictiveCell, ...]
    n_draws: int = 25
    extended_draws: int = 5
    extended_starts: int = 100
    random_seed: int = 20261220
    max_artifact_bytes: int = 3 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    schema_version: int = 1

    def __post_init__(self):
        PredictiveBootstrapStudy(
            self.model,
            self.screening,
            self.cells,
            self.n_draws,
            self.random_seed,
            self.max_artifact_bytes,
            self.min_free_bytes,
            self.schema_version,
        )
        if (
            type(self.extended_draws) is not int
            or not 1 <= self.extended_draws <= self.n_draws
            or type(self.extended_starts) is not int
            or not self.model.n_init < self.extended_starts <= 100
        ):
            raise ValueError("Prespecify more starts and a bounded paired draw prefix")
        if any(c.law not in {"complementary", "independent_gaussian"} for c in self.cells):
            raise ValueError("Fresh diagnostic supports four-group and Gaussian-control laws")

    def to_dict(self):
        d = asdict(self)
        d.update(
            model=self.model.to_dict(),
            screening=self.screening.to_dict(),
            cells=[c.to_dict() for c in self.cells],
        )
        return d

    @classmethod
    def from_dict(cls, value):
        d = _options(cls, value, {"schema_version", "model", "screening", "cells"})
        d["model"] = IntegrativeConfig.from_dict(d["model"])
        d["screening"] = PredictiveScreenConfig.from_dict(d["screening"])
        d["cells"] = tuple(PredictiveCell.from_dict(c) for c in d["cells"])
        return cls(**d)

    def plan(self):
        return [
            dict(
                cell=c.name,
                cell_config=c.to_dict(),
                repetition=i,
                data_seed=derived_seed(self.random_seed, c.name, i, "training_data"),
                split_seed=derived_seed(self.random_seed, c.name, i, "fold_split"),
                draw_stream_seed=derived_seed(self.random_seed, c.name, i, "bootstrap"),
                planned_searches=6 * (1 + 2 * self.n_draws + 1 + self.extended_draws),
                planned_oracles=6 * (1 + self.extended_draws) if c.role == "alternative" else 0,
            )
            for c in self.cells
            for i in range(c.n_datasets)
        ]

    def draw_seeds(self, declaration):
        return PredictiveBootstrapConfig(self.n_draws, declaration["draw_stream_seed"]).draw_seeds()

    def probe(self, seed=20261221):
        return replace(
            self, cells=tuple(replace(c, n_datasets=1) for c in self.cells), random_seed=seed
        )
