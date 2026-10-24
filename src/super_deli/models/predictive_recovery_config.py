"""Frozen diagnostic design; no changes to any production model configuration."""

from dataclasses import dataclass, replace

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.predictive_study_config import PredictiveStudyConfig


@dataclass(frozen=True)
class PredictiveRecoveryConfig:
    study: PredictiveStudyConfig
    partitions: int = 5
    extended_starts: int = 100
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported recovery diagnostic schema")
        if not isinstance(self.study, PredictiveStudyConfig):
            raise ValueError("Explicit frozen predictive study required")
        if self.study.screening.n_folds != 5:
            raise ValueError("Five whole-locality folds required")
        if type(self.partitions) is not int or not 2 <= self.partitions <= 5:
            raise ValueError("Use two to five prespecified partitions")
        if (
            type(self.extended_starts) is not int
            or not self.study.model.n_init < self.extended_starts <= 100
        ):
            raise ValueError("Extended starts must exceed baseline, at most 100")
        if any(c.law not in {"complementary", "independent_gaussian"} for c in self.study.cells):
            raise ValueError(
                "Fresh diagnostic supports complementary and independent Gaussian laws"
            )

    def to_dict(self):
        return {
            "schema_version": 1,
            "study": self.study.to_dict(),
            "partitions": self.partitions,
            "extended_starts": self.extended_starts,
        }

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"schema_version", "study"})
        options["study"] = PredictiveStudyConfig.from_dict(options["study"])
        return cls(**options)

    def plan(self):
        return [
            {
                **{k: v for k, v in r.items() if k != "planned_fits"},
                "partition_seeds": [r["split_seed"]]
                + [
                    derived_seed(
                        self.study.random_seed, r["cell"], r["repetition"], f"extra_split_{j}"
                    )
                    for j in range(1, self.partitions)
                ],
                "planned_searches": 1 + 5 * self.partitions + 6,
                "planned_oracles": 6 if r["cell_config"]["role"] == "alternative" else 0,
            }
            for r in self.study.plan()
        ]

    def probe(self, seed=20261202):
        return replace(self, study=self.study.probe(seed))
