"""Frozen point-only Gaussian-control diagnostic; no calibration policy."""

from __future__ import annotations

from dataclasses import dataclass, replace

from super_deli.benchmarking.config import derived_seed
from super_deli.models.empirical_config import _options
from super_deli.models.integrative_config import IntegrativeConfig

CASES = ("independent_gaussian", "correlated_gaussian", "zero_conditional_gaussian")
ARMS = ("baseline", "homogeneous_mean_noise", "starts_100")


@dataclass(frozen=True)
class AICDiagnosticConfig:
    model: IntegrativeConfig
    n_datasets: int = 50
    n_scores: int = 150
    n_test: int = 2048
    sizes: tuple[int, ...] = (64, 256)
    cases: tuple[str, ...] = CASES
    long_n_init: int = 100
    random_seed: int = 20261124
    max_artifact_bytes: int = 1024**3
    min_free_bytes: int = 8 * 1024**3
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported AIC diagnostic schema")
        if not isinstance(self.model, IntegrativeConfig):
            raise ValueError("An integrative point model is required")
        m = self.model
        if (
            m.covariance_type != "full"
            or m.selection_criterion != "aic"
            or m.n_bootstrap != 0
            or m.n_threads != 1
            or m.n_pcs != 4
            or m.max_k != 4
            or m.target_level != "locality"
            or len(m.views) != 2
            or tuple(v.kind for v in m.views) != ("morphological", "ecological")
            or any(v.n_pcs != 2 or v.weight != 1 for v in m.views)
            or m.views[0].pca_fit != "target_means"
        ):
            raise ValueError("Use the prespecified four-PC full-covariance grouped AIC point model")
        for key in ("n_datasets", "n_scores", "n_test", "long_n_init"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_test < 20 or self.n_test % 4 or self.long_n_init < m.n_init:
            raise ValueError("Use >=20 held-out units and a nondecreasing diagnostic start budget")
        if (
            not isinstance(self.sizes, tuple)
            or not self.sizes
            or any(type(v) is not int for v in self.sizes)
            or len(set(self.sizes)) != len(self.sizes)
        ):
            raise ValueError("Declare distinct sample sizes")
        if any(type(n) is not int or n < 20 or n % 4 for n in self.sizes):
            raise ValueError("Sample sizes must be >=20 and divisible by four")
        if (
            not isinstance(self.cases, tuple)
            or not self.cases
            or any(not isinstance(v, str) for v in self.cases)
            or len(set(self.cases)) != len(self.cases)
        ):
            raise ValueError("Declare distinct Gaussian laws")
        if any(c not in CASES for c in self.cases):
            raise ValueError("Unknown diagnostic Gaussian law")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("Use an unsigned 32-bit study seed")
        if any(
            type(v) is not int or v < 1024**2
            for v in (self.max_artifact_bytes, self.min_free_bytes)
        ):
            raise ValueError("Explicit storage byte budgets must be >=1 MiB")

        if self.n_test * 1024 > m.max_working_bytes:
            raise ValueError("Held-out prediction exceeds working-memory budget")

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            **{
                k: getattr(self, k)
                for k in (
                    "n_datasets",
                    "n_scores",
                    "n_test",
                    "sizes",
                    "cases",
                    "long_n_init",
                    "random_seed",
                    "max_artifact_bytes",
                    "min_free_bytes",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        options = _options(cls, value, {"model", "schema_version"})
        options["model"] = IntegrativeConfig.from_dict(options["model"])
        for key in ("cases", "sizes"):
            if key in options:
                if not isinstance(options[key], (list, tuple)):
                    raise ValueError(f"{key} must be a sequence")
                options[key] = tuple(options[key])
        return cls(**options)

    def arm(self, name):
        if name not in ARMS:
            raise ValueError("Unknown diagnostic arm")
        return replace(self.model, n_init=self.long_n_init) if name == "starts_100" else self.model

    def plan(self):
        rows = []
        for case in (*self.cases, "gaussian_scores"):
            for n in self.sizes:
                cell = f"{case}_n{n}"
                for i in range(self.n_scores if case == "gaussian_scores" else self.n_datasets):
                    rows.append(
                        {
                            "cell": cell,
                            "case": case,
                            "n": n,
                            "repetition": i,
                            "data_seed": derived_seed(self.random_seed, cell, i, "training_data"),
                            "test_seed": derived_seed(self.random_seed, cell, i, "heldout_data"),
                            "planned_fits": 1 if case == "gaussian_scores" else len(ARMS),
                        }
                    )
        return rows
