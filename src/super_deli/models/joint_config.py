"""Declared block-genotype and conditional continuous-phenotype null settings."""

from dataclasses import dataclass, replace

from super_deli.models.empirical_config import _options, validate_study_dimensions
from super_deli.models.integrative_config import (
    IntegrativeConfig,
    configuration_dict,
    require_calibrated_pca,
)
from super_deli.models.linked_config import LinkedCalibrationConfig

CASES = ("zero_association", "linear_association", "complementary", "heavy_tailed", "cline")


@dataclass(frozen=True)
class JointCalibrationConfig:
    model: IntegrativeConfig
    block_ids: tuple[str, ...]
    predictor_blocks: tuple[str, ...]
    assume_unrelated_specimens: bool
    assume_independent_blocks: bool
    assume_fixed_panel: bool
    assume_prespecified_blocks: bool
    assume_conditional_gaussian: bool
    assume_prespecified_predictors: bool
    n_simulations: int = 99
    alpha: float = 0.05
    random_seed: int = 20261028
    rank_tol: float = 1e-12
    max_archive_bytes: int = 2147483648
    schema_version: int = 1

    def __post_init__(self):
        import math

        if not isinstance(self.model, IntegrativeConfig):
            raise ValueError("An integrative model is required")
        require_calibrated_pca(self.model)
        m = self.model
        genomic = [v for v in m.views if v.kind == "genomic"]
        if (
            len(genomic) != 1
            or not 2 <= len(m.views) <= 3
            or any(v.kind not in {"genomic", "morphological", "ecological"} for v in m.views)
            or m.views[0].kind != "genomic"
            or m.target_level != "specimen"
            or m.missing_views != "error"
            or m.n_bootstrap
            or m.max_k < 2
            or any(
                v.aggregation != "none"
                or v.missing_policy != "error"
                or v.count_transform != "identity"
                for v in m.views
            )
        ):
            raise ValueError(
                "Use genomic-first matched complete continuous views, K>=2, no stability"
            )
        LinkedCalibrationConfig(
            model=replace(genomic[0].genomic_config(), assume_unlinked=False),
            modality=genomic[0].modality,
            block_ids=self.block_ids,
            assume_unrelated_specimens=self.assume_unrelated_specimens,
            assume_independent_blocks=self.assume_independent_blocks,
            assume_fixed_panel=self.assume_fixed_panel,
            assume_prespecified_blocks=self.assume_prespecified_blocks,
            n_simulations=self.n_simulations,
            alpha=self.alpha,
            random_seed=self.random_seed,
            max_working_bytes=m.max_working_bytes,
            max_archive_bytes=self.max_archive_bytes,
            schema_version=self.schema_version,
        )
        if (
            self.assume_conditional_gaussian is not True
            or self.assume_prespecified_predictors is not True
        ):
            raise ValueError("Declare conditional Gaussian residuals and prespecified predictors")
        if (
            not isinstance(self.predictor_blocks, tuple)
            or not self.predictor_blocks
            or any(not isinstance(b, str) or b not in self.block_ids for b in self.predictor_blocks)
            or len(set(self.predictor_blocks)) != len(self.predictor_blocks)
        ):
            raise ValueError("predictor_blocks must be distinct declared block IDs")
        if (
            type(self.rank_tol) not in (int, float)
            or not math.isfinite(self.rank_tol)
            or not 0 < self.rank_tol < 1
        ):
            raise ValueError("rank_tol must be finite and in (0,1)")

    @classmethod
    def from_dict(cls, value):
        o = _options(
            cls,
            value,
            {
                "schema_version",
                "model",
                "block_ids",
                "predictor_blocks",
                "assume_unrelated_specimens",
                "assume_independent_blocks",
                "assume_fixed_panel",
                "assume_prespecified_blocks",
                "assume_conditional_gaussian",
                "assume_prespecified_predictors",
            },
        )
        for name in ("block_ids", "predictor_blocks"):
            if not isinstance(o[name], list):
                raise ValueError(f"{name} must be an array")
            o[name] = tuple(o[name])
        o["model"] = IntegrativeConfig.from_dict(o["model"])
        return cls(**o)

    def to_dict(self):
        return {
            **configuration_dict(self),
            "block_ids": list(self.block_ids),
            "predictor_blocks": list(self.predictor_blocks),
        }

    def genomic_policy(self):
        v = self.model.views[0]
        return LinkedCalibrationConfig(
            replace(v.genomic_config(), assume_unlinked=False),
            self.block_ids,
            self.assume_unrelated_specimens,
            self.assume_independent_blocks,
            self.assume_fixed_panel,
            self.assume_prespecified_blocks,
            modality=v.modality,
            n_simulations=self.n_simulations,
            alpha=self.alpha,
            random_seed=self.random_seed,
            max_working_bytes=self.model.max_working_bytes,
            max_archive_bytes=self.max_archive_bytes,
        )


@dataclass(frozen=True)
class JointStudyConfig:
    calibration: JointCalibrationConfig
    n_datasets: int = 25
    n_units: int = 64
    random_seed: int = 20261028
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.calibration, JointCalibrationConfig):
            raise ValueError("A joint calibration policy is required")
        validate_study_dimensions(self)
        c = self.calibration
        if (
            len(c.block_ids) != 128
            or c.block_ids != tuple(f"block_{j // 8:02d}" for j in range(128))
            or c.predictor_blocks != tuple(f"block_{j:02d}" for j in range(4))
            or tuple(v.kind for v in c.model.views) != ("genomic", "morphological", "ecological")
            or tuple(v.modality for v in c.model.views)
            != ("genomic", "morphological", "ecological")
            or any(v.feature_names for v in c.model.views)
        ):
            raise ValueError(
                "Study uses 128 loci/16 eight-SNP blocks, first four predictors, "
                "three kind-named views"
            )

    @classmethod
    def from_dict(cls, value):
        o = _options(cls, value, {"schema_version", "calibration"})
        o["calibration"] = JointCalibrationConfig.from_dict(o["calibration"])
        return cls(**o)

    def to_dict(self):
        return {**configuration_dict(self), "calibration": self.calibration.to_dict()}
