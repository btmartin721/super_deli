"""Independent opt-in contract for paired encoders and a shared Gaussian mixture."""

from dataclasses import dataclass, fields

from super_deli.models.separate_config import FAMILY as DENSITY_FAMILY
from super_deli.models.separate_config import SeparateNeuralConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig

FAMILY = "matched_separate_autoencoders_shared_gmm"


@dataclass(frozen=True)
class SeparateGMMConfig(SeparateNeuralConfig):
    model_family: str = FAMILY
    max_k: int = 4
    min_cluster_size: int = 5
    n_init: int = 10
    max_iter: int = 500
    tol: float = 0.001
    reg_covar: float = 1e-6
    mixture_seed: int = 71323

    def __post_init__(self):
        if self.model_family != FAMILY:
            raise ValueError("Unsupported shared-mixture encoder family")
        self.encoder_config()
        # Reuse the established four-coordinate GMM validation, without PCA.
        TabularClusterConfig(
            assume_independent_units=self.assume_independent_units,
            n_pcs=4,
            max_k=self.max_k,
            min_cluster_size=self.min_cluster_size,
            n_bootstrap=0,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_seed=self.mixture_seed,
            max_working_bytes=self.max_working_bytes,
        )

    def encoder_config(self):
        value = {f.name: getattr(self, f.name) for f in fields(SeparateNeuralConfig)}
        return SeparateNeuralConfig(**{**value, "model_family": DENSITY_FAMILY})
