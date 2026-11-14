"""Dependency-free, separately versioned complete-genotype neural controls."""

from dataclasses import asdict, dataclass, fields

from super_deli.models.config import BaselineConfig

TRAINING = dict(
    encoder=[16, 8, 2],
    decoder=[8, 16],
    output="three logits per retained locus",
    activation="tanh",
    loss="mean per-locus categorical cross entropy",
    updates=500,
    learning_rate=0.001,
    weight_decay=0.0001,
    dtype="float64",
    device="cpu",
)


@dataclass(frozen=True)
class GenomicNeuralConfig:
    ploidy: int
    missing_policy: str
    assume_independent_specimens: bool
    schema_version: int = 1
    n_pcs: int = 2
    max_k: int = 4
    min_cluster_size: int = 3
    n_init: int = 10
    max_iter: int = 500
    tol: float = 0.001
    reg_covar: float = 1e-6
    random_seed: int = 0
    mixture_seed: int = 71323
    encoding_seed: int = 0
    subsampling_seed: int = 0
    n_subsamples: int = 0
    max_loci: int = 512
    max_working_bytes: int = 268435456
    max_pairwise_bytes: int = 268435456

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.n_pcs != 2
            or type(self.n_pcs) is not int
        ):
            raise ValueError("Genomic neural schema 1 has two fixed coordinates")
        if self.missing_policy != "error":
            raise ValueError("Complete called genotypes required; no imputation in this workflow")
        self.baseline_config()
        if self.min_cluster_size < 3:
            raise ValueError("Two-coordinate shared mixtures require minimum size at least three")
        for name in ("random_seed", "mixture_seed", "encoding_seed", "subsampling_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError("Seeds must be uint32 integers")
        if type(self.n_subsamples) is not int or not 0 <= self.n_subsamples <= 10000:
            raise ValueError("n_subsamples must be an integer from zero to 10000")
        if type(self.max_loci) is not int or not 2 <= self.max_loci <= 4096:
            raise ValueError("Declare an original-panel limit between 2 and 4096 loci")
        if type(self.max_working_bytes) is not int or self.max_working_bytes < 1:
            raise ValueError("Positive max_working_bytes required")

    def baseline_config(self):
        return BaselineConfig(
            self.ploidy,
            self.missing_policy,
            self.assume_independent_specimens,
            n_pcs=self.n_pcs,
            max_k=self.max_k,
            min_cluster_size=self.min_cluster_size,
            scaling="patterson",
            n_bootstrap=0,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            random_seed=self.encoding_seed,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "ploidy", "missing_policy", "assume_independent_specimens"}
        if (
            not isinstance(value, dict)
            or required - value.keys()
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Missing or unknown genomic neural configuration fields")
        return cls(**value)


@dataclass(frozen=True)
class GenomicComparisonConfig:
    model: GenomicNeuralConfig
    neural_seeds: tuple[int, ...] = (0, 1, 2)
    schema_version: int = 1

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.model) is not GenomicNeuralConfig
        ):
            raise ValueError("Unsupported genomic comparison schema/model")
        GenomicNeuralConfig.from_dict(self.model.to_dict())
        if (
            not isinstance(self.neural_seeds, tuple)
            or not 1 <= len(self.neural_seeds) <= 16
            or any(type(s) is not int or not 0 <= s < 2**32 for s in self.neural_seeds)
            or len(set(self.neural_seeds)) != len(self.neural_seeds)
        ):
            raise ValueError("Prespecify one to sixteen distinct neural uint32 seeds")

    def to_dict(self):
        return dict(
            schema_version=1, model=self.model.to_dict(), neural_seeds=list(self.neural_seeds)
        )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "model", "neural_seeds"}
            or not isinstance(value["neural_seeds"], (list, tuple))
        ):
            raise ValueError("Comparison requires schema_version, model and neural_seeds")
        return cls(
            GenomicNeuralConfig.from_dict(value["model"]),
            tuple(value["neural_seeds"]),
            value["schema_version"],
        )
