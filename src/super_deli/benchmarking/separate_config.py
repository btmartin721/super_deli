"""Prespecified design; workflow calls, trainings and independent data stay distinct."""

from dataclasses import dataclass

from super_deli.benchmarking.view_weight_config import ViewWeightStudy

CASES = (
    "morphology_only",
    "ecology_only",
    "four_complementary",
    "four_redundant",
    "gaussian_independent",
    "gaussian_correlated",
    "discordant",
    "moons",
)
PRIMARY = CASES[:6]
SEEDS = (0, 101, 202)
NATIVE = tuple(f"{family}_{seed}" for family in ("separate", "joint2", "joint4") for seed in SEEDS)
ARMS = (
    NATIVE
    + ("pca4",)
    + tuple(f"{role}_{seed}" for role in ("morphology", "ecology") for seed in SEEDS)
    + ("integrative_gmm",)
)
ROOTS = {"main": [100003, 100019, 100043, 100049], "preflight": [100057, 100069, 100103, 100109]}


@dataclass(frozen=True)
class SeparateStudy(ViewWeightStudy):
    """The same approval ceilings; a distinct scientific design and frozen grid."""


def tasks(config, phase):
    if phase not in ROOTS:
        raise ValueError("Unknown separate-encoder phase")
    cases, sizes = (("morphology_only",), (32,)) if config.fixture else (CASES, (64, 256))
    r = ROOTS[phase]
    return [
        {
            "case": c,
            "n": n,
            "replicate": i,
            "key": f"{c}_n{n}_{i:04d}",
            "data_seed": [r[0], CASES.index(c), n, i],
            "noise_seed": [r[1], CASES.index(c), n, i],
            "permutation_seed": [r[2], CASES.index(c), n, i],
            "ecology_order_seed": [r[2], CASES.index(c), n, i, 1],
            "subsampling_path": [r[3], CASES.index(c), n, i],
            "draws": (1 if config.fixture else 3)
            if i == 0 and c in ("morphology_only", "four_complementary")
            else 0,
        }
        for c in cases
        for n in sizes
        for i in range(1 if config.fixture or phase == "preflight" else 5)
    ]


def counts(config, phase):
    grid = tasks(config, phase)
    observations = len(grid)
    draws = 9 * sum(t["draws"] for t in grid)
    replays = (1 if config.fixture else 12) if phase == "main" else 0
    return {
        "observations": observations,
        "point_workflows": 17 * observations,
        "subsample_refits": draws,
        "legacy_replays": replays,
        "workflows": 17 * observations + draws + replays,
        "neural_fits": 18 * observations + 4 * draws // 3 + replays,
        "density_fits": 16 * observations + draws + replays,
        "gmm_searches": observations,
    }


def model_for(arm, subsampling_seed, draws=0):
    from dataclasses import replace

    from super_deli.benchmarking.matched_data import configuration
    from super_deli.benchmarking.separate_controls import ControlConfig
    from super_deli.models.separate_config import FAMILY, SeparateNeuralConfig

    if arm in NATIVE or arm == "pca4":
        seed = int(arm.rsplit("_", 1)[1]) if arm != "pca4" else 0
        base = configuration("paired_" + str(seed), subsampling_seed, draws if arm in NATIVE else 0)
        if arm.startswith("joint2_"):
            return replace(base, schema_version=2, view_loss_weights=(0.5, 0.5))
        if arm.startswith("separate_"):
            return SeparateNeuralConfig.from_dict({**base.to_dict(), "model_family": FAMILY})
        family = (
            "benchmark_joint_four_autoencoder_hdbscan"
            if arm.startswith("joint4_")
            else "benchmark_per_view_pca_hdbscan"
        )
        return ControlConfig.from_dict({**base.to_dict(), "model_family": family})
    return configuration(arm, subsampling_seed, 0)
