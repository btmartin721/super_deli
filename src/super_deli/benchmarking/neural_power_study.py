"""Additive fixed power study built on the reusable sealed-calibration lifecycle."""

from super_deli.benchmarking.neural_calibration_config import model_for
from super_deli.benchmarking.neural_power_config import (
    ARMS,
    NAMESPACE,
    NeuralPowerStudy,
    archived_jobs,
    counts,
    tasks,
)
from super_deli.models.separate_neural import SEPARATE_TRAINING


def design():
    return dict(
        config_type=NeuralPowerStudy,
        arms=ARMS,
        native_arms=("neural",),
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        training=SEPARATE_TRAINING,
        settings=dict(
            rng_namespace=NAMESPACE,
            neural_seed=0,
            mixture_seed=71323,
            raw_features=[4, 8],
            n_simulations=99,
            family="tied",
            alpha=0.05,
            primary_rates=36,
            paired_comparisons=18,
            recovery="K=4 AND ARI>=0.8",
            signal_multipliers=[0.5, 1.0, 1.5],
            sample_sizes=[64, 256],
            missingness="error",
            stability="disabled",
            interpretation="Descriptive operating range; not selected-K/species significance",
            statistic="Maximum eligible BIC improvement over K=1",
        ),
    )


def jobs(config, phase, binding):
    result = [dict(kind="fresh", task=t) for t in tasks(config, phase)]
    return result + (archived_jobs(binding) if phase == "main" else [])


def engine(root, source):
    from super_deli.benchmarking.neural_power_audit import audit_unit
    from super_deli.benchmarking.neural_power_run import run_unit
    from super_deli.benchmarking.sealed_calibration_study import SealedCalibrationStudy

    return SealedCalibrationStudy(
        root, source, design(), jobs=jobs, runner=run_unit, auditor=audit_unit
    )


def check_frozen(root):
    from super_deli.benchmarking.neural_lifecycle import check_frozen as check

    return check(root, design())
