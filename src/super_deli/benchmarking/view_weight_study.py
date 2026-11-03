"""Legacy view-weight protocol; shared lifecycle preserves its defaults."""

from super_deli.benchmarking import neural_lifecycle as engine
from super_deli.benchmarking.neural_analysis import require
from super_deli.io.config import read_json


def design():
    from super_deli.benchmarking.view_weight_config import (
        ARMS,
        FRACTIONS,
        JOINT,
        ROOTS,
        SEEDS,
        ViewWeightStudy,
        counts,
        model_for,
        tasks,
    )

    return dict(
        config_type=ViewWeightStudy,
        arms=ARMS,
        native_arms=JOINT,
        counts=counts,
        model_for=model_for,
        tasks=tasks,
        audit_replay=audit_replay,
        settings=dict(
            fractions=FRACTIONS,
            neural_seeds=SEEDS,
            rng_roots=ROOTS,
            primary_rates=108,
            paired_comparisons=72,
        ),
        runner="super_deli.benchmarking.view_weight_run",
        auditor="super_deli.benchmarking.view_weight_audit",
    )


def declaration(root, config):
    return engine.declaration(root, config, design())


def freeze(root, config):
    return engine.freeze(root, config, design())


def check_frozen(root):
    return engine.check_frozen(root, design())


def run_phase(root, source, phase, *, workers=1):
    return engine.run_phase(root, source, phase, workers=workers, design=design())


def audit_phase(root, source, phase):
    return engine.audit_phase(root, source, phase, design())


def gate(root, *, staging_bytes):
    return engine.gate(root, staging_bytes=staging_bytes, design=design())


def audit_replay(out, source, entry, cache):
    import numpy as np

    from super_deli.benchmarking.matched_audit import audit_sources
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.neural_moons_coordinates import safe_path
    from super_deli.benchmarking.view_weight_audit import audit_arm

    require(read_json(out / "replay.json") == entry, "Planned replay entry")
    original = safe_path(source, entry["path"])
    data = audit_sources(original, entry["task"])
    task = {**entry["task"], "draws": 0}
    rows = read_json(out / "fits.json")
    require(len(rows) == 1, "One replay workflow")
    row = rows[0]
    require(row["exact_numeric_replay"], "Legacy numeric replay passed")
    folder = out / row["arm"]
    result = audit_arm(folder, task, data, row, cache)
    actual, expected = arrays(folder / "point.npz"), arrays(original / entry["arm"] / "point.npz")
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)
    initial, old = (
        arrays(folder / "optimizer-checks.npz"),
        arrays(original / entry["arm"] / "initial-weights.npz"),
    )
    for name, value in old.items():
        if name.startswith("fit_0_"):
            np.testing.assert_array_equal(
                initial[name.replace("fit_0_", "fit_0_initial_", 1)], value
            )
    return [result]
