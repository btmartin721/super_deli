"""Approved, immutable Gaussian-reference feasibility grid; no model defaults."""

from super_deli.benchmarking.unknown_k_config import CASES, CONDITIONS, StudyConfig

ARMS = ("spectral_5", "spectral_15", "reference_5", "reference_15", "gmm", "density")
SETTINGS = {
    "schema_version": 1,
    "pilot_draws": 49,
    "reference_draws": 99,
    "alpha": 0.05,
    "sd_guard": 1e-12,
    "gap_tolerance": 1e-8,
    "rank_multiplier": 64,
    "model_seed": 71323,
    "primary_rates": 192,
    "main_roots": [95111, 95117, 95131],
    "preflight_roots": [95143, 95147, 95153],
    "noise_sd": 0.15,
}


def tasks(config, phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown phase")
    roots = SETTINGS[phase + "_roots"]
    cases = ("gaussian", "moons") if config.fixture else CASES
    sizes = (32,) if config.fixture else (64, 256)
    count = 1 if config.fixture or phase == "preflight" else 10
    replay = {
        ("gaussian", sizes[0], "clean"),
        ("moons", sizes[-1], "noisy"),
    }
    if not config.fixture:
        replay |= {("gaussian_8d", 256, "noisy"), ("four_spherical", 64, "clean")}
    return [
        {
            "key": f"{case}_n{n}_{i:04d}",
            "case": case,
            "n": n,
            "replicate": i,
            "data_seed": [roots[0], CASES.index(case), n, i],
            "noise_seed": [roots[1], CASES.index(case), n, i],
            "permutation_seed": [roots[0], CASES.index(case), n, i, 1],
            "draw_root": roots[2],
            "replay_conditions": [c for c in CONDITIONS if i == 0 and (case, n, c) in replay],
        }
        for case in cases
        for n in sizes
        for i in range(count)
    ]


def draw_seed(task, condition, role, index):
    if condition not in CONDITIONS or role not in {"pilot", "reference"}:
        raise ValueError("Undeclared condition/reference role")
    limit = SETTINGS[role + "_draws"]
    if type(index) is not int or not 0 <= index < limit:
        raise ValueError("Reference index outside frozen budget")
    return [
        task["draw_root"],
        CASES.index(task["case"]),
        task["n"],
        task["replicate"],
        CONDITIONS.index(condition),
        0 if role == "pilot" else 1,
        index,
    ]


def counts(config, phase):
    grid = tasks(config, phase)
    tables = 2 * len(grid)
    replays = sum(len(t["replay_conditions"]) for t in grid)
    return {
        "observations": len(grid),
        "tables": tables,
        "table_replays": replays,
        "evaluations": 300 * (tables + replays),
        "point_outputs": 6 * tables,
        "replay_outputs": 6 * replays,
        "embedding_step_bound": 4 * (tables + replays),
    }


__all__ = ["ARMS", "CASES", "CONDITIONS", "SETTINGS", "StudyConfig", "counts", "draw_seed", "tasks"]
