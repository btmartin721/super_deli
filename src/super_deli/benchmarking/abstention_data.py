"""Fresh independent known, unfamiliar and indistinguishable measurement cohorts."""

import numpy as np

from super_deli.benchmarking.abstention_config import COHORTS
from super_deli.benchmarking.matched_config import SETTINGS
from super_deli.benchmarking.neural_power_data import generate as original_generate


def parameters(task, cohort):
    if cohort not in COHORTS:
        raise ValueError("Unknown reference-evaluation cohort")
    n = (
        task["n"]
        if cohort == "train"
        else (80 if task["fixture"] else 160)
        if cohort == "calibration"
        else (64 if task["fixture"] else 512)
        if cohort in ("near", "far", "overlap")
        else (64 if task["fixture"] else 2048)
    )
    path = [*task["seed_path"], COHORTS.index(cohort)]
    return dict(
        case="four_complementary",
        n=n,
        signal_multiplier=0.5 if task["case"] in ("linear_weak", "confounded_size") else 1.0,
        **{
            name: path + [i]
            for i, name in enumerate(
                ("data_seed", "noise_seed", "permutation_seed", "ecology_order_seed", "size_seed")
            )
        },
    )


def generate(task, cohort):
    config = parameters(task, cohort)
    d = {
        k: v.copy()
        for k, v in original_generate(config).items()
        if k not in ("morphology_truth", "ecology_truth")
    }
    original_truth = d["truth"].copy()
    if cohort in ("near", "far", "overlap"):
        a = 3 * config["signal_multiplier"]
        target = (a, 0) if cohort == "near" else (6, 6) if cohort == "far" else (a, a)
        for role, axis, bit, sl in (
            ("morphology", 0, original_truth // 2, slice(0, 4)),
            ("ecology", 1, original_truth % 2, slice(4, 12)),
        ):
            d["latent_" + role][:, axis] += target[axis] - np.where(bit == 0, -a, a)
            load = np.asarray(SETTINGS["loading_" + role], dtype=float)
            load /= np.linalg.norm(load, axis=1)[:, None]
            d[role] = d["latent_" + role] @ load.T + d["noise"][:, sl]
        d["truth"] = np.full(config["n"], 4, dtype=int)
    d["base_morphology"], d["base_ecology"] = d["morphology"].copy(), d["ecology"].copy()
    size = np.random.default_rng(config["size_seed"]).normal(0, 0.5, config["n"])[d["order"]]
    if task["case"] == "confounded_size" and cohort != "shifted":
        size += 0.8 * (
            np.ones(config["n"])
            if cohort in ("near", "far", "overlap")
            else np.where(d["truth"] // 2 == 0, -1, 1)
        )
    d["size"] = size
    if task["case"] in ("combined", "confounded_size"):
        d["morphology"] = np.exp(
            0.35 * d["morphology"] + size[:, None] * np.array([0.8, 1.0, 1.2, 1.4])
        )
    if task["case"] == "combined":
        d["ecology"] = 20 + 5 * np.tanh(d["ecology"] / 2)
    if task["case"] == "confounded_size":
        d["ecology"] += 0.4 * size[:, None] * np.array([1.0, -1.0] * 4)
    return d


def seed_registry(config):
    from super_deli.benchmarking.abstention_config import cohort_names, tasks

    rows, paths = [], []
    for phase in ("preflight", "main"):
        for task in tasks(config, phase):
            for cohort in cohort_names(task):
                p = parameters(task, cohort)
                seeds = {k: v for k, v in p.items() if k.endswith("_seed")}
                rows.append(
                    dict(phase=phase, key=task["key"], cohort=cohort, n=p["n"], seeds=seeds)
                )
                paths.extend(tuple(v) for v in seeds.values())
    if len(paths) != len(set(paths)):
        raise ValueError("Seed domains overlap")
    return dict(
        passed=True,
        materialized_before_generation=True,
        seed_paths=len(paths),
        cohorts=len(rows),
        records=rows,
    )
