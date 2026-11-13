"""Disjoint new diagnostic roles around unchanged archived measurement laws."""

import numpy as np

from super_deli.benchmarking.abstention_data import generate as original_generate
from super_deli.benchmarking.abstention_data import parameters
from super_deli.benchmarking.observed_feasibility_config import roles, tasks


def generator_task(task, role):
    definitions = roles(task)
    names = [r["name"] for r in definitions]
    if role not in names:
        raise ValueError("Unknown diagnostic role")
    spec = definitions[names.index(role)]
    return dict(
        task,
        key=task["key"] + "_" + role,
        n=spec["n"],
        seed_path=[*task["seed_path"], names.index(role)],
    ), spec


def generate_role(task, role):
    source, spec = generator_task(task, role)
    data = original_generate(source, spec["generator_cohort"])
    if len(data["truth"]) != spec["n"]:
        raise ValueError("Frozen role size")
    return data


def measured(data):
    x = np.column_stack((data["morphology"], data["ecology"]))
    if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all():
        raise ValueError("Twelve finite measured traits required")
    with np.errstate(over="raise", invalid="raise"):
        native = x.astype(np.float32)
    if not np.isfinite(native).all():
        raise ValueError("Native float32 overflow")
    return x


def role_ids(task, name):
    _, spec = generator_task(task, name)
    # Phase/domain/role prefixes keep reused original heads separate from new specimens.
    tag = "observed750_" + "_".join(map(str, task["seed_path"])) + "_" + name
    return [f"{tag}_{i:05d}" for i in range(spec["n"])]


def seed_registry(config):
    records = []
    paths = []
    for phase in ("preflight", "main"):
        for task in tasks(config, phase):
            for role in roles(task):
                source, spec = generator_task(task, role["name"])
                p = parameters(source, spec["generator_cohort"])
                seeds = {k: v for k, v in p.items() if k.endswith("_seed")}
                records.append(dict(phase=phase, key=task["key"], role=spec, n=p["n"], seeds=seeds))
                paths.extend(tuple(v) for v in seeds.values())
    if len(paths) != len(set(paths)) or any(p[:2] != (20261112, 750) for p in paths):
        raise ValueError("Disjoint diagnostic seed paths")
    return dict(
        passed=True,
        materialized_before_generation=True,
        cohorts=len(records),
        rows=sum(r["n"] for r in records),
        seed_paths=len(paths),
        records=records,
    )
