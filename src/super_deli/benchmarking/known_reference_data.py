"""Independent parent cohorts and explicit whole-group exclusions before fitting."""

import numpy as np

from super_deli.benchmarking.abstention_data import generate as original_generate
from super_deli.benchmarking.abstention_data import parameters
from super_deli.benchmarking.known_reference_config import roles, tasks


def source_task(task, role):
    names = [k for k, _ in roles(task)]
    if role not in names:
        raise ValueError("Unknown parent cohort role")
    return {**task, "seed_path": [*task["seed_path"], names.index(role)]}


def generate(task, role):
    return original_generate(source_task(task, role), "iid" if role == "overlap_pool" else role)


def matrix(data):
    return np.column_stack((data["morphology"], data["ecology"]))


def indices(data, heldout, *, unknown=False, overlap=False):
    if type(heldout) is not int or heldout not in range(4):
        raise ValueError("One original group 0..3 must be held out")
    y = np.asarray(data["truth"])
    known = [c for c in range(4) if c != heldout]
    if overlap:
        return np.flatnonzero(y == known[0])
    return np.flatnonzero(y == heldout if unknown else y != heldout)


def subset(data, rows, known):
    # Only measurement columns and explicit retained labels reach classifier I/O.
    y = np.asarray(data["truth"])[rows]
    mapping = {c: j for j, c in enumerate(known)}
    return dict(
        morphology=np.asarray(data["morphology"])[rows].copy(),
        ecology=np.asarray(data["ecology"])[rows].copy(),
        ecology_order=np.arange(len(rows) - 1, -1, -1),
        truth=np.asarray([mapping[int(c)] for c in y], dtype=int),
    )


def training_data(parent, heldout):
    known = [c for c in range(4) if c != heldout]
    train_rows = indices(parent["train"], heldout)
    cal_rows = indices(parent["calibration"], heldout)
    train = subset(parent["train"], train_rows, known)
    cal = subset(parent["calibration"], cal_rows, known)
    selection = dict(train=train_rows.tolist(), calibration=cal_rows.tolist())
    return dict(known=known, train=train, calibration=cal, selection=selection)


def fold_data(parent, heldout):
    fitted = training_data(parent, heldout)
    known, train, cal, selection = (
        fitted["known"],
        fitted["train"],
        fitted["calibration"],
        fitted["selection"],
    )
    cohorts = {}
    for role in [
        "iid",
        "heldout",
        "near",
        "far",
        "overlap",
        *(["shifted"] if "shifted" in parent else []),
    ]:
        source = "iid" if role == "heldout" else "overlap_pool" if role == "overlap" else role
        data = parent[source]
        rows = (
            indices(data, heldout, unknown=role == "heldout", overlap=role == "overlap")
            if role in ("iid", "heldout", "overlap", "shifted")
            else np.arange(len(data["truth"]))
        )
        is_known = role in ("iid", "shifted")
        y = subset(data, rows, known)["truth"] if is_known else np.full(len(rows), 3, dtype=int)
        cohorts[role] = dict(
            x=matrix(data)[rows].copy(),
            y=y,
            known=is_known,
            parent_role=source,
            parent_rows=rows.tolist(),
        )
    return dict(
        heldout=heldout,
        known_original_groups=known,
        training=train,
        calibration=cal,
        selected_training_rows=selection,
        cohorts=cohorts,
        original_to_reference={str(c): j for j, c in enumerate(known)},
        overlap_matching_reference=0,
    )


def seed_registry(config):
    records = []
    seen = set()
    for phase in ("preflight", "main"):
        for task in tasks(config, phase):
            for role, n in roles(task):
                p = parameters(source_task(task, role), "iid" if role == "overlap_pool" else role)
                if p["n"] != n:
                    raise ValueError("Frozen parent cohort size differs from original generator")
                seeds = {k: v for k, v in p.items() if k.endswith("_seed")}
                for value in seeds.values():
                    key = tuple(value)
                    if key in seen:
                        raise ValueError("Overlapping parent innovations")
                    seen.add(key)
                records.append(dict(phase=phase, key=task["key"], role=role, n=n, seeds=seeds))
    return dict(
        passed=True,
        materialized_before_generation=True,
        cohorts=len(records),
        rows=sum(r["n"] for r in records),
        seed_paths=len(seen),
        records=records,
    )
