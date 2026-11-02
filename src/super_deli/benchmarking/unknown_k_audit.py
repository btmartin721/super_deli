"""Fit-free inventory, input, graph, baseline and exact-replay verification."""

import csv
import time
from collections import Counter

import numpy as np

from super_deli.benchmarking.neural_analysis import arrays, audit_point, read_raw, require
from super_deli.benchmarking.unknown_k_config import ARMS, CONDITIONS, tasks
from super_deli.benchmarking.unknown_k_data import model_config
from super_deli.benchmarking.unknown_k_graph_audit import (
    audit_embedding,
    audit_geometry,
    audit_spectral,
)
from super_deli.benchmarking.unknown_k_reference import audit_candidates, reference
from super_deli.benchmarking.unknown_k_study import check_frozen
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def audit_phase(root, phase):
    start = time.monotonic()
    f, config = check_frozen(root)
    expected = read_json(root / f"{phase}-inventory.json")
    actual = {
        str(p.relative_to(root / phase)): fingerprint_file(p).sha256
        for p in (root / phase).rglob("*")
        if p.is_file()
    }
    require(actual == expected, "Complete result inventory")
    statuses = Counter()
    total = replays = 0
    for task in tasks(config, phase):
        out = root / phase / task["key"]
        require(read_json(out / "task.json") == task, "Frozen source task")
        ref = reference(task)
        stored = arrays(out / "generating-reference.npz")
        require(set(ref) == set(stored), "Source inventory")
        for key, value in ref.items():
            np.testing.assert_allclose(value, stored[key], atol=1e-11)
        records = read_json(out / "fits.json")
        require(len(records) == 8 * (1 + task["replay"]), "Frozen workflow budget")
        observed_keys = []
        for row in records:
            condition, arm, kind = row["condition"], row["arm"], row["kind"]
            observed_keys.append((condition, arm, kind))
            folder = out / condition / (arm + ("_replay" if kind == "replay" else ""))
            ids, x = read_raw(out / condition / "raw")
            require(
                ids == [f"invented_{i:04d}" for i in range(task["n"])], "Original source identities"
            )
            np.testing.assert_allclose(x, ref[condition], atol=1e-11)
            summary = (
                read_json(folder / "summary.json") if (folder / "summary.json").exists() else None
            )
            if row["status"] == "failed":
                error = row.get("error") or (summary or {}).get("error")
                require(bool(error), "Retained numerical/workflow failure")
                require(
                    not (folder / "assignments.csv").exists(), "Failure cannot export assignments"
                )
            elif arm.startswith("spectral"):
                require(
                    summary["status"] == row["status"]
                    and summary["selected_k"] == row["selected_k"],
                    "Spectral status export",
                )
                require(
                    summary["provenance"]["configuration"]["model"] == f["arms"][arm],
                    "Frozen spectral model",
                )
                a = arrays(folder / "point.npz")
                audit_spectral(x, ids, model_config(arm), summary, a)
                require(
                    summary["inference"]["p_value"] is None
                    and summary["stability"]["status"] == "not_computed",
                    "Point-only contract",
                )
                assignment = folder / "assignments.csv"
                require(
                    assignment.exists() == (row["status"] == "available"), "Assignment availability"
                )
                if assignment.exists():
                    with assignment.open(newline="") as handle:
                        exported = list(csv.DictReader(handle))
                    require([r["unit_id"] for r in exported] == ids, "Assignment identities")
                    np.testing.assert_array_equal(
                        [int(r["cluster"]) for r in exported], a["labels"]
                    )
            else:
                a = arrays(folder / ("mixture.npz" if arm == "gmm" else "point.npz"))
                audit_point(x, folder, arm, {**row, "labels": a["labels"]})
                require(
                    summary["provenance"]["configuration"]["model"] == f["arms"][arm],
                    "Frozen baseline",
                )
            if arm == "gmm":
                audit_candidates(folder, len(x))
            if (
                row["status"] == "failed"
                and arm.startswith("spectral")
                and (folder / "point.npz").exists()
            ):
                audit_spectral(x, ids, model_config(arm), summary, arrays(folder / "point.npz"))
            if kind == "replay":
                original = out / condition / arm
                previous = next(
                    r
                    for r in records
                    if r["condition"] == condition and r["arm"] == arm and r["kind"] == "point"
                )
                require(
                    [previous.get(k) for k in ("status", "selected_k", "reason")]
                    == [row.get(k) for k in ("status", "selected_k", "reason")],
                    "Replay decisions",
                )
                names = {str(p.relative_to(folder)) for p in folder.rglob("*.npz")}
                require(
                    names == {str(p.relative_to(original)) for p in original.rglob("*.npz")},
                    "Replay numeric inventory",
                )
                for name in names:
                    a, b = arrays(folder / name), arrays(original / name)
                    require(set(a) == set(b), "Replay array names")
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key])
                replays += 1
            statuses[row["status"]] += 1
            total += 1
        required = {
            (c, a, k)
            for c in CONDITIONS
            for a in ARMS
            for k in (("point", "replay") if task["replay"] else ("point",))
        }
        require(
            len(set(observed_keys)) == len(records) and set(observed_keys) == required,
            "Exact task accounting",
        )
    probes = 0
    if phase == "preflight":
        for n in (32, 48) if config.fixture else (64, 256):
            for target in range(2, 7):
                folder = root / phase / "probes" / f"n{n}_k{target}"
                r, a = read_json(folder / "result.json"), arrays(folder / "arrays.npz")
                require(
                    r["engineering_only"] and r["target"] == target and r["n"] == n,
                    "Timing probe scope",
                )
                if r["status"] == "complete":
                    ids = [f"probe_{i:04d}" for i in range(n)]
                    expected = np.random.default_rng([93431, n]).normal(size=(n, 8))
                    expected = (expected - expected.mean(0)) / expected.std(0)
                    np.testing.assert_array_equal(expected, a["scores"])
                    lap, values, vectors, _ = audit_geometry(
                        a["scores"], ids, model_config("spectral_15"), r, a
                    )
                    require(
                        audit_embedding(a, r, lap, values, vectors, target) == r["reason"],
                        "Probe equations",
                    )
                else:
                    require(bool(r["error"]), "Preserved failed timing probe")
                probes += 1
    require(
        total == f["counts"][phase]["workflows"]
        and total + probes == f["counts"][phase]["evaluations"],
        "Evaluation count",
    )
    execution = read_json(root / f"{phase}-execution.json")
    expected_datasets = [
        {"key": t["key"], "n": t["n"], "records": read_json(root / phase / t["key"] / "fits.json")}
        for t in tasks(config, phase)
    ]
    expected_probes = (
        [
            read_json(root / phase / "probes" / f"n{n}_k{k}" / "result.json")
            for n in ((32, 48) if config.fixture else (64, 256))
            for k in range(2, 7)
        ]
        if phase == "preflight"
        else []
    )
    require(
        execution["datasets"] == expected_datasets and execution["probe_fits"] == expected_probes,
        "Execution ledger matches every audited output",
    )
    require(all(execution[k] == v for k, v in f["counts"][phase].items()), "Execution counts")
    require(
        1 <= execution["workers"] <= config.max_workers
        and np.isfinite(execution["seconds"])
        and execution["seconds"] >= 0,
        "Execution timing scope",
    )
    result = {
        "passed": True,
        "phase": phase,
        "workflows": total,
        "probes": probes,
        "evaluations": total + probes,
        "statuses": dict(statuses),
        "exact_workflow_replays": replays,
        "observations": len(tasks(config, phase)),
        "execution_sha256": fingerprint_file(root / f"{phase}-execution.json").sha256,
        "seconds": time.monotonic() - start,
        "audit_boundary": (
            "Independent source/graph/QR/density equations, not second library implementations"
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    return result
