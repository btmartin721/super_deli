"""Independent source/draw/decision and baseline audits with exact replay checks."""

import time
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.gaussian_graph_config import ARMS, tasks
from super_deli.benchmarking.gaussian_graph_reference import (
    audit_profiles,
    gaussian_audit,
    tail_reference,
)
from super_deli.benchmarking.gaussian_graph_study import check_frozen, guard, inventory
from super_deli.benchmarking.neural_analysis import arrays, audit_point, read_raw, require
from super_deli.benchmarking.unknown_k_data import arm_declarations, model_config
from super_deli.benchmarking.unknown_k_graph_audit import (
    audit_embedding,
    audit_geometry,
    audit_spectral,
)
from super_deli.benchmarking.unknown_k_reference import audit_candidates, reference
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def audit_reference_point(x, ids, neighbors, r, a, profiles, null_failure):
    if null_failure:
        require(
            r["reason"] == "unresolved_gaussian_fit"
            and r["p_reference"] is None
            and r["status"] == "withheld"
            and r["selected_k"] is None
            and "labels" not in a,
            "No rank after unresolved null fit",
        )
        return
    lap, values, vectors, comp = audit_geometry(
        (x - x.mean(0)) / x.std(0), ids, model_config(f"spectral_{neighbors}"), r, a
    )
    expected = tail_reference(
        a["eigenvalues"],
        profiles[f"n{neighbors}_pilot_spectra"],
        profiles[f"n{neighbors}_reference_spectra"],
    )
    require(expected["p_reference"] == r["p_reference"], "Exact reference tail rank")
    if r["p_reference"] is None:
        require(
            r["status"] == "withheld" and r["reason"] == expected["reason"] and "labels" not in a,
            "Incomplete references withhold partition",
        )
        return
    require(
        r["proposed_k"] == expected["proposed_k"] and r["exceedances"] == expected["exceedances"],
        "Reference proposal/tail accounting",
    )
    require(r["screen_pass"] == (r["p_reference"] <= 0.05), "Frozen split threshold")
    for key in ("z", "pilot_mean", "pilot_sd", "reference_maxima"):
        np.testing.assert_allclose(a[key], expected[key], atol=1e-10, rtol=1e-10)
    k, reason = expected["proposed_k"], expected["reason"]
    if reason is None:
        if k == 1:
            require(r["interpretation"] == "no_reference_supported_split", "K=1 interpretation")
            np.testing.assert_array_equal(a["proposed_labels"], np.zeros(len(x), dtype=int))
        elif len(set(comp)) not in (1, k):
            reason = "unsupported_component_count"
        elif r["status"] == "failed":
            require(r["solver_called"] and bool(r["error"]), "Preserved assignment failure")
            return
        elif len(set(comp)) == k:
            np.testing.assert_array_equal(
                a["proposed_labels"][:, None] == a["proposed_labels"], comp[:, None] == comp
            )
        else:
            require(r["solver_called"], "Required connected embedding")
            reason = audit_embedding(a, r, lap, values, vectors, k)
    if reason is None:
        sizes = [int(sum(a["proposed_labels"] == j)) for j in sorted(set(a["proposed_labels"]))]
        require(r["cluster_sizes"] == sizes, "Reference sizes")
        if min(sizes) < 5:
            reason = "small_cluster"
    require(r["reason"] == reason, "Reference guard priority")
    if reason is None:
        require(r["status"] == "available" and r["selected_k"] == k, "Reference availability")
        np.testing.assert_array_equal(a["labels"], a["proposed_labels"])
    else:
        require(
            r["status"] == "withheld" and r["selected_k"] is None and "labels" not in a,
            "Count abstention is not K=1",
        )


def audit_table(task, condition, raw, folder):
    ids, x = read_raw(raw)
    require(ids == [f"invented_{i:04d}" for i in range(task["n"])], "Unit identities")
    rows = read_json(folder / "fits.json")
    require(len(rows) == 6 and {r["arm"] for r in rows} == set(ARMS), "Six fixed arms")
    null_failure = (folder / "null-fit-failure.json").exists()
    profiles = None
    graph_attempts = 0
    if null_failure:
        require(
            bool(read_json(folder / "null-fit-failure.json")["error"]), "Retained raw-null failure"
        )
    else:
        fitted = arrays(folder / "gaussian.npz")
        gaussian_audit(x, fitted)
        profiles = arrays(folder / "reference-spectra.npz")
        ledger = read_json(folder / "draw-ledger.json")
        audit_profiles(task, condition, ids, fitted, ledger, profiles)
        graph_attempts = sum(len(r["graphs"]) for r in ledger)
    for row in rows:
        arm = row["arm"]
        out = folder / arm
        if row["status"] == "failed":
            require(bool(row.get("error")) and row["selected_k"] is None, "Preserved failure")
            continue
        if arm.startswith(("spectral_", "reference_")):
            require(
                read_json(out / "summary.json") == {k: v for k, v in row.items() if k != "arm"},
                "Exact point summary export",
            )
            a = arrays(out / "point.npz")
            if arm.startswith("spectral_"):
                audit_spectral(x, ids, model_config(arm), row, a)
            else:
                audit_reference_point(
                    x, ids, int(arm.split("_")[1]), row, a, profiles, null_failure
                )
        else:
            a = arrays(out / ("mixture.npz" if arm == "gmm" else "point.npz"))
            audit_point(x, out, arm, {**row, "labels": a["labels"]})
            summary = read_json(out / "summary.json")
            require(
                summary["provenance"]["configuration"]["model"] == arm_declarations()[arm],
                "Unchanged baseline configuration",
            )
            if arm == "gmm":
                audit_candidates(out, len(x))
    return rows, graph_attempts + 4


def replay_check(original, repeated):
    names = {str(p.relative_to(original)) for p in original.rglob("*.npz")}
    require(
        names == {str(p.relative_to(repeated)) for p in repeated.rglob("*.npz")}, "Replay arrays"
    )
    for name in names:
        a, b = arrays(original / name), arrays(repeated / name)
        require(set(a) == set(b), "Replay array schema")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    for name in ("draw-ledger.json",):
        require(read_json(original / name) == read_json(repeated / name), "Replay seeds/hashes")
    a, b = read_json(original / "fits.json"), read_json(repeated / "fits.json")
    for x, y in zip(a, b, strict=True):
        require(
            {k: v for k, v in x.items() if k != "seconds"}
            == {k: v for k, v in y.items() if k != "seconds"},
            "Exact replay decisions",
        )


def audit_phase(root, phase):
    root = Path(root)
    start = time.monotonic()
    frozen, config = check_frozen(root)
    require(
        inventory(root / phase) == read_json(root / f"{phase}-inventory.json"), "Complete inventory"
    )
    statuses = Counter()
    actual = steps = outputs = replays = 0
    datasets = []
    with threadpool_limits(limits=1):
        for number, task in enumerate(tasks(config, phase), 1):
            out = root / phase / task["key"]
            require(read_json(out / "task.json") == task, "Source task")
            source = reference(task)
            stored = arrays(out / "generating-reference.npz")
            require(set(source) == set(stored), "Generating reference schema")
            for key in source:
                np.testing.assert_allclose(stored[key], source[key], atol=1e-11)
            table_rows = read_json(out / "tables.json")
            expected_keys = {(c, "point") for c in ("clean", "noisy")} | {
                (c, "replay") for c in task["replay_conditions"]
            }
            require(
                len(table_rows) == len(expected_keys)
                and {(r["condition"], r["kind"]) for r in table_rows} == expected_keys,
                "Table schedule",
            )
            for table in table_rows:
                condition, kind = table["condition"], table["kind"]
                folder = out / condition / kind
                _, x = read_raw(out / condition / "raw")
                np.testing.assert_allclose(x, source[condition], rtol=0, atol=1e-11)
                rows, attempts = audit_table(task, condition, out / condition / "raw", folder)
                require(
                    read_json(folder / "execution.json")
                    == {k: v for k, v in table.items() if k != "kind"},
                    "Table execution ledger",
                )
                require(table["records"] == rows, "Point execution ledger")
                if kind == "replay":
                    replay_check(out / condition / "point", folder)
                    replays += 1
                actual += attempts
                outputs += len(rows)
                steps += sum(
                    r.get("method") == "components" or r.get("solver_called", False) for r in rows
                )
                statuses.update(r["status"] for r in rows)
            datasets.append({"key": task["key"], "n": task["n"], "tables": table_rows})
            guard(root, config)
            if number % 10 == 0 or number == len(tasks(config, phase)):
                print({"audit": phase, "observations": number}, flush=True)
            if time.monotonic() - start > config.max_projected_seconds:
                raise ValueError("Audit runtime limit exceeded")
    execution = read_json(root / f"{phase}-execution.json")
    require(execution["datasets"] == datasets, "Complete execution ledger")
    require(all(execution[k] == v for k, v in frozen["counts"][phase].items()), "Frozen counts")
    require(
        outputs
        == frozen["counts"][phase]["point_outputs"] + frozen["counts"][phase]["replay_outputs"]
        and replays == frozen["counts"][phase]["table_replays"],
        "Output accounting",
    )
    result = {
        "passed": True,
        "phase": phase,
        **frozen["counts"][phase],
        "actual_evaluation_attempts": actual,
        "assignment_steps": steps,
        "statuses_including_replays": dict(statuses),
        "seconds": time.monotonic() - start,
        "execution_sha256": fingerprint_file(root / f"{phase}-execution.json").sha256,
        "audit_boundary": (
            "Independent equations and source reconstruction, not a second eigensolver"
        ),
    }
    write_json(result, root / f"{phase}-audit.json")
    return result
