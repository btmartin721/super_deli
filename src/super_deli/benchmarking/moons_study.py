"""Exclusive frozen diagnostic execution and measured storage gating."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.density_study import retained_bytes
from super_deli.benchmarking.moons_archive import archive_plan, simulate_latents, source_matrix
from super_deli.benchmarking.moons_config import ARMS, MoonsDiagnosticConfig, arm_settings
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def diagnostic_plan(config, archive_root, phase):
    rows, replays = archive_plan(
        archive_root, config.sample_sizes, draws_per_size=1 if phase == "preflight" else 5
    )
    fresh = config.fresh_plan(phase)
    return fresh + (rows if phase == "main" else []) + replays


def run_observation(task):
    import numpy as np

    from super_deli.benchmarking.density_study import recovery_metrics
    from super_deli.benchmarking.moons_archive import write_analytical
    from super_deli.benchmarking.moons_fit import fit_diagnostic_arm
    from super_deli.benchmarking.moons_geometry import (
        branch_reference_diagnostics,
        graph_diagnostics,
    )
    from super_deli.io.tabular import ingest_tabular

    output, archive, row = task
    out = Path(output) / row["key"]
    out.mkdir()
    write_json(row, out / "plan.json")
    data = simulate_latents(row["case"], row["n"], row["seed"])
    if row["origin"] != "fresh":
        p = Path(archive) / "study" / row["source_key"]
        x, _ = source_matrix(p / "raw/traits.tsv")
        np.testing.assert_array_equal(data["x"], x)
    source_ids = tuple(f"unit_{i:04d}" for i in range(row["n"]))
    expected = None
    if row["origin"] == "replay":
        record = read_json(p / "hdbscan.json")["draws"][row["draw"]]
        ids = np.asarray(record["training_indices"], dtype=int)
        source_ids = tuple(source_ids[i] for i in ids)
        data = {k: v[ids] if len(v) else v for k, v in data.items()}
        with np.load(p / "hdbscan.npz", allow_pickle=False) as a:
            expected = {
                key: a[f"draw_{row['draw']}_{key}"]
                for key in ("labels", "strength", "scores", "means", "scales", "components")
            }
    elif row["origin"] == "archived":
        with np.load(p / "hdbscan.npz", allow_pickle=False) as a:
            expected = {
                key: a[f"point_{key}"]
                for key in ("labels", "strength", "scores", "means", "scales", "components")
            }
    dataset = write_analytical(data["x"], source_ids, out / "raw")
    # Keep all latent/reference information separate from analytical inputs.
    np.savez_compressed(out / "reference.npz", **{k: v for k, v in data.items() if k != "x"})
    table = ingest_tabular(dataset, base_dir=out / "raw", output_dir=out / "ingestion").tables[0]
    result = {
        "key": row["key"],
        "origin": row["origin"],
        "case": row["case"],
        "n": row["n"],
        "fitted_units": len(table.unit_ids),
        "arms": {},
    }
    for arm in ("baseline",) if row["origin"] == "replay" else ARMS:
        try:
            metadata, arrays = fit_diagnostic_arm(table, arm)
            if expected is not None and arm == "baseline":
                np.testing.assert_array_equal(arrays["labels"], expected["labels"])
                for key in ("strength", "scores", "means", "scales", "components"):
                    np.testing.assert_allclose(arrays[key], expected[key], rtol=1e-10, atol=1e-12)
                metadata["archived_replay"] = "matched"
            # These functions receive truth only after fitting and selection are complete.
            graph, core, edges = graph_diagnostics(
                arrays["scores"], data["truth"], metadata["parameters"]["min_samples"]
            )
            metadata.update(metrics=recovery_metrics(arrays["labels"], data["truth"]), graph=graph)
            if metadata["hierarchy_status"] == "available":
                metadata["branch_reference"] = branch_reference_diagnostics(
                    arrays["condensed"], data["truth"]
                )
            arrays.update(core_distances=core, mst_edges=edges)
            np.savez_compressed(out / f"{arm}.npz", **arrays)
        except (ValueError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            metadata = {"status": "failed", "error": str(exc)}
        write_json(metadata, out / f"{arm}.json")
        result["arms"][arm] = {
            k: metadata[k]
            for k in ("status", "selected_k", "noise_count", "hierarchy_status", "metrics", "graph")
            if k in metadata
        }
    write_json(result, out / "summary.json")
    write_json(
        {
            p.relative_to(out).as_posix(): fingerprint_file(p).sha256
            for p in out.rglob("*")
            if p.is_file()
        },
        out / "checksums.json",
    )
    return result


def safe_run_observation(task):
    """Preserve input/setup failures and account for every unattempted arm."""
    try:
        return run_observation(task)
    except Exception as exc:
        output, _, row = task
        out = Path(output) / row["key"]
        out.mkdir(exist_ok=True)
        result = {
            "key": row["key"],
            "origin": row["origin"],
            "case": row["case"],
            "n": row["n"],
            "setup_status": "failed",
            "error": str(exc),
            "arms": {},
        }
        if not (out / "plan.json").exists():
            write_json(row, out / "plan.json")
        write_json(result, out / "summary.json")
        if not (out / "checksums.json").exists():
            write_json(
                {
                    p.relative_to(out).as_posix(): fingerprint_file(p).sha256
                    for p in out.rglob("*")
                    if p.is_file()
                },
                out / "checksums.json",
            )
        return result


def run_moons_diagnostic(
    config, *, archive_root, phase, output_dir, archive_audit, workers=1, gate=None
):
    c = MoonsDiagnosticConfig.from_dict(config.to_dict())
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    archive, out = Path(archive_root).resolve(), Path(output_dir).resolve()
    if out == archive or out.is_relative_to(archive):
        raise ValueError("Diagnostic output must be outside preserved source evidence")
    audit = read_json(archive_audit)
    if audit["status"] != "passed":
        raise ValueError("Source archive audit must pass")
    for name, digest in audit["source_binding"].items():
        if fingerprint_file(archive / name).sha256 != digest:
            raise ValueError("Source binding changed")
    if phase == "main":
        receipt = read_json(gate) if gate else {}
        if (
            receipt.get("status") != "passed"
            or receipt.get("configuration_sha256") != configuration_hash(c.to_dict())
            or not out.is_relative_to(Path(receipt["artifact_root"]))
        ):
            raise ValueError("Matching independently audited measured preflight required")
        if fingerprint_file(receipt["audit_file"]).sha256 != receipt["audit_sha256"]:
            raise ValueError("Preflight audit changed")
    if not out.parent.is_dir() or shutil.disk_usage(out.parent).free < c.min_free_bytes:
        raise ValueError("Existing parent and sufficient free space required")
    plan = diagnostic_plan(c, archive, phase)
    out.mkdir(exist_ok=False)
    arms = {
        name: {"model": arm_settings(name)[0].to_dict(), "scaling": arm_settings(name)[1]}
        for name in ARMS
    }
    write_json(
        {
            "schema_version": 1,
            "configuration": c.to_dict(),
            "phase": phase,
            "plan": plan,
            "arms": arms,
            "source_binding": audit["source_binding"],
            "software": {n: version(n) for n in ("super-deli", "numpy", "scipy", "scikit-learn")},
        },
        out / "frozen-plan.json",
    )
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (out / "ledger.jsonl").open("x") as handle,
    ):
        for row in pool.map(safe_run_observation, [(str(out), str(archive), r) for r in plan]):
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if (
                retained_bytes(out.parent) > c.artifact_cap_bytes
                or shutil.disk_usage(out).free < c.min_free_bytes
            ):
                raise RuntimeError("Retained-storage/free-space bound failed; evidence preserved")
    fits = [a for r in rows for a in r["arms"].values()]
    result = {
        "status": "complete",
        "phase": phase,
        "observations": len(rows),
        "attempted_fits": len(fits),
        "successful_fits": sum(a["status"] == "complete" for a in fits),
        "failed_fits": sum(a["status"] == "failed" for a in fits),
        "unavailable_hierarchies": sum(a.get("hierarchy_status") == "unavailable" for a in fits),
        "unattempted_fits": sum(1 if r["origin"] == "replay" else len(ARMS) for r in plan)
        - len(fits),
    }
    write_json(result, out / "summary.json")
    return result


def moons_storage_gate(config, *, probe_dir, audit_file, artifact_root, output_file):
    root, probe = Path(artifact_root).resolve(), Path(probe_dir).resolve()
    audit = read_json(audit_file)
    frozen = read_json(probe / "frozen-plan.json")
    digest = configuration_hash(config.to_dict())
    if (
        audit.get("status") != "passed"
        or audit.get("phase") != "preflight"
        or audit.get("configuration_sha256") != digest
        or configuration_hash(frozen["configuration"]) != digest
    ):
        raise ValueError("Matching audited preflight required")
    if not probe.is_relative_to(root) or not Path(output_file).resolve().is_relative_to(root):
        raise ValueError("Gate/probe must be inside the declared artifact root")
    # Account separately for point inputs/eight arms and cheap single-draw replays.
    n = len(config.sample_sizes)
    planned_main = (3 * n * config.datasets_per_cell + audit["archived_observations"]) * len(
        ARMS
    ) + 5 * n
    planned_probe = 3 * n * len(ARMS) + n
    if audit["failed_fits"] or audit["unavailable_hierarchies"] or audit["unattempted_fits"]:
        raise ValueError("Unresolved preflight fits/hierarchies; evidence retained")
    actual = retained_bytes(root)
    projection = (
        actual + int(retained_bytes(probe) * planned_main / planned_probe * 1.25) + 16 * 1024**2
    )
    passed = (
        projection <= config.artifact_cap_bytes
        and shutil.disk_usage(root).free >= config.min_free_bytes
    )
    result = {
        "status": "passed" if passed else "failed",
        "artifact_root": str(root),
        "configuration_sha256": digest,
        "measured_probe_bytes": retained_bytes(probe),
        "projected_bytes_with_25pct_allowance": projection,
        "retained_bytes": actual,
        "artifact_cap_bytes": config.artifact_cap_bytes,
        "free_bytes": shutil.disk_usage(root).free,
        "audit_file": str(Path(audit_file).resolve()),
        "audit_sha256": fingerprint_file(audit_file).sha256,
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError("Measured gate failed; explicit budget amendment required")
    return result
