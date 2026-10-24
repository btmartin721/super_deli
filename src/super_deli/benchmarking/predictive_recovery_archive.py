"""Read-only diagnosis of existing predictive archives with independent checks."""

import gzip
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize, write_progress
from super_deli.benchmarking.predictive_recovery_audit import audit_diagnostics
from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.io.config import read_json, write_json
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig
from super_deli.models.predictive_study_config import PredictiveStudyConfig


def condense(metrics):
    result = {"gates": metrics["gates"], "spaces": []}
    for row in metrics["spaces"]:
        keep = {
            k: row[k]
            for k in (
                "prefix",
                "status",
                "n",
                "dimension",
                "candidates",
                "joint_axis_r2",
                "fused_axis_r2",
                "generating_counts",
                "maps",
            )
            if k in row
        }
        if "upstream" in row:
            keep["upstream_axes"] = {
                kind: {
                    k: v[k]
                    for k in (
                        "raw_group_axis_r2",
                        "source_axis_r2",
                        "profile_axis_r2",
                        "count_axis_r2",
                    )
                }
                for kind, v in row["upstream"].items()
            }
        result["spaces"].append(keep)
    return result


def independent_folds(workflow, model):
    import numpy as np

    from super_deli.benchmarking.aic_diagnostic_audit import check_pool

    rows = []
    for a in read_json(workflow / "fold_assignments.json"):
        p = workflow / f"fold_{a['fold']:03d}"
        rec = read_json(p / "validation_record.json")
        cs = read_json(p / "run/candidates.json") if (p / "run/candidates.json").exists() else None
        density = {}
        if rec["status"] == "complete":
            with np.load(p / "validation.npz", allow_pickle=False) as held:
                _, density = check_pool(
                    (p / "run/gmm_candidates.npz").read_bytes(),
                    cs,
                    counts=None,
                    config=model,
                    heldout=held,
                )
        rows.append({"status": rec["status"], "candidates": cs, "density": density})
    return rows


def _diagnose(task):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.benchmarking.predictive_audit import audit_observation
    from super_deli.benchmarking.predictive_recovery_metrics import diagnose_workflow

    source, row, old_config, new_config, destination, scratch = task
    old = PredictiveStudyConfig.from_dict(old_config)
    new = PredictiveRecoveryConfig.from_dict(new_config)
    declaration = {
        k: row[k]
        for k in (
            "cell",
            "cell_config",
            "repetition",
            "data_seed",
            "test_seed",
            "split_seed",
            "planned_fits",
        )
    }
    root = Path(destination)
    base = root / ".working" / f"{row['cell']}_{row['repetition']:04d}"
    base.mkdir(parents=True)
    with threadpool_limits(limits=1):
        audit_observation((source, declaration, row, old_config, scratch))
        with materialize(
            Path(source) / row["archive"],
            row,
            scratch=scratch,
            min_free_bytes=new.study.min_free_bytes,
        ) as original:
            direct = row["cell_config"]["law"] == "gaussian_scores"
            if direct:
                with np.load(original / "inputs.npz", allow_pickle=False) as p:
                    tests = p["external"].copy()
                test_layout = None
                truth = None
            else:
                with np.load(original / "external_values.npz", allow_pickle=False) as p:
                    tests = {k: p[k].copy() for k in p.files}
                test_layout = read_json(original / "external_layout.json")
                with np.load(original / "truth.npz", allow_pickle=False) as p:
                    truth = (
                        p["training"].copy()
                        if row["cell_config"]["law"] in {"complementary", "unequal_variance"}
                        else None
                    )
            workflow = original / "workflow"
            metrics = diagnose_workflow(
                workflow, truth, tests, test_layout, old.model, base / "diagnostics"
            )
            full = read_json(workflow / "full/run/candidates.json")
            audit_diagnostics(
                workflow,
                truth,
                tests,
                test_layout,
                old.model,
                base / "diagnostics",
                full,
                independent_folds(workflow, old.model),
            )
            if metrics["gates"]["actual_k"] != row["selection"]["screened_k"]:
                raise ValueError("Archived actual selection changed")
            record = {
                "cell": row["cell"],
                "cell_config": row["cell_config"],
                "repetition": row["repetition"],
                "source_archive_sha256": row["archive_sha256"],
                "source_manifest_sha256": row["manifest_sha256"],
                "diagnostic_audit": "passed",
                "no_refits": True,
                **condense(metrics),
            }
            write_json(record, base / "record.json")
            write_json({"source": source, "observation": row}, base / "private-source-pin.json")
    relative = f"{row['cell']}/dataset_{row['repetition']:04d}.tar.gz"
    target = root / relative
    target.parent.mkdir(exist_ok=True)
    packed = compact_directory(base, target)
    shutil.rmtree(base)
    return {**record, **packed, "archive": relative, "manifest": relative + ".manifest.json"}


def diagnose_archives(config, *, source, output_dir, scratch, artifact_root, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four one-thread workers")
    prior = Path(source).resolve()
    root = Path(output_dir).resolve()
    budget = Path(artifact_root).resolve()
    if prior == root or root.is_relative_to(prior) or prior.is_relative_to(root):
        raise ValueError("Keep prior evidence immutable and separate")
    if root == budget or not root.is_relative_to(budget):
        raise ValueError("Use a new output directory inside the artifact budget root")
    old = PredictiveStudyConfig.from_dict(read_json(prior / "configuration.json"))
    ledger = [json.loads(s) for s in (prior / "observations.jsonl").read_text().splitlines()]
    if read_json(prior / "summary.json")["status"] != "complete" or len(ledger) != len(old.plan()):
        raise ValueError("Completed prior study required")
    if {(r["cell"], r["repetition"]) for r in ledger} != {
        (r["cell"], r["repetition"]) for r in old.plan()
    }:
        raise ValueError("Prior identities differ")
    root.mkdir(parents=True, exist_ok=False)
    write_json(config.to_dict(), root / "configuration.json")
    rows = []
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        gzip.open(root / "points.jsonl.gz", "xt") as points,
    ):
        tasks = (
            (
                str(prior),
                r,
                old.to_dict(),
                config.to_dict(),
                str(root),
                str(Path(scratch).resolve()),
            )
            for r in ledger
        )
        for row in pool.map(_diagnose, tasks, chunksize=1):
            rows.append(row)
            points.write(json.dumps(row, sort_keys=True) + "\n")
            points.flush()
            write_progress(
                {"observations_audited": len(rows), "requested": len(ledger)},
                root / "progress.json",
            )
            if retained_bytes(budget) > config.study.max_artifact_bytes:
                raise RuntimeError("Artifact cap reached; retain partial diagnosis")
    cells = []
    for name in dict.fromkeys(r["cell"] for r in ledger):
        group = [r for r in rows if r["cell"] == name]
        ks = [next(c for c in r["gates"]["candidates"] if c["k"] == 4) for r in group]
        cells.append(
            {
                "cell": name,
                "requested": len(group),
                "audited": len(group),
                "k4_full_eligible": sum(c["full_status"] == "eligible" for c in ks),
                "k4_every_fold_eligible": sum(c["fold_ineligible_count"] == 0 for c in ks),
                "k4_full_and_every_fold_eligible": sum(c["passes"]["eligibility_only"] for c in ks),
                "k4_positive_gain_all_fitted": sum(
                    c["pooled_gain_all_fitted_folds"] is not None
                    and c["pooled_gain_all_fitted_folds"] > 0
                    for c in ks
                ),
                "selected_k4": {
                    rule: sum(r["gates"][rule + "_k"] == 4 for r in group)
                    for rule in ("ordinary", "eligibility_only", "gain_only", "actual")
                },
            }
        )
    result = {
        "observations": len(rows),
        "source_searches_audited": sum(r["planned_fits"] for r in ledger),
        "new_searches": 0,
        "diagnostic_audits_passed": len(rows),
        "cells": cells,
        "scope": "Read-only archival diagnosis; counterfactual rules are not production methods",
    }
    write_json(result, root / "summary.json")
    return result
