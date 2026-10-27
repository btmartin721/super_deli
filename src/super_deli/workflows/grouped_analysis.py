"""One frozen grouped cohort, native selections and coverage-explicit reporting."""

from collections import Counter
from itertools import combinations
from pathlib import Path

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.grouped_analysis import freeze_grouped_inputs
from super_deli.models.grouped_analysis_config import GroupedAnalysisConfig
from super_deli.workflows.grouped_analysis_adapters import read_grouped_arm, run_grouped_arm
from super_deli.workflows.tabular_analysis import _csv, _same_preprocessing
from super_deli.workflows.tabular_comparison_report import write_comparison_report

LIMITATIONS = [
    "Prespecified native configurations; no winning model, consensus species "
    "partition or new p-value.",
    "Group membership/counts and complete continuous inputs are fixed; "
    "independent targets are assumed, not verified.",
    "Conditional GMM membership, native stability and significance are different evidence types.",
    "Predictive stability is restricted to shared-covariance locality analyses "
    "and conditions on original folds and seeds.",
    "Unsupported stability combinations remain unavailable; failure never becomes K=1.",
    "OOB frequencies have workflow-specific coverage thresholds; "
    "sparse support is not zero or certainty.",
]


def run_grouped_tabular_analysis(dataset, *, analysis_config, base_dir, output_dir):
    config = GroupedAnalysisConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        frozen, cohort, sources = freeze_grouped_inputs(
            dataset, config.arms[0].model, base_dir, out / "inputs"
        )
        write_json(
            {
                "schema_version": 1,
                "dataset": dataset.to_dict(),
                "frozen_dataset": frozen.to_dict(),
                "analysis": config.to_dict(),
                "sources": sources,
                "cohort": cohort,
                "configuration_sha256": configuration_hash(config.to_dict()),
            },
            out / "manifest.json",
        )
        (out / "arms").mkdir()
        rows = []
        points = {}
        reference = None
        for arm in config.arms:
            row = {
                "name": arm.name,
                "family": f"{arm.model.covariance_type}_{arm.kind}_{arm.model.selection_criterion}",
                "covariance_type": arm.model.covariance_type,
                "criterion": arm.model.selection_criterion,
                "status": "failed",
                "selected_k": None,
                "ordinary_k": None,
                "ordinary_available": False,
                "screen_status": "not_available",
                "resolved_count": None,
                "noise_count": 0,
                "unresolved_count": None,
                "stability_status": "not_available",
                "configuration": arm.to_dict(),
                "stability": None,
                "error": None,
            }
            child = out / "arms" / arm.name
            try:
                for item in sources:
                    if (
                        fingerprint_file(out / "inputs" / item["snapshot"]).sha256
                        != item["source"]["sha256"]
                    ):
                        raise ValueError("Frozen grouped source hash mismatch")
                try:
                    run_grouped_arm(arm, frozen, out / "inputs", child)
                except Exception as exc:
                    # Recover native ordinary evidence after a late screen failure.
                    row["execution_error"] = f"{type(exc).__name__}: {exc}"
                    if arm.kind != "predictive":
                        raise
                native = read_grouped_arm(arm, child, cohort)
                for item in sources:
                    if (
                        fingerprint_file(out / "inputs" / item["snapshot"]).sha256
                        != item["source"]["sha256"]
                    ):
                        raise ValueError("Frozen grouped source hash mismatch")
                if reference is None:
                    reference = native["parameters"]
                else:
                    _same_preprocessing(reference, native["parameters"])
                stability = native["stability"]
                row.update(
                    preprocessing=native["preprocessing"],
                    ordinary_k=native["ordinary_k"],
                    ordinary_available=True,
                    screen_status=native["screen_status"],
                    selected_k=native["selected_k"],
                    stability=stability,
                    stability_status=stability["status"],
                    native_summary=f"arms/{arm.name}/summary.json",
                    ordinary_summary=(native["run"] / "summary.json").relative_to(out).as_posix(),
                )
                if native["point"] is None:
                    row.update(
                        status="withheld",
                        error="Native predictive screen withheld; ordinary fit retained",
                    )
                else:
                    point = native["point"]
                    points[arm.name] = point
                    row.update(
                        status="partial" if stability.get("failed", 0) else "complete",
                        resolved_count=len(cohort["unit_ids"]),
                        unresolved_count=0,
                        cluster_sizes=dict(Counter(map(str, point["labels"]))),
                        native_assignments=native["selected_assignments"]
                        .relative_to(out)
                        .as_posix(),
                    )
                    write_json(point, child / "comparison-point.json")
                if "selection" in native:
                    row["selection"] = native["selection"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                write_json(
                    {"status": "failed", "error": row["error"]}, out / f"{arm.name}-failure.json"
                )
            rows.append(row)
        comparisons = []
        for a, b in combinations(config.arms, 2):
            pair = {"left": a.name, "right": b.name}
            if a.name in points and b.name in points:
                x, y = points[a.name], points[b.name]
                pair.update(
                    compare_partitions(x["labels"], y["labels"], x["resolved"], y["resolved"])
                )
            else:
                pair.update(
                    status="unavailable",
                    reason="Native point fit/screen failed, withheld or mismatched",
                )
            comparisons.append(pair)
        summary = {
            "schema_version": 1,
            "workflow": "grouped_tabular_analysis_v1",
            "status": "complete"
            if all(r["status"] == "complete" for r in rows)
            else "partial"
            if points
            else "failed",
            "dataset_id": dataset.dataset_id,
            "level": cohort["target_level"],
            "unit_ids": cohort["unit_ids"],
            "groups": cohort["groups"],
            "n_pcs": config.arms[0].model.n_pcs,
            "models": rows,
            "comparisons": comparisons,
            "limitations": LIMITATIONS,
            "inference": {"status": "not_computed", "p_value": None},
            "report_columns": ["covariance_type", "criterion", "ordinary_k", "screen_status"],
        }
        write_json(comparisons, out / "comparisons.json")
        write_json(cohort["groups"], out / "groups.json")
        columns = (
            "name",
            "family",
            "status",
            "selected_k",
            "ordinary_k",
            "ordinary_available",
            "screen_status",
            "resolved_count",
            "noise_count",
            "unresolved_count",
            "stability_status",
            "error",
        )
        _csv(out / "models.csv", [{k: r[k] for k in columns} for r in rows])
        _csv(
            out / "groups.csv",
            [
                {k: g[k] for k in ("target_id", "morphology_count", "ecology_count")}
                for g in cohort["groups"]
            ],
        )
        keys = list(dict.fromkeys(k for r in comparisons for k in r))
        _csv(out / "comparisons.csv", [{k: r.get(k) for k in keys} for r in comparisons])
        write_comparison_report(summary, points, out / "report.html")
        summary["artifacts"] = {
            p.relative_to(out).as_posix(): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        write_json(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, out / "failure.json"
        )
        raise
