"""One immutable matched cohort, unchanged native fits and evidence-separated comparison."""

from itertools import combinations
from pathlib import Path

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.matched_analysis import freeze_matched_inputs, verify_sources
from super_deli.models.matched_analysis_config import MatchedAnalysisConfig
from super_deli.workflows.matched_analysis_adapters import read_matched_arm, run_matched_arm
from super_deli.workflows.matched_analysis_report import write_matched_report
from super_deli.workflows.tabular_analysis import _csv

LIMITATIONS = [
    (
        "Prespecified native PCA/shared-GMM and separate-encoder/shared-GMM "
        "configurations; no automatic winner or consensus "
        "partition."
    ),
    (
        "Same raw selected features and specimen cohort; fitted "
        "representations differ. Compare BIC only within each native "
        "search."
    ),
    (
        "Conditional mixture memberships, whole-unit resampling stability and "
        "significance are different evidence types."
    ),
    (
        "PCA whole-unit bootstrap and neural 80% subsampling retain native "
        "inclusion/OOB counts, failures and conditioning."
    ),
    (
        "All coordinates enter clustering; two-dimensional displays are "
        "partial views. Component colors are local to each "
        "arm."
    ),
    (
        "Independent specimens and individually measured ecology are assumed, "
        "not established by matching IDs."
    ),
    (
        "No reference labels, species probabilities, species-boundary test or "
        "new p-value. Empirical evidence gates remain "
        "unchanged."
    ),
]


def run_matched_analysis(dataset, *, analysis_config, base_dir, output_dir):
    config = MatchedAnalysisConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        frozen, cohort, sources = freeze_matched_inputs(dataset, config, base_dir, out / "inputs")
        write_json(
            dict(
                schema_version=1,
                dataset=dataset.to_dict(),
                frozen_dataset=frozen.to_dict(),
                analysis=config.to_dict(),
                configuration_sha256=configuration_hash(config.to_dict()),
                cohort=cohort,
                sources=sources,
            ),
            out / "manifest.json",
        )
        (out / "arms").mkdir()
        rows = []
        points = {}
        memberships = {}
        for arm in config.arms:
            verify_sources(out / "inputs", sources)
            row = dict(
                name=arm.name,
                family=arm.family,
                status="failed",
                selected_k=None,
                search_boundary=None,
                resolved_count=None,
                stability_status="unavailable",
                stability=None,
                configuration=arm.model.to_dict(),
                error=None,
            )
            child = out / "arms" / arm.name
            try:
                run_matched_arm(arm, frozen, out / "inputs", child)
                native = read_matched_arm(arm, child, cohort, frozen)
                stability = native["stability"]
                point = native["point"]
                summary = native["summary"]
                row.update(
                    status="partial" if stability.get("failed", 0) else "complete",
                    selected_k=summary["selected_k"],
                    search_boundary=summary["selected_k"] == arm.model.max_k,
                    resolved_count=len(point["labels"]),
                    stability_status=stability["status"],
                    stability=stability,
                    preprocessing=native["preprocessing"],
                    candidates=native["candidates"],
                    native_summary=f"arms/{arm.name}/summary.json",
                    native_assignments=f"arms/{arm.name}/assignments.csv",
                )
                points[arm.name] = point
                memberships[arm.name] = native["membership"]
                write_json(point, child / "comparison-point.json")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                write_json(
                    dict(status="failed", error=row["error"]), out / f"{arm.name}-failure.json"
                )
            verify_sources(out / "inputs", sources)
            rows.append(row)
        comparisons = []
        for left, right in combinations(config.arms, 2):
            pair = dict(left=left.name, right=right.name)
            if left.name in points and right.name in points:
                a, b = points[left.name], points[right.name]
                pair.update(
                    compare_partitions(a["labels"], b["labels"], a["resolved"], b["resolved"])
                )
            else:
                pair.update(
                    status="unavailable", reason="A point fit or cohort verification failed"
                )
            comparisons.append(pair)
        status = (
            "complete"
            if all(r["status"] == "complete" for r in rows)
            else "partial"
            if points
            else "failed"
        )
        summary = dict(
            schema_version=1,
            status=status,
            dataset_id=dataset.dataset_id,
            n_units=len(cohort["unit_ids"]),
            level="specimen",
            unit_ids=cohort["unit_ids"],
            models=rows,
            comparisons=comparisons,
            limitations=LIMITATIONS,
            inference=dict(status="not_computed", p_value=None),
            configuration_sha256=configuration_hash(config.to_dict()),
        )
        _csv(
            out / "models.csv",
            [
                {
                    k: r[k]
                    for k in (
                        "name",
                        "family",
                        "status",
                        "selected_k",
                        "search_boundary",
                        "resolved_count",
                        "stability_status",
                        "error",
                    )
                }
                for r in rows
            ],
        )
        write_json(comparisons, out / "comparisons.json")
        keys = list(dict.fromkeys(k for r in comparisons for k in r))
        _csv(out / "comparisons.csv", [{k: r.get(k) for k in keys} for r in comparisons])
        assignments = []
        for i, u in enumerate(cohort["unit_ids"]):
            r = dict(
                unit_id=u,
                morphology_observation_id=cohort["views"][0]["observation_ids"][i],
                ecology_observation_id=cohort["views"][1]["observation_ids"][i],
            )
            for arm in config.arms:
                r[arm.name + "__cluster"] = (
                    points[arm.name]["labels"][i] if arm.name in points else None
                )
                r[arm.name + "__maximum_conditional_membership"] = (
                    memberships[arm.name][i] if arm.name in points else None
                )
            assignments.append(r)
        _csv(out / "assignments.csv", assignments)
        write_matched_report(summary, points, out / "report.html")
        summary["artifacts"] = {
            p.relative_to(out).as_posix(): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        write_json(
            dict(status="failed", error=f"{type(exc).__name__}: {exc}"), out / "failure.json"
        )
        raise
