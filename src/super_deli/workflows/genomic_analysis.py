"""Compare native genomic/phenotypic discovery on one immutable matched cohort."""

from itertools import combinations
from pathlib import Path

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import write_json
from super_deli.io.genomic_analysis import freeze_genomic_analysis
from super_deli.io.matched_analysis import verify_sources
from super_deli.models.genomic_analysis_config import GenomicAnalysisConfig
from super_deli.workflows.genomic_analysis_adapters import read_genomic_arm, run_genomic_arm
from super_deli.workflows.genomic_analysis_report import write_genomic_report
from super_deli.workflows.tabular_analysis import _csv

LIMITATIONS = [
    "Candidate partitions do not establish species boundaries; no automatic winner or consensus.",
    "Explicit specimen matching does not establish independence or individually measured ecology.",
    "Genomic diagonal, tabular full and declared integrative covarianc"
    "e families differ; differences are not integration effects alone.",
    "Compare BIC only within each native candidate search, never across representations or arms.",
    "Membership is conditional on a selected fitted mixture, not speci"
    "es probability or resampling support.",
    "Locus-block bootstrap conditions on specimens; specimen bootstrap"
    " conditions on source loci. Do not pool their frequencies.",
    "Correlated/redundant views and arbitrary weights may emphasize sh"
    "ared signal. Conflict need not indicate species.",
    "No missing-data, grouped/unmatched, shape or neural adapter; no n"
    "ew p-values or empirical gate changes.",
]


def run_genomic_analysis(dataset, *, analysis_config, base_dir, output_dir):
    config = GenomicAnalysisConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        frozen, cohort, sources = freeze_genomic_analysis(dataset, config, base_dir, out / "inputs")
        digest = configuration_hash(config.to_dict())
        write_json(
            dict(
                schema_version=1,
                dataset=dataset.to_dict(),
                frozen_dataset=frozen.to_dict(),
                analysis=config.to_dict(),
                configuration_sha256=digest,
                cohort=cohort,
                sources=sources,
            ),
            out / "manifest.json",
        )
        (out / "arms").mkdir()
        rows, points, memberships = [], {}, {}
        for arm in config.arms:
            verify_sources(out / "inputs", sources)
            row = dict(
                name=arm.name,
                family=arm.family,
                modalities=list(arm.modalities),
                configuration=arm.model.to_dict(),
                status="failed",
                selected_k=None,
                search_boundary=None,
                error=None,
            )
            child = out / "arms" / arm.name
            try:
                run_genomic_arm(arm, frozen, out / "inputs", child)
                native = read_genomic_arm(arm, child, cohort, frozen)
                points[arm.name] = native.pop("point")
                memberships[arm.name] = native.pop("membership")
                row.update(native)
                row["search_boundary"] = row["selected_k"] == arm.model.max_k
                write_json(points[arm.name], child / "comparison-point.json")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                write_json(
                    dict(status="failed", error=row["error"]), out / f"{arm.name}-failure.json"
                )
            verify_sources(out / "inputs", sources)
            rows.append(row)
        comparisons = []
        for a, b in combinations(config.arms, 2):
            row = dict(left=a.name, right=b.name)
            if a.name in points and b.name in points:
                x, y = points[a.name], points[b.name]
                row.update(
                    compare_partitions(x["labels"], y["labels"], x["resolved"], y["resolved"])
                )
            else:
                row.update(status="unavailable", reason="A native point/cohort verification failed")
            comparisons.append(row)
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
            unit_ids=cohort["unit_ids"],
            n_units=len(cohort["unit_ids"]),
            models=rows,
            comparisons=comparisons,
            limitations=LIMITATIONS,
            inference=dict(status="not_computed", p_value=None),
            configuration_sha256=digest,
        )
        assignments = []
        for i, unit in enumerate(cohort["unit_ids"]):
            row = dict(unit_id=unit)
            for view in cohort["views"]:
                row[view["modality"] + "__observation_id"] = view["observation_ids"][i]
            for arm in config.arms:
                row[arm.name + "__cluster"] = (
                    points[arm.name]["labels"][i] if arm.name in points else None
                )
                row[arm.name + "__maximum_conditional_membership"] = (
                    memberships[arm.name][i] if arm.name in points else None
                )
            assignments.append(row)
        _csv(out / "assignments.csv", assignments)
        keys = ("name", "family", "status", "selected_k", "search_boundary", "error")
        _csv(out / "models.csv", [{k: r[k] for k in keys} for r in rows])
        write_json(comparisons, out / "comparisons.json")
        keys = list(dict.fromkeys(k for r in comparisons for k in r))
        _csv(out / "comparisons.csv", [{k: r.get(k) for k in keys} for r in comparisons])
        write_genomic_report(summary, points, out / "report.html")
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
