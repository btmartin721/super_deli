"""Coverage-explicit policy comparison on immutable, possibly incomplete inputs."""

import csv
import html
import math
import shutil
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from super_deli.core.data import TabularReaderOptions
from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.tabular_analysis_config import TabularAnalysisConfig
from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig


def cohort_rows(table, names, policy):
    if policy not in ("complete_case", "mean"):
        raise ValueError("Unknown missing-data policy")
    by_name = {f.name: i for i, f in enumerate(table.features)}
    if (
        not names
        or set(names) - by_name.keys()
        or any(
            table.features[by_name[f]].dtype != "continuous"
            or table.features[by_name[f]].role != "feature"
            for f in names
        )
    ):
        raise ValueError("Missing-data comparison requires declared continuous analytical traits")
    if len(set(table.unit_ids)) != len(table.unit_ids):
        raise ValueError("Repeated units require explicit aggregation outside this workflow")
    result = []
    for i, (obs, unit, values) in enumerate(
        zip(table.observation_ids, table.unit_ids, table.values, strict=True)
    ):
        selected = [values[by_name[f]] for f in names]
        if any(
            v is not None and (type(v) not in (int, float) or not math.isfinite(v))
            for v in selected
        ):
            raise ValueError(
                "Continuous traits require finite numbers or declared None missing cells"
            )
        absent = [f for f, v in zip(names, selected, strict=True) if v is None]
        reason = (
            "all_selected_missing"
            if len(absent) == len(names)
            else "incomplete_selected_features"
            if absent and policy == "complete_case"
            else None
        )
        result.append(
            dict(
                source_index=i,
                observation_id=obs,
                unit_id=unit,
                eligible=reason is None,
                exclusion_reason=reason,
                missing_features=absent,
                values=selected,
            )
        )
    return result


def policy_dataset(dataset, modality, table, names, records, directory):
    root = Path(directory)
    root.mkdir()
    selected = [r for r in records if r["eligible"]]
    by_obs = {o.observation_id: o for o in modality.observations}
    features = {f.name: f for f in table.features}
    with (root / "traits.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", *names])
        writer.writerows(
            [r["observation_id"], *("" if v is None else v for v in r["values"])] for r in selected
        )
    result = replace(
        dataset,
        modalities=(
            replace(
                modality,
                path="traits.tsv",
                format="tsv",
                features=tuple(features[f] for f in names),
                observations=tuple(by_obs[r["observation_id"]] for r in selected),
                tabular_options=TabularReaderOptions(),
            ),
        ),
    )
    write_json(result.to_dict(), root / "dataset.json")
    return result


def run_missing_analysis(dataset, *, analysis_config, base_dir, output_dir, modality_name=None):
    from super_deli.workflows.tabular_analysis import run_tabular_analysis

    config = TabularMissingAnalysisConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    modalities = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(modalities) != 1:
        raise ValueError("Select exactly one morphology/ecology table")
    out = Path(output_dir).absolute()
    source = resolve_input(modalities[0].path, base_dir)
    if out.resolve() == source.resolve() or out.resolve() in source.resolve().parents:
        raise ValueError("Outputs must not contain or replace the source")
    out.mkdir(exist_ok=False)
    try:
        modality = modalities[0]
        before = fingerprint_file(source)
        snapshot = out / ("input." + modality.format)
        with source.open("rb") as src, snapshot.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if (
            fingerprint_file(source).sha256 != before.sha256
            or fingerprint_file(snapshot).sha256 != before.sha256
        ):
            raise ValueError("Input changed while freezing source")
        frozen = replace(dataset, modalities=(replace(modality, path=snapshot.name),))
        write_json(
            dict(
                schema_version=2,
                source_sha256=before.sha256,
                dataset=dataset.to_dict(),
                frozen_dataset=frozen.to_dict(),
                analysis=config.to_dict(),
                configuration_sha256=configuration_hash(config.to_dict()),
            ),
            out / "manifest.json",
        )
        loaded = ingest_tabular(frozen, base_dir=out, output_dir=out / "source-ingestion")
        table = loaded.tables[0]
        names = config.arms[0].model.feature_names
        (out / "policies").mkdir()
        policies = {}
        points = {}
        models = []
        for policy in config.policies:
            root = out / "policies" / policy
            root.mkdir()
            records = cohort_rows(table, names, policy)
            write_json(records, root / "eligibility.json")
            write_json(
                dict(
                    schema_version=2,
                    policy=policy,
                    selected_features=list(names),
                    source_units=len(records),
                    eligible_units=sum(r["eligible"] for r in records),
                    missing_mask=[[f in r["missing_features"] for f in names] for r in records],
                    fully_unobserved_rows_are_excluded_not_noise=True,
                ),
                root / "missingness.json",
            )
            eligible = [r for r in records if r["eligible"]]
            state = dict(
                status="failed",
                source_units=len(records),
                eligible_units=len(eligible),
                excluded_units=len(records) - len(eligible),
                error=None,
            )
            try:
                policy_models = []
                policy_points = {}
                if not eligible:
                    raise ValueError("No eligible units under fixed policy")
                if any(all(r["values"][j] is None for r in eligible) for j in range(len(names))):
                    raise ValueError("A selected training feature has no observed values")
                child = policy_dataset(frozen, modality, table, names, records, root / "raw")
                result = run_tabular_analysis(
                    child,
                    analysis_config=TabularAnalysisConfig(config.arms),
                    base_dir=root / "raw",
                    output_dir=root / "analysis",
                    modality_name=modality.name,
                    _preprocessing_policy=policy,
                )
                state.update(
                    status=result["status"],
                    native_summary="policies/" + policy + "/analysis/summary.json",
                )
                for arm, row in zip(config.arms, result["models"], strict=True):
                    item = dict(
                        row, policy=policy, eligible_units=len(eligible), source_units=len(records)
                    )
                    policy_models.append(item)
                    pointfile = root / "analysis/arms" / arm.name / "comparison-point.json"
                    if pointfile.exists():
                        point = read_json(pointfile)
                        ids = [r["unit_id"] for r in eligible]
                        policy_points[policy + "/" + arm.name] = dict(
                            zip(
                                ids,
                                zip(point["labels"], point["resolved"], strict=True),
                                strict=True,
                            )
                        )
                models.extend(policy_models)
                points.update(policy_points)
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
                for arm in config.arms:
                    models.append(
                        dict(
                            name=arm.name,
                            family=arm.family,
                            policy=policy,
                            status="failed",
                            selected_k=None,
                            error=state["error"],
                            source_units=len(records),
                            eligible_units=len(eligible),
                            stability_status="not_attempted",
                        )
                    )
                write_json(state, root / "failure.json")
            policies[policy] = state
        comparisons = []
        units = list(table.unit_ids)
        for left, right in combinations(points, 2):
            a, b = points[left], points[right]
            comparisons.append(
                dict(
                    left=left,
                    right=right,
                    left_eligible=len(a),
                    right_eligible=len(b),
                    **compare_partitions(
                        [a.get(u, (-1, False))[0] for u in units],
                        [b.get(u, (-1, False))[0] for u in units],
                        [a.get(u, (-1, False))[1] for u in units],
                        [b.get(u, (-1, False))[1] for u in units],
                    ),
                )
            )
        status = (
            "complete"
            if all(s["status"] == "complete" for s in policies.values())
            else "partial"
            if points
            else "failed"
        )
        summary = dict(
            schema_version=2,
            status=status,
            source_units=len(units),
            unit_ids=units,
            policies=policies,
            models=models,
            comparisons=comparisons,
            inference=dict(status="not_computed", p_value=None),
            limitations=[
                "Single imputation; no missing-value uncertainty integration",
                "Stability conditional on original mask/policy and independence assumption",
                "Excluded units have no assignment; noise is not a shared cluster",
                "No species claim, policy winner or absolute BIC comparison across cohorts",
                "Empirical readiness gates are not opened by numerical eligibility",
            ],
        )
        if (
            fingerprint_file(snapshot).sha256 != before.sha256
            or fingerprint_file(source).sha256 != before.sha256
        ):
            raise ValueError("Frozen or original input changed during analysis")
        write_json(comparisons, out / "comparisons.json")
        write_json(summary, out / "summary.json")
        lines = [
            "<!doctype html><meta charset='utf-8'><title>Missing-data comparison</title>",
            "<h1>Missing-data policy comparison</h1><p>Single imputation; no species significance. "
            "Excluded units and density noise remain separate. "
            "Stability is conditional on the observed mask.</p>",
            "<table><tr><th>Policy</th><th>Model</th><th>Eligible/source</th><th>K</th><th>Status</th></tr>",
        ]
        for m in models:
            lines.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(v)) + "</td>"
                    for v in (
                        m["policy"],
                        m["name"],
                        f"{m['eligible_units']}/{m['source_units']}",
                        m["selected_k"],
                        m["status"],
                    )
                )
                + "</tr>"
            )
        lines.append(
            "</table><p>Per-fit preprocessing and native coverage/failure records "
            "accompany this report.</p>"
        )
        (out / "report.html").write_text("\n".join(lines), encoding="utf-8")
        write_json(
            {
                p.relative_to(out).as_posix(): fingerprint_file(p).sha256
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
            out / "checksums.json",
        )
        return summary
    except Exception as exc:
        (out / "summary.json").unlink(missing_ok=True)
        write_json(dict(status="failed", error=str(exc)), out / "failure.json")
        raise
