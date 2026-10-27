"""Prespecified one-cohort comparison through unchanged standalone workflows."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

from super_deli.core.partition_comparison import compare_partitions
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.tabular_analysis_config import TabularAnalysisConfig
from super_deli.workflows.tabular_comparison_report import write_comparison_report

LIMITATIONS = [
    "Prespecified configurations; no automatic model selection or species delimitation claim.",
    "GMM membership is conditional posterior membership; HDBSCAN strength is "
    "algorithmic density membership.",
    "GMM bootstrap projects all original units, including in-bag units, with "
    "separate OOB denominators.",
    "HDBSCAN subsampling labels only included units; noise is never a shared cluster.",
    "Parameter sensitivity, sampling stability and significance are different "
    "quantities; supports are not pooled.",
    "No p-values are computed or transferred; independent units are assumed, not verified.",
]


def _csv(path, rows):
    if not rows:
        rows = [{"status": "not_requested"}]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _native(arm, dataset, base_dir, output_dir, modality_name, context=None):
    if arm.family == "gmm":
        from super_deli.workflows.tabular_gmm import run_tabular_gmm

        workflow = run_tabular_gmm
    else:
        from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan

        workflow = run_tabular_hdbscan
    kwargs = {}
    model = arm.model
    if context is not None:
        kwargs["preprocessing_context"] = context
        if arm.family == "gmm":
            model = replace(model, missing_policy="mean" if context.policy == "mean" else "error")
    return workflow(
        dataset,
        model_config=model,
        base_dir=base_dir,
        output_dir=output_dir,
        modality_name=modality_name,
        **kwargs,
    )


def _read_point(arm, directory, expected_ids, expected_observations, expected_hash):
    import numpy as np

    summary = read_json(directory / "summary.json")
    if summary["unit_ids"] != list(expected_ids) or summary["observation_ids"] != list(
        expected_observations
    ):
        raise ValueError("Child unit/observation order does not match the frozen cohort")
    inputs = summary["provenance"]["ingestion"]["inputs"]
    if len(inputs) != 1 or inputs[0]["sha256"] != expected_hash:
        raise ValueError("Child input does not match the frozen source hash")
    prep_file = "preprocessing.npz" if arm.family == "gmm" else "point.npz"
    with np.load(directory / prep_file, allow_pickle=False) as arrays:
        prep = {
            k: arrays[k].copy()
            for k in ("kept_indices", "means", "scales", "pca_mean", "components")
        }
    with np.load(
        directory / ("mixture.npz" if arm.family == "gmm" else "point.npz"), allow_pickle=False
    ) as arrays:
        labels, scores = arrays["labels"].copy(), arrays["scores"].copy()
        strength = (
            arrays["membership"].max(axis=1) if arm.family == "gmm" else arrays["strength"].copy()
        )
    resolved = labels >= 0
    if arm.family == "gmm":
        resolved &= strength >= arm.model.assignment_threshold
    return (
        summary,
        prep,
        {"labels": labels.tolist(), "scores": scores.tolist(), "resolved": resolved.tolist()},
    )


def _same_preprocessing(reference, current):
    import numpy as np

    for name in reference:
        if reference[name].shape != current[name].shape or not np.allclose(
            reference[name], current[name], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"Child preprocessing mismatch: {name}")


def run_tabular_analysis(
    dataset,
    *,
    analysis_config,
    base_dir,
    output_dir,
    modality_name=None,
    _preprocessing_policy=None,
):
    """Freeze one input, run independent arms and retain partial results on failures.

    Native numerical engines, seeds and output formats are preserved. The input
    is copied before fitting; later changes to the caller's source cannot alter
    the cohort seen by subsequent arms.
    """
    from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig

    if isinstance(analysis_config, TabularMissingAnalysisConfig):
        from super_deli.workflows.tabular_missing import run_missing_analysis

        return run_missing_analysis(
            dataset,
            analysis_config=analysis_config,
            base_dir=base_dir,
            output_dir=output_dir,
            modality_name=modality_name,
        )
    config = TabularAnalysisConfig.from_dict(analysis_config.to_dict())
    dataset = config_from_dict(dataset.to_dict())
    selected = [
        m
        for m in dataset.modalities
        if m.kind in {"morphological", "ecological"}
        and (modality_name is None or m.name == modality_name)
    ]
    if len(selected) != 1:
        raise ValueError("Select exactly one morphology/ecology table with --modality")
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        modality = selected[0]
        source = resolve_input(modality.path, base_dir)
        before = fingerprint_file(source)
        snapshot = out / f"input.{modality.format}"
        with source.open("rb") as src, snapshot.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if (
            fingerprint_file(source).sha256 != before.sha256
            or fingerprint_file(snapshot).sha256 != before.sha256
        ):
            raise OSError("Input changed while freezing the source")
        frozen = replace(dataset, modalities=(replace(modality, path=snapshot.name),))
        manifest = {
            "schema_version": 1,
            "source": asdict(before),
            "dataset": dataset.to_dict(),
            "frozen_dataset": frozen.to_dict(),
            "analysis": config.to_dict(),
            "configuration_sha256": configuration_hash(config.to_dict()),
        }
        if _preprocessing_policy is not None:
            manifest["schema_version"] = 2
            manifest["effective_preprocessing_policy"] = _preprocessing_policy
            manifest["configuration_sha256"] = configuration_hash(
                {
                    "analysis": config.to_dict(),
                    "effective_preprocessing_policy": _preprocessing_policy,
                }
            )
        write_json(manifest, out / "manifest.json")
        loaded = ingest_tabular(frozen, base_dir=out, output_dir=out / "cohort")
        table = loaded.tables[0]
        # Reuse the density family's complete-continuous validation for every arm,
        # including a GMM-only analysis. It performs no model fitting.
        from super_deli.models.hdbscan_config import HDBSCANConfig
        from super_deli.models.tabular_hdbscan import prepare_density_table

        common = config.arms[0].model
        validation_kwargs = {}
        if _preprocessing_policy is not None:
            from super_deli.models.tabular_missing import TrainingContext

            validation_kwargs["context"] = TrainingContext(_preprocessing_policy)
        prepare_density_table(
            table,
            HDBSCANConfig(
                True,
                feature_names=common.feature_names,
                n_pcs=common.n_pcs,
                rank_tol=common.rank_tol,
                max_working_bytes=common.max_working_bytes,
                max_pairwise_bytes=common.max_pairwise_bytes,
            ),
            **validation_kwargs,
        )
        (out / "arms").mkdir()
        rows, points, reference = [], {}, None
        for arm in config.arms:
            child = out / "arms" / arm.name
            row = {
                "name": arm.name,
                "family": arm.family,
                "status": "failed",
                "selected_k": None,
                "resolved_count": None,
                "noise_count": None,
                "unresolved_count": None,
                "stability_status": None,
                "configuration": arm.model.to_dict(),
                "stability": None,
                "error": None,
            }
            try:
                if fingerprint_file(snapshot).sha256 != before.sha256:
                    raise ValueError("Frozen input hash mismatch")
                if _preprocessing_policy is None:
                    _native(arm, frozen, out, child, modality.name)
                else:
                    _native(
                        arm,
                        frozen,
                        out,
                        child,
                        modality.name,
                        TrainingContext(_preprocessing_policy),
                    )
                summary, prep, point = _read_point(
                    arm, child, table.unit_ids, table.observation_ids, before.sha256
                )
                if fingerprint_file(snapshot).sha256 != before.sha256:
                    raise ValueError("Frozen input hash mismatch")
                if reference is not None:
                    _same_preprocessing(reference, prep)
                else:
                    reference = prep
                stability = summary["bootstrap" if arm.family == "gmm" else "stability"]
                incomplete = bool(stability.get("failed", 0))
                row.update(
                    status="partial" if incomplete else "complete",
                    selected_k=summary["selected_k"],
                    resolved_count=sum(point["resolved"]),
                    noise_count=sum(v < 0 for v in point["labels"]),
                    unresolved_count=sum(
                        not ok and label >= 0
                        for ok, label in zip(point["resolved"], point["labels"], strict=True)
                    ),
                    cluster_sizes=dict(Counter(str(v) for v in point["labels"] if v >= 0)),
                    stability_status=stability["status"],
                    stability=stability,
                )
                row["preprocessing"] = read_json(child / "preprocessing.json")
                row["native_summary"] = f"arms/{arm.name}/summary.json"
                row["native_stability_arrays"] = (
                    f"arms/{arm.name}/stability.npz" if (child / "stability.npz").exists() else None
                )
                points[arm.name] = point
                write_json(point, child / "comparison-point.json")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                write_json(
                    {"status": "failed", "error": row["error"]}, out / f"{arm.name}-failure.json"
                )
            rows.append(row)
        comparisons = []
        for left, right in combinations(config.arms, 2):
            pair = {"left": left.name, "right": right.name}
            if left.name in points and right.name in points:
                a, b = points[left.name], points[right.name]
                pair.update(
                    compare_partitions(a["labels"], b["labels"], a["resolved"], b["resolved"])
                )
            else:
                pair.update(
                    status="unavailable", reason="Point fit failed or cohort/preprocessing mismatch"
                )
            comparisons.append(pair)
        status = (
            "complete"
            if all(r["status"] == "complete" for r in rows)
            else "partial"
            if points
            else "failed"
        )
        summary = {
            "schema_version": 1,
            "status": status,
            "n_units": len(table.unit_ids),
            "dataset_id": dataset.dataset_id,
            "modality": modality.name,
            "kind": modality.kind,
            "level": modality.level,
            "unit_ids": table.unit_ids,
            "observation_ids": table.observation_ids,
            "feature_names": common.feature_names,
            "n_pcs": common.n_pcs,
            "models": rows,
            "comparisons": comparisons,
            "limitations": LIMITATIONS,
            "inference": {"status": "not_computed", "p_value": None},
            "source_sha256": before.sha256,
        }
        if _preprocessing_policy is not None:
            summary["schema_version"] = 2
            summary["effective_preprocessing_policy"] = _preprocessing_policy
        write_json(comparisons, out / "comparisons.json")
        columns = (
            "name",
            "family",
            "status",
            "selected_k",
            "resolved_count",
            "noise_count",
            "unresolved_count",
            "stability_status",
            "error",
        )
        _csv(out / "models.csv", [{k: r[k] for k in columns} for r in rows])
        allkeys = list(dict.fromkeys(k for row in comparisons for k in row))
        _csv(out / "comparisons.csv", [{k: r.get(k) for k in allkeys} for r in comparisons])
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
