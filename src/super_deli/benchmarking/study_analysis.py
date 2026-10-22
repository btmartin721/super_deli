"""Audit scenario shards and summarize a completed study without fitting models."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

from super_deli.benchmarking.config import BenchmarkConfig, derived_seed
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json


def rate_summary(values, attempted):
    """Pointwise Wilson Monte Carlo intervals for independent dataset events."""
    from super_deli.benchmarking.metrics import summarize_values

    available = [v for v in values if v is not None]
    if any(v not in (0, 1) for v in available):
        raise ValueError("Rates require Bernoulli events or missing values")
    result = summarize_values(values, attempted)
    n = len(available)
    if n:
        p = sum(available) / n
        z = NormalDist().inv_cdf(0.975)
        divisor = 1 + z * z / n
        center = (p + z * z / (2 * n)) / divisor
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
        interval = [max(0.0, center - half), min(1.0, center + half)]
    else:
        interval = None
    return {**result, "wilson_95": interval, "events": sum(available)}


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _child(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Study paths must stay inside their shard")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Study paths cannot follow links outside their shard")
    return result


def audit_bootstrap(directory, analysis_seed, n_bootstrap):
    """Reconstruct draws, paired source multiplicities and sufficient statistics."""
    import numpy as np

    summary = read_json(directory / "summary.json")
    units = summary["unit_ids"]
    n = len(units)
    with np.load(directory / "mixture.npz", allow_pickle=False) as arrays:
        point = arrays["labels"]
    logs = read_json(directory / "bootstrap_replicates.json")
    if len(logs) != n_bootstrap:
        raise ValueError("Bootstrap log does not cover every requested draw")
    seeds = np.random.SeedSequence(analysis_seed).spawn(n_bootstrap + 1)
    if summary["point_seed"] != int(seeds[0].generate_state(1)[0]):
        raise ValueError("Point seed differs from the declared analysis stream")
    groups = []
    for i, view in enumerate(summary["views"]):
        rows = read_json(directory / f"view_{i:03d}/source_rows.json")
        targets = [rows["target_ids"][j] for j in view["retained_source_indices"]]
        mapping = {}
        for j, target in enumerate(targets):
            mapping.setdefault(target, []).append(j)
        groups.append((view["modality"], mapping))
    counts, oob_counts, requested, successful = (
        np.zeros((n, n), dtype=np.uint64) for _ in range(4)
    )
    point_sets = [frozenset(np.flatnonzero(point == k)) for k in range(summary["selected_k"])]
    recoveries = np.zeros(len(point_sets), dtype=int)
    selected, reasons, failed = Counter(), Counter(), 0
    for i, (log, seed) in enumerate(zip(logs, seeds[1:], strict=True)):
        draw_seed, fit_seed = map(int, seed.generate_state(2))
        draw = np.random.default_rng(draw_seed).integers(n, size=n)
        absent = np.flatnonzero(np.bincount(draw, minlength=n) == 0)
        if (
            log["replicate"] != i
            or log["draw_seed"] != draw_seed
            or log["fit_seed"] != fit_seed
            or log["training_indices"] != draw.tolist()
            or log["oob_indices"] != absent.tolist()
        ):
            raise ValueError("Bootstrap draw or seed differs from its declared stream")
        mask = np.zeros(n, dtype=bool)
        mask[absent] = True
        oob = np.logical_and.outer(mask, mask)
        requested += oob
        if log["status"] == "failed":
            failed += 1
            if not log.get("error"):
                raise ValueError("Failed bootstrap needs a recorded reason")
            reasons[log["error"]] += 1
            continue
        labels = np.asarray(log["labels"])
        k = log["selected_k"]
        if (
            log["status"] != "complete"
            or labels.shape != (n,)
            or labels.dtype.kind not in "iu"
            or not k >= 1
            or np.any(labels < 0)
            or np.any(labels >= k)
        ):
            raise ValueError("Invalid saved bootstrap partition")
        for (name, mapping), view in zip(groups, log["views"], strict=True):
            expected = [j for index in draw for j in mapping[units[index]]]
            if view["modality"] != name or view["source_training_indices"] != expected:
                raise ValueError("Source members do not match the paired target draw")
        same = np.equal.outer(labels, labels)
        counts += same
        oob_counts += same & oob
        successful += oob
        predicted_sets = {frozenset(np.flatnonzero(labels == j)) for j in range(k)}
        recoveries += [group in predicted_sets for group in point_sets]
        selected[k] += 1
    completed = n_bootstrap - failed
    if summary["bootstrap"]["successful"] != completed or summary["bootstrap"]["failed"] != failed:
        raise ValueError("Bootstrap completion counts disagree with raw logs")
    if _canonical(summary["bootstrap"]["selected_k_counts"]) != _canonical(dict(selected)):
        raise ValueError("Bootstrap K counts disagree with raw logs")
    with np.load(directory / "stability.npz", allow_pickle=False) as arrays:
        for name, expected in (
            ("coassignment_counts", counts),
            ("oob_coassignment_counts", oob_counts),
            ("oob_requested", requested),
            ("oob_successful", successful),
        ):
            np.testing.assert_array_equal(arrays[name], expected)
        for prefix, numerator, denominator, valid in (
            (
                "coassignment",
                counts,
                np.full_like(counts, n_bootstrap),
                np.full_like(counts, completed),
            ),
            ("oob_coassignment", oob_counts, requested, successful),
        ):
            low = np.divide(
                numerator, denominator, out=np.full((n, n), np.nan), where=denominator > 0
            )
            high = np.divide(
                numerator + denominator - valid,
                denominator,
                out=np.full((n, n), np.nan),
                where=denominator > 0,
            )
            frequency = np.divide(
                numerator,
                denominator,
                out=np.full((n, n), np.nan),
                where=(denominator > 0) & (valid == denominator),
            )
            if prefix == "coassignment":
                np.fill_diagonal(low, 1)
                np.fill_diagonal(high, 1)
            for name, expected in (
                (prefix, frequency),
                (prefix + "_lower", low),
                (prefix + "_upper", high),
            ):
                np.testing.assert_allclose(arrays[name], expected, atol=0, rtol=0, equal_nan=True)
    clusters = read_json(directory / "clusters.json")
    if [c["exact_recovery_count"] for c in clusters] != recoveries.tolist():
        raise ValueError("Exact-set recovery counts disagree with saved partitions")
    for c, count in zip(clusters, recoveries, strict=True):
        if c["frequency"] != (int(count) / n_bootstrap if not failed else None):
            raise ValueError("Exact-set recovery frequency has the wrong denominator")
    pair_indices = np.triu_indices(n, 1)
    return {
        "requested": n_bootstrap,
        "successful": completed,
        "failed": failed,
        "failure_reasons": dict(reasons),
        "oob_requested_min": int(requested[pair_indices].min()),
        "oob_requested_median": float(np.median(requested[pair_indices])),
        "oob_requested_max": int(requested[pair_indices].max()),
    }


def audit_study(study_dir):
    """Require complete identities, re-score fits, and validate every requested draw."""
    from super_deli.benchmarking.metrics import score_run

    source = Path(study_dir).resolve()
    config = BenchmarkConfig.from_dict(read_json(source / "study.json"))
    outcomes = read_json(source / "execution-outcomes.json")
    if outcomes["status"] != "complete" or any(r["returncode"] != 0 for r in outcomes["outcomes"]):
        raise ValueError("Study execution is incomplete")
    if read_json(source / "execution.json")["configuration_sha256"] != configuration_hash(
        config.to_dict()
    ):
        raise ValueError("Execution declaration differs from the study configuration")
    expected_names = [s.name for s in config.scenarios]
    if [r["scenario"] for r in outcomes["outcomes"]] != expected_names:
        raise ValueError("Scenario outcomes differ from the study declaration")
    records, ledgers = [], []
    failed_point_logged_draws = 0
    draws = Counter()
    for i, scenario in enumerate(config.scenarios):
        shard = source / scenario.name / "runs"
        declared = replace(config, scenarios=(scenario,))
        if BenchmarkConfig.from_dict(read_json(shard / "config.json")) != declared:
            raise ValueError("Scenario shard changed study settings")
        summary = read_json(shard / "summary.json")
        ledger = shard / "runs.jsonl"
        if fingerprint_file(ledger).sha256 != summary["runs_sha256"] or summary[
            "configuration_sha256"
        ] != configuration_hash(declared.to_dict()):
            raise ValueError("Shard provenance hash mismatch")
        if (
            fingerprint_file(shard / "summary.json").sha256
            != outcomes["outcomes"][i]["summary_sha256"]
        ):
            raise ValueError("Execution summary hash mismatch")
        child = [json.loads(line) for line in ledger.read_text().splitlines()]
        keys = [(r["repetition"], r["arm"]) for r in child]
        expected = {(j, a.name) for j in range(config.n_datasets) for a in config.arms}
        if (
            len(keys) != len(expected)
            or set(keys) != expected
            or summary["attempted"] != len(expected)
        ):
            raise ValueError("Missing, duplicate or unexpected dataset/arm records")
        manifests = set()
        for record in child:
            repetition, arm_name = record["repetition"], record["arm"]
            if record["scenario"] != scenario.name or record["status"] not in {
                "complete",
                "failed",
            }:
                raise ValueError("Invalid scenario record identity or state")
            for purpose in ("simulation", "analysis"):
                if record[purpose + "_seed"] != derived_seed(
                    config.random_seed, scenario.name, repetition, purpose
                ):
                    raise ValueError("Dataset seed differs from study declaration")
            run = _child(shard, record["run_path"])
            if run != shard / f"{scenario.name}_{repetition:04d}" / arm_name:
                raise ValueError("Run path differs from its declared dataset/arm")
            simulation = run.parent / "simulation"
            manifest_path = simulation / "manifest.json"
            if record["simulation_manifest_sha256"] != fingerprint_file(manifest_path).sha256:
                raise ValueError("Simulation manifest hash mismatch")
            manifest = read_json(manifest_path)
            if manifest_path not in manifests:
                for relative, digest in manifest["input_hashes"].items():
                    if fingerprint_file(_child(simulation, relative)).sha256 != digest:
                        raise ValueError("Generated input or truth hash mismatch")
                manifests.add(manifest_path)
            if record["status"] == "complete":
                actual = read_json(run / "summary.json")
                arm = next(a for a in config.arms if a.name == arm_name)
                grouped = scenario.linkage == "grouped"
                model = replace(
                    config.model,
                    target_level="locality" if grouped else "specimen",
                    missing_views="error",
                    random_seed=record["analysis_seed"],
                    views=tuple(
                        replace(
                            v, aggregation="mean" if grouped and v.kind != "ecological" else "none"
                        )
                        for v in config.model.views
                        if v.kind in arm.kinds
                    ),
                )
                expected_configuration = {
                    "dataset": config_from_dict(
                        read_json(simulation / "cohort/dataset.json")
                    ).to_dict(),
                    "model": model.to_dict(),
                }
                if actual["unit_ids"] != manifest["common_ids"] or _canonical(
                    actual["provenance"]["configuration"]
                ) != _canonical(expected_configuration):
                    raise ValueError("Fitted cohort or configuration changed")
                if actual["provenance"]["configuration_sha256"] != configuration_hash(
                    expected_configuration
                ):
                    raise ValueError("Fit configuration hash mismatch")
                for name in ("mixture", "fusion"):
                    if (
                        fingerprint_file(run / f"{name}.npz").sha256
                        != actual["provenance"][name + "_sha256"]
                    ):
                        raise ValueError("Point-fit numerical artifact hash mismatch")
                for j, view in enumerate(actual["views"]):
                    directory = run / f"view_{j:03d}"
                    rows = read_json(directory / "source_rows.json")
                    if (
                        fingerprint_file(directory / "source_values.npy").sha256
                        != rows["array_sha256"]
                    ):
                        raise ValueError("Prepared source array hash mismatch")
                    if read_json(directory / "parameters/summary.json") != view:
                        raise ValueError("View parameter summary changed")
                    for name, digest in view["arrays"].items():
                        if (
                            fingerprint_file(_child(directory / "parameters", name)).sha256
                            != digest
                        ):
                            raise ValueError("View parameter array hash mismatch")
                truth = {r["unit_id"]: r for r in read_json(simulation / "truth.json")["units"]}
                evaluation = _canonical(score_run(run, truth))
                if evaluation != record["evaluation"] or evaluation != read_json(
                    run / "evaluation.json"
                ):
                    raise ValueError("Saved metrics differ from independent re-scoring")
                draw_counts = audit_bootstrap(
                    run, record["analysis_seed"], config.model.n_bootstrap
                )
                draws.update({k: draw_counts[k] for k in ("requested", "successful", "failed")})
                record = {
                    **record,
                    "evaluation": evaluation,
                    "bootstrap_failure_reasons": draw_counts["failure_reasons"],
                    "oob_draw_counts": {
                        k: draw_counts[k]
                        for k in ("oob_requested_min", "oob_requested_median", "oob_requested_max")
                    },
                }
            else:
                if (run / "summary.json").exists():
                    raise ValueError("A failed ledger entry has a completion summary")
                if (run / "failure.json").exists() and read_json(run / "failure.json")[
                    "error"
                ] != record["error"]:
                    raise ValueError("Point-failure reason disagrees with its artifact")
                if (run / "bootstrap_replicates.json").exists():
                    failed_point_logged_draws += len(read_json(run / "bootstrap_replicates.json"))
            records.append({**record, "run_path": str(run.relative_to(source))})
        if sum(r["status"] == "failed" for r in child) != summary["point_failed"]:
            raise ValueError("Point failure counts disagree with ledger")
        ledgers.append({"scenario": scenario.name, "sha256": fingerprint_file(ledger).sha256})
    return (
        config,
        records,
        {
            "source_ledgers": ledgers,
            "bootstrap": dict(draws),
            "expected_point_fits": len(config.scenarios) * config.n_datasets * len(config.arms),
            "expected_bootstrap_draws": len(config.scenarios)
            * config.n_datasets
            * len(config.arms)
            * config.model.n_bootstrap,
            "unscored_logged_draws_after_failed_points": failed_point_logged_draws,
            "draws_not_logged_after_failed_points": (
                len(config.scenarios)
                * config.n_datasets
                * len(config.arms)
                * config.model.n_bootstrap
                - draws["requested"]
                - failed_point_logged_draws
            ),
        },
    )


def summarize_study(config, records, audit):
    from super_deli.benchmarking.metrics import summarize_runs, summarize_values

    summary = {
        "schema_version": 1,
        "status": "complete",
        "audit": audit,
        "configuration_sha256": configuration_hash(config.to_dict()),
        "point_failed": sum(r["status"] == "failed" for r in records),
        "point_failure_records": [
            {k: r[k] for k in ("scenario", "arm", "repetition", "error", "error_type")}
            for r in records
            if r["status"] == "failed"
        ],
        **summarize_runs(records, config),
    }
    bins = []
    for group in summary["groups"]:
        subset = [
            r for r in records if (r["scenario"], r["arm"]) == (group["scenario"], group["arm"])
        ]
        complete = [r for r in subset if r["status"] == "complete"]
        available = [r for r in complete if r["evaluation"]["bootstrap"]["status"] == "available"]
        group["rates"] = {
            "point_success": rate_summary(
                [int(r["status"] == "complete") for r in subset], len(subset)
            ),
            "fit_and_k_correct": rate_summary(
                [
                    int(r["status"] == "complete" and r["evaluation"]["metrics"]["k_correct"] == 1)
                    for r in subset
                ],
                len(subset),
            ),
            **{
                name: rate_summary(
                    [r["evaluation"]["metrics"][name] for r in complete], len(subset)
                )
                for name in ("k_correct", "false_split")
            },
            "any_stable_cluster": rate_summary(
                [int(r["evaluation"]["stable_cluster_count"] > 0) for r in available], len(subset)
            ),
            "any_stable_wrong_cluster": rate_summary(
                [int(r["evaluation"]["stable_wrong_cluster_count"] > 0) for r in available],
                len(subset),
            ),
        }
        group["stable_clusters"] = sum(r["evaluation"]["stable_cluster_count"] for r in available)
        group["stable_wrong_clusters"] = sum(
            r["evaluation"]["stable_wrong_cluster_count"] for r in available
        )
        reasons = Counter()
        for record in complete:
            reasons.update(record["bootstrap_failure_reasons"])
        group["bootstrap_failure_reasons"] = dict(reasons)
        group["oob_draw_counts"] = {
            k: summarize_values(
                [r.get("oob_draw_counts", {}).get(k) for r in complete], len(subset)
            )
            for k in ("oob_requested_min", "oob_requested_median", "oob_requested_max")
        }
        # Bin means weight contributing datasets equally, not their pair counts.
        for scope in ("all_pairs", "oob_pairs"):
            for j in range(10):
                selected = [
                    b
                    for r in complete
                    for b in r["evaluation"]["coassignment_bins"]
                    if b["scope"] == scope and b["lower"] == j / 10 and b["pairs"] > 0
                ]
                bins.append(
                    {
                        "scenario": group["scenario"],
                        "arm": group["arm"],
                        "scope": scope,
                        "lower": j / 10,
                        "upper": (j + 1) / 10,
                        "total_pairs_descriptive": sum(b["pairs"] for b in selected),
                        "frequency": summarize_values(
                            [b["mean_frequency"] for b in selected], len(subset)
                        ),
                        "truth_same": summarize_values(
                            [b["true_same_fraction"] for b in selected], len(subset)
                        ),
                    }
                )
    summary["support_bins"] = bins
    return summary


def write_analysis(study_dir, *, output_dir):
    """Write compact audited results; leave every raw study artifact unchanged."""
    try:
        from super_deli.benchmarking.metrics import METRIC_NAMES
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for study analysis") from exc

    destination = Path(output_dir).absolute()
    if destination.resolve().is_relative_to(Path(study_dir).resolve()):
        raise ValueError("Analysis output must be outside the immutable raw study directory")
    destination.mkdir(exist_ok=False)
    try:
        config, records, audit = audit_study(study_dir)
        summary = summarize_study(config, records, audit)
        with (destination / "datasets.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            base = (
                "scenario",
                "arm",
                "repetition",
                "status",
                "simulation_seed",
                "analysis_seed",
                "point_seed",
                "error",
                "error_type",
                "selected_k",
                "truth_k_retained",
                "n_retained",
                "stable_cluster_count",
                "stable_wrong_cluster_count",
                "bootstrap_requested",
                "bootstrap_successful",
                "bootstrap_failed",
            )
            writer.writerow((*base, *METRIC_NAMES))
            for record in records:
                evaluation = record.get("evaluation", {})
                row = {
                    **evaluation,
                    **record,
                    **{
                        "bootstrap_" + k: evaluation.get("bootstrap", {}).get(k)
                        for k in ("requested", "successful", "failed")
                    },
                }
                writer.writerow(
                    (
                        *(row.get(k) for k in base),
                        *(evaluation.get("metrics", {}).get(k) for k in METRIC_NAMES),
                    )
                )
        with (destination / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                ("scenario", "arm", "metric", "mean", "mcse", "n_available", "n_missing")
            )
            for group in summary["groups"]:
                for name, value in group["metrics"].items():
                    writer.writerow(
                        (
                            group["scenario"],
                            group["arm"],
                            name,
                            *(value[k] for k in ("mean", "mcse", "n_available", "n_missing")),
                        )
                    )
        write_json(config.to_dict(), destination / "study.json")
        with (destination / "paired.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scenario",
                    "reference",
                    "comparator",
                    "metric",
                    "mean_difference",
                    "mcse",
                    "n_available",
                    "n_missing",
                )
            )
            for pair in summary["paired_comparisons"]:
                for name, values in pair["metrics"].items():
                    writer.writerow(
                        (
                            pair["scenario"],
                            pair["reference"],
                            pair["comparator"],
                            name,
                            *(values[k] for k in ("mean", "mcse", "n_available", "n_missing")),
                        )
                    )
        with (destination / "rates.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scenario",
                    "arm",
                    "event",
                    "events",
                    "n_available",
                    "n_missing",
                    "rate",
                    "wilson_95_lower",
                    "wilson_95_upper",
                )
            )
            for group in summary["groups"]:
                for name, values in group["rates"].items():
                    interval = values["wilson_95"] or [None, None]
                    writer.writerow(
                        (
                            group["scenario"],
                            group["arm"],
                            name,
                            *(values[k] for k in ("events", "n_available", "n_missing", "mean")),
                            *interval,
                        )
                    )
        with (destination / "support-bins.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scenario",
                    "arm",
                    "scope",
                    "lower",
                    "upper",
                    "n_available",
                    "n_missing",
                    "mean_frequency",
                    "frequency_mcse",
                    "mean_truth_same_fraction",
                    "truth_same_mcse",
                    "total_pairs_descriptive",
                )
            )
            for row in summary["support_bins"]:
                writer.writerow(
                    (
                        *(row[k] for k in ("scenario", "arm", "scope", "lower", "upper")),
                        *(
                            row["frequency"][k]
                            for k in ("n_available", "n_missing", "mean", "mcse")
                        ),
                        row["truth_same"]["mean"],
                        row["truth_same"]["mcse"],
                        row["total_pairs_descriptive"],
                    )
                )
        write_json(summary, destination / "summary.json")
        with (destination / "files.jsonl").open("x", encoding="utf-8") as handle:
            root = Path(study_dir).resolve()
            for path in sorted(root.rglob("*")):
                if path.is_file() and "cache" not in path.relative_to(root).parts:
                    fingerprint = fingerprint_file(path)
                    handle.write(
                        json.dumps(
                            {
                                "path": str(path.relative_to(root)),
                                "sha256": fingerprint.sha256,
                                "size_bytes": fingerprint.size_bytes,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        write_json(
            {
                "status": "complete",
                "files_manifest_sha256": fingerprint_file(destination / "files.jsonl").sha256,
                "summary_sha256": fingerprint_file(destination / "summary.json").sha256,
                "datasets_sha256": fingerprint_file(destination / "datasets.csv").sha256,
            },
            destination / "audit.json",
        )
        return summary
    except Exception as exc:
        write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
        raise
