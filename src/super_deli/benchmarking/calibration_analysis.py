"""Audit a four-scenario score study without refitting or modifying saved runs."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.score_calibration import SCENARIOS, simulate_scores, study_options
from super_deli.benchmarking.study_analysis import rate_summary
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def _selection(candidates, search, n):
    from super_deli.models.gaussian_calibration import bic_improvement

    if [c["k"] for c in candidates] != list(range(1, search.max_k + 1)):
        raise ValueError("Candidate log does not cover the configured K search")
    for c in candidates:
        if c["status"] == "insufficient_samples":
            if n >= c["k"] * search.min_cluster_size or c["bic"] is not None:
                raise ValueError("Invalid insufficient-sample candidate")
        elif c["status"] in {"eligible", "small_component"}:
            sizes = c["hard_cluster_sizes"]
            if (
                len(sizes) != c["k"]
                or sum(sizes) != n
                or min(sizes) < 0
                or (min(sizes) >= search.min_cluster_size) != (c["status"] == "eligible")
                or not c["converged"]
            ):
                raise ValueError("Candidate sizes/status disagree")
        else:
            raise ValueError("Unknown candidate status")
    eligible = [c for c in candidates if c["status"] == "eligible"]
    return bic_improvement(candidates), min(eligible, key=lambda c: (c["bic"], c["k"]))["k"]


def audit_score_study(root, *, output_dir):
    try:
        import numpy as np

        from super_deli.models.gaussian_calibration import monte_carlo_summary, score_hash
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for score-study analysis") from exc

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Analysis must be outside immutable study shards")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    rows, rates, inventory, original_config = [], [], {}, None
    for scenario in SCENARIOS:
        directory = root / scenario
        options = read_json(directory / "study.json")
        config, search = study_options(options)
        if original_config is not None and options != original_config:
            raise ValueError("Scenario shards differ in study configuration")
        original_config = options
        shard = read_json(directory / "summary.json")
        if (
            shard["status"] != "complete"
            or shard["scenario"] != scenario
            or shard["configuration_sha256"] != configuration_hash(options)
            or shard["requested"] != options["n_repetitions"]
            or len(shard["datasets"]) != options["n_repetitions"]
        ):
            raise ValueError("Study shard is incomplete or differs from its declaration")
        events = []
        for i, row in enumerate(shard["datasets"]):
            data_seed = derived_seed(options["random_seed"], scenario, i, "data")
            fit_seed = derived_seed(options["random_seed"], scenario, i, "calibration")
            if (
                row["scenario"] != scenario
                or row["repetition"] != i
                or row["data_seed"] != data_seed
                or row["calibration_seed"] != fit_seed
                or row["directory"] != f"rep_{i:04d}"
            ):
                raise ValueError("Dataset identity or seed stream mismatch")
            ledger = read_json(directory / f"ledger_{i:04d}.json")
            if ledger != shard["datasets"][: i + 1]:
                raise ValueError("Attempt ledger differs from completion summary")
            scores = simulate_scores(scenario, options["n_units"], data_seed)
            if score_hash(scores) != row["scores_sha256"]:
                raise ValueError("Observed dataset checksum differs from its generator")
            if row["status"] == "point_failed":
                if not row.get("error") or row["successful_draws"] or row["failed_draws"]:
                    raise ValueError("Invalid point failure record")
                events.append(None)
                rows.append(row)
                continue
            result_dir = directory / row["directory"]
            if fingerprint_file(result_dir / "summary.json").sha256 != row["summary_sha256"]:
                raise ValueError("Calibration summary checksum mismatch")
            result = read_json(result_dir / "summary.json")
            policy = replace(config, random_seed=fit_seed)
            if result["calibration"] != policy.to_dict() or result[
                "search_configuration"
            ] != json.loads(json.dumps(search.to_dict())):
                raise ValueError("Calibration settings differ from the study declaration")
            for name in ("null_model.npz", "null_replicates.json"):
                if fingerprint_file(result_dir / name).sha256 != result["artifacts"][name]:
                    raise ValueError("Calibration numerical artifact checksum mismatch")
            with np.load(result_dir / "null_model.npz", allow_pickle=False) as arrays:
                if not np.array_equal(scores, arrays["scores"]):
                    raise ValueError("Saved observed scores differ from the declared generator")
                mean, covariance = arrays["mean"], arrays["covariance"]
            expected_mean = scores.mean(axis=0)
            centered = scores - expected_mean
            if (
                not np.array_equal(mean, expected_mean)
                or not np.array_equal(covariance, centered.T @ centered / len(scores))
                or score_hash(scores) != row["scores_sha256"]
                or result["scores_sha256"] != row["scores_sha256"]
            ):
                raise ValueError("Fitted Gaussian parameters or score checksum mismatch")
            records = read_json(result_dir / "null_replicates.json")
            if len(records) != config.n_simulations:
                raise ValueError("Null simulation log does not cover every requested draw")
            seeds = np.random.SeedSequence(fit_seed).spawn(config.n_simulations + 1)
            if result["point_seed"] != int(seeds[0].generate_state(1)[0]):
                raise ValueError("Point seed differs from its declared stream")
            t, k = _selection(result["observed_candidates"], search, len(scores))
            if (
                t != result["observed_statistic"]
                or k != result["observed_selected_k"]
                or t != row["statistic"]
                or k != row["selected_k"]
            ):
                raise ValueError("Observed statistic or selected K mismatch")
            factor = np.linalg.cholesky(covariance)
            for j, (record, seed) in enumerate(zip(records, seeds[1:], strict=True)):
                draw_seed, em_seed = map(int, seed.generate_state(2))
                draw = (
                    mean + np.random.default_rng(draw_seed).standard_normal(scores.shape) @ factor.T
                )
                if (
                    record["replicate"] != j
                    or record["draw_seed"] != draw_seed
                    or record["fit_seed"] != em_seed
                    or record["scores_sha256"] != score_hash(draw)
                ):
                    raise ValueError("Null draw or seed checksum mismatch")
                if record["status"] == "complete":
                    t_null, k_null = _selection(record["candidates"], search, len(scores))
                    if t_null != record["statistic"] or k_null != record["selected_k"]:
                        raise ValueError("Null statistic or selected K mismatch")
                elif record["status"] != "failed" or not record.get("error"):
                    raise ValueError("Invalid failed null-draw record")
            inference = monte_carlo_summary(t, records, alpha=config.alpha)
            if (
                inference != result["inference"]
                or inference["status"] != row["status"]
                or inference["p_value"] != row["p_value"]
                or inference["reject_single_gaussian"] != row["reject"]
                or inference["successful"] != row["successful_draws"]
                or inference["failed"] != row["failed_draws"]
            ):
                raise ValueError("Inference differs from reconstructed simulation outcomes")
            events.append(int(row["reject"]) if row["reject"] is not None else None)
            rows.append(row)
        expected_counts = {
            "planned_null_draws": options["n_repetitions"] * config.n_simulations,
            "attempted_null_draws": sum(
                r["successful_draws"] + r["failed_draws"] for r in shard["datasets"]
            ),
            "successful_null_draws": sum(r["successful_draws"] for r in shard["datasets"]),
        }
        if any(shard[key] != value for key, value in expected_counts.items()):
            raise ValueError("Shard requested/attempted/successful denominators disagree")
        rate = rate_summary(events, options["n_repetitions"])
        if rate != shard["rejection_rate"]:
            raise ValueError("Rejection-rate summary differs from independent dataset events")
        rates.append({"scenario": scenario, "interpretation": shard["rate_interpretation"], **rate})
    for path in sorted(root.rglob("*")):
        if path.is_file():
            inventory[str(path.relative_to(root))] = fingerprint_file(path).sha256
    destination.mkdir(exist_ok=False)
    attempted_draws = sum(r["successful_draws"] + r["failed_draws"] for r in rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "study": original_config,
        "datasets": len(rows),
        "attempted_null_draws": attempted_draws,
        "unattempted_null_draws": len(rows) * original_config["calibration"]["n_simulations"]
        - attempted_draws,
        "point_failures": sum(r["status"] == "point_failed" for r in rows),
        "planned_null_draws": len(rows) * original_config["calibration"]["n_simulations"],
        "successful_null_draws": sum(r["successful_draws"] for r in rows),
        "failed_null_draws": sum(r["failed_draws"] for r in rows),
        "rates": rates,
        "raw_files": len(inventory),
        "audit_scope": (
            "Reconstructed generators, parameters, seeds, draws, selection and p-values; no refits"
        ),
    }
    for name, values in (("datasets.csv", rows), ("rates.csv", rates)):
        fields = sorted(set().union(*(r.keys() for r in values)))
        with (destination / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    write_json(inventory, destination / "files.json")
    write_json(summary, destination / "summary.json")
    return summary
