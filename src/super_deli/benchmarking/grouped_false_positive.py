"""Streaming, no-refit audits and prespecified null rejection-rate intervals."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_boundary_analysis import (
    audit_arm,
    verify_counts,
    verify_reference,
)
from super_deli.benchmarking.grouped_boundary_metrics import raw_metrics
from super_deli.benchmarking.grouped_boundary_reference import CASES, known_model, load_model
from super_deli.benchmarking.grouped_likelihood import case_policy
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig

SCOPE = (
    "Fixed-procedure Gaussian-law rejection rates from independent simulated observations. "
    "Known-law controls are simulation diagnostics, not empirical replacements. "
    "No correction, species-boundary claim, default change, or stability resampling."
)


def exact_interval(events, n, *, error=0.05):
    """Two-sided Clopper-Pearson interval; no denominator means unavailable."""
    from scipy.stats import beta

    if type(events) is not int or type(n) is not int or not 0 <= events <= n:
        raise ValueError("Use integer events and denominator with 0 <= events <= n")
    if not 0 < error < 1:
        raise ValueError("Interval error must be strictly between zero and one")
    if not n:
        return None
    return [
        0.0 if not events else float(beta.ppf(error / 2, events, n - events + 1)),
        1.0 if events == n else float(beta.ppf(1 - error / 2, events + 1, n - events)),
    ]


def rejection_rate(events, available, requested, *, family_size=10):
    if type(requested) is not int or not 0 <= available <= requested or requested < 1:
        raise ValueError("Availability must fit a positive requested denominator")
    if type(family_size) is not int or family_size < 1:
        raise ValueError("Family size must be a positive integer")
    pointwise = exact_interval(events, available)
    adjusted = exact_interval(events, available, error=0.05 / family_size)
    missing = requested - available
    rate = events / available if available else None
    return {
        "events": events,
        "available": available,
        "requested": requested,
        "withheld": missing,
        "rate_among_available": rate,
        "mcse_among_available": math.sqrt(rate * (1 - rate) / available) if available else None,
        "exact_interval_95": pointwise,
        "bonferroni_familywise_95": adjusted,
        "family_size": family_size,
        "withheld_outcome_bounds": [events / requested, (events + missing) / requested],
        "all_requested_pointwise_envelope": [
            exact_interval(events, requested)[0],
            exact_interval(events + missing, requested)[1],
        ],
        "scope": "Conditional-on-availability intervals; missing-outcome bounds are not CIs",
    }


def paired_decisions(left, right, *, error=0.05):
    """Known-minus-fitted rate difference using exact marginal discordance bounds."""
    if left.keys() != right.keys():
        raise ValueError("Paired arms must contain identical observation identifiers")
    pairs = [
        (left[k], right[k]) for k in sorted(left) if left[k] is not None and right[k] is not None
    ]
    if any(type(v) is not bool for pair in pairs for v in pair):
        raise ValueError("Available decisions must be Boolean")
    n = len(pairs)
    known_only = sum(not a and b for a, b in pairs)
    fitted_only = sum(a and not b for a, b in pairs)
    interval = None
    if n:
        a = exact_interval(known_only, n, error=error / 2)
        b = exact_interval(fitted_only, n, error=error / 2)
        interval = [a[0] - b[1], a[1] - b[0]]
    return {
        "available_pairs": n,
        "requested_pairs": len(left),
        "withheld_pairs": len(left) - n,
        "known_only_rejections": known_only,
        "fitted_only_rejections": fitted_only,
        "both_reject": sum(a and b for a, b in pairs),
        "both_do_not_reject": sum(not a and not b for a, b in pairs),
        "known_minus_fitted": (known_only - fitted_only) / n if n else None,
        "conservative_exact_difference_interval": interval,
        "confidence": 1 - error,
        "method": "Bonferroni exact marginal discordance bounds; no arm independence assumption",
    }


def summarize_tails(rows, requested_by_case):
    rates, paired = [], []
    for case in CASES:
        decisions = {}
        for arm in ("fitted", "known"):
            group = [
                r
                for r in rows
                if r["case"] == case and r["arm"] == arm and r["statistic_kind"] == "likelihood"
            ]
            ids = [r["repetition"] for r in group]
            if sorted(ids) != list(range(requested_by_case[case])):
                raise ValueError("Missing, duplicated or unexpected primary tail identity")
            decisions[arm] = {r["repetition"]: r["reject"] for r in group}
            available = [r for r in group if r["p_value"] is not None]
            rates.append(
                {
                    "case": case,
                    "arm": arm,
                    **rejection_rate(
                        sum(r["reject"] for r in available), len(available), requested_by_case[case]
                    ),
                }
            )
        paired.append(
            {
                "case": case,
                **paired_decisions(decisions["fitted"], decisions["known"]),
                "simultaneous_across_five_cases": paired_decisions(
                    decisions["fitted"], decisions["known"], error=0.05 / len(CASES)
                ),
            }
        )
    return {
        "primary_rates": rates,
        "paired_decisions": paired,
        "multiple_comparisons": "Ten primary rate intervals use Bonferroni exact coverage",
        "BIC": "Retained descriptive comparator; excluded from primary interval family",
    }


def _audit_observation(task):
    """Audit one observed dataset and both references, without fitting any model."""
    import numpy as np

    from super_deli.benchmarking.grouped_analysis import verify_model

    source, case, i, ledger, declaration = task
    study = GroupedLikelihoodStudyConfig.from_dict(declaration)
    base = Path(source) / case / f"dataset_{i:04d}"
    policy = case_policy(study, case, i)
    seed = derived_seed(study.random_seed, case, i, "observed_data")
    if (ledger["case"], ledger["repetition"], ledger["data_seed"], ledger["null_stream_seed"]) != (
        case,
        i,
        seed,
        policy.random_seed,
    ):
        raise ValueError("Observation identity or seed differs")
    if hashlib.sha256((base / "values.npz").read_bytes()).hexdigest() != ledger["values_sha256"]:
        raise ValueError("Observed array checksum differs")
    values, layout, _ = simulate_grouped(
        case, study.n_units, seed, target_level=policy.model.target_level
    )
    with np.load(base / "values.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(values):
            raise ValueError("Observed array fields differ")
        for name, expected in values.items():
            np.testing.assert_array_equal(arrays[name], expected)
    if not (base / "calibration/null_model.npz").exists():
        if ledger["status"] != "failed" or not (base / "failure.json").exists():
            raise ValueError("Missing raw fit without retained failure")
        verify_counts(ledger, [], policy, "fresh")
        tails = [
            {
                "case": case,
                "repetition": i,
                "arm": arm,
                "statistic_kind": kind,
                "p_value": None,
                "reject": None,
                "exceedances": None,
                "requested_draws": policy.n_simulations,
                "successful_draws": 0,
                "failed_draws": 0,
                "unattempted_draws": policy.n_simulations,
                "conditional_tail_interval_95": None,
                "statistic": None,
            }
            for arm in ("fitted", "known")
            for kind in ("likelihood", "bic")
        ]
        return {
            "tails": tails,
            "covariance": None,
            "pipelines": [],
            "members": 0,
            "case": case,
            "repetition": i,
            "ledger": ledger,
        }
    verify_model(base, base / "calibration", values, layout, policy)
    fitted = load_model(base / "calibration")
    truth, known = known_model(fitted, case), load_model(base / "known")
    verify_reference(truth, case)
    verify_reference(known, case)
    for name in ("morphology", "ecology", "targets", "counts", "values"):
        np.testing.assert_array_equal(getattr(known, name), getattr(fitted, name))
    if read_json(base / "known/observed.json") != read_json(base / "calibration/observed.json"):
        raise ValueError("Shared observed fit differs")
    left, right = [
        read_json(base / arm / "archive_inventory.json") for arm in ("calibration", "known")
    ]
    if {k: v for k, v in left.items() if k.startswith("observed/")} != {
        k: v for k, v in right.items() if k.startswith("observed/")
    }:
        raise ValueError("Shared observed archive bytes differ")
    pipelines, rows, members = [], [], 0
    for arm, directory, model in (
        ("fitted", base / "calibration", fitted),
        ("known", base / "known", known),
    ):
        points, inference, count = audit_arm(
            directory, model, values, policy, cohort="fresh", case=case, repetition=i, arm=arm
        )
        pipelines.extend(points)
        rows.extend({"case": case, "repetition": i, "arm": arm, **r} for r in inference)
        members += count
    verify_counts(ledger, pipelines, policy, "fresh")
    return {
        "tails": rows,
        "covariance": {"case": case, "repetition": i, **raw_metrics(fitted, truth)},
        "pipelines": pipelines,
        "members": members,
        "case": case,
        "repetition": i,
        "ledger": ledger,
    }


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit_observation(task)


def _csv_write(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            for r in rows
        )


def audit_study(root, *, output_dir, workers=1):
    """Stream unique point records to compressed evidence; bound audit concurrency."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four single-thread audit workers")
    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    destination.mkdir(exist_ok=False)
    tasks, requested, policies, totals = [], {}, [], {}
    for case in CASES:
        declaration = read_json(root / case / "study.json")
        study = GroupedLikelihoodStudyConfig.from_dict(declaration)
        normalized = study.to_dict()
        policies.append(normalized)
        if normalized != policies[0]:
            raise ValueError("Scenario configurations differ")
        ledgers = [json.loads(s) for s in (root / case / "datasets.jsonl").read_text().splitlines()]
        if len(ledgers) != study.n_datasets:
            raise ValueError("Observed dataset ledger count differs")
        requested[case] = study.n_datasets
        summary = read_json(root / case / "summary.json")
        for key in (
            "planned_fits",
            "attempted_fits",
            "successful_fits",
            "failed_fits",
            "unattempted_fits",
        ):
            value = sum(r[key] for r in ledgers)
            if summary[key] != value:
                raise ValueError("Scenario summary and retained ledgers differ")
            totals[key] = totals.get(key, 0) + value
        tasks.extend((str(root), case, i, ledger, declaration) for i, ledger in enumerate(ledgers))
    rows, covariances, observations, members, records = [], [], [], 0, 0
    handles, writers = {}, {}
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = (
        pool.map(audit_observation, tasks, chunksize=1) if pool else map(audit_observation, tasks)
    )
    try:
        for result in results:
            case = result["case"]
            rows.extend(result["tails"])
            if result["covariance"] is not None:
                covariances.append(result["covariance"])
            observations.append(result["ledger"])
            members += result["members"]
            for record in result["pipelines"]:
                if case not in handles:
                    handles[case] = gzip.open(
                        destination / f"pipelines-{case}.csv.gz", "xt", newline=""
                    )
                    # Failed records may precede complete records; fix the schema explicitly.
                    fields = [
                        "cohort",
                        "case",
                        "repetition",
                        "arm",
                        "replicate",
                        "status",
                        "statistic",
                        "bic_statistic",
                        "selected_k",
                        "bic_selected_k",
                        "lr_maximizer_k",
                        "eligible_ks",
                        "maximum_iterations",
                        "count_association",
                        "lr_count_association",
                        "covariance_min_over_regularizer",
                        "morphology_retained_variance",
                        "ecology_retained_variance",
                        "joint_retained_variance",
                    ]
                    writers[case] = csv.DictWriter(handles[case], fieldnames=fields)
                    writers[case].writeheader()
                writers[case].writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in record.items()
                    }
                )
                records += 1
            if len(observations) % 25 == 0:
                (destination / "progress.json").write_text(
                    json.dumps(
                        {
                            "observations_audited": len(observations),
                            "unique_pipelines": records,
                            "archive_members": members,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()
        if pool:
            pool.shutdown(cancel_futures=True)
    if records != totals["attempted_fits"]:
        raise ValueError("Unique audited pipeline count differs")
    for name, values in (
        ("tails", rows),
        ("covariance", covariances),
        ("observations", observations),
    ):
        _csv_write(destination / f"{name}.csv", values)
    result = {
        "scope": SCOPE,
        "no_model_refits": True,
        "observations": len(observations),
        "requested_by_case": requested,
        "unique_pipeline_records_audited": records,
        "archive_members": members,
        "fit_accounting": totals,
        **summarize_tails(rows, requested),
    }
    write_json(inventory(root), destination / "files.json")
    write_json(result, destination / "summary.json")
    return result
