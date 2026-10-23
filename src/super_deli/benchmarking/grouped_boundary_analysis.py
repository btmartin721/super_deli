"""No-refit known-law, dense raw likelihood, all-candidate and exact-tail audits."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.full_grouped_diagnostic_analysis import csv_rows
from super_deli.benchmarking.grouped_analysis import verify_model
from super_deli.benchmarking.grouped_boundary_diagnostics import SCOPE
from super_deli.benchmarking.grouped_boundary_metrics import point_metrics, raw_metrics, summaries
from super_deli.benchmarking.grouped_boundary_reference import CASES, known_model, load_model
from super_deli.benchmarking.grouped_likelihood import case_policy
from super_deli.benchmarking.grouped_likelihood_analysis import audit_candidates
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig


def verify_reference(model, case):
    """Independent direct covariance construction, without reference-factory calls."""
    import numpy as np

    if case not in CASES:
        raise ValueError("Unknown reference law")
    if case in CASES[:2]:
        scales = [0.5 + 1.5 * i / 7 for i in range(8)]
        b = (
            np.eye(8)
            if case == CASES[0]
            else np.array(
                [[scales[i] * scales[j] * 0.6 ** abs(i - j) for j in range(8)] for i in range(8)]
            )
        )
        mean = np.zeros(8) if case == CASES[0] else np.array([-2 + 4 * i / 7 for i in range(8)])
        r = (
            0.75 * np.eye(4)
            if case == CASES[0]
            else np.array([[1.5 * 0.35 ** abs(i - j) for j in range(4)] for i in range(4)])
        )
    else:
        vectors = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
        spectrum = (
            [0, 0, 0, 0]
            if case == CASES[2]
            else [0.8, 0.4, 0, 0]
            if case == CASES[3]
            else [0.8, 0.4, 0.02, 0.005]
        )
        c = sum(x * np.outer(vectors[:, i], vectors[:, i]) for i, x in enumerate(spectrum))
        a = np.zeros((4, 4))
        for i in range(4):
            a[i, i], a[i, (i + 1) % 4] = 0.6, 0.2
        b = np.block([[c + a @ a.T, a], [a.T, np.eye(4)]])
        mean, r = np.zeros(8), 0.75 * np.eye(4)
    for actual, expected in ((model.mean, mean), (model.covariance, b), (model.within, r)):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(model.sampling_factor @ model.sampling_factor.T, b, atol=2e-14)
    np.testing.assert_allclose(
        model.within_sampling_factor @ model.within_sampling_factor.T, r, atol=2e-14
    )


def regenerated(model, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(model.counts), 8))
    residuals = rng.standard_normal(model.morphology.shape)
    latent = model.mean + z @ model.sampling_factor.T
    order = {u: i for i, u in enumerate(model.layout["unit_ids"])}
    ecology_ids = next(v["target_ids"] for v in model.layout["views"] if v["kind"] == "ecological")
    return {
        "morphological": latent[model.targets, :4] + residuals @ model.within_sampling_factor.T,
        "ecological": latent[[order[u] for u in ecology_ids], 4:],
    }


def manual_statistics(record, dimensions):
    import math

    if record["status"] != "complete":
        return None
    eligible = []
    for c in record["candidates"]:
        if c["status"] == "insufficient_samples":
            continue
        k, ll = c["k"], c["log_likelihood"]
        q = k - 1 + k * dimensions + k * dimensions * (dimensions + 1) // 2
        if not math.isclose(c["bic"], -2 * ll + q * math.log(record["n_retained"]), abs_tol=1e-8):
            raise ValueError("Independent BIC equation differs")
        if not math.isclose(c["aic"], -2 * ll + 2 * q, abs_tol=1e-8):
            raise ValueError("Independent AIC equation differs")
        if c["status"] == "eligible":
            eligible.append(c)
    one = next(c for c in eligible if c["k"] == 1)
    result = {
        "likelihood": 2 * (max(c["log_likelihood"] for c in eligible) - one["log_likelihood"]),
        "bic": one["bic"] - min(c["bic"] for c in eligible),
    }
    if result["likelihood"] != record["statistic"] or result["bic"] != record["bic_statistic"]:
        raise ValueError("Independent statistic differs")
    for name, key in (("aic", "selected_k"), ("bic", "bic_selected_k")):
        if min(eligible, key=lambda c: (c[name], c["k"]))["k"] != record[key]:
            raise ValueError("Independent selected K differs")
    return result


def tails(observed, records, policy):
    from scipy.stats import beta, binom

    obs = manual_statistics(observed, policy.model.n_pcs)
    values = [manual_statistics(r, policy.model.n_pcs) for r in records]
    complete = (
        obs is not None
        and len(values) == policy.n_simulations
        and all(v is not None for v in values)
    )
    output = []
    for key in ("likelihood", "bic"):
        x = sum(v[key] >= obs[key] for v in values if v is not None) if obs is not None else None
        n = policy.n_simulations
        p = (1 + x) / (1 + n) if complete else None
        interval = None
        if complete:
            lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
            hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
            if x and abs(binom.sf(x - 1, n, lo) - 0.025) > 1e-9:
                raise ValueError("Independent lower tail interval equation differs")
            if x < n and abs(binom.cdf(x, n, hi) - 0.025) > 1e-9:
                raise ValueError("Independent upper tail interval equation differs")
            interval = [lo, hi]
        output.append(
            {
                "statistic_kind": key,
                "statistic": obs[key] if obs is not None else None,
                "p_value": p,
                "reject": p <= policy.alpha if p is not None else None,
                "exceedances": x,
                "requested_draws": n,
                "successful_draws": sum(v is not None for v in values),
                "failed_draws": sum(v is None for v in values),
                "unattempted_draws": n - len(values),
                "conditional_tail_interval_95": interval,
            }
        )
    return output


def audit_arm(directory, model, values, policy, *, cohort, case, repetition, arm):

    observed = read_json(directory / "observed.json")
    records = [read_json(p) for p in sorted(directory.glob("draw_*.json"))]
    if (directory / "null_replicates.json").exists() and records != read_json(
        directory / "null_replicates.json"
    ):
        raise ValueError("Replicate ledger differs from retained draws")
    rows, members = [], 0
    declared = read_json(directory / "archive_inventory.json")
    with ZipFile(directory / "pipeline_artifacts.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(
            archive.namelist()
        ) != set(declared):
            raise ValueError("Archive has duplicate, missing or unexpected members")
        for name, entry in declared.items():
            data = archive.read(name)
            if (
                hashlib.sha256(data).hexdigest() != entry["sha256"]
                or len(data) != entry["size_bytes"]
            ):
                raise ValueError("Archive content checksum differs")
        members = len(declared)
        for record in [observed, *records]:
            is_observed = record is observed
            if not is_observed:
                i = record["replicate"]
                if i != len(rows) - (0 if arm == "known" else 1):
                    raise ValueError("Draw order changed or draws replaced")
                seed = derived_seed(policy.random_seed, "grouped_raw_gaussian_v2", i, "null_data")
                if record["simulation_seed"] != seed:
                    raise ValueError("Draw seed differs")
                raw = regenerated(model, seed)
            else:
                raw = values
            audit_candidates(archive, record, policy, raw, model.layout)
            if arm == "known" and is_observed:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "case": case,
                    "repetition": repetition,
                    "arm": arm,
                    "replicate": -1 if is_observed else record["replicate"],
                    **point_metrics(archive, record, model.counts, policy.model),
                }
            )
    inference = tails(observed, records, policy)
    if (directory / "summary.json").exists():
        primary = read_json(directory / "summary.json")["inference"]
        if primary["p_value"] != inference[0]["p_value"]:
            raise ValueError("Independent primary p-value differs")
        if primary["reject_grouped_raw_gaussian"] != inference[0]["reject"]:
            raise ValueError("Independent primary decision differs")
        if (
            primary.get("conditional_null_tail_interval_95")
            != inference[0]["conditional_tail_interval_95"]
        ):
            raise ValueError("Independent tail interval differs")
        if (
            primary.get("exceedances") != inference[0]["exceedances"]
            and observed["status"] == "complete"
        ):
            raise ValueError("Independent exceedance count differs")
    return rows, inference, members


def audit_study(root, *, output_dir, archived_study=None):
    import numpy as np

    root, destination = Path(root), Path(output_dir)
    destination.mkdir(exist_ok=False)
    rows, covariances, pipelines, cases = [], [], [], []
    archive_members = 0
    sources = [("fresh", root)]
    if archived_study is not None:
        sources.insert(0, ("archived", Path(archived_study)))
    for cohort, source in sources:
        for case in CASES:
            study = GroupedLikelihoodStudyConfig.from_dict(read_json(source / case / "study.json"))
            ledgers = [
                json.loads(s) for s in (source / case / "datasets.jsonl").read_text().splitlines()
            ]
            if len(ledgers) != study.n_datasets:
                raise ValueError("Dataset ledger count differs")
            for i, ledger in enumerate(ledgers):
                first = len(pipelines)
                base, policy = source / case / f"dataset_{i:04d}", case_policy(study, case, i)
                seed = derived_seed(study.random_seed, case, i, "observed_data")
                if ledger["data_seed"] != seed or ledger["null_stream_seed"] != policy.random_seed:
                    raise ValueError("Observed seed differs")
                if ledger["case"] != case or ledger["repetition"] != i:
                    raise ValueError("Dataset identity differs")
                if inventory_hash(base / "values.npz") != ledger["values_sha256"]:
                    raise ValueError("Observed values checksum differs")
                values, layout, _ = simulate_grouped(
                    case, study.n_units, seed, target_level=policy.model.target_level
                )
                with np.load(base / "values.npz", allow_pickle=False) as arrays:
                    for name, expected in values.items():
                        np.testing.assert_array_equal(arrays[name], expected)
                if not (base / "calibration/null_model.npz").exists():
                    if not (base / "failure.json").exists() and ledger["status"] != "failed":
                        raise ValueError("Missing fit without failure record")
                    for arm in ("fitted", "known") if cohort == "fresh" else ("fitted",):
                        for key in ("likelihood", "bic"):
                            rows.append(
                                {
                                    "cohort": cohort,
                                    "case": case,
                                    "repetition": i,
                                    "arm": arm,
                                    "statistic_kind": key,
                                    "p_value": None,
                                    "reject": None,
                                }
                            )
                    verify_counts(ledger, [], policy, cohort)
                    continue
                verify_model(base, base / "calibration", values, layout, policy)
                fitted = load_model(base / "calibration")
                truth = known_model(fitted, case)
                verify_reference(truth, case)
                covariances.append(
                    {"cohort": cohort, "case": case, "repetition": i, **raw_metrics(fitted, truth)}
                )
                arms = [("fitted", base / "calibration", fitted)]
                if cohort == "fresh":
                    known = load_model(base / "known")
                    verify_reference(known, case)
                    for name in ("morphology", "ecology", "targets", "counts", "values"):
                        np.testing.assert_array_equal(getattr(known, name), getattr(fitted, name))
                    if read_json(base / "known/observed.json") != read_json(
                        base / "calibration/observed.json"
                    ):
                        raise ValueError("Shared observed record differs")
                    left = read_json(base / "calibration/archive_inventory.json")
                    right = read_json(base / "known/archive_inventory.json")
                    if {k: v for k, v in left.items() if k.startswith("observed/")} != {
                        k: v for k, v in right.items() if k.startswith("observed/")
                    }:
                        raise ValueError("Shared observed bytes differ")
                    arms.append(("known", base / "known", known))
                for arm, directory, model in arms:
                    points, inference, members = audit_arm(
                        directory,
                        model,
                        values,
                        policy,
                        cohort=cohort,
                        case=case,
                        repetition=i,
                        arm=arm,
                    )
                    pipelines.extend(points)
                    archive_members += members
                    rows.extend(
                        {"cohort": cohort, "case": case, "repetition": i, "arm": arm, **r}
                        for r in inference
                    )
                verify_counts(ledger, pipelines[first:], policy, cohort)
            if cohort == "fresh":
                summary = read_json(source / case / "summary.json")
                for key in (
                    "planned_fits",
                    "attempted_fits",
                    "successful_fits",
                    "failed_fits",
                    "unattempted_fits",
                ):
                    if summary[key] != sum(r[key] for r in ledgers):
                        raise ValueError("Unique fit accounting differs from ledger")
                cases.append(summary)
    files = inventory(root)
    result = {
        "scope": SCOPE,
        "no_model_refits": True,
        "cases": cases,
        "new_unique_planned_fits": sum(s["planned_fits"] for s in cases),
        "new_unique_successful_fits": sum(s["successful_fits"] for s in cases),
        "unique_pipeline_records_audited": len(pipelines),
        "archive_members": archive_members,
        "study_files": len(files),
        **summaries(rows, covariances, pipelines),
    }
    for name, records in (("tails", rows), ("covariance", covariances), ("pipelines", pipelines)):
        csv_rows(
            [
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                for r in records
            ],
            destination / f"{name}.csv",
        )
    write_json(files, destination / "files.json")
    write_json(result, destination / "summary.json")
    return result


def inventory_hash(path):
    from super_deli.core.provenance import fingerprint_file

    return fingerprint_file(path).sha256


def verify_counts(ledger, points, policy, cohort):
    expected = 1 + policy.n_simulations * (2 if cohort == "fresh" else 1)
    checks = {
        "planned_fits": expected,
        "attempted_fits": len(points),
        "successful_fits": sum(r["status"] == "complete" for r in points),
        "failed_fits": sum(r["status"] == "failed" for r in points),
        "unattempted_fits": expected - len(points),
    }
    if any(ledger[k] != v for k, v in checks.items()):
        raise ValueError("Independent unique-fit accounting differs")
