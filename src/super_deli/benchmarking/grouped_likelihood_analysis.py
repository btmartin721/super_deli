"""Independent densities for every candidate, raw-null audits and paired criteria/tails."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from super_deli.benchmarking.empirical_analysis import audit_fitted_study
from super_deli.benchmarking.full_grouped_diagnostic_analysis import csv_rows, describe
from super_deli.benchmarking.grouped_analysis import point, verify_model
from super_deli.benchmarking.grouped_likelihood import case_policy, simulator, summarize
from super_deli.benchmarking.recovery_analysis import _ari, _dense_mixture, _wilson
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_likelihood_config import CASES, GroupedLikelihoodStudyConfig
from super_deli.workflows.grouped_likelihood import TAIL_OPTIONS


def audit_candidates(archive, record, policy, values, layout):
    import numpy as np

    point(archive, record, policy, values, layout)
    if record["status"] != "complete":
        return
    prefix = record["archive_prefix"] + "/run/"
    raw = archive.read(prefix + "gmm_candidates.npz")
    run = json.loads(archive.read(prefix + "summary.json"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["candidate_sha256"] or digest != run["provenance"]["candidate_sha256"]:
        raise ValueError("Candidate-pool checksum mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as pool:
        with np.load(
            io.BytesIO(archive.read(prefix + "mixture.npz")), allow_pickle=False
        ) as chosen:
            np.testing.assert_array_equal(pool["scores"], chosen["scores"])
        n, d = pool["scores"].shape
        expected = {"scores"}
        for candidate in record["candidates"]:
            if candidate["status"] == "insufficient_samples":
                continue
            k = candidate["k"]
            arrays = {
                name: pool[f"k{k}_{name}"]
                for name in ("weights", "means", "covariances", "membership", "labels")
            }
            expected.update(f"k{k}_{name}" for name in arrays)
            arrays["scores"] = pool["scores"]
            if arrays["means"].shape != (k, d) or arrays["covariances"].shape != (k, d, d):
                raise ValueError("Candidate dimensions disagree")
            np.testing.assert_allclose(arrays["weights"].sum(), 1, atol=1e-12)
            ll = _dense_mixture(arrays)
            q = k - 1 + k * d + k * d * (d + 1) // 2
            np.testing.assert_allclose(candidate["log_likelihood"], ll, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(candidate["aic"], -2 * ll + 2 * q, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(
                candidate["bic"], -2 * ll + q * np.log(n), rtol=1e-10, atol=1e-7
            )
            sizes = np.bincount(arrays["labels"], minlength=k).tolist()
            if (
                candidate["hard_cluster_sizes"] != sizes
                or candidate["distinct_unit_sizes"] != sizes
            ):
                raise ValueError("Candidate memberships disagree with size constraints")
        if set(pool.files) != expected:
            raise ValueError("Candidate pool has missing or unexpected arrays")


def paired_rows(root, study):
    import numpy as np

    from super_deli.models.monte_carlo import summarize_upper_tail

    output = []
    for case in CASES:
        for i in range(study.n_datasets):
            base = root / case / f"dataset_{i:04d}"
            calibration = base / "calibration"
            if (calibration / "failure.json").exists():
                failure = read_json(calibration / "failure.json")
                for criterion in ("bic", "aic"):
                    output.append(
                        {
                            "case": case,
                            "repetition": i,
                            "criterion": criterion,
                            "status": "failed",
                            "error": failure["error"],
                        }
                    )
                continue
            result = read_json(calibration / "summary.json")
            observed = read_json(calibration / "observed.json")
            draws = read_json(calibration / "null_replicates.json")
            truth = np.array(read_json(base / "truth.json")["labels"])
            groups = case in {"complementary", "complementary_256", "unequal_variance"}
            for criterion in ("bic", "aic"):
                row = {
                    "case": case,
                    "repetition": i,
                    "criterion": criterion,
                    "n": study.large_n if case == "complementary_256" else study.n_units,
                }
                if observed["status"] == "complete":
                    t = observed["bic_statistic" if criterion == "bic" else "statistic"]
                    replicas = [
                        {**r, "statistic": r["bic_statistic"]}
                        if r["status"] == "complete" and criterion == "bic"
                        else r
                        for r in draws
                    ]
                    inference = summarize_upper_tail(
                        t, replicas, alpha=study.calibration.alpha, **TAIL_OPTIONS
                    )
                    if criterion == "aic" and inference != result["inference"]:
                        raise ValueError("Primary likelihood tail differs from paired analysis")
                    k = observed["bic_selected_k" if criterion == "bic" else "selected_k"]
                    with ZipFile(calibration / "pipeline_artifacts.zip") as archive:
                        with np.load(
                            io.BytesIO(archive.read("observed/run/gmm_candidates.npz")),
                            allow_pickle=False,
                        ) as pool:
                            ari = _ari(truth, pool[f"k{k}_labels"]) if groups else None
                    interval = inference["conditional_null_tail_interval_95"]
                    row.update(
                        statistic=t,
                        selected_k=k,
                        ari=ari,
                        status=inference["status"],
                        p_value=inference["p_value"],
                        reject=inference["reject_grouped_raw_gaussian"],
                        exceedances=inference["exceedances"],
                        successful_draws=inference["successful"],
                        failed_draws=inference["failed"],
                        tail_lower=interval[0] if interval else None,
                        tail_upper=interval[1] if interval else None,
                        failed_p_lower=inference["failed_outcome_bounds"][0],
                        failed_p_upper=inference["failed_outcome_bounds"][1],
                    )
                else:
                    row.update(status="point_failed", p_value=None, reject=None)
                output.append(row)
    return output


def paired_summary(rows):
    result, changes = [], []
    for case in CASES:
        for criterion in ("bic", "aic"):
            planned = [r for r in rows if r["case"] == case and r["criterion"] == criterion]
            available = [r for r in planned if r["status"] == "available"]
            points = [r for r in planned if r.get("selected_k") is not None]
            events = sum(r["reject"] for r in available)
            counts = {str(k): sum(r["selected_k"] == k for r in points) for k in range(1, 5)}
            result.append(
                {
                    "case": case,
                    "criterion": criterion,
                    "requested": len(planned),
                    "available_tests": len(available),
                    "withheld_tests": len(planned) - len(available),
                    "rejected": events,
                    "rejection_wilson_95": _wilson(events, len(available)),
                    "rejection_withheld_bounds": [
                        events / len(planned),
                        (events + len(planned) - len(available)) / len(planned),
                    ],
                    "available_points": len(points),
                    "selected_k_counts": counts,
                    "selected_four_wilson_95": _wilson(counts["4"], len(points)),
                    "ari": describe([r.get("ari") for r in points]),
                }
            )
        left = {r["repetition"]: r for r in rows if r["case"] == case and r["criterion"] == "bic"}
        right = {r["repetition"]: r for r in rows if r["case"] == case and r["criterion"] == "aic"}
        ids = [i for i in left if left[i]["status"] == right[i]["status"] == "available"]
        changes.append(
            {
                "case": case,
                "available_pairs": len(ids),
                "likelihood_only_reject": sum(
                    right[i]["reject"] and not left[i]["reject"] for i in ids
                ),
                "bic_only_reject": sum(left[i]["reject"] and not right[i]["reject"] for i in ids),
                "gained_four": sum(
                    left[i]["selected_k"] != 4 and right[i]["selected_k"] == 4 for i in ids
                ),
                "lost_four": sum(
                    left[i]["selected_k"] == 4 and right[i]["selected_k"] != 4 for i in ids
                ),
            }
        )
    return {"summaries": result, "paired_changes": changes}


def audit_grouped_likelihood(root, *, output_dir):
    from super_deli.models.grouped_null import write_raw_views

    root, destination = Path(root), Path(output_dir)
    study = GroupedLikelihoodStudyConfig.from_dict(read_json(root / CASES[0] / "study.json"))
    backend = {
        "cases": CASES,
        "study_type": GroupedLikelihoodStudyConfig,
        "policy_for": case_policy,
        "summarize": summarize,
        "simulator": lambda c, n, s: simulator(study, c, n, s),
        "point_auditor": audit_candidates,
        "tail_options": TAIL_OPTIONS,
        "namespace": "grouped_raw_gaussian_v2",
        "decision_key": "reject_grouped_raw_gaussian",
        "target": "grouped_gaussian_raw_continuous_views",
        "writer": write_raw_views,
        "verify_model": verify_model,
        "audit_scope": (
            "No model refits. Regenerated raw observations/draws; dense raw likelihood, "
            "constrained optimality, counts/identities, PCA moments/eigen-equations/projections, "
            "fusion and every candidate's independent Gaussian density, membership, likelihood, "
            "AIC/BIC, selection/statistics, seeds and complete upper-tail accounting. "
            "No global-optimization certification or species-boundary claim."
        ),
    }
    result = audit_fitted_study(root, output_dir=destination, backend=backend)
    rows = paired_rows(root, study)
    csv_rows(rows, destination / "paired.csv")
    write_json(paired_summary(rows), destination / "paired.json")
    return result
