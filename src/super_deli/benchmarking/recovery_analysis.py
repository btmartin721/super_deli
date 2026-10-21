"""Read-only reconstruction of recovery evidence, without model refits."""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.joint_analysis import point
from super_deli.benchmarking.joint_study import simulate_joint
from super_deli.benchmarking.recovery_inputs import DirectoryArchive, inventory
from super_deli.benchmarking.recovery_metrics import recovery_metrics
from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.recovery_config import COHORTS, VARIANTS, RecoveryConfig, variants


def _ari(truth, predicted):
    import numpy as np

    # Independent contingency-table implementation, rather than sklearn's metric.
    _, a = np.unique(truth, return_inverse=True)
    _, b = np.unique(predicted, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(table, (a, b), 1)
    choose = lambda x: (x * (x - 1) // 2).sum()  # noqa: E731
    total = len(a) * (len(a) - 1) / 2
    index, left, right = choose(table), choose(table.sum(0)), choose(table.sum(1))
    expected, upper = left * right / total, (left + right) / 2
    return 1.0 if upper == expected else float((index - expected) / (upper - expected))


def _axis_check(scores, labels, reported):
    import numpy as np

    centered = scores - scores.mean(0)
    projector = centered @ np.linalg.pinv(centered)
    for name, axis in (("a", labels // 2), ("b", labels % 2)):
        y = axis - axis.mean()
        residual = y - projector @ y
        np.testing.assert_allclose(reported[name], 1 - residual @ residual / (y @ y), atol=1e-10)


def _dense_mixture(arrays):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    logs = np.stack(
        [
            np.log(w) + multivariate_normal.logpdf(arrays["scores"], mean=m, cov=c)
            for w, m, c in zip(
                arrays["weights"], arrays["means"], arrays["covariances"], strict=True
            )
        ],
        axis=1,
    )
    likelihood = logsumexp(logs, axis=1)
    np.testing.assert_allclose(np.exp(logs - likelihood[:, None]), arrays["membership"], atol=1e-9)
    np.testing.assert_array_equal(logs.argmax(1), arrays["labels"])
    return float(likelihood.sum())


def _oracle(directory, labels, baseline, model, record):
    import numpy as np

    if record["status"] != "complete":
        if record["status"] == "failed" and not record.get("error"):
            raise ValueError("Failed truth-initialized fit lacks error")
        return
    with (
        np.load(directory / "mixture.npz", allow_pickle=False) as fitted,
        np.load(directory / "initialization.npz", allow_pickle=False) as initial,
    ):
        np.testing.assert_array_equal(fitted["scores"], baseline["scores"])
        means = np.stack([baseline["scores"][labels == k].mean(0) for k in range(4)])
        np.testing.assert_allclose(initial["means"], means, atol=1e-12)
        np.testing.assert_allclose(
            initial["weights"], np.bincount(labels, minlength=4) / len(labels)
        )
        for k in range(4):
            residual = baseline["scores"][labels == k] - means[k]
            covariance = sum(np.outer(r, r) for r in residual) / len(
                residual
            ) + model.reg_covar * np.eye(means.shape[1])
            np.testing.assert_allclose(initial["covariances"][k], covariance, atol=1e-12)
        likelihood = _dense_mixture(fitted)
        d = baseline["scores"].shape[1]
        np.testing.assert_allclose(
            record["bic"],
            -2 * likelihood + (4 * (d + d * (d + 1) // 2) + 3) * np.log(len(labels)),
            atol=1e-7,
        )
        sizes = np.bincount(fitted["labels"], minlength=4)
        if (
            record["hard_cluster_sizes"] != sizes.tolist()
            or record["eligible"] != bool(sizes.min() >= model.min_cluster_size)
            or not record["converged"]
        ):
            raise ValueError("Truth-initialized constraint or convergence mismatch")
        np.testing.assert_allclose(record["ari"], _ari(labels, fitted["labels"]), atol=1e-12)


def _wilson(events, n):
    if not n:
        return None
    z, p = 1.959963984540054, events / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - radius), min(1, center + radius)]


def summaries(rows, requested):
    import numpy as np

    output = []
    for cohort in COHORTS:
        for name in VARIANTS:
            selected = [r for r in rows if r["cohort"] == cohort and r["variant"] == name]
            available = [r for r in selected if r["status"] == "complete"]
            events = sum(r["selected_k"] == 4 for r in available)
            row = {
                "cohort": cohort,
                "variant": name,
                "requested": requested,
                "available": len(available),
                "failed": len(selected) - len(available),
                "selected_k_counts": {
                    str(k): sum(r["selected_k"] == k for r in available) for k in range(1, 5)
                },
                "selected_four_rate": events / len(available) if available else None,
                "selected_four_wilson_95": _wilson(events, len(available)),
            }
            for key in (
                "ari",
                "joint_a_r2",
                "joint_b_r2",
                "k4_bic_minus_selected",
                "k4_twice_ll_gain",
                "k4_extra_penalty",
            ):
                values = [r[key] for r in available if r[key] is not None]
                row[key] = (
                    {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "min": float(min(values)),
                        "max": float(max(values)),
                    }
                    if values
                    else None
                )
            row["k4_eligible"] = sum(r["k4_status"] == "eligible" for r in available)
            row["k4_small_component"] = sum(r["k4_status"] == "small_component" for r in available)
            output.append(row)
    return output


def audit_recovery(root, *, output_dir):
    from tempfile import TemporaryDirectory

    import numpy as np

    from super_deli.models.joint_null import write_raw_joint

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Audit must be separate from immutable diagnostic artifacts")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    expected = read_json(root / "files.json")
    actual = inventory(root)
    actual.pop("files.json")
    if actual != expected:
        raise ValueError("Diagnostic artifact inventory mismatch")
    config = RecoveryConfig.from_dict(read_json(root / "configuration.json"))
    execution = read_json(root / "execution.json")
    if execution["source_pins"] != {
        "study.json": config.source_study_sha256,
        "study-analysis/files.json": config.source_inventory_sha256,
        "runtime_sources.json": config.source_modules_sha256,
    }:
        raise ValueError("Source pins disagree with frozen configuration")
    study = JointStudyConfig.from_dict(read_json(root / "source_study.json"))
    grid = variants(study.calibration.model)
    if read_json(root / "grid.json") != {
        name: json.loads(json.dumps(m.to_dict())) for name, m in grid.items()
    }:
        raise ValueError("Frozen diagnostic grid mismatch")
    declarations = read_json(root / "datasets.json")
    keys = [(c, i) for c in COHORTS for i in range(config.n_datasets)]
    if [(r["cohort"], r["repetition"]) for r in declarations] != keys:
        raise ValueError("Unexpected dataset identities or replacements")
    points, oracles, rows, candidate_rows, oracle_rows = [], [], [], [], []
    for declaration in declarations:
        cohort, i = declaration["cohort"], declaration["repetition"]
        n = config.large_n if cohort == "fresh_256" else config.small_n
        seed = derived_seed(
            study.random_seed if cohort == "archived" else config.random_seed,
            "complementary" if cohort == "archived" else cohort,
            i,
            "observed_data",
        )
        if declaration["seed"] != seed or declaration["n"] != n:
            raise ValueError("Dataset size or seed mismatch")
        directory = root / cohort / f"dataset_{i:04d}"
        values, layout, labels = simulate_joint("complementary", n, seed)
        truth = read_json(directory / "truth.json")
        if truth["unit_ids"] != layout["unit_ids"] or truth["labels"] != labels.tolist():
            raise ValueError("Evaluation identity or generating labels disagree")
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            for key in values:
                np.testing.assert_array_equal(arrays[key], values[key])
        with TemporaryDirectory(prefix="super-deli-recovery-audit-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_joint(values, layout, generated)
            if inventory(generated) != inventory(directory / "raw"):
                raise ValueError("Regenerated raw inputs or identities disagree")
        for name, model in grid.items():
            record = read_json(directory / name / "record.json")
            full = {"cohort": cohort, "repetition": i, "variant": name, **record}
            points.append(full)
            row = {
                "cohort": cohort,
                "repetition": i,
                "variant": name,
                "n": n,
                "status": record["status"],
            }
            if record["status"] == "complete":
                point(
                    DirectoryArchive(directory),
                    {**record, "archive_prefix": name},
                    SimpleNamespace(model=model),
                    values,
                    layout,
                )
                metrics = recovery_metrics(directory / name / "run", labels)
                if metrics != record["metrics"]:
                    raise ValueError("Saved recovery metrics disagree")
                with np.load(directory / name / "run/mixture.npz", allow_pickle=False) as fitted:
                    np.testing.assert_allclose(
                        metrics["ari"], _ari(labels, fitted["labels"]), atol=1e-12
                    )
                    _axis_check(fitted["scores"], labels, metrics["joint_axis_r2"])
                with np.load(directory / name / "run/fusion.npz", allow_pickle=False) as fusion:
                    _axis_check(fusion["fused"], labels, metrics["fused_axis_r2"])
                for j, view in enumerate(model.views):
                    with np.load(
                        directory / name / f"run/view_{j:03d}/parameters/profiles.npz",
                        allow_pickle=False,
                    ) as profile:
                        _axis_check(profile["profiles"], labels, metrics["view_axis_r2"][view.kind])
                k4 = next(c for c in metrics["bic_terms"] if c["k"] == 4)
                row.update(
                    selected_k=metrics["selected_k"],
                    ari=metrics["ari"],
                    joint_a_r2=metrics["joint_axis_r2"]["a"],
                    joint_b_r2=metrics["joint_axis_r2"]["b"],
                    fused_a_r2=metrics["fused_axis_r2"]["a"],
                    fused_b_r2=metrics["fused_axis_r2"]["b"],
                    axis_a_homogeneity=metrics["axis_homogeneity"]["a"],
                    axis_b_homogeneity=metrics["axis_homogeneity"]["b"],
                    k4_status=k4["status"],
                    k4_bic_minus_selected=k4["bic_minus_selected"],
                    k4_twice_ll_gain=k4["twice_log_likelihood_gain"],
                    k4_extra_penalty=k4["penalty_minus_selected"],
                )
                for c in metrics["bic_terms"]:
                    candidate_rows.append(
                        {
                            "cohort": cohort,
                            "repetition": i,
                            "variant": name,
                            **{
                                k: v
                                for k, v in c.items()
                                if k not in {"hard_cluster_sizes", "distinct_unit_sizes"}
                            },
                        }
                    )
            else:
                if not record.get("error") or "metrics" in record:
                    raise ValueError("Invalid failed point record")
            rows.append(row)
        oracle = read_json(directory / "truth_initialized/result.json")
        oracles.append({"cohort": cohort, "repetition": i, **oracle})
        baseline = read_json(directory / "baseline/record.json")
        oracle_row = {"cohort": cohort, "repetition": i, **oracle}
        if baseline["status"] == "complete":
            with np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as arrays:
                _oracle(
                    directory / "truth_initialized", labels, arrays, study.calibration.model, oracle
                )
            if oracle["status"] == "complete":
                selected = next(
                    c for c in baseline["candidates"] if c["k"] == baseline["selected_k"]
                )
                original = next(c for c in baseline["candidates"] if c["k"] == 4)
                oracle_row.update(
                    bic_minus_baseline_selected=oracle["bic"] - selected["bic"],
                    bic_minus_unsupervised_k4=oracle["bic"] - original["bic"],
                )
        elif oracle["status"] != "unattempted" or oracle["attempted"]:
            raise ValueError("Truth-initialized fit attempted without baseline scores")
        oracle_rows.append(
            {k: v for k, v in oracle_row.items() if k not in {"hard_cluster_sizes", "role"}}
        )
    if points != read_json(root / "points.json") or oracles != read_json(
        root / "truth_initialized.json"
    ):
        raise ValueError("Global ledgers differ from individual records")
    expected_counts = {
        "schema_version": 1,
        "datasets": len(keys),
        "planned_points": len(keys) * 7,
        "successful_points": sum(r["status"] == "complete" for r in points),
        "failed_points": sum(r["status"] == "failed" for r in points),
        "planned_truth_initialized": len(keys),
        "successful_truth_initialized": sum(r["status"] == "complete" for r in oracles),
        "failed_truth_initialized": sum(r["status"] == "failed" for r in oracles),
        "unattempted_truth_initialized": sum(r["status"] == "unattempted" for r in oracles),
        "replayed_archived": config.n_datasets,
        "bootstrap_or_calibration_fits": 0,
    }
    if read_json(root / "summary.json") != expected_counts:
        raise ValueError("Study accounting mismatch")
    # Independently check the original archived baseline snapshots and their replay.
    for i in range(config.n_datasets):
        snapshot = root / "source_snapshots" / f"dataset_{i:04d}"
        old = read_json(snapshot / "record.json")
        new = read_json(root / "archived" / f"dataset_{i:04d}/baseline/record.json")
        point(
            DirectoryArchive(snapshot),
            old,
            study.calibration,
            *simulate_joint("complementary", config.small_n, declarations[i]["seed"])[:2],
        )
        if old["selected_k"] != new["selected_k"]:
            raise ValueError("Archived selection replay mismatch")
        for a, b in zip(old["candidates"], new["candidates"], strict=True):
            if {k: v for k, v in a.items() if k != "bic"} != {
                k: v for k, v in b.items() if k != "bic"
            }:
                raise ValueError("Archived constraint replay mismatch")
            np.testing.assert_allclose(a["bic"], b["bic"], atol=1e-7, rtol=1e-10)
        with (
            np.load(snapshot / "observed/run/mixture.npz", allow_pickle=False) as a,
            np.load(
                root / "archived" / f"dataset_{i:04d}/baseline/run/mixture.npz", allow_pickle=False
            ) as b,
        ):
            np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-11, rtol=1e-9)
            np.testing.assert_array_equal(a["labels"], b["labels"])
    result = {
        **expected_counts,
        "status": "complete",
        "verified_files": len(expected),
        "summaries": summaries(rows, config.n_datasets),
        "audit_scope": (
            "Regenerated inputs/labels/identities; inventory, configurations, seeds, baseline "
            "replay, fusion/PCA projection and selected GMM dense likelihood, BIC/membership; "
            "independent contingency ARI, pseudoinverse axis recovery and "
            "truth-initialization/dense likelihood. No model refits. Nonselected GMM "
            "likelihoods inferred from preserved BIC logs, not independently recomputed; view "
            "PCA fits and convergence histories preserved, not independently refitted."
        ),
        "interpretation": (
            "Exploratory recovery diagnostics under one constructed law, not species "
            "validation; no across-score-space absolute BIC comparisons or automatic "
            "configuration selection. Wilson intervals are pointwise descriptive, unadjusted "
            "for the diagnostic grid."
        ),
    }
    destination.mkdir()
    for name, data in (
        ("points.csv", rows),
        ("candidates.csv", candidate_rows),
        ("truth_initialized.csv", oracle_rows),
    ):
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (destination / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write_json(result, destination / "summary.json")
    write_json(expected, destination / "files.json")
    return result
