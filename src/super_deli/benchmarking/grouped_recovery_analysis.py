"""Read-only reconstruction of recovery evidence, without model refits."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_recovery_inputs import (
    first_specimen,
    point,
    replay,
    source_pins,
)
from super_deli.benchmarking.grouped_recovery_metrics import recovery_metrics
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_analysis import _ari, _axis_check, _dense_mixture, _wilson
from super_deli.benchmarking.recovery_inputs import DirectoryArchive, inventory
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_recovery_config import (
    COHORTS,
    VARIANTS,
    GroupedRecoveryConfig,
    variants,
)


def _oracle(directory, labels, baseline, model, record):
    import numpy as np

    if record["status"] != "complete":
        if record["status"] != "failed" or not record.get("error"):
            raise ValueError("Invalid attempted truth-initialized fit")
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
        d = baseline["scores"].shape[1]
        residuals = baseline["scores"] - means[labels]
        covariance = sum(np.outer(r, r) for r in residuals) / len(
            labels
        ) + model.reg_covar * np.eye(d)
        np.testing.assert_allclose(
            initial["covariances"], np.repeat(covariance[None], 4, axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            fitted["covariances"], np.repeat(fitted["covariances"][:1], 4, axis=0), atol=0
        )
        likelihood = _dense_mixture(fitted)
        np.testing.assert_allclose(
            record["bic"],
            -2 * likelihood + (4 * d + d * (d + 1) // 2 + 3) * np.log(len(labels)),
            atol=1e-7,
        )
        sizes = np.bincount(fitted["labels"], minlength=4)
        if (
            record["hard_cluster_sizes"] != sizes.tolist()
            or record["eligible"] != bool(sizes.min() >= model.min_cluster_size)
            or not record["converged"]
            or record["attempted"] is not True
        ):
            raise ValueError("Truth-initialized constraints/convergence mismatch")
        np.testing.assert_allclose(record["ari"], _ari(labels, fitted["labels"]), atol=1e-12)


def _upstream(directory, values, layout, labels, metrics):
    """Independent raw moments, PCA eigen-equations and profile/axis checks."""
    import numpy as np

    order = {u: i for i, u in enumerate(layout["unit_ids"])}
    for j, view in enumerate(layout["views"]):
        kind = view["kind"]
        base = directory / f"view_{j:03d}"
        raw = values[kind]
        target = np.array([order[u] for u in view["target_ids"]])
        counts = np.bincount(target, minlength=len(labels))
        means = np.stack([np.sum(raw[target == g], axis=0) / counts[g] for g in range(len(labels))])
        reported = metrics["upstream"][kind]
        if reported["counts"] != counts.tolist() or reported["counts_by_label"] != [
            counts[labels == k].tolist() for k in range(4)
        ]:
            raise ValueError("Count diagnostic mismatch")
        np.testing.assert_allclose(reported["raw_group_mean"], means, atol=1e-12)
        _axis_check(means, labels, reported["raw_group_axis_r2"])
        if np.ptp(counts):
            _axis_check(counts[:, None], labels, reported["count_axis_r2"])
        elif reported["count_axis_r2"] != {"a": 0.0, "b": 0.0}:
            raise ValueError("Constant count diagnostic must be zero")
        with np.load(base / "parameters/pca.npz", allow_pickle=False) as pca:
            np.testing.assert_array_equal(pca["kept_indices"], np.arange(raw.shape[1]))
            np.testing.assert_allclose(pca["feature_means"], raw.mean(0), atol=1e-12)
            np.testing.assert_allclose(pca["feature_scales"], raw.std(0), atol=1e-12)
            standardized = (raw - raw.mean(0)) / raw.std(0)
            np.testing.assert_allclose(pca["mean"], standardized.mean(0), atol=1e-12)
            centered = standardized - standardized.mean(0)
            covariance = centered.T @ centered / (len(raw) - 1)
            c = pca["components"]
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            np.testing.assert_allclose(c @ c.T, np.eye(len(c)), atol=1e-11)
            np.testing.assert_allclose(covariance @ c.T, c.T * eigenvalues[: len(c)], atol=1e-10)
            np.testing.assert_allclose(pca["explained_variance"], eigenvalues[: len(c)], atol=1e-11)
            np.testing.assert_allclose(
                reported["source_variance_ratio"],
                eigenvalues[: len(c)] / eigenvalues.sum(),
                atol=1e-11,
            )
            np.testing.assert_allclose(reported["source_pca_components"], c, atol=0)
            scores = centered @ c.T
        _axis_check(scores, labels[target], reported["source_axis_r2"])
        profiles = np.stack([scores[target == g].mean(0) for g in range(len(labels))])
        with np.load(base / "parameters/profiles.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(profiles, saved["profiles"], atol=1e-11)
        _axis_check(profiles, labels, reported["profile_axis_r2"])


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
                "morphological_raw_group_a_r2",
                "morphological_raw_group_b_r2",
                "morphological_source_a_r2",
                "morphological_profile_a_r2",
                "ecological_raw_group_b_r2",
                "ecological_profile_b_r2",
                "fused_a_r2",
                "fused_b_r2",
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


def audit_grouped_recovery(root, *, output_dir):
    from tempfile import TemporaryDirectory

    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

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
    config = GroupedRecoveryConfig.from_dict(read_json(root / "configuration.json"))
    execution = read_json(root / "execution.json")
    if execution["source_pins"] != source_pins(config):
        raise ValueError("Source pins disagree with frozen configuration")
    study = GroupedStudyConfig.from_dict(read_json(root / "source_study.json"))
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
        values, layout, labels = simulate_grouped(
            "complementary", n, seed, target_level=study.calibration.model.target_level
        )
        truth = read_json(directory / "truth.json")
        if truth["unit_ids"] != list(layout["unit_ids"]) or truth["labels"] != labels.tolist():
            raise ValueError("Evaluation identity or generating labels disagree")
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            for key in values:
                np.testing.assert_array_equal(arrays[key], values[key])
        with TemporaryDirectory(prefix="super-deli-recovery-audit-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_views(values, layout, generated)
            if inventory(generated) != inventory(directory / "raw"):
                raise ValueError("Regenerated raw inputs or identities disagree")
        control, control_layout = first_specimen(values, layout)
        with TemporaryDirectory(prefix="super-deli-first-audit-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_views(control, control_layout, generated)
            if inventory(generated) != inventory(directory / "first_raw"):
                raise ValueError("First-specimen inputs/identities disagree")
        if read_json(directory / "layout.json") != json.loads(json.dumps(layout)):
            raise ValueError("Dataset layout mismatch")
        for name, model in grid.items():
            selected_values, selected_layout = (
                (control, control_layout) if name == "first_specimen" else (values, layout)
            )
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
                    selected_values,
                    selected_layout,
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
                _upstream(
                    directory / name / "run", selected_values, selected_layout, labels, metrics
                )
                for c in metrics["bic_terms"]:
                    d = c["d"]
                    parameters = c["k"] - 1 + c["k"] * d + d * (d + 1) // 2
                    if c["parameters"] != parameters:
                        raise ValueError("Shared covariance parameter-count mismatch")
                    np.testing.assert_allclose(c["penalty"], parameters * np.log(n), atol=1e-12)
                    if c["bic"] is not None:
                        np.testing.assert_allclose(
                            c["bic_minus_selected"],
                            c["penalty_minus_selected"] - c["twice_log_likelihood_gain"],
                            atol=1e-10,
                        )
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
                for kind, upstream in metrics["upstream"].items():
                    for stage in ("raw_group", "source", "profile", "count"):
                        for axis, value in upstream[f"{stage}_axis_r2"].items():
                            row[f"{kind}_{stage}_{axis}_r2"] = value
                for j, variance in enumerate(metrics["joint_variance_ratio"]):
                    row[f"joint_pc{j + 1}_variance_ratio"] = variance
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
        "planned_points": len(keys) * len(VARIANTS),
        "successful_points": sum(r["status"] == "complete" for r in points),
        "failed_points": sum(r["status"] == "failed" for r in points),
        "planned_truth_initialized": len(keys),
        "successful_truth_initialized": sum(r["status"] == "complete" for r in oracles),
        "failed_truth_initialized": sum(r["status"] == "failed" for r in oracles),
        "unattempted_truth_initialized": sum(r["status"] == "unattempted" for r in oracles),
        "replayed_archived": sum(
            r["status"] == "reproduced" for r in replay(root, config.n_datasets)
        ),
        "bootstrap_or_calibration_fits": 0,
    }
    if read_json(root / "summary.json") != expected_counts:
        raise ValueError("Study accounting mismatch")
    reproduced = replay(root, config.n_datasets)
    if reproduced != read_json(root / "replay.json"):
        raise ValueError("Baseline replay accounting mismatch")
    for i in range(config.n_datasets):
        snapshot = root / "source_snapshots" / f"dataset_{i:04d}"
        point(
            DirectoryArchive(snapshot),
            read_json(snapshot / "record.json"),
            study.calibration,
            *simulate_grouped(
                "complementary",
                config.small_n,
                declarations[i]["seed"],
                target_level=study.calibration.model.target_level,
            )[:2],
        )
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
            "PCA eigen-equations and raw moments independently checked without refitting; "
            "convergence histories preserved, not independently refitted."
        ),
        "interpretation": (
            "Exploratory recovery diagnostics under one constructed law, not species "
            "validation; no across-score-space absolute BIC comparisons or automatic "
            "configuration selection. First-specimen control removes replication and unequal "
            "source weighting together, not a pure weighting intervention. Wilson intervals are "
            "pointwise descriptive, unadjusted "
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
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    write_json(result, destination / "summary.json")
    write_json(expected, destination / "files.json")
    return result
