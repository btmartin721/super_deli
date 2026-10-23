"""Read-only audit of fitted raw parameters, simulated inputs and pipeline results."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_study import (
    case_policy,
    simulate_continuous,
    summarize_empirical,
)
from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.empirical_config import CASES, EmpiricalStudyConfig
from super_deli.workflows.calibration_artifacts import fit_counts


def _raw_hashes(values, layout, writer=None):
    from super_deli.models.empirical_null import write_raw_views

    if writer is None:
        writer = write_raw_views

    with TemporaryDirectory(prefix="super-deli-empirical-audit-") as temporary:
        directory = Path(temporary) / "raw"
        writer(values, layout, directory)
        return {
            p.relative_to(directory).as_posix(): fingerprint_file(p).sha256
            for p in directory.rglob("*")
            if p.is_file()
        }


def _point(archive, record, policy, values, layout):
    import numpy as np

    audit_pipeline_point(archive, record, policy)
    if record["status"] != "complete":
        return
    prefix = record["archive_prefix"] + "/run/"
    run = json.loads(archive.read(prefix + "summary.json"))
    if (
        run["unit_ids"] != layout["unit_ids"]
        or len(run["views"]) != 2
        or record["n_retained"] != len(values)
    ):
        raise ValueError("Retained empirical unit identities disagree")
    offset, order = 0, {u: i for i, u in enumerate(layout["unit_ids"])}
    for i, view in enumerate(layout["views"]):
        rows = json.loads(archive.read(prefix + f"view_{i:03d}/source_rows.json"))
        with io.BytesIO(archive.read(prefix + f"view_{i:03d}/source_values.npy")) as handle:
            source = np.load(handle, allow_pickle=False)
        width = len(view["feature_names"])
        if rows["unit_ids"] != view["unit_ids"] or rows["feature_names"] != view["feature_names"]:
            raise ValueError("Raw source view identities/schema disagree")
        np.testing.assert_array_equal(
            source, values[[order[u] for u in rows["unit_ids"]], offset : offset + width]
        )
        offset += width


def audit_empirical_study(root, *, output_dir):
    return audit_fitted_study(root, output_dir=output_dir)


def audit_fitted_study(root, *, output_dir, grouped=False, backend=None):
    try:
        import numpy as np

        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.empirical_calibration import TAIL_OPTIONS, _dependencies

        _dependencies()
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for empirical study analysis") from exc
    cases, study_type = CASES, EmpiricalStudyConfig
    policy_for, summarize, simulator, point_auditor = (
        case_policy,
        summarize_empirical,
        simulate_continuous,
        _point,
    )
    tail_options = TAIL_OPTIONS
    namespace, decision_key = "empirical_raw_gaussian_v1", "reject_joint_raw_gaussian"
    target, writer = "joint_gaussian_raw_continuous_views", None
    if grouped:
        from super_deli.benchmarking.grouped_analysis import point, verify_model
        from super_deli.benchmarking.grouped_study import case_policy as policy_for
        from super_deli.benchmarking.grouped_study import simulate_grouped
        from super_deli.benchmarking.grouped_study import summarize_grouped as summarize
        from super_deli.models.grouped_config import GroupedStudyConfig as study_type
        from super_deli.models.grouped_config import study_cases
        from super_deli.models.grouped_null import write_raw_views as writer
        from super_deli.workflows.grouped_calibration import TAIL_OPTIONS as tail_options
        from super_deli.workflows.grouped_calibration import _dependencies as grouped_dependencies

        grouped_dependencies()
        point_auditor = point
        namespace, decision_key = "grouped_raw_gaussian_v1", "reject_grouped_raw_gaussian"
        target = "grouped_gaussian_raw_continuous_views"
    if backend is not None:
        cases, study_type = backend["cases"], backend["study_type"]
        policy_for, summarize, simulator = (
            backend["policy_for"],
            backend["summarize"],
            backend["simulator"],
        )
        point_auditor, tail_options = backend["point_auditor"], backend["tail_options"]
        namespace, decision_key = backend["namespace"], backend["decision_key"]
        target, writer, verify_model = backend["target"], backend["writer"], backend["verify_model"]
    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Analysis must be outside immutable study shards")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    original, rows, rates, members, successes = None, [], [], 0, 0
    if grouped:
        first = read_json(root / "independent_gaussian/study.json")
        cases = study_cases(study_type.from_dict(first))
    for case in cases:
        directory = root / case
        declaration = read_json(directory / "study.json")
        study = study_type.from_dict(declaration)
        if grouped and study.calibration.schema_version == 2:
            from super_deli.workflows.grouped_calibration import BOUNDARY_TAIL_OPTIONS

            tail_options = BOUNDARY_TAIL_OPTIONS
            namespace = "grouped_raw_gaussian_v2"
        if original is not None and original != declaration:
            raise ValueError("Study shard declarations disagree")
        original = declaration
        shard = read_json(directory / "summary.json")
        records = [
            json.loads(line) for line in (directory / "datasets.jsonl").read_text().splitlines()
        ]
        if len(records) != study.n_datasets or shard != summarize(records, study, case):
            raise ValueError("Study ledger/completion summary disagree")
        for i, row in enumerate(records):
            policy = policy_for(study, case, i)
            seed = derived_seed(study.random_seed, case, i, "observed_data")
            if any(
                row[key] != value
                for key, value in {
                    "case": case,
                    "repetition": i,
                    "data_seed": seed,
                    "null_stream_seed": policy.random_seed,
                    "directory": f"dataset_{i:04d}",
                }.items()
            ):
                raise ValueError("Observed dataset identity/seed mismatch")
            base = directory / row["directory"]
            output = base / "calibration"
            if read_json(output / "configuration.json") != json.loads(json.dumps(policy.to_dict())):
                raise ValueError("Empirical calibration configuration mismatch")
            if (
                row["outcome_file"] not in {"summary.json", "failure.json"}
                or fingerprint_file(output / row["outcome_file"]).sha256 != row["outcome_sha256"]
            ):
                raise ValueError("Calibration outcome checksum mismatch")
            if grouped:
                values, layout, labels = simulate_grouped(
                    case, study.n_units, seed, target_level=policy.model.target_level
                )
            else:
                values, layout, labels = simulator(case, study.n_units, seed)
            values_path = base / ("values.npz" if isinstance(values, dict) else "values.npy")
            if isinstance(values, dict):
                with np.load(values_path, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(values):
                        raise ValueError("Observed raw array keys mismatch")
                    for kind, expected in values.items():
                        np.testing.assert_array_equal(expected, arrays[kind])
            else:
                with values_path.open("rb") as handle:
                    np.testing.assert_array_equal(values, np.load(handle, allow_pickle=False))
            if (
                fingerprint_file(values_path).sha256 != row["values_sha256"]
                or read_json(base / "truth.json")["labels"] != labels.tolist()
            ):
                raise ValueError("Observed generator/truth checksum mismatch")
            for name, digest in _raw_hashes(values, layout, writer).items():
                if fingerprint_file(base / "observed" / name).sha256 != digest:
                    raise ValueError("Observed raw input differs from declared generator")
            if row["outcome_file"] == "failure.json":
                failure = read_json(output / "failure.json")
                point = (
                    read_json(output / "observed.json")
                    if (output / "observed.json").exists()
                    else None
                )
                draws = [read_json(p) for p in sorted(output.glob("draw_*.json"))]
                counts = fit_counts(point, draws, policy.n_simulations)
                if (
                    row["status"] != "failed"
                    or row["p_value"] is not None
                    or not failure["error"]
                    or any(row[k] != v or failure[k] != v for k, v in counts.items())
                ):
                    raise ValueError("Failed outcome or attempt denominators disagree")
                rows.append(row)
                continue
            result = read_json(output / "summary.json")
            if result["configuration_sha256"] != configuration_hash(policy.to_dict()):
                raise ValueError("Configuration digest mismatch")
            for name, digest in result["artifacts"].items():
                if fingerprint_file(output / name).sha256 != digest:
                    raise ValueError("Empirical numerical artifact checksum mismatch")
            for path, digest in result["observed_input_sha256"].items():
                if fingerprint_file(path).sha256 != digest:
                    raise ValueError("Observed source checksum mismatch")
            if grouped or backend is not None:
                fitted_layout, draw_values, n_units, n_features = verify_model(
                    base, output, values, layout, policy
                )
            else:
                fitted_layout = read_json(output / "layout.json")
                for key in ("unit_ids", "views"):
                    if fitted_layout[key] != json.loads(json.dumps(layout[key])):
                        raise ValueError("Fitted Gaussian layout mismatch")
                with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["values"], values)
                    mean = values.mean(axis=0)
                    centered = values - mean
                    covariance = centered.T @ centered / len(values)
                    np.testing.assert_array_equal(arrays["mean"], mean)
                    np.testing.assert_array_equal(arrays["covariance"], covariance)
                if result["n_units"] != len(values) or result["n_raw_features"] != values.shape[1]:
                    raise ValueError("Fitted Gaussian dimensions disagree")
                scales = np.sqrt(np.diag(covariance))
                factor = scales[:, None] * np.linalg.cholesky(
                    covariance / scales[:, None] / scales[None, :]
                )

                def draw_values(seed, mean=mean, shape=values.shape, factor=factor):
                    return mean + np.random.default_rng(seed).standard_normal(shape) @ factor.T

                n_units, n_features = values.shape
            if result["n_units"] != n_units or result["n_raw_features"] != n_features:
                raise ValueError("Fitted null dimensions disagree")
            point = read_json(output / "observed.json")
            draws = read_json(output / "null_replicates.json")
            if (
                point != result["point"]
                or point["archive_prefix"] != "observed"
                or len(draws) != (policy.n_simulations if point["status"] == "complete" else 0)
            ):
                raise ValueError("Point/draw ledger differs from requested design")
            inventory = read_json(output / "archive_inventory.json")
            with ZipFile(output / "pipeline_artifacts.zip") as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(inventory):
                    raise ValueError("Archive membership mismatch")
                for name in names:
                    raw = archive.read(name)
                    path = PurePosixPath(name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or inventory[name]
                        != {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
                    ):
                        raise ValueError("Archive inventory/path mismatch")
                members += len(names)
                point_auditor(archive, point, policy, values, fitted_layout)
                successes += point["status"] == "complete"
                for j, draw in enumerate(draws):
                    null_seed = derived_seed(policy.random_seed, namespace, j, "null_data")
                    if (
                        draw["replicate"] != j
                        or draw["simulation_seed"] != null_seed
                        or draw["archive_prefix"] != f"null_{j:04d}"
                        or draw != read_json(output / f"draw_{j:04d}.json")
                    ):
                        raise ValueError("Null draw identity/seed mismatch")
                    sample = draw_values(null_seed)
                    for name, digest in _raw_hashes(sample, fitted_layout, writer).items():
                        if (
                            hashlib.sha256(
                                archive.read(draw["archive_prefix"] + "/raw/" + name)
                            ).hexdigest()
                            != digest
                        ):
                            raise ValueError("Null raw generator checksum mismatch")
                    point_auditor(archive, draw, policy, sample, fitted_layout)
                    successes += draw["status"] == "complete"
            expected = (
                summarize_upper_tail(point["statistic"], draws, alpha=policy.alpha, **tail_options)
                if point["status"] == "complete"
                else {
                    "status": "point_failed",
                    "requested": policy.n_simulations,
                    "successful": 0,
                    "failed": 0,
                    "p_value": None,
                    decision_key: None,
                }
            )
            if (
                result["inference"] != expected
                or row["status"] != expected["status"]
                or row["p_value"] != expected["p_value"]
                or row["reject"] != expected[decision_key]
                or row["statistic"] != point.get("statistic")
                or row["selected_k"] != point.get("selected_k")
            ):
                raise ValueError("Reconstructed empirical inference differs from exports")
            if any(
                result[k] != v or row[k] != v
                for k, v in fit_counts(point, draws, policy.n_simulations).items()
            ):
                raise ValueError("Fit requested/attempted/successful denominators disagree")
            evidence = read_json(output / "evidence.json")
            if evidence["target"] != target or evidence["value"] != expected["p_value"]:
                raise ValueError("Empirical evidence target/value mismatch")
            rows.append(row)
        rates.append(
            {"case": case, "interpretation": shard["interpretation"], **shard["rejection_rate"]}
        )
    files = {
        p.relative_to(root).as_posix(): fingerprint_file(p).sha256
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "study": original,
        "datasets": len(rows),
        "rates": rates,
        "archive_members": members,
        "verified_successful_pipelines": successes,
        "raw_files": len(files),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                "planned_fits",
                "attempted_fits",
                "successful_fits",
                "failed_fits",
                "unattempted_fits",
            )
        },
        "audit_scope": backend["audit_scope"]
        if backend
        else (
            "Regenerated raw observations/draws; checked raw ML parameters "
            "(dense likelihood/score for grouped fits), "
            "selected likelihoods, fusion/projection, selection and p-values; no pipeline fits. "
            "View PCA and nonselected likelihoods preserved/checksummed, not refitted. "
            "Outer failures retain attempt-ledger counts without complete numerical re-audit."
        ),
    }
    if grouped and any(v.pca_fit == "target_means" for v in study.calibration.model.views):
        summary["audit_scope"] += (
            " Independently reconstructed both views' PCA feature selection, training "
            "means/scales, eigen-equations, target identities/counts, raw projections "
            "and aggregated profiles for every successful observed/null pipeline."
        )
    destination.mkdir(exist_ok=False)
    for name, values in (("datasets.csv", rows), ("rates.csv", rates)):
        with (destination / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted(set().union(*(r.keys() for r in values))),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(values)
    write_json(files, destination / "files.json")
    write_json(summary, destination / "summary.json")
    return summary
