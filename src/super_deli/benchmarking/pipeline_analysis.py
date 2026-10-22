"""Read-only audit of raw generators, archives, fusion and reference-test accounting."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from importlib import import_module
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from super_deli.benchmarking.calibration_analysis import _selection
from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.pipeline_study import case_config, summarize_records
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.pipeline_config import CASES, PipelineStudyConfig


def _json(archive, name):
    return json.loads(archive.read(name))


def _arrays(archive, name):
    import numpy as np

    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _generated(reference, seed, pattern):
    from super_deli.benchmarking.simulation import simulate, write_simulation

    with TemporaryDirectory(prefix="super-deli-generator-audit-") as temporary:
        directory = Path(temporary) / "simulation"
        write_simulation(simulate(reference.scenario(pattern), reference, seed), directory)
        return {
            p.relative_to(directory).as_posix(): fingerprint_file(p).sha256
            for p in directory.rglob("*")
            if p.is_file()
        }


def audit_pipeline_point(archive, record, policy, *, allow_stability=False):
    import numpy as np
    from scipy.special import logsumexp

    prefix = record["archive_prefix"] + "/run/"
    if record["status"] == "failed":
        if not record.get("error") or "statistic" in record:
            raise ValueError("Invalid failed pipeline record")
        return
    run = _json(archive, prefix + "summary.json")
    candidates = _json(archive, prefix + "candidates.json")
    n = len(run["unit_ids"])
    t, k = _selection(candidates, policy.model, n)
    seed = int(np.random.SeedSequence(policy.model.random_seed).spawn(1)[0].generate_state(1)[0])
    if (
        record["statistic"] != t
        or record["selected_k"] != k
        or record["candidates"] != candidates
        or run["selected_k"] != k
        or record["n_retained"] != n
        or record["point_seed"] != seed
        or run["point_seed"] != seed
        or run["bootstrap"]["requested"] != (policy.model.n_bootstrap if allow_stability else 0)
        or run["provenance"]["configuration"]["model"]
        != json.loads(json.dumps(policy.model.to_dict()))
    ):
        raise ValueError("Pipeline selection, seed or configuration mismatch")
    blocks = []
    for i, view in enumerate(run["views"]):
        path = prefix + f"view_{i:03d}/parameters/"
        for name, digest in view["arrays"].items():
            if hashlib.sha256(archive.read(path + name)).hexdigest() != digest:
                raise ValueError("View parameter checksum mismatch")
        profile = _arrays(archive, path + "profiles.npz")
        centered = profile["profiles"] - profile["mean"]
        np.testing.assert_allclose(profile["profiles"].mean(axis=0), profile["mean"], atol=1e-12)
        np.testing.assert_allclose(
            np.sqrt(np.sum(centered**2) / (n - 1)), profile["scale"], rtol=1e-10
        )
        blocks.append(centered * np.sqrt(profile["normalized_weight"]) / profile["scale"])
    fusion = _arrays(archive, prefix + "fusion.npz")
    mixture = _arrays(archive, prefix + "mixture.npz")
    for name in ("fusion", "mixture"):
        digest = hashlib.sha256(archive.read(prefix + name + ".npz")).hexdigest()
        if digest != record[name + "_sha256"] or digest != run["provenance"][name + "_sha256"]:
            raise ValueError("Point numerical checksum mismatch")
    np.testing.assert_allclose(np.concatenate(blocks, axis=1), fusion["fused"], atol=1e-12)
    scores = (fusion["fused"] - fusion["mean"]) @ fusion["components"].T
    np.testing.assert_allclose(scores, mixture["scores"], rtol=1e-10, atol=1e-12)
    # Reconstruct the selected dense Gaussian likelihood; other K values are
    # checked through their preserved complete candidate logs, without refits.
    log_components = []
    for weight, mean, covariance in zip(
        mixture["weights"], mixture["means"], mixture["covariances"], strict=True
    ):
        centered = scores - mean
        sign, determinant = np.linalg.slogdet(covariance)
        if sign != 1:
            raise ValueError("Selected covariance is not positive definite")
        quadratic = np.sum(centered * np.linalg.solve(covariance, centered.T).T, axis=1)
        log_components.append(
            np.log(weight) - 0.5 * (scores.shape[1] * np.log(2 * np.pi) + determinant + quadratic)
        )
    log_components = np.stack(log_components, axis=1)
    likelihood = logsumexp(log_components, axis=1)
    membership = np.exp(log_components - likelihood[:, None])
    np.testing.assert_allclose(membership, mixture["membership"], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(np.argmax(membership, axis=1), mixture["labels"])
    d = scores.shape[1]
    from super_deli.models.mixture_parameters import mixture_parameter_count

    covariance_type = getattr(policy.model, "covariance_type", "full")
    if run.get("covariance_type") != covariance_type:
        raise ValueError("Covariance family differs from the declared model")
    if covariance_type == "tied":
        np.testing.assert_array_equal(
            mixture["covariances"],
            np.repeat(mixture["covariances"][:1], k, axis=0),
        )
    parameters = mixture_parameter_count(k, d, covariance_type)
    selected = next(c for c in candidates if c["k"] == k)
    np.testing.assert_allclose(
        -2 * likelihood.sum() + parameters * np.log(n), selected["bic"], rtol=1e-8, atol=1e-7
    )


def audit_pipeline_study(root, *, output_dir):
    try:
        from super_deli.benchmarking import simulation
        from super_deli.models.monte_carlo import summarize_upper_tail
        from super_deli.workflows.pipeline_calibration import TAIL_OPTIONS

        import_module("sklearn")
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install 'super-deli[clustering]' for pipeline-study analysis") from exc

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Analysis must be outside immutable study shards")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    original, rows, rates = None, [], []
    archive_members = successful_points = 0
    for case in CASES:
        directory = root / case
        options = read_json(directory / "study.json")
        study = PipelineStudyConfig.from_dict(options)
        if original is not None and original != options:
            raise ValueError("Study shards have different declarations")
        original = options
        shard = read_json(directory / "summary.json")
        records = [
            json.loads(line) for line in (directory / "datasets.jsonl").read_text().splitlines()
        ]
        if len(records) != study.n_datasets or shard != summarize_records(records, study, case):
            raise ValueError("Shard ledger, completion or denominators disagree")
        for i, row in enumerate(records):
            policy = case_config(study, case, i)
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
                raise ValueError("Dataset identity or seed mismatch")
            base = directory / row["directory"]
            result_dir = base / "calibration"
            result = read_json(result_dir / "summary.json")
            if (
                fingerprint_file(result_dir / "summary.json").sha256 != row["summary_sha256"]
                or read_json(result_dir / "configuration.json")
                != json.loads(json.dumps(policy.to_dict()))
                or result["configuration_sha256"] != configuration_hash(policy.to_dict())
                or result["generator_sha256"] != fingerprint_file(simulation.__file__).sha256
            ):
                raise ValueError("Calibration configuration or source checksum mismatch")
            for name, digest in result["artifacts"].items():
                if fingerprint_file(result_dir / name).sha256 != digest:
                    raise ValueError("Calibration artifact checksum mismatch")
            generated = _generated(
                policy.reference, seed, "null" if case.endswith("_null") else "complementary"
            )
            for name, digest in generated.items():
                if fingerprint_file(base / "observed_simulation" / name).sha256 != digest:
                    raise ValueError("Observed raw generator mismatch")
            for name, digest in result["observed_input_sha256"].items():
                if fingerprint_file(name).sha256 != digest:
                    raise ValueError("Observed input checksum mismatch")
            point = read_json(result_dir / "observed.json")
            draws = read_json(result_dir / "null_replicates.json")
            if point != result["point"] or len(draws) != (
                policy.n_simulations if point["status"] == "complete" else 0
            ):
                raise ValueError("Point or requested draw ledger mismatch")
            inventory = read_json(result_dir / "archive_inventory.json")
            with ZipFile(result_dir / "pipeline_artifacts.zip") as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(inventory):
                    raise ValueError("Archive membership mismatch")
                for name in names:
                    path = PurePosixPath(name)
                    data = archive.read(name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or inventory[name]
                        != {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                    ):
                        raise ValueError("Archive path or inventory checksum mismatch")
                archive_members += len(names)
                for j, draw in enumerate(draws):
                    null_seed = derived_seed(
                        policy.random_seed, policy.reference.law, j, "raw_null"
                    )
                    if (
                        draw["replicate"] != j
                        or draw["simulation_seed"] != null_seed
                        or draw != read_json(result_dir / f"draw_{j:04d}.json")
                        or draw["archive_prefix"] != f"null_{j:04d}"
                    ):
                        raise ValueError("Null draw identity or seed mismatch")
                    for name, digest in _generated(policy.reference, null_seed, "null").items():
                        raw = archive.read(draw["archive_prefix"] + "/simulation/" + name)
                        if hashlib.sha256(raw).hexdigest() != digest:
                            raise ValueError("Null raw generator mismatch")
                for fit in (point, *draws):
                    audit_pipeline_point(archive, fit, policy)
                    successful_points += fit["status"] == "complete"
            inference = (
                summarize_upper_tail(point["statistic"], draws, alpha=policy.alpha, **TAIL_OPTIONS)
                if point["status"] == "complete"
                else result["inference"]
            )
            if point["status"] == "failed" and inference != {
                "status": "point_failed",
                "requested": policy.n_simulations,
                "successful": 0,
                "failed": 0,
                "p_value": None,
                "reject_reference_population": None,
            }:
                raise ValueError("Invalid inference after a failed observed pipeline")
            if (
                row["statistic"] != point.get("statistic")
                or row["selected_k"] != point.get("selected_k")
                or inference != result["inference"]
                or inference["p_value"] != row["p_value"]
                or inference["reject_reference_population"] != row["reject"]
                or inference["status"] != row["status"]
            ):
                raise ValueError("Reconstructed reference inference mismatch")
            counts = {
                "planned_fits": policy.n_simulations + 1,
                "attempted_fits": len(draws) + 1,
                "successful_fits": sum(r["status"] == "complete" for r in (point, *draws)),
                "failed_fits": sum(r["status"] == "failed" for r in (point, *draws)),
                "unattempted_fits": policy.n_simulations - len(draws),
            }
            if any(row[key] != value or result[key] != value for key, value in counts.items()):
                raise ValueError("Fit requested/attempted/successful denominators disagree")
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
        "archive_members": archive_members,
        "verified_successful_pipelines": successful_points,
        **{
            key: sum(r[key] for r in rows)
            for key in (
                "planned_fits",
                "attempted_fits",
                "successful_fits",
                "failed_fits",
                "unattempted_fits",
            )
        },
        "rates": rates,
        "raw_files": len(files),
        "audit_scope": (
            "Regenerated raw inputs; verified archived bytes and parameter hashes; reconstructed "
            "block scaling/fusion, joint PCA projection, selected GMM likelihood/BIC/membership, "
            "complete K-selection bookkeeping, seeds and Monte Carlo inference; no model refits. "
            "View preprocessing/GPA/PCA fits and nonselected GMM likelihoods are preserved "
            "and checksummed, not independently refitted."
        ),
    }
    destination.mkdir(exist_ok=False)
    for name, values in (("datasets.csv", rows), ("rates.csv", rates)):
        with (destination / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(values[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    write_json(files, destination / "files.json")
    write_json(summary, destination / "summary.json")
    return summary
