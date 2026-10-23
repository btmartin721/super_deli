"""Paired fitted/known-law raw references with one shared observed point fit."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_boundary_reference import CASES, known_model, load_model
from super_deli.benchmarking.grouped_likelihood import case_policy
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_likelihood_config import GroupedLikelihoodStudyConfig
from super_deli.workflows.calibration_artifacts import archive_tree
from super_deli.workflows.grouped_likelihood import TAIL_OPTIONS, point_fitter

SCOPE = (
    "Paired simulation diagnostic of estimated versus known Gaussian nuisance parameters. "
    "Same counts, identities, model settings and Gaussian innovations per paired draw. "
    "One shared observed point fit. Known-law results are not empirical p-values, a "
    "correction, general size guarantee or species-boundary evidence. No stability resampling."
)


def run_reference(fitted_dir, reference, policy, destination):
    import numpy as np

    from super_deli.models.grouped_null import simulate_null, write_raw_views
    from super_deli.models.monte_carlo import summarize_upper_tail
    from super_deli.workflows.integrative import run_integrative

    destination.mkdir(exist_ok=False)
    write_json(policy.to_dict(), destination / "configuration.json")
    write_json(reference.layout, destination / "layout.json")
    with (destination / "null_model.npz").open("xb") as handle:
        np.savez(handle, **reference.model_arrays())
    observed = read_json(fitted_dir / "observed.json")
    write_json(observed, destination / "observed.json")
    records, inventory = [], {}
    with ZipFile(destination / "pipeline_artifacts.zip", "x", ZIP_DEFLATED, compresslevel=4) as out:
        source_inventory = read_json(fitted_dir / "archive_inventory.json")
        with ZipFile(fitted_dir / "pipeline_artifacts.zip") as source:
            for name in source.namelist():
                if name.startswith("observed/"):
                    out.writestr(name, source.read(name))
                    inventory[name] = source_inventory[name]
        if observed["status"] == "complete":
            for i in range(policy.n_simulations):
                seed = derived_seed(policy.random_seed, "grouped_raw_gaussian_v2", i, "null_data")
                with TemporaryDirectory(prefix="known-", dir=destination) as temporary:
                    staging = Path(temporary)
                    values = simulate_null(reference, seed)
                    declaration = write_raw_views(values, reference.layout, staging / "raw")
                    record = point_fitter(
                        staging,
                        declaration,
                        staging / "raw",
                        f"null_{i:04d}",
                        model=policy.model,
                        runner=run_integrative,
                    )
                    record.update(replicate=i, simulation_seed=seed)
                    records.append(record)
                    write_json(record, destination / f"draw_{i:04d}.json")
                    archive_tree(
                        out, staging, record["archive_prefix"], inventory, policy.max_archive_bytes
                    )
    write_json(records, destination / "null_replicates.json")
    write_json(inventory, destination / "archive_inventory.json")
    inference = (
        summarize_upper_tail(observed["statistic"], records, alpha=policy.alpha, **TAIL_OPTIONS)
        if observed["status"] == "complete"
        else {"status": "point_failed", "p_value": None, "reject_grouped_raw_gaussian": None}
    )
    inference["p_value_kind"] = "Known-law simulation diagnostic; not empirical calibration"
    result = {"scope": SCOPE, "inference": inference, "shared_observed_fit": True}
    write_json(result, destination / "summary.json")
    return records


def run_case(study, *, case, output_dir):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.grouped_likelihood import run_grouped_likelihood

    study = GroupedLikelihoodStudyConfig.from_dict(study.to_dict())
    if case not in CASES:
        raise ValueError("Unknown Gaussian boundary diagnostic case")
    root = Path(output_dir).resolve()
    root.mkdir(exist_ok=False)
    write_json(study.to_dict(), root / "study.json")
    records = []
    with (root / "datasets.jsonl").open("x") as ledger:
        for i in range(study.n_datasets):
            policy = case_policy(study, case, i)
            seed = derived_seed(study.random_seed, case, i, "observed_data")
            base = root / f"dataset_{i:04d}"
            base.mkdir()
            values, layout, labels = simulate_grouped(
                case, study.n_units, seed, target_level=policy.model.target_level
            )
            dataset = write_raw_views(values, layout, base / "observed")
            with (base / "values.npz").open("xb") as handle:
                np.savez(handle, **values)
            write_json(
                {"labels": labels.tolist(), "scope": "Single Gaussian law"}, base / "truth.json"
            )
            row = {
                "case": case,
                "repetition": i,
                "data_seed": seed,
                "null_stream_seed": policy.random_seed,
                "values_sha256": fingerprint_file(base / "values.npz").sha256,
                "planned_fits": 1 + 2 * policy.n_simulations,
            }
            try:
                run_grouped_likelihood(
                    dataset,
                    config=policy,
                    base_dir=base / "observed",
                    output_dir=base / "calibration",
                )
                reference = known_model(load_model(base / "calibration"), case)
                run_reference(base / "calibration", reference, policy, base / "known")
                inference = [
                    read_json(base / arm / "summary.json")["inference"]["status"]
                    for arm in ("calibration", "known")
                ]
                row["status"] = (
                    "complete" if all(x == "available" for x in inference) else "incomplete"
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                row.update(status="failed", error=str(exc))
                write_json(row, base / "failure.json")
            attempted = []
            if (base / "calibration/observed.json").exists():
                attempted.append(read_json(base / "calibration/observed.json"))
            for arm in ("calibration", "known"):
                attempted.extend(read_json(p) for p in sorted((base / arm).glob("draw_*.json")))
            row.update(
                attempted_fits=len(attempted),
                successful_fits=sum(r["status"] == "complete" for r in attempted),
                failed_fits=sum(r["status"] == "failed" for r in attempted),
                unattempted_fits=row["planned_fits"] - len(attempted),
            )
            records.append(row)
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
            ledger.flush()
    result = {
        "scope": SCOPE,
        "case": case,
        "datasets": len(records),
        **{
            key: sum(r[key] for r in records)
            for key in (
                "planned_fits",
                "attempted_fits",
                "successful_fits",
                "failed_fits",
                "unattempted_fits",
            )
        },
        "failed_observations": sum(r["status"] == "failed" for r in records),
        "withheld_observations": sum(r["status"] != "complete" for r in records),
        "replacement_draws": 0,
        "replacement_observations": 0,
    }
    write_json(result, root / "summary.json")
    return result
