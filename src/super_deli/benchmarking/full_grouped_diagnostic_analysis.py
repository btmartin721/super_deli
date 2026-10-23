"""No-refit audit of point recovery, optimization diagnostics and archived power."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.full_grouped_diagnostics import accounting, metrics
from super_deli.benchmarking.full_grouped_oracle import audit_oracle
from super_deli.benchmarking.full_grouped_power import audit_tests
from super_deli.benchmarking.grouped_recovery_inputs import point, replay, source_pins
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_analysis import _ari, _axis_check, _wilson
from super_deli.benchmarking.recovery_inputs import DirectoryArchive, inventory
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.full_grouped_diagnostic_config import (
    ARMS,
    COHORTS,
    PARTITIONS,
    FullGroupedDiagnosticConfig,
    variants,
)
from super_deli.models.grouped_config import GroupedStudyConfig


def csv_rows(rows, path):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    import numpy as np

    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean_mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else None,
    }


def summaries(rows):
    output, paired = [], []
    for cohort in COHORTS:
        for arm in ARMS:
            planned = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
            complete = [r for r in planned if r["status"] == "complete"]
            counts = {str(k): sum(r["selected_k"] == k for r in complete) for k in range(1, 5)}
            output.append(
                {
                    "cohort": cohort,
                    "arm": arm,
                    "requested": len(planned),
                    "available": len(complete),
                    "failed": len(planned) - len(complete),
                    "selected_k_counts": counts,
                    "selected_four_wilson_95": _wilson(counts["4"], len(complete)),
                    "withheld_four_bounds": [
                        counts["4"] / len(planned),
                        (counts["4"] + len(planned) - len(complete)) / len(planned),
                    ],
                    **{
                        key: describe([r.get(key) for r in complete])
                        for key in (
                            "ari",
                            "joint_a_r2",
                            "joint_b_r2",
                            "fused_a_r2",
                            "fused_b_r2",
                            "k4_bic_minus_selected",
                            "k4_twice_ll_gain",
                            "k4_extra_penalty",
                        )
                    },
                    "k4_eligible": sum(r["k4_status"] == "eligible" for r in complete),
                }
            )
        left = {
            r["repetition"]: r
            for r in rows
            if r["cohort"] == cohort and r["arm"] == "baseline" and r["status"] == "complete"
        }
        for arm in ARMS[1:]:
            right = {
                r["repetition"]: r
                for r in rows
                if r["cohort"] == cohort and r["arm"] == arm and r["status"] == "complete"
            }
            ids = sorted(left.keys() & right.keys())
            paired.append(
                {
                    "cohort": cohort,
                    "left": "baseline",
                    "right": arm,
                    "available_pairs": len(ids),
                    "selected_k_changes": sum(
                        left[i]["selected_k"] != right[i]["selected_k"] for i in ids
                    ),
                    "gained_four": sum(
                        left[i]["selected_k"] != 4 and right[i]["selected_k"] == 4 for i in ids
                    ),
                    "lost_four": sum(
                        left[i]["selected_k"] == 4 and right[i]["selected_k"] != 4 for i in ids
                    ),
                    "ari_change": describe([right[i]["ari"] - left[i]["ari"] for i in ids]),
                }
            )
    return output, paired


def _audit_point(directory, record, model, values, layout, labels):
    import numpy as np

    run = directory / "run"
    point(
        DirectoryArchive(directory.parent),
        {**record, "archive_prefix": directory.name},
        SimpleNamespace(model=model),
        values,
        layout,
    )
    result = metrics(run, labels, model)
    if result != record["metrics"]:
        raise ValueError("Saved diagnostic metrics disagree")
    with np.load(run / "mixture.npz", allow_pickle=False) as fitted:
        np.testing.assert_allclose(result["ari"], _ari(labels, fitted["labels"]), atol=1e-12)
        _axis_check(fitted["scores"], labels, result["joint_axis_r2"])
    with np.load(run / "fusion.npz", allow_pickle=False) as fusion:
        _axis_check(fusion["fused"], labels, result["fused_axis_r2"])
    order = {u: j for j, u in enumerate(layout["unit_ids"])}
    for j, view in enumerate(layout["views"]):
        target = np.array([order[u] for u in view["target_ids"]])
        raw_means = np.stack(
            [values[view["kind"]][target == g].mean(0) for g in range(len(labels))]
        )
        _axis_check(raw_means, labels, result["upstream"][view["kind"]]["raw_group_axis_r2"])
        with np.load(run / f"view_{j:03d}/parameters/profiles.npz", allow_pickle=False) as profiles:
            _axis_check(profiles["profiles"], labels, result["view_axis_r2"][view["kind"]])
    for c in result["bic_terms"]:
        d, k = c["d"], c["k"]
        q = k - 1 + k * d + (k if model.covariance_type == "full" else 1) * d * (d + 1) // 2
        if c["parameters"] != q:
            raise ValueError("Dimension/family-specific BIC parameter mismatch")
        np.testing.assert_allclose(c["penalty"], q * np.log(len(labels)), atol=1e-12)
        if c["bic"] is not None:
            np.testing.assert_allclose(
                c["bic_minus_selected"],
                c["penalty_minus_selected"] - c["twice_log_likelihood_gain"],
                atol=1e-10,
            )
    return result


def audit_full_grouped_diagnostics(root, *, output_dir):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("Audit must be separate from immutable diagnostic artifacts")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    expected, actual = read_json(root / "files.json"), inventory(root)
    actual.pop("files.json")
    if expected != actual:
        raise ValueError("Diagnostic artifact inventory mismatch")
    config = FullGroupedDiagnosticConfig.from_dict(read_json(root / "configuration.json"))
    for original, name in (
        ("study.json", "source_declaration.json"),
        ("study-audit/files.json", "source_inventory.json"),
        ("runtime_sources.json", "source_runtime.json"),
    ):
        if fingerprint_file(root / name).sha256 != source_pins(config)[original]:
            raise ValueError("Pinned source declaration/inventory/runtime mismatch")
    study = GroupedStudyConfig.from_dict(read_json(root / "source_study.json"))
    if study != GroupedStudyConfig.from_dict(read_json(root / "source_declaration.json")) or (
        study.n_units,
        study.n_datasets,
    ) != (config.small_n, config.n_datasets):
        raise ValueError("Source declaration disagrees with diagnostic cohort")
    grid = variants(study.calibration.model)
    if read_json(root / "grid.json") != json.loads(
        json.dumps({k: m.to_dict() for k, m in grid.items()})
    ):
        raise ValueError("Frozen grid mismatch")
    execution = read_json(root / "execution.json")
    if (
        execution["source_pins"] != source_pins(config)
        or execution["replacement_fits"]
        or execution["threads_per_job"] != 1
    ):
        raise ValueError("Execution declaration mismatch")
    source_inventory = read_json(root / "source_inventory.json")
    for path in (root / "retained_tests").rglob("*.json"):
        key = path.relative_to(root / "retained_tests")
        original = key.parent / "calibration" / key.name
        if fingerprint_file(path).sha256 != source_inventory[original.as_posix()]:
            raise ValueError("Retained test differs from pinned source")
    tails = audit_tests(root / "retained_tests", study)
    if tails != read_json(root / "retained_tails.json"):
        raise ValueError("Retained tail summary mismatch")
    declarations = read_json(root / "datasets.json")
    if [(r["cohort"], r["repetition"]) for r in declarations] != [
        (c, i) for c in COHORTS for i in range(config.n_datasets)
    ]:
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
            raise ValueError("Diagnostic dataset seed/size mismatch")
        directory = root / cohort / f"dataset_{i:04d}"
        values, layout, labels = simulate_grouped(
            "complementary", n, seed, target_level=study.calibration.model.target_level
        )
        truth = read_json(directory / "truth.json")
        if (
            truth["labels"] != labels.tolist()
            or truth["unit_ids"] != list(layout["unit_ids"])
            or read_json(directory / "layout.json") != json.loads(json.dumps(layout))
        ):
            raise ValueError("Generating truth/layout mismatch")
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            for kind, value in values.items():
                np.testing.assert_array_equal(arrays[kind], value)
        with TemporaryDirectory(prefix="super-deli-full-diagnostic-audit-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_views(values, layout, generated)
            if inventory(generated) != inventory(directory / "raw"):
                raise ValueError("Regenerated raw inputs/identities mismatch")
        for arm, model in grid.items():
            record = read_json(directory / arm / "record.json")
            points.append({"cohort": cohort, "repetition": i, "arm": arm, **record})
            row = {
                "cohort": cohort,
                "repetition": i,
                "arm": arm,
                "n": n,
                "status": record["status"],
            }
            if record["status"] == "complete":
                result = _audit_point(directory / arm, record, model, values, layout, labels)
                k4 = next(c for c in result["bic_terms"] if c["k"] == 4)
                row.update(
                    selected_k=result["selected_k"],
                    ari=result["ari"],
                    joint_a_r2=result["joint_axis_r2"]["a"],
                    joint_b_r2=result["joint_axis_r2"]["b"],
                    fused_a_r2=result["fused_axis_r2"]["a"],
                    fused_b_r2=result["fused_axis_r2"]["b"],
                    morphology_a_r2=result["view_axis_r2"]["morphological"]["a"],
                    ecology_b_r2=result["view_axis_r2"]["ecological"]["b"],
                    k4_status=k4["status"],
                    k4_bic_minus_selected=k4["bic_minus_selected"],
                    k4_twice_ll_gain=k4["twice_log_likelihood_gain"],
                    k4_extra_penalty=k4["penalty_minus_selected"],
                    covariance_condition_max=max(result["selected_covariance_condition_numbers"]),
                )
                candidate_rows.extend(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "arm": arm,
                        **{
                            k: v
                            for k, v in c.items()
                            if k not in {"hard_cluster_sizes", "distinct_unit_sizes"}
                        },
                    }
                    for c in result["bic_terms"]
                )
            elif record["status"] != "failed" or not record.get("error") or "metrics" in record:
                raise ValueError("Invalid failed point record")
            rows.append(row)
        if cohort == "archived":
            snapshot = root / "source_snapshots" / directory.name
            manifest = snapshot / "archive_inventory.json"
            if (
                fingerprint_file(manifest).sha256
                != source_inventory[
                    f"complementary/{directory.name}/calibration/archive_inventory.json"
                ]
            ):
                raise ValueError("Source archive inventory mismatch")
            declared_members = read_json(manifest)
            actual_members = inventory(snapshot / "observed")
            if {"observed/" + k: v for k, v in actual_members.items()} != {
                k: v["sha256"] for k, v in declared_members.items() if k.startswith("observed/")
            }:
                raise ValueError("Source snapshot differs from pinned archive inventory")
            old_record = read_json(snapshot / "record.json")
            if old_record != read_json(
                root / "retained_tests/complementary" / directory.name / "observed.json"
            ):
                raise ValueError("Archived observation record mismatch")
            point(DirectoryArchive(snapshot), old_record, study.calibration, values, layout)
        baseline = read_json(directory / "baseline/record.json")
        for name in PARTITIONS:
            record = read_json(directory / f"truth_{name}/result.json")
            oracle = {"cohort": cohort, "repetition": i, **record}
            oracles.append(dict(oracle))
            if baseline["status"] == "complete":
                with np.load(directory / "baseline/run/mixture.npz", allow_pickle=False) as fitted:
                    audit_oracle(
                        directory / f"truth_{name}",
                        labels,
                        fitted["scores"],
                        name,
                        study.calibration.model,
                        record,
                    )
                if record["status"] == "complete":
                    selected = next(
                        c for c in baseline["candidates"] if c["k"] == baseline["selected_k"]
                    )
                    matching = next(c for c in baseline["candidates"] if c["k"] == record["k"])
                    oracle.update(
                        bic_minus_selected=record["bic"] - selected["bic"],
                        bic_minus_unsupervised_same_k=record["bic"] - matching["bic"],
                    )
            elif record["status"] != "unattempted" or record["attempted"]:
                raise ValueError("Truth fit attempted without baseline scores")
            oracle_rows.append(
                {k: v for k, v in oracle.items() if k not in {"role", "hard_cluster_sizes"}}
            )
    if points != read_json(root / "points.json") or oracles != read_json(
        root / "truth_initialized.json"
    ):
        raise ValueError("Global ledgers differ from individual records")
    reproduced = replay(root, config.n_datasets)
    if reproduced != read_json(root / "replay.json"):
        raise ValueError("Archived replay mismatch")
    expected_counts = accounting(points, oracles, len(declarations), reproduced)
    if expected_counts != read_json(root / "summary.json"):
        raise ValueError("Diagnostic fit accounting mismatch")
    summary, paired = summaries(rows)
    result = {
        **expected_counts,
        "status": "complete",
        "verified_files": len(expected),
        "archived_tests": len(tails),
        "archived_draws": sum(r["successful_draws"] for r in tails),
        "summaries": summary,
        "paired": paired,
        "audit_scope": (
            "No model refits. Raw regeneration and identities; independent PCA moments, "
            "eigen-equations/projections, fusion and selected dense GMM likelihood, "
            "membership/BIC; candidate-log algebra; independent ARI/axis recovery; truth "
            "initialization and dense likelihood; exact retained-test hashes, seeds and "
            "upper-tail recomputation. Nonselected likelihoods inferred from logged BIC, "
            "not independently recomputed. No significance/stability reruns."
        ),
    }
    destination.mkdir()
    for name, data in (
        ("points.csv", rows),
        ("candidates.csv", candidate_rows),
        ("truth_initialized.csv", oracle_rows),
        ("retained_tails.csv", tails),
    ):
        csv_rows(data, destination / name)
    write_json(result, destination / "summary.json")
    write_json(expected, destination / "files.json")
    return result
