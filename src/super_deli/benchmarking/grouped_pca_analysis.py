"""Independent no-refit audit of equal-group PCA, recovery and stability evidence."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_pca_study import replay, simulate
from super_deli.benchmarking.grouped_recovery_metrics import recovery_metrics
from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
from super_deli.benchmarking.recovery_analysis import _ari, _axis_check, _wilson
from super_deli.benchmarking.recovery_inputs import DirectoryArchive, inventory
from super_deli.benchmarking.shared_analysis import audit_stability
from super_deli.benchmarking.shared_study import accounting, point_metrics
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_pca_study_config import (
    ARMS,
    COHORTS,
    GroupedPCAStudyConfig,
    study_model,
)


def _pca_moments(matrix, parameters, scaling):
    import numpy as np

    keep = np.flatnonzero(matrix.var(0) > 0)
    np.testing.assert_array_equal(parameters["kept_indices"], keep)
    expected_mean = matrix[:, keep].mean(0)
    scale = matrix[:, keep].std(0) if scaling in {None, "standard"} else np.ones(len(keep))
    np.testing.assert_allclose(parameters["feature_means"], expected_mean, atol=1e-11)
    np.testing.assert_allclose(parameters["feature_scales"], scale, atol=1e-11)
    standardized = (matrix[:, keep] - expected_mean) / scale
    np.testing.assert_allclose(parameters["mean"], standardized.mean(0), atol=1e-12)
    centered = standardized - standardized.mean(0)
    covariance = centered.T @ centered / (len(matrix) - 1)
    components = np.asarray(parameters["components"])
    values = np.linalg.eigvalsh(covariance)[::-1][: len(components)]
    np.testing.assert_allclose(components @ components.T, np.eye(len(components)), atol=1e-11)
    np.testing.assert_allclose(covariance @ components.T, components.T * values, atol=1e-10)
    np.testing.assert_allclose(parameters["explained_variance"], values, atol=1e-10)


def _target_parameters(diagnostic, means, counts, units, draw, view):

    if (
        diagnostic["pca_fit"] != "target_means"
        or diagnostic["pca_training_target_ids"] != [units[i] for i in draw]
        or diagnostic["pca_training_counts"] != counts[draw].tolist()
    ):
        raise ValueError("Target-mean PCA training identities/counts/multiplicities disagree")
    parameters = {
        key: diagnostic[value]
        for key, value in {
            "kept_indices": "pca_kept_indices",
            "feature_means": "pca_feature_means",
            "feature_scales": "pca_feature_scales",
            "components": "pca_components",
            "mean": "pca_center",
            "explained_variance": "pca_explained_variance",
        }.items()
    }
    _pca_moments(means[draw], parameters, view.scaling)
    return parameters


def _point(run, record, model, values, layout, *, allow_stability=True):
    import numpy as np

    audit_pipeline_point(
        DirectoryArchive(run.parent.parent),
        record,
        SimpleNamespace(model=model),
        allow_stability=allow_stability,
    )
    summary = read_json(run / "summary.json")
    units = list(layout["unit_ids"])
    if summary["unit_ids"] != units or record["n_retained"] != len(units):
        raise ValueError("Retained group identities disagree")
    order = {u: i for i, u in enumerate(units)}
    projections = {}
    for j, (view, design) in enumerate(zip(model.views, layout["views"], strict=True)):
        path = run / f"view_{j:03d}"
        source = read_json(path / "source_rows.json")
        for key in ("unit_ids", "observation_ids", "target_ids", "feature_names"):
            if source[key] != list(design[key]):
                raise ValueError("Source identities/features disagree")
        raw = np.load(path / "source_values.npy", allow_pickle=False)
        np.testing.assert_array_equal(raw, values[view.kind])
        target = np.array([order[u] for u in design["target_ids"]])
        counts = np.bincount(target, minlength=len(units))
        means = np.stack([raw[target == i].mean(0) for i in range(len(units))])
        matrix = means if view.pca_fit == "target_means" else raw
        with np.load(path / "parameters/pca.npz", allow_pickle=False) as pca:
            parameters = {k: pca[k] for k in pca.files}
        _pca_moments(matrix, parameters, view.scaling)
        diagnostic = read_json(path / "parameters/summary.json")["diagnostics"]
        if view.pca_fit == "target_means":
            other = _target_parameters(
                diagnostic, means, counts, units, np.arange(len(units)), view
            )
            for key, value in other.items():
                np.testing.assert_allclose(value, parameters[key], atol=1e-12)
        scores = (
            (raw[:, parameters["kept_indices"]] - parameters["feature_means"])
            / parameters["feature_scales"]
            - parameters["mean"]
        ) @ parameters["components"].T
        profiles = np.stack([scores[target == i].mean(0) for i in range(len(units))])
        with np.load(path / "parameters/profiles.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["profiles"], profiles, atol=1e-11)
        projections[view.kind] = (raw, target, counts, means, scores, profiles)
    return projections


def _bootstrap_pca(run, model, layout, projections):
    import numpy as np

    units = list(layout["unit_ids"])
    for row in read_json(run / "bootstrap_replicates.json"):
        if row["status"] != "complete":
            continue
        draw = np.array(row["training_indices"])
        for view, diagnostic in zip(model.views, row["views"], strict=True):
            _, target, counts, means, _, _ = projections[view.kind]
            expected = np.concatenate([np.flatnonzero(target == i) for i in draw])
            np.testing.assert_array_equal(diagnostic["source_training_indices"], expected)
            if view.pca_fit == "target_means":
                _target_parameters(diagnostic, means, counts, units, draw, view)


def summaries(rows):
    import numpy as np

    result = []
    for cohort in COHORTS:
        for arm in ARMS:
            selected = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
            available = [r for r in selected if r["status"] == "complete"]
            n = len(available)
            counts = {str(k): sum(r["selected_k"] == k for r in available) for k in range(1, 5)}
            rates = {}
            for name, events in (("four", counts["4"]), ("non_single", n - counts["1"])):
                rates[name] = {
                    "events": events,
                    "available": n,
                    "rate": events / n if n else None,
                    "wilson_95": _wilson(events, n),
                    "failed_outcome_bounds": [
                        events / len(selected),
                        (events + len(selected) - n) / len(selected),
                    ],
                }
            row = {
                "cohort": cohort,
                "arm": arm,
                "requested": len(selected),
                "available": n,
                "failed": len(selected) - n,
                "selected_k_counts": counts,
                "rates": rates,
            }
            for key in (
                "ari",
                "joint_a_r2",
                "joint_b_r2",
                "morphology_a_r2",
                "ecology_b_r2",
                "k4_minus_k2_bic",
            ):
                values = [r[key] for r in available if r[key] is not None]
                row[key] = (
                    {
                        "mean": float(np.mean(values)),
                        "mean_mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else None,
                        "median": float(np.median(values)),
                        "min": float(min(values)),
                        "max": float(max(values)),
                    }
                    if values
                    else None
                )
            result.append(row)
    paired = []
    for cohort in COHORTS:
        for left, right in (
            ("source_2", "source_4"),
            ("source_2", "target_2"),
            ("source_4", "target_4"),
            ("source_2", "target_4"),
        ):
            a = {
                r["repetition"]: r
                for r in rows
                if r["cohort"] == cohort and r["arm"] == left and r["status"] == "complete"
            }
            b = {
                r["repetition"]: r
                for r in rows
                if r["cohort"] == cohort and r["arm"] == right and r["status"] == "complete"
            }
            common = sorted(a.keys() & b.keys())
            differences = [b[i]["ari"] - a[i]["ari"] for i in common if a[i]["ari"] is not None]
            paired.append(
                {
                    "cohort": cohort,
                    "left": left,
                    "right": right,
                    "available_pairs": len(common),
                    "mean_ari_change": float(np.mean(differences)) if differences else None,
                    "median_ari_change": float(np.median(differences)) if differences else None,
                    "ari_improved": sum(d > 1e-9 for d in differences),
                    "ari_worsened": sum(d < -1e-9 for d in differences),
                    "gained_four": sum(
                        a[i]["selected_k"] != 4 and b[i]["selected_k"] == 4 for i in common
                    ),
                    "lost_four": sum(
                        a[i]["selected_k"] == 4 and b[i]["selected_k"] != 4 for i in common
                    ),
                }
            )
    return result, paired


def audit_grouped_pca_study(root, *, output_dir):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("Audit must be outside immutable study")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    expected = read_json(root / "files.json")
    actual = inventory(root)
    actual.pop("files.json")
    if expected != actual:
        raise ValueError("Study artifact inventory mismatch")
    config = GroupedPCAStudyConfig.from_dict(read_json(root / "configuration.json"))
    prior = GroupedStudyConfig.from_dict(read_json(root / "source_study.json"))
    if read_json(root / "grid.json") != {
        arm: json.loads(json.dumps(study_model(config, "archived", 0, arm).to_dict()))
        for arm in ARMS
    }:
        raise ValueError("Frozen study grid mismatch")
    declarations = read_json(root / "datasets.json")
    if [(r["cohort"], r["repetition"]) for r in declarations] != [
        (c, i) for c in COHORTS for i in range(config.n_datasets)
    ]:
        raise ValueError("Dataset identities or replacements differ")
    records, rows, candidates, stability = [], [], [], []
    for declaration in declarations:
        cohort, i = declaration["cohort"], declaration["repetition"]
        n = config.large_n if cohort == "fresh_256" else config.small_n
        seed = (
            derived_seed(prior.random_seed, "complementary", i, "observed_data")
            if cohort == "archived"
            else derived_seed(config.random_seed, cohort, i, "observed_data")
        )
        structured = cohort in {"archived", "fresh_64", "fresh_256", "unequal_variance"}
        if declaration != {
            "cohort": cohort,
            "repetition": i,
            "n": n,
            "seed": seed,
            "has_groups": structured,
        }:
            raise ValueError("Seed, size or evaluation declaration mismatch")
        directory = root / cohort / f"dataset_{i:04d}"
        values, layout, labels = simulate(cohort, n, seed, target_level=config.model.target_level)
        truth = read_json(directory / "truth.json")
        if (
            truth["labels"] != labels.tolist()
            or truth["unit_ids"] != list(layout["unit_ids"])
            or truth["has_groups"] != structured
        ):
            raise ValueError("Generating truth/identities disagree")
        if read_json(directory / "layout.json") != json.loads(json.dumps(layout)):
            raise ValueError("Raw group layout/counts mismatch")
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            for key, value in values.items():
                np.testing.assert_array_equal(value, arrays[key])
        with TemporaryDirectory(prefix="super-deli-grouped-pca-audit-") as temporary:
            write_raw_views(values, layout, Path(temporary) / "raw")
            if inventory(Path(temporary) / "raw") != inventory(directory / "raw"):
                raise ValueError("Regenerated raw values/identities differ")
        for arm in ARMS:
            model = study_model(config, cohort, i, arm)
            record = read_json(directory / arm / "record.json")
            records.append({"cohort": cohort, "repetition": i, "arm": arm, **record})
            row = {
                "cohort": cohort,
                "repetition": i,
                "arm": arm,
                "n": n,
                "status": record["status"],
                "selected_k": None,
                "ari": None,
                "joint_a_r2": None,
                "joint_b_r2": None,
                "morphology_a_r2": None,
                "ecology_b_r2": None,
                "k4_minus_k2_bic": None,
                "bootstrap_requested": model.n_bootstrap,
                "bootstrap_failed": record["bootstrap"]["failed"],
            }
            if record["status"] == "failed":
                if (
                    not record.get("error")
                    or "metrics" in record
                    or record["bootstrap"]
                    != {
                        "requested": model.n_bootstrap,
                        "attempted": 0,
                        "successful": 0,
                        "failed": 0,
                    }
                ):
                    raise ValueError("Malformed failed point record")
                rows.append(row)
                continue
            run = directory / arm / "run"
            projections = _point(run, record, model, values, layout)
            metrics = (
                recovery_metrics(run, labels) if structured else point_metrics(run, labels, False)
            )
            if metrics != record["metrics"]:
                raise ValueError("Point metrics disagree")
            with np.load(run / "mixture.npz", allow_pickle=False) as mixture:
                if structured:
                    np.testing.assert_allclose(
                        metrics["ari"], _ari(labels, mixture["labels"]), atol=1e-12
                    )
                    _axis_check(mixture["scores"], labels, metrics["joint_axis_r2"])
                    with np.load(run / "fusion.npz", allow_pickle=False) as f:
                        _axis_check(f["fused"], labels, metrics["fused_axis_r2"])
                    for kind, (_, target, _, means, scores, profiles) in projections.items():
                        upstream = metrics["upstream"][kind]
                        _axis_check(means, labels, upstream["raw_group_axis_r2"])
                        _axis_check(scores, labels[target], upstream["source_axis_r2"])
                        _axis_check(profiles, labels, upstream["profile_axis_r2"])
                verified, recoveries = audit_stability(
                    run, model, mixture["labels"], record["selected_k"]
                )
            _bootstrap_pca(run, model, layout, projections)
            if verified != record["bootstrap"]:
                raise ValueError("Stability ledger mismatch")
            c2, c4 = (next(c for c in metrics["bic_terms"] if c["k"] == k) for k in (2, 4))
            row.update(
                selected_k=record["selected_k"],
                ari=metrics["ari"],
                k4_minus_k2_bic=c4["bic"] - c2["bic"],
            )
            if structured:
                row.update(
                    joint_a_r2=metrics["joint_axis_r2"]["a"],
                    joint_b_r2=metrics["joint_axis_r2"]["b"],
                    morphology_a_r2=metrics["upstream"]["morphological"]["profile_axis_r2"]["a"],
                    ecology_b_r2=metrics["upstream"]["ecological"]["profile_axis_r2"]["b"],
                )
            rows.append(row)
            for c in metrics["bic_terms"]:
                d, k = c["d"], c["k"]
                if c["parameters"] != k - 1 + k * d + d * (d + 1) // 2:
                    raise ValueError("Tied-family parameter count mismatch")
                candidates.append(
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
                )
            if model.n_bootstrap:
                stability.append(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "arm": arm,
                        **verified,
                        "selected_k": record["selected_k"],
                        "mean_exact_set_frequency": float(np.mean(recoveries) / model.n_bootstrap)
                        if not verified["failed"]
                        else None,
                    }
                )
        if cohort == "archived":
            old = read_json(directory / "source_baseline/record.json")
            _point(
                directory / "source_baseline/run",
                {**old, "archive_prefix": "source_baseline"},
                config.model,
                values,
                layout,
                allow_stability=False,
            )
    if records != read_json(root / "points.json"):
        raise ValueError("Global ledger differs")
    reproduced = replay(root, config.n_datasets)
    if reproduced != read_json(root / "replay.json"):
        raise ValueError("Replay ledger mismatch")
    total = {
        "schema_version": 1,
        "datasets": len(declarations),
        "reproduced_archived": sum(r["status"] == "reproduced" for r in reproduced),
        **accounting(records),
    }
    if total != read_json(root / "summary.json"):
        raise ValueError("Study accounting mismatch")
    grouped, paired = summaries(rows)
    result = {
        **total,
        "status": "complete",
        "verified_files": len(expected),
        "summaries": grouped,
        "paired": paired,
        "audit_scope": (
            "Regenerated raw values/identities/counts/truth; full inventories, fixed "
            "settings, seeds and archived replay; independent PCA raw moments/eigen-equations "
            "for source rows and group means, projections/profiles/fusion, selected dense GMM "
            "density/BIC/membership; independent ARI/axis checks; bootstrap seeds/source "
            "members/draw multiplicities and target-mean PCA parameters, distinct-unit "
            "constraints, partitions, exact-set counts, OOB denominators and failure bounds. "
            "No model refits. Nonselected GMM and source-row bootstrap PCA parameters are not "
            "independently refitted."
        ),
        "interpretation": (
            "Prespecified four-arm comparison under constructed laws; unchanged "
            "defaults/p-values. K selection without generating discrete groups is "
            "descriptive, not a type-I error test or species significance. Wilson intervals "
            "are pointwise. Stability is separate from significance and conditions on fixed "
            "observed group members."
        ),
    }
    destination.mkdir()
    for name, data in (
        ("points.csv", rows),
        ("candidates.csv", candidates),
        ("stability.csv", stability),
        ("paired.csv", paired),
    ):
        with (destination / name).open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(dict.fromkeys(k for row in data for k in row)),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(data)
    write_json(result, destination / "summary.json")
    write_json(expected, destination / "files.json")
    return result
