"""Private per-space prediction, constraints and signal-retention evidence."""

from super_deli.io.config import read_json, write_json


def workflow_folds(workflow):
    import numpy as np

    assignments = (
        read_json(workflow / "fold_assignments.json")
        if (workflow / "fold_assignments.json").exists()
        else []
    )
    folds = []
    for a in assignments:
        directory = workflow / f"fold_{a['fold']:03d}"
        p = directory / "validation_record.json"
        record = (
            read_json(p) if p.exists() else {"status": "failed", "error": "Missing fold evidence"}
        )
        logs = {}
        if (directory / "validation.npz").exists():
            with np.load(directory / "validation.npz", allow_pickle=False) as held:
                logs = {k[1:]: held[k].tolist() for k in held.files if k.startswith("k")}
        folds.append({**a, **record, "log_densities": logs})
    return folds


def diagnose_workflow(workflow, truth, tests, test_layout, model, destination):
    import numpy as np
    from scipy.linalg import subspace_angles
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.aic_diagnostic_metrics import candidate_metrics
    from super_deli.benchmarking.grouped_recovery_metrics import upstream_metrics
    from super_deli.benchmarking.predictive_recovery_gates import decompose_gates
    from super_deli.benchmarking.recovery_metrics import axis_r2
    from super_deli.io.predictive_inputs import project_exported
    from super_deli.workflows.predictive import save_arrays

    destination.mkdir(parents=True, exist_ok=False)
    p = workflow / "full/run/candidates.json"
    if not p.exists():
        result = {"status": "withheld", "gates": None, "spaces": []}
        write_json(result, destination / "metrics.json")
        return result
    candidates = read_json(p)
    folds = workflow_folds(workflow)
    n = read_json(workflow / "full/run/summary.json")["unit_ids"]
    gates = decompose_gates(candidates, folds, len(n))
    definitions = [("full", None)] + [
        (f"fold_{f['fold']:03d}", f["training_indices"]) for f in folds
    ]
    spaces = []
    full_matrices = {}
    for prefix, indices in definitions:
        run = workflow / prefix / "run"
        if not (run / "gmm_candidates.npz").exists():
            spaces.append({"prefix": prefix, "status": "unavailable"})
            continue
        direct = test_layout is None
        scores = tests if direct else project_exported(run, tests, test_layout)
        metrics, logs = candidate_metrics(
            (run / "gmm_candidates.npz").read_bytes(),
            read_json(run / "candidates.json"),
            test_scores=scores,
            reg_covar=model.reg_covar,
        )
        save_arrays(
            destination / (prefix + "_external.npz"),
            scores=scores,
            **{"k" + k: v for k, v in logs.items()},
        )
        labels = None if truth is None else truth if indices is None else truth[indices]
        with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
            for c in metrics["candidates"]:
                if labels is not None and c["status"] != "insufficient_samples":
                    predicted = pool[f"k{c['k']}_labels"]
                    c["ari"] = float(adjusted_rand_score(labels, predicted))
                    c["generating_by_component"] = [
                        [int(np.sum((labels == a) & (predicted == b))) for b in range(c["k"])]
                        for a in range(4)
                    ]
            if labels is not None:
                metrics["joint_axis_r2"] = axis_r2(pool["scores"], labels)
                metrics["generating_counts"] = np.bincount(labels, minlength=4).tolist()
        if not direct:
            if labels is not None:
                metrics.update(upstream_metrics(run, labels))
            maps = []
            blocks = []
            for i in range(2):
                directory = run / f"view_{i:03d}/parameters"
                with np.load(directory / "pca.npz", allow_pickle=False) as p:
                    matrix = np.zeros((len(p["components"]), 4))
                    matrix[:, p["kept_indices"]] = p["components"] / p["feature_scales"]
                    if prefix == "full":
                        full_matrices[i] = matrix
                    angles = subspace_angles(full_matrices[i].T, matrix.T)
                    maps.append(
                        {
                            "view": i,
                            "angles_from_full_radians": angles.tolist(),
                            "loadings_raw_coordinates": matrix.tolist(),
                            "explained_variance_ratio": p["explained_variance_ratio"].tolist(),
                        }
                    )
                with np.load(directory / "profiles.npz", allow_pickle=False) as p:
                    blocks.append(
                        (p["profiles"] - p["mean"]) * np.sqrt(p["normalized_weight"]) / p["scale"]
                    )
            metrics["maps"] = maps
            if labels is not None:
                metrics["fused_axis_r2"] = axis_r2(np.column_stack(blocks), labels)
        spaces.append({"prefix": prefix, "status": "complete", **metrics})
    result = {"status": gates["status"], "gates": gates, "spaces": spaces}
    write_json(result, destination / "metrics.json")
    return result


def oracle_diagnostics(workflow, truth, tests, test_layout, model, destination):
    import numpy as np

    from super_deli.benchmarking.full_grouped_oracle import fit_oracle
    from super_deli.io.predictive_inputs import project_exported
    from super_deli.models.predictive_selection import gaussian_predictions
    from super_deli.workflows.predictive import fold_model, save_arrays

    destination.mkdir()
    assignments = (
        read_json(workflow / "fold_assignments.json")
        if (workflow / "fold_assignments.json").exists()
        else []
    )
    for prefix, indices, trained in [("full", None, model)] + [
        (f"fold_{f['fold']:03d}", f["training_indices"], fold_model(model, f["fold"]))
        for f in assignments
    ]:
        source = workflow / prefix / "run/gmm_candidates.npz"
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as pool:
            scores = pool["scores"].copy()
        labels = truth if indices is None else truth[indices]
        target = destination / prefix
        try:
            if set(labels.tolist()) != {0, 1, 2, 3}:
                raise ValueError("All four generating groups required for K=4 initialization")
            record = fit_oracle(scores, labels, "four", trained, target)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            target.mkdir(exist_ok=True)
            record = {
                "attempted": True,
                "partition": "four",
                "k": 4,
                "status": "failed",
                "error": str(exc),
                "role": "Truth-initialized optimization diagnostic; never selection",
            }
            write_json(record, target / "result.json")
        if record["status"] != "complete":
            continue
        external = project_exported(source.parent, tests, test_layout)
        with np.load(target / "mixture.npz", allow_pickle=False) as pool:
            parameters = {"k4_" + k: pool[k] for k in ("weights", "means", "covariances")}
            ext, _ = gaussian_predictions(external, parameters, 4)
            train, _ = gaussian_predictions(scores, parameters, 4)
            arrays = {
                "external_scores": external,
                "external_logdensity": ext,
                "training_logdensity": train,
            }
            if indices is not None:
                with np.load(workflow / prefix / "validation.npz", allow_pickle=False) as held:
                    val, _ = gaussian_predictions(held["scores"], parameters, 4)
                    arrays.update(validation_scores=held["scores"], validation_logdensity=val)
            save_arrays(target / "predictions.npz", **arrays)
