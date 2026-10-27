"""Thin native workflow adapters; screened failures retain their ordinary fits."""

from pathlib import Path

from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json


def run_grouped_arm(arm, dataset, base_dir, output_dir):
    if arm.kind == "ordinary":
        from super_deli.workflows.integrative import run_integrative

        return run_integrative(
            dataset, model_config=arm.model, base_dir=base_dir, output_dir=output_dir
        )
    options = {} if arm.fold_size_policy is None else {"fold_size_policy": arm.fold_size_policy}
    if arm.bootstrap is None:
        from super_deli.workflows.predictive import run_predictive

        return run_predictive(
            dataset,
            model_config=arm.model,
            screen_config=arm.screen,
            base_dir=base_dir,
            output_dir=output_dir,
            **options,
        )
    from super_deli.workflows.predictive_bootstrap import run_predictive_bootstrap

    return run_predictive_bootstrap(
        dataset,
        model_config=arm.model,
        screen_config=arm.screen,
        bootstrap_config=arm.bootstrap,
        base_dir=base_dir,
        output_dir=output_dir,
        **options,
    )


def read_grouped_arm(arm, directory, cohort):
    import numpy as np

    root = Path(directory)
    native = read_json(root / "summary.json") if (root / "summary.json").exists() else {}
    point = root / ("point" if arm.bootstrap is not None else "")
    run = root if arm.kind == "ordinary" else point / "full/run"
    ordinary = read_json(run / "summary.json")
    if ordinary["unit_ids"] != cohort["unit_ids"]:
        raise ValueError("Child target order mismatch")
    # Verify all native source identities/features/target memberships before comparing.
    for i, expected in enumerate(cohort["views"]):
        rows = read_json(run / f"view_{i:03d}/source_rows.json")
        values = np.load(run / f"view_{i:03d}/source_values.npy", allow_pickle=False)
        if configuration_hash(values.tolist()) != expected["values_sha256"]:
            raise ValueError("Child selected source values differ from the frozen cohort")
        for key in ("observation_ids", "unit_ids", "target_ids", "feature_names"):
            if rows[key] != expected[key]:
                raise ValueError(f"Child source membership mismatch: {key}")
    params = {}
    for filename in [
        "fusion.npz",
        *(f"view_{i:03d}/parameters/{name}.npz" for i in (0, 1) for name in ("pca", "profiles")),
    ]:
        with np.load(run / filename, allow_pickle=False) as a:
            params.update({f"{filename}:{k}": a[k].copy() for k in a.files})
    result = {
        "ordinary_k": ordinary["selected_k"],
        "ordinary_available": True,
        "native_summary": str((root / "summary.json").name),
        "point_available": True,
        "screen_status": "not_requested",
        "stability": ordinary["bootstrap"],
        "selected_k": ordinary["selected_k"],
        "selected_assignments": run / "assignments.csv",
    }
    with np.load(run / "mixture.npz", allow_pickle=False) as a:
        scores = a["scores"].copy()
        labels = a["labels"].tolist()
    if arm.kind == "predictive":
        screen = (
            read_json(point / "summary.json")
            if (point / "summary.json").exists()
            else {
                "selection": {
                    "status": "withheld",
                    "screened_k": None,
                    "error": "Native screen terminated after its ordinary fit",
                }
            }
        )
        selection = screen["selection"]
        result.update(
            screen_status=selection["status"],
            selection=selection,
            selected_k=selection.get("screened_k"),
            point_available=selection["status"] == "complete",
            selected_assignments=point / "selected_assignments.csv",
        )
        if result["point_available"]:
            with np.load(point / "selected_memberships.npz", allow_pickle=False) as a:
                labels = a["labels"].tolist()
        if arm.bootstrap is not None:
            result["stability"] = native.get(
                "stability", {"status": "unavailable", "requested": arm.bootstrap.n_bootstrap}
            )
        else:
            result["stability"] = {
                "status": "not_computed"
                if arm.model.covariance_type == "tied"
                and arm.model.target_level == "locality"
                and arm.screen.n_folds == 5
                else "unsupported",
                "requested": 0,
                "reason": "Screened stability requires explicit shared/locality bootstrap settings",
            }
    elif arm.model.selection_criterion == "aic" and arm.model.covariance_type == "tied":
        result["stability"] = {
            "status": "unsupported",
            "requested": 0,
            "reason": "Ordinary shared AIC stability is not validated",
        }
    result["point"] = (
        {"labels": labels, "scores": scores.tolist(), "resolved": [True] * len(labels)}
        if result["point_available"]
        else None
    )
    result["preprocessing"] = ordinary["views"]
    result["parameters"] = params
    result["run"] = run
    return result
