"""Frozen grouped-orchestration verification, separate from significance studies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.config import derived_seed
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_analysis_config import GroupedAnalysisArm, GroupedAnalysisConfig
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
from super_deli.models.predictive_bootstrap_config import PredictiveBootstrapConfig
from super_deli.models.predictive_config import PredictiveScreenConfig

CASES = (
    "independent_gaussian",
    "correlated_gaussian",
    "complementary",
    "heavy_tailed",
    "unequal_variance",
    "cline",
)
SETTINGS = {
    "schema_version": 1,
    "cases": list(CASES),
    "sample_sizes": [64, 256],
    "datasets_per_cell": 3,
    "draws": 25,
    "data_seed": 79531,
    "main_seed": 79571,
    "preflight_seed": 79579,
    "model_seed": 71323,
    "split_seed": 71324,
    "bootstrap_seed": 79561,
    "artifact_cap_bytes": 2147483648,
    "min_free_bytes": 8589934592,
    "storage_allowance": 1.25,
    "reserve_bytes": 134217728,
    "main_searches": 2568,
    "preflight_searches": 364,
}


def model_arms(draws=0, *, target_level="locality"):
    views = tuple(
        ViewConfig(
            k,
            k,
            feature_names=tuple(f"feature_{i}" for i in range(4)),
            aggregation="mean" if k == "morphological" else "none",
            pca_fit="target_means" if k == "morphological" else "source_rows",
        )
        for k in ("morphological", "ecological")
    )
    base = IntegrativeConfig(
        True,
        n_pcs=4,
        n_bootstrap=0,
        views=views,
        target_level=target_level,
        random_seed=SETTINGS["model_seed"],
    )
    arms = []
    for criterion in ("bic", "aic", "predictive"):
        for covariance, name in (("full", "full"), ("tied", "shared")):
            predictive = criterion == "predictive"
            model = replace(
                base,
                covariance_type=covariance,
                selection_criterion="aic" if predictive else criterion,
            )
            arms.append(
                GroupedAnalysisArm(
                    f"{name}_{criterion}",
                    "predictive" if predictive else "ordinary",
                    model,
                    screen=PredictiveScreenConfig(random_seed=SETTINGS["split_seed"])
                    if predictive
                    else None,
                    bootstrap=PredictiveBootstrapConfig(draws, SETTINGS["bootstrap_seed"])
                    if predictive and covariance == "tied" and draws
                    else None,
                )
            )
    return GroupedAnalysisConfig(tuple(arms))


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown study phase")
    cases = ("unequal_variance",) if phase == "preflight" else CASES
    return [
        {
            "phase": phase,
            "case": case,
            "n": n,
            "dataset": i,
            "key": f"{case}_n{n}_{i:02d}",
            "data_seed": derived_seed(SETTINGS["data_seed"], phase, CASES.index(case), f"{n}_{i}"),
            "draws": SETTINGS["draws"] if phase == "preflight" or i == 0 else 0,
            "replay": phase == "preflight" or i == 0,
        }
        for case in cases
        for n in SETTINGS["sample_sizes"]
        for i in range(1 if phase == "preflight" else SETTINGS["datasets_per_cell"])
    ]


def generate(task):
    if task["case"] not in CASES:
        raise ValueError("Unknown grouped law")
    if task["case"] in {"unequal_variance", "cline"}:
        from super_deli.benchmarking.grouped_pca_study import simulate

        return simulate(task["case"], task["n"], task["data_seed"], target_level="locality")
    from super_deli.benchmarking.grouped_study import simulate_grouped

    return simulate_grouped(task["case"], task["n"], task["data_seed"])


def native_point_paths(arm, directory):
    directory = Path(directory)
    point = directory / "point" if arm.bootstrap is not None else directory
    run = point / "full/run" if arm.kind == "predictive" else directory
    return point, run


def compare_replay(arm, original, replay):
    import numpy as np

    a, arun = native_point_paths(arm, original)
    b, brun = native_point_paths(replace(arm, bootstrap=None), replay)
    names = [
        "mixture.npz",
        "fusion.npz",
        *(f"view_{i:03d}/parameters/{f}.npz" for i in (0, 1) for f in ("pca", "profiles")),
    ]
    if (arun / "gmm_candidates.npz").exists():
        names.append("gmm_candidates.npz")
    for name in names:
        with np.load(arun / name) as x, np.load(brun / name) as y:
            assert x.files == y.files
            for key in x.files:
                np.testing.assert_array_equal(x[key], y[key])
    assert (arun / "assignments.csv").read_bytes() == (brun / "assignments.csv").read_bytes()
    if arm.kind == "predictive":
        assert read_json(a / "fold_assignments.json") == read_json(b / "fold_assignments.json")
        for folder in sorted(a.glob("fold_*")):
            if not folder.is_dir():
                continue
            for source in sorted(folder.rglob("*.npz")):
                with np.load(source) as x, np.load(b / source.relative_to(a)) as y:
                    assert x.files == y.files
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key])
        assert (
            read_json(a / "summary.json")["selection"] == read_json(b / "summary.json")["selection"]
        )
        if (a / "selected_assignments.csv").exists():
            assert (a / "selected_assignments.csv").read_bytes() == (
                b / "selected_assignments.csv"
            ).read_bytes()


def _observation(task):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views
    from super_deli.workflows.grouped_analysis import run_grouped_tabular_analysis
    from super_deli.workflows.grouped_analysis_adapters import run_grouped_arm

    d = Path(task["output"])
    d.mkdir()
    record = {"task": task, "status": "failed", "error": None, "replays": []}
    try:
        values, layout, truth = generate(task)
        dataset = write_raw_views(values, layout, d / "raw")
        write_json(layout, d / "layout.json")
        if truth is not None:
            with (d / "truth.npz").open("xb") as handle:
                np.savez_compressed(handle, labels=truth)
        config = model_arms(task["draws"])
        summary = run_grouped_tabular_analysis(
            dataset, analysis_config=config, base_dir=d / "raw", output_dir=d / "analysis"
        )
        record["status"] = summary["status"]
        if task["replay"]:
            (d / "replays").mkdir()
            for arm in config.arms:
                run_grouped_arm(
                    replace(arm, bootstrap=None), dataset, d / "raw", d / "replays" / arm.name
                )
                compare_replay(arm, d / "analysis/arms" / arm.name, d / "replays" / arm.name)
                record["replays"].append(arm.name)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    write_json(record, d / "validation.json")
    return record


def run_study(phase, *, output_dir, workers=1, gate_file=None):
    import shutil

    from super_deli.core.provenance import fingerprint_file

    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("One to four single-thread workers")
    out = Path(output_dir)
    if phase == "main":
        gate = read_json(gate_file)
        if gate["status"] != "passed" or gate["settings"] != SETTINGS:
            raise ValueError("Matching audited storage gate required")
        if fingerprint_file(gate["audit_path"]).sha256 != gate["audit_sha256"]:
            raise ValueError("Audit changed after gate")
    if shutil.disk_usage(out.parent).free < SETTINGS["min_free_bytes"]:
        raise ValueError("Insufficient free storage")
    out.mkdir()
    jobs = [{**t, "output": str(out / t["key"])} for t in tasks(phase)]
    write_json(
        {"settings": SETTINGS, "phase": phase, "tasks": jobs, "arms": model_arms(25).to_dict()},
        out / "plan.json",
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_observation, jobs))
    result = {
        "phase": phase,
        "observations": len(results),
        "complete": sum(r["status"] == "complete" for r in results),
        "results": results,
    }
    write_json(result, out / "study-summary.json")
    return result


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def storage_gate(preflight, audit_path, artifact_root, output_file, *, artifact_cap_bytes=None):
    import shutil

    from super_deli.core.provenance import fingerprint_file

    cap = SETTINGS["artifact_cap_bytes"] if artifact_cap_bytes is None else artifact_cap_bytes
    if type(cap) is not int or cap < 1:
        raise ValueError("artifact_cap_bytes must be a positive integer")
    root, probe = Path(artifact_root), Path(preflight)
    audit = read_json(audit_path)
    if audit["status"] != "passed" or audit["searches"] != 364:
        raise ValueError("Audited complete preflight required")
    sizes = []
    for t in tasks("preflight"):
        base = probe / t["key"]
        analysis = base / "analysis"
        child = analysis / "arms/shared_predictive"
        point = tree_bytes(analysis) - tree_bytes(child) + tree_bytes(child / "point")
        stability = (tree_bytes(child) - tree_bytes(child / "point")) / 25
        replays = tree_bytes(base / "replays")
        overhead = tree_bytes(base) - tree_bytes(analysis) - replays
        sizes.append(
            {
                "n": t["n"],
                "point_bytes": point,
                "draw_bytes": stability,
                "replay_bytes": replays,
                "overhead_bytes": overhead,
            }
        )
    projected = 0
    for t in tasks("main"):
        size = next(s for s in sizes if s["n"] == t["n"])
        projected += (
            size["point_bytes"]
            + t["draws"] * size["draw_bytes"]
            + (size["replay_bytes"] if t["replay"] else 0)
            + size["overhead_bytes"]
        )
    retained = tree_bytes(root)
    total = int((projected + retained) * SETTINGS["storage_allowance"]) + SETTINGS["reserve_bytes"]
    free = shutil.disk_usage(root).free
    result = {
        "status": "passed"
        if total <= cap and free >= max(total, SETTINGS["min_free_bytes"])
        else "failed",
        "settings": SETTINGS,
        "artifact_cap_bytes": cap,
        "budget_amendment": cap != SETTINGS["artifact_cap_bytes"],
        "measured_classes": sizes,
        "projected_total_bytes": total,
        "retained_bytes": retained,
        "free_bytes": free,
        "audit_path": str(Path(audit_path).absolute()),
        "audit_sha256": fingerprint_file(audit_path).sha256,
    }
    write_json(result, output_file)
    return result
