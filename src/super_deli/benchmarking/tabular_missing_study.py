"""Frozen missing-data sensitivity study; raw truth is never passed to analysis."""

import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.tabular_analysis_config import TabularAnalysisArm, TabularAnalysisConfig
from super_deli.models.tabular_cluster_config import TabularClusterConfig
from super_deli.models.tabular_missing_config import TabularMissingAnalysisConfig

CASES = ("gaussian", "four_spherical", "moons")
MASKS = ("mcar10", "mcar30", "mar", "mnar", "pattern")
NAMES = tuple(f"trait_{i}" for i in range(4))
SETTINGS = dict(
    schema_version=1,
    cases=list(CASES),
    sample_sizes=[64, 256],
    datasets_per_cell=10,
    masks=list(MASKS),
    data_seed=80303,
    preflight_seed=80309,
    main_seed=80317,
    model_seed=80323,
    mask_seed=80329,
    draws=10,
    stability_datasets_per_cell=2,
    main_pipeline_fits=1836,
    preflight_pipeline_fits=408,
    artifact_cap_bytes=4 * 1024**3,
    min_free_bytes=8 * 1024**3,
    max_workers=4,
    storage_allowance=1.25,
    storage_reserve_bytes=128 * 1024**2,
)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def arms(draws=0):
    return (
        TabularAnalysisArm(
            "gmm",
            TabularClusterConfig(
                True, feature_names=NAMES, n_bootstrap=draws, random_seed=SETTINGS["model_seed"]
            ),
        ),
        TabularAnalysisArm(
            "density",
            HDBSCANConfig(
                True, feature_names=NAMES, n_subsamples=draws, random_seed=SETTINGS["model_seed"]
            ),
        ),
    )


def tasks(phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown validation phase")
    return [
        dict(
            key=f"{case}_n{n}_{i:02}",
            case=case,
            n=n,
            replicate=i,
            phase=phase,
            data_seed=[80303, SETTINGS[phase + "_seed"], c, n, i],
            mask_seed=[80329, SETTINGS[phase + "_seed"], c, n, i],
            stability=phase == "preflight" or i < 2,
            replay=phase == "preflight" or i == 0,
        )
        for c, case in enumerate(CASES)
        for n in (64, 256)
        for i in range(1 if phase == "preflight" else 10)
    ]


def generate(task):
    import numpy as np

    from super_deli.benchmarking.density_study import generate_raw

    z, truth = generate_raw(task["case"], task["n"], task["data_seed"])
    z = z - np.array([0.5, 0.25] if task["case"] == "moons" else [0.0, 0.0])
    x = np.column_stack(
        [z[:, 0], z[:, 1], (z[:, 0] + z[:, 1]) / np.sqrt(2), (z[:, 0] - z[:, 1]) / np.sqrt(2)]
    )
    x += np.random.default_rng(task["data_seed"] + [41]).normal(0, 0.05, x.shape)
    uniforms = np.random.default_rng(task["mask_seed"] + [0]).random(x.shape)
    mar = np.random.default_rng(task["mask_seed"] + [1]).random(x.shape)
    mar_mask = mar < np.where(x[:, 0:1] <= 0, 0.15, 0.65)
    mar_mask[:, 0] = False
    mnar = np.random.default_rng(task["mask_seed"] + [2]).random(x.shape) < np.where(
        x <= 0, 0.10, 0.50
    )
    rng = np.random.default_rng(task["mask_seed"] + [3])
    nuisance = rng.integers(2, size=len(x))
    selected = (np.arange(4)[None, :] // 2) == nuisance[:, None]
    pattern = selected & (rng.random(x.shape) < 0.60)
    return (
        x,
        truth,
        dict(
            mcar10=uniforms < 0.10, mcar30=uniforms < 0.30, mar=mar_mask, mnar=mnar, pattern=pattern
        ),
        nuisance,
    )


def write_data(x, mask, directory):
    root = Path(directory)
    root.mkdir()
    ids = [f"unit_{i:04}" for i in range(len(x))]
    with (root / "traits.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", *NAMES])
        writer.writerows(
            [u, *("" if absent else value for value, absent in zip(row, missing, strict=True))]
            for u, row, missing in zip(ids, x, mask, strict=True)
        )
    d = config_from_dict(
        dict(
            schema_version=1,
            dataset_id="invented_missingness",
            specimens=[dict(specimen_id=u) for u in ids],
            modalities=[
                dict(
                    name="traits",
                    kind="morphological",
                    level="specimen",
                    format="tsv",
                    path="traits.tsv",
                    features=[dict(name=f, dtype="continuous") for f in NAMES],
                    observations=[dict(observation_id=u, unit_id=u) for u in ids],
                )
            ],
        )
    )
    write_json(d.to_dict(), root / "dataset.json")
    return d


def _replays(directory):
    import numpy as np

    from super_deli.models.tabular_missing import TrainingContext
    from super_deli.workflows.tabular_analysis import _native

    records = []
    specs = [("oracle", None, directory / "raw/oracle", directory / "oracle/arms")]
    specs.extend(
        (
            p,
            p,
            directory / "analyses/mcar30/policies" / p / "raw",
            directory / "analyses/mcar30/policies" / p / "analysis/arms",
        )
        for p in ("complete_case", "mean")
    )
    (directory / "replays").mkdir()
    for tag, policy, raw, original in specs:
        for arm in arms():
            target = directory / "replays" / (tag + "_" + arm.name)
            record = dict(policy=tag, arm=arm.name, status="prerequisite_failed")
            if (original / arm.name / "summary.json").exists():
                try:
                    d = config_from_dict(read_json(raw / "dataset.json"))
                    _native(
                        arm, d, raw, target, "traits", TrainingContext(policy) if policy else None
                    )
                    assert (target / "assignments.csv").read_bytes() == (
                        original / arm.name / "assignments.csv"
                    ).read_bytes()
                    files = (
                        ("mixture.npz", "preprocessing.npz")
                        if arm.family == "gmm"
                        else ("point.npz",)
                    )
                    for name in files:
                        with (
                            np.load(target / name, allow_pickle=False) as a,
                            np.load(original / arm.name / name, allow_pickle=False) as b,
                        ):
                            assert a.files == b.files
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key])
                    record["status"] = "exact"
                except Exception as exc:
                    record.update(status="failed", error=str(exc))
            records.append(record)
    return records


def observation(task):
    import numpy as np

    from super_deli.workflows.tabular_analysis import run_tabular_analysis

    out = Path(task["output"])
    out.mkdir()
    write_json(task, out / "task.json")
    result = dict(key=task["key"], status="failed", error=None)
    try:
        x, truth, masks, nuisance = generate(task)
        with (out / "simulation.npz").open("xb") as handle:
            np.savez_compressed(handle, complete=x, truth=truth, nuisance=nuisance, **masks)
        (out / "raw").mkdir()
        (out / "analyses").mkdir()
        d = write_data(x, np.zeros(x.shape, dtype=bool), out / "raw/oracle")
        oracle = run_tabular_analysis(
            d,
            analysis_config=TabularAnalysisConfig(arms()),
            base_dir=out / "raw/oracle",
            output_dir=out / "oracle",
        )
        outcomes = {}
        for name, mask in masks.items():
            d = write_data(x, mask, out / "raw" / name)
            config = TabularMissingAnalysisConfig(
                arms(10 if task["stability"] and name == "mcar30" else 0), ("complete_case", "mean")
            )
            outcomes[name] = run_tabular_analysis(
                d,
                analysis_config=config,
                base_dir=out / "raw" / name,
                output_dir=out / "analyses" / name,
            )["status"]
        replays = _replays(out) if task["replay"] else []
        result.update(
            status="retained",
            oracle_status=oracle["status"],
            masked_status=outcomes,
            replays=replays,
            planned_pipeline_fits=22
            + (40 if task["stability"] else 0)
            + (6 if task["replay"] else 0),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(result, out / "validation.json")
    return result


def run_validation(phase, *, output_dir, workers=1, gate_file=None):
    import shutil

    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    out = Path(output_dir)
    if phase == "main":
        gate = read_json(gate_file)
        if (
            gate["status"] != "passed"
            or gate["settings"] != SETTINGS
            or gate["source_identity"] != source_identity()
        ):
            raise ValueError("Matching audited storage/source gate required")
        if (
            hashlib.sha256(Path(gate["preflight_audit"]).read_bytes()).hexdigest()
            != gate["preflight_audit_sha256"]
        ):
            raise ValueError("Preflight audit changed")
    if shutil.disk_usage(out.parent).free < 8 * 1024**3:
        raise ValueError("At least 8 GiB free required")
    out.mkdir(exist_ok=False)
    write_json(
        dict(settings=SETTINGS, source_identity=source_identity(), phase=phase), out / "frozen.json"
    )
    plan = [dict(t, output=str(out / t["key"])) for t in tasks(phase)]
    if workers == 1:
        records = [observation(t) for t in plan]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(observation, plan))
    result = dict(
        status="retained",
        phase=phase,
        observations=len(records),
        records=records,
        planned_pipeline_fits=SETTINGS[phase + "_pipeline_fits"],
        settings=SETTINGS,
    )
    write_json(result, out / "summary.json")
    return result


def storage_gate(preflight_dir, audit_file, *, checkpoint_dir, scratch_dir, output_file):
    import json
    import shutil
    import zipfile

    root = Path(preflight_dir)
    audit = read_json(audit_file)
    if audit["status"] != "passed" or audit["planned_pipeline_fits"] != 408:
        raise ValueError("Independently audited full-settings preflight required")
    if read_json(root / "frozen.json")["source_identity"] != source_identity():
        raise ValueError("Source changed since frozen preflight")
    projection = 0
    classes = []
    for record in audit["records"]:
        d = root / record["key"]

        def size(folder):
            return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())

        total = size(d)
        replay = size(d / "replays")
        resampling = 0
        for p in (d / "analyses/mcar30/policies").rglob("*"):
            if not p.is_file():
                continue
            if p.name in {"stability.npz", "bootstrap_replicates.json", "subsamples.json"}:
                resampling += p.stat().st_size
            elif p.name == "training-context.npz":
                with zipfile.ZipFile(p) as z:
                    point = (
                        sum(
                            m.compress_size + len(m.filename) + 150
                            for m in z.infolist()
                            if m.filename.startswith("fit_0_")
                        )
                        + 1024
                    )
                resampling += max(0, p.stat().st_size - point)
            elif p.name == "training-context.json":
                v = read_json(p)
                point = (
                    len(
                        json.dumps(
                            dict(declaration=v["declaration"], fits=v["fits"][:1]), indent=2
                        ).encode()
                    )
                    + 1024
                )
                resampling += max(0, p.stat().st_size - point)
        base = max(0, total - replay - resampling)
        projected = 10 * base + 2 * resampling + replay
        projection += projected
        classes.append(
            dict(
                key=record["key"],
                point_and_raw_bytes=base,
                resampling_bytes=resampling,
                replay_bytes=replay,
                projected_main_bytes=projected,
            )
        )
    retained = sum(
        p.stat().st_size
        for folder in (Path(checkpoint_dir), Path(scratch_dir))
        for p in folder.rglob("*")
        if p.is_file()
    )
    estimated = retained + int(projection * 1.25) + 128 * 1024**2
    free = shutil.disk_usage(checkpoint_dir).free
    result = dict(
        status="passed" if estimated <= 4 * 1024**3 and free >= 8 * 1024**3 else "failed",
        settings=SETTINGS,
        source_identity=source_identity(),
        preflight_audit=str(Path(audit_file).absolute()),
        preflight_audit_sha256=hashlib.sha256(Path(audit_file).read_bytes()).hexdigest(),
        retained_before_main_bytes=retained,
        measured_class_projection_bytes=projection,
        estimated_retained_bytes=estimated,
        artifact_cap_bytes=4 * 1024**3,
        free_bytes=free,
        classes=classes,
    )
    write_json(result, output_file)
    return result
