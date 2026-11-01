"""Bounded, frozen transductive neural comparison with a mandatory preflight gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json

CASES = ("gaussian", "four_spherical", "moons", "cline")
ARMS = ("gmm", "density", "neural_0", "neural_101", "neural_202")
FEATURES = tuple(f"trait_{i}" for i in range(1, 9))
SETTINGS = {
    "schema_version": 1,
    "data_seed": 91003,
    "noise_seed": 91009,
    "subsampling_seed": 91019,
    "neural_seeds": [0, 101, 202],
    "sample_sizes": [64, 256],
    "replicates": 10,
    "model_seed": 71323,
    "preflight_fits": 120,
    "main_fits": 1210,
    "cap_bytes": 2 * 1024**3,
    "min_free_bytes": 8 * 1024**3,
    "max_projected_seconds": 7200,
    "max_workers": 2,
    "linear_map": [[1, 0], [0, 1], [1, 1], [1, -1], [0.5, 1], [-1, 0.5], [2, 0.5], [0.5, -2]],
    "measurement_noise_sd": 0.15,
    "nonlinear_map": ["z1", "z2", "z1*z2", "z1^2", "z2^2", "sin(z1)", "cos(z2)", "sin(z1+z2)"],
}


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(package)): fingerprint_file(p).sha256
        for p in sorted(package.rglob("*.py"))
    }


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown study phase")
    namespace = 2 if phase == "preflight" else 3
    return [
        {
            "case": case,
            "n": n,
            "replicate": i,
            "namespace": namespace,
            "draws": 2 if phase == "preflight" else 20 if i == 0 else 0,
            "key": f"{case}_n{n}_{i:04d}",
            "data_seed": [91003, namespace, c, n, i],
            "noise_seed": [91009, namespace, c, n, i],
            "subsampling_seed": [91019, namespace, c, n, i],
            "replay": phase == "main"
            and i == 0
            and ((case == "gaussian" and n == 64) or (case == "moons" and n == 256)),
        }
        for c, case in enumerate(CASES)
        for n in (64, 256)
        for i in range(1 if phase == "preflight" else 10)
    ]


def generate(task):
    import numpy as np

    from super_deli.benchmarking.density_study import generate_raw

    z, truth = generate_raw(task["case"], task["n"], task["data_seed"])
    if task["case"] == "moons":
        a, b = z.T
        x = np.column_stack([a, b, a * b, a**2, b**2, np.sin(a), np.cos(b), np.sin(a + b)])
    else:
        x = z @ np.asarray(SETTINGS["linear_map"]).T
    x += np.random.default_rng(task["noise_seed"]).normal(0, 0.15, x.shape)
    return x, z, truth


def write_dataset(x, directory, *, unit_ids=None):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    ids = list(unit_ids) if unit_ids is not None else [f"invented_{i:04d}" for i in range(len(x))]
    if len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError("One unique source identity per row required")
    with (directory / "traits.tsv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["observation_id", *FEATURES])
        writer.writerows([u, *row] for u, row in zip(ids, x, strict=True))
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "invented_neural_comparison",
            "specimens": [{"specimen_id": u} for u in ids],
            "modalities": [
                {
                    "name": "traits",
                    "kind": "morphological",
                    "level": "specimen",
                    "format": "tsv",
                    "path": "traits.tsv",
                    "features": [{"name": f, "dtype": "continuous"} for f in FEATURES],
                    "observations": [{"observation_id": u, "unit_id": u} for u in ids],
                }
            ],
        }
    )
    write_json(dataset.to_dict(), directory / "dataset.json")
    return dataset


def run_arm(dataset, raw_dir, directory, arm):
    import numpy as np

    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.neural_config import NeuralConfig
    from super_deli.models.tabular_cluster_config import TabularClusterConfig
    from super_deli.workflows.tabular_gmm import run_tabular_gmm
    from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan
    from super_deli.workflows.tabular_neural import run_tabular_neural

    start = time.monotonic()
    try:
        if arm == "gmm":
            summary = run_tabular_gmm(
                dataset,
                model_config=TabularClusterConfig(
                    assume_independent_units=True,
                    feature_names=FEATURES,
                    n_bootstrap=0,
                    n_pcs=2,
                    max_k=4,
                    min_cluster_size=5,
                    n_init=10,
                    random_seed=71323,
                ),
                base_dir=raw_dir,
                output_dir=directory,
            )
        elif arm == "density":
            summary = run_tabular_hdbscan(
                dataset,
                model_config=HDBSCANConfig(
                    assume_independent_units=True, feature_names=FEATURES, n_subsamples=0
                ),
                base_dir=raw_dir,
                output_dir=directory,
            )
        elif arm in ARMS[2:]:
            summary = run_tabular_neural(
                dataset,
                model_config=NeuralConfig(FEATURES, True, random_seed=int(arm.split("_")[1])),
                base_dir=raw_dir,
                output_dir=directory,
            )
        else:
            raise ValueError("Unknown study arm")
        with np.load(
            directory / ("mixture.npz" if arm == "gmm" else "point.npz"), allow_pickle=False
        ) as f:
            labels = f["labels"].tolist()
        result = {"status": "complete", "selected_k": summary["selected_k"], "labels": labels}
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    result.update(arm=arm, seconds=time.monotonic() - start)
    write_json(result, directory.parent / f"{arm}-attempt.json")
    return result


def run_observation(job):
    import numpy as np

    from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples

    task, root = job
    out = Path(root) / task["key"]
    out.mkdir(exist_ok=False)
    write_json(task, out / "task.json")
    x, z, truth = generate(task)
    # Reference arrays never enter a fitting configuration or analytical table.
    np.savez_compressed(
        out / "generating-reference.npz",
        x=x,
        z=z,
        truth=np.asarray([] if truth is None else truth, dtype=int),
    )
    dataset = write_dataset(x, out / "raw")
    ids = [s.specimen_id for s in dataset.specimens]
    records = []
    (out / "point").mkdir()
    for arm in ARMS:
        result = run_arm(dataset, out / "raw", out / "point" / arm, arm)
        records.append({**result, "kind": "point", "training_indices": list(range(len(x)))})
    (out / "draws").mkdir()
    for i, indices in enumerate(
        subsample_indices(len(x), task["draws"], 0.8, task["subsampling_seed"])
        if task["draws"]
        else ()
    ):
        d = out / "draws" / f"draw_{i:03d}"
        d.mkdir()
        sub = write_dataset(x[indices], d / "raw", unit_ids=[ids[j] for j in indices])
        for arm in ARMS:
            result = run_arm(sub, d / "raw", d / arm, arm)
            records.append(
                {**result, "kind": "draw", "draw": i, "training_indices": indices.tolist()}
            )
    (out / "support").mkdir()
    for arm in ARMS:
        support = summarize_subsamples(
            len(x), [r for r in records if r["arm"] == arm and r["kind"] == "draw"]
        )
        write_json(support.summary, out / "support" / f"{arm}.json")
        np.savez_compressed(out / "support" / f"{arm}.npz", **support.arrays)
    if task["replay"]:
        (out / "replay").mkdir()
        for arm in ARMS:
            result = run_arm(dataset, out / "raw", out / "replay" / arm, arm)
            records.append({**result, "kind": "replay", "training_indices": list(range(len(x)))})
    write_json(records, out / "fits.json")
    write_json(
        {
            str(p.relative_to(out)): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
        out / "checksums.json",
    )
    return {
        "key": task["key"],
        "attempts": len(records),
        "failures": sum(r["status"] != "complete" for r in records),
        "fit_seconds": sum(r["seconds"] for r in records),
    }


def retained_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def check_free(root):
    if shutil.disk_usage(root).free < SETTINGS["min_free_bytes"]:
        raise ValueError("At least 8 GiB free is required")


def assert_frozen(root):
    freeze = read_json(root / "frozen.json")
    if freeze["settings"] != SETTINGS or freeze["source_sha256"] != source_hashes():
        raise ValueError("Scientific source/settings differ from the frozen execution")
    if freeze["software"] != {n: version(n) for n in freeze["software"]}:
        raise ValueError("Runtime differs from the frozen execution")
    if freeze["python"] != sys.version or freeze["platform"] != platform.platform():
        raise ValueError("Python/platform differs from the frozen execution")
    return freeze


def freeze(root):
    import tarfile

    root.mkdir(exist_ok=True)
    write_json(
        {
            "settings": SETTINGS,
            "python": sys.version,
            "platform": platform.platform(),
            "source_sha256": source_hashes(),
            "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
            "software": {
                n: version(n)
                for n in ("super-deli", "numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")
            },
        },
        root / "frozen.json",
    )
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as archive:
        for p in sorted(package.rglob("*.py")):
            archive.add(p, arcname=str(p.relative_to(package)))
    check_free(root)


def run_phase(root, phase, workers):
    assert_frozen(root)
    check_free(root)
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("At most two single-thread workers permitted")
    if phase == "main":
        gate = read_json(root / "storage-runtime-gate.json")
        audit = read_json(root / "preflight-audit.json")
        if (
            gate["status"] != "passed"
            or not audit["passed"]
            or (gate["audit_sha256"] != fingerprint_file(root / "preflight-audit.json").sha256)
        ):
            raise ValueError("A matching passed preflight audit and resource gate are required")
    directory = root / phase
    directory.mkdir(exist_ok=False)
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_observation, [(t, str(directory)) for t in tasks(phase)]):
            results.append(result)
            print(json.dumps(result), flush=True)
            check_free(root)
            if retained_bytes(root) > SETTINGS["cap_bytes"]:
                raise ValueError("Retained evidence exceeded the approved cap")
    expected = SETTINGS[f"{phase}_fits"]
    if sum(r["attempts"] for r in results) != expected:
        raise ValueError("Workflow accounting differs from the frozen plan")
    write_json(
        {
            "phase": phase,
            "workers": workers,
            "observations": results,
            "expected_fits": expected,
            "elapsed_seconds": time.monotonic() - start,
        },
        root / f"{phase}-execution.json",
    )


def gate(root, staging_bytes):
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Staging byte count must be a nonnegative integer")
    assert_frozen(root)
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    factor = SETTINGS["main_fits"] / SETTINGS["preflight_fits"]
    projected = int(
        retained_bytes(root / "preflight") * factor * 1.25
        + retained_bytes(root)
        + staging_bytes
        + 128 * 1024**2
    )
    seconds = sum(r["fit_seconds"] for r in execution["observations"]) * factor * 1.25 / 2
    free = shutil.disk_usage(root).free
    passed = (
        audit["passed"]
        and projected <= SETTINGS["cap_bytes"]
        and seconds <= SETTINGS["max_projected_seconds"]
        and free >= SETTINGS["min_free_bytes"]
    )
    result = {
        "status": "passed" if passed else "failed",
        "projected_bytes": projected,
        "projected_main_seconds_two_workers": seconds,
        "staging_bytes": staging_bytes,
        "free_bytes": free,
        "settings": SETTINGS,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("freeze", "preflight", "audit-preflight", "gate", "main", "audit-main", "report"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--staging-bytes", type=int, default=0)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.phase == "freeze":
        freeze(root)
    elif args.phase in {"main", "preflight"}:
        run_phase(root, args.phase, args.workers)
    elif args.phase == "gate":
        if gate(root, args.staging_bytes)["status"] != "passed":
            raise SystemExit(1)
    else:
        from super_deli.benchmarking.neural_analysis import audit_phase, report_study

        if args.phase.startswith("audit-"):
            audit_phase(root, args.phase.removeprefix("audit-"))
        else:
            report_study(root)


if __name__ == "__main__":
    main()
