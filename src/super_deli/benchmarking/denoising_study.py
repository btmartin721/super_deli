"""Frozen paired feature-noise study with mandatory audited resource gates."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_study import (
    FEATURES,
    check_free,
    retained_bytes,
    source_hashes,
    write_dataset,
)
from super_deli.core.provenance import fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json

CASES = ("gaussian", "four_spherical", "moons")
ARMS = (
    "density",
    "neural_0",
    "neural_101",
    "neural_202",
    "denoising_0",
    "denoising_101",
    "denoising_202",
)
CONDITIONS = ("clean", "noisy")
SETTINGS = {
    "schema_version": 1,
    "source_seed": 93001,
    "measurement_seed": 93019,
    "subsampling_seed": 93047,
    "corruption_seed": 93053,
    "replicates": 10,
    "neural_seeds": [0, 101, 202],
    "sample_sizes": [64, 256],
    "measurement_sd_raw_units": 0.15,
    "training_sd_standardized_units": 0.15,
    "main_fits": 2618,
    "preflight_fits": 266,
    "main_neural_fits": 2190,
    "preflight_neural_fits": 222,
    "cap_bytes": 2 * 1024**3,
    "min_free_bytes": 8 * 1024**3,
    "max_projected_seconds": 7200,
    "max_workers": 2,
    "linear_map": [[1, 0], [0, 1], [1, 1], [1, -1], [0.5, 1], [-1, 0.5], [2, 0.5], [0.5, -2]],
    "nonlinear_map": ["z1", "z2", "z1*z2", "z1^2", "z2^2", "sin(z1)", "cos(z2)", "sin(z1+z2)"],
    "primary_rate_intervals": 36,
    "primary_paired_differences": 12,
    "simultaneous_binomial_intervals": 60,
}


def tasks(phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown study phase")
    ns = 11 if phase == "preflight" else 13
    return [
        {
            "case": case,
            "n": n,
            "replicate": i,
            "namespace": ns,
            "key": f"{case}_n{n}_{i:04d}",
            "data_seed": [93001, ns, c, n, i],
            "noise_seed": [93019, ns, c, n, i],
            "subsampling_seed": [93047, ns, c, n, i],
            "corruption_seed": [93053, ns, c, n, i],
            "draws": 2 if phase == "preflight" else 20 if i == 0 else 0,
            "replay": phase == "main"
            and i == 0
            and ((case == "gaussian" and n == 64) or (case == "moons" and n == 256)),
        }
        for c, case in enumerate(CASES)
        for n in (64, 256)
        for i in range(1 if phase == "preflight" else 10)
    ]


def seed32(value):
    import numpy as np

    return int(np.random.SeedSequence(value).generate_state(1)[0])


def generate(task):
    import numpy as np

    from super_deli.benchmarking.density_study import generate_raw

    z, truth = generate_raw(task["case"], task["n"], task["data_seed"])
    if task["case"] == "moons":
        a, b = z.T
        clean = np.column_stack([a, b, a * b, a**2, b**2, np.sin(a), np.cos(b), np.sin(a + b)])
    else:
        clean = z @ np.array(SETTINGS["linear_map"]).T
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, clean.shape)
    return clean, noise, z, truth


def run_arm(dataset, raw_dir, directory, arm, corruption_seed):
    import numpy as np

    from super_deli.models.denoising_config import DenoisingConfig
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.neural_config import NeuralConfig
    from super_deli.workflows.tabular_denoising import run_tabular_denoising
    from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan
    from super_deli.workflows.tabular_neural import run_tabular_neural

    start = time.monotonic()
    try:
        kwargs = {"base_dir": raw_dir, "output_dir": directory}
        if arm == "density":
            config = HDBSCANConfig(True, feature_names=FEATURES)
            summary = run_tabular_hdbscan(dataset, model_config=config, **kwargs)
        elif arm.startswith("neural_") and arm in ARMS:
            config = NeuralConfig(FEATURES, True, random_seed=int(arm.split("_")[1]))
            summary = run_tabular_neural(dataset, model_config=config, **kwargs)
        elif arm.startswith("denoising_") and arm in ARMS:
            config = DenoisingConfig(
                FEATURES, True, random_seed=int(arm.split("_")[1]), corruption_seed=corruption_seed
            )
            summary = run_tabular_denoising(dataset, model_config=config, **kwargs)
        else:
            raise ValueError("Unknown study arm")
        with np.load(directory / "point.npz", allow_pickle=False) as f:
            labels = f["labels"].tolist()
        result = {"status": "complete", "selected_k": summary["selected_k"], "labels": labels}
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    result.update(arm=arm, seconds=time.monotonic() - start)
    write_json(result, directory.parent / f"{arm}-attempt.json")
    return result


def run_source(z, directory):
    import numpy as np
    from threadpoolctl import threadpool_limits

    from super_deli.models.density_scores import fit_density_scores
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.mixture import FIT_LOCK

    directory.mkdir()
    start = time.monotonic()
    try:
        scores = (z - z.mean(0)) / z.std(0)
        with FIT_LOCK, threadpool_limits(limits=1):
            labels, strength = fit_density_scores(scores, HDBSCANConfig(True))
        np.savez_compressed(
            directory / "point.npz", scores=scores, labels=labels, strength=strength
        )
        result = {
            "status": "complete",
            "labels": labels.tolist(),
            "selected_k": len(set(labels) - {-1}),
        }
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    result.update(arm="source_reference", seconds=time.monotonic() - start)
    write_json(result, directory / "attempt.json")
    return result


def run_observation(job):
    import numpy as np

    from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples

    task, root = job
    out = Path(root) / task["key"]
    out.mkdir()
    write_json(task, out / "task.json")
    clean, noise, z, truth = generate(task)
    np.savez_compressed(
        out / "generating-reference.npz", clean=clean, noise=noise, z=z, truth=truth
    )
    indices = (
        subsample_indices(len(z), task["draws"], 0.8, task["subsampling_seed"])
        if task["draws"]
        else ()
    )
    records = []
    for condition in CONDITIONS:
        x = clean if condition == "clean" else clean + noise
        c = out / condition
        c.mkdir()
        dataset = write_dataset(x, c / "raw")
        ids = [s.specimen_id for s in dataset.specimens]
        rows = [
            ("point", None, np.arange(len(x))),
            *(("draw", i, ix) for i, ix in enumerate(indices)),
        ]
        if task["replay"]:
            rows.append(("replay", None, np.arange(len(x))))
        for kind, draw, ix in rows:
            d = c / (f"draw_{draw:03d}" if kind == "draw" else kind)
            d.mkdir()
            raw_dir = d / "raw" if kind == "draw" else c / "raw"
            sub = (
                write_dataset(x[ix], raw_dir, unit_ids=[ids[j] for j in ix])
                if kind == "draw"
                else dataset
            )
            for arm in ARMS:
                corruption = seed32(
                    [*task["corruption_seed"], int(arm.split("_")[-1]) if arm != "density" else 0]
                )
                r = run_arm(sub, raw_dir, d / arm, arm, corruption)
                records.append(
                    {
                        **r,
                        "condition": condition,
                        "kind": kind,
                        "draw": draw,
                        "training_indices": ix.tolist(),
                        "corruption_seed": corruption,
                        "path": str((d / arm).relative_to(out)),
                    }
                )
        for arm in ARMS:
            support = summarize_subsamples(
                len(x),
                [
                    r
                    for r in records
                    if r["condition"] == condition and r["arm"] == arm and r["kind"] == "draw"
                ],
            )
            write_json(support.summary, c / f"{arm}-support.json")
            np.savez_compressed(c / f"{arm}-support.npz", **support.arrays)
    for kind in ("point", "replay") if task["replay"] else ("point",):
        d = out / f"source-{kind}"
        r = run_source(z, d)
        records.append(
            {
                **r,
                "kind": kind,
                "condition": "source",
                "draw": None,
                "training_indices": list(range(len(z))),
                "path": d.name,
            }
        )
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


def legacy_replays(root, phase):
    import numpy as np

    archive = Path(read_json(root / "frozen.json")["archive_dir"])
    out = root / phase / "legacy-replays"
    out.mkdir()
    records = []
    for key in ("gaussian_n64_0000", "moons_n256_0000"):
        source = archive / "main" / key
        dataset = config_from_dict(read_json(source / "raw" / "dataset.json"))
        parent = out / key
        parent.mkdir()
        for arm in ARMS[:4]:
            r = run_arm(dataset, source / "raw", parent / arm, arm, 0)
            r.update(key=key, path=str((parent / arm).relative_to(root / phase)))
            if r["status"] == "complete":
                with (
                    np.load(parent / arm / "point.npz", allow_pickle=False) as new,
                    np.load(source / "point" / arm / "point.npz", allow_pickle=False) as old,
                ):
                    if set(new.files) != set(old.files):
                        raise ValueError("Legacy numeric array inventory changed")
                    for name in old.files:
                        np.testing.assert_array_equal(new[name], old[name])
                r["exact_numeric_replay"] = True
            records.append(r)
    write_json(records, out / "fits.json")
    return {
        "key": "legacy-replays",
        "attempts": 8,
        "failures": sum(r["status"] != "complete" for r in records),
        "fit_seconds": sum(r["seconds"] for r in records),
    }


def freeze(root, archive_dir):
    root.mkdir(exist_ok=True)
    archive_dir = Path(archive_dir).resolve()
    old = read_json(archive_dir / "frozen.json")
    for name, expected in old["software"].items():
        if name != "super-deli" and version(name) != expected:
            raise ValueError("Legacy replay dependency runtime differs")
    if old["python"] != sys.version or old["platform"] != platform.platform():
        raise ValueError("Legacy replay Python/platform differs")
    lock = {
        "settings": SETTINGS,
        "source_sha256": source_hashes(),
        "python": sys.version,
        "platform": platform.platform(),
        "plan_sha256": fingerprint_file(root / "approved-plan.md").sha256,
        "software": {n: version(n) for n in old["software"]},
        "archive_dir": str(archive_dir),
        "archive_locks": {
            n: fingerprint_file(archive_dir / n).sha256
            for n in ("frozen.json", "frozen-sources.tar.gz", "main-audit.json", "completion.json")
        },
    }
    write_json(lock, root / "frozen.json")
    package = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "frozen-sources.tar.gz", "x:gz") as bundle:
        for p in sorted(package.rglob("*.py")):
            bundle.add(p, arcname=str(p.relative_to(package)))
    check_free(root)


def assert_frozen(root):
    lock = read_json(root / "frozen.json")
    if lock["settings"] != SETTINGS or lock["source_sha256"] != source_hashes():
        raise ValueError("Denoising source/settings differ from the freeze")
    if (
        lock["software"] != {n: version(n) for n in lock["software"]}
        or lock["python"] != sys.version
        or lock["platform"] != platform.platform()
    ):
        raise ValueError("Frozen runtime differs")
    if fingerprint_file(root / "approved-plan.md").sha256 != lock["plan_sha256"]:
        raise ValueError("Approved plan differs")
    for name, digest in lock["archive_locks"].items():
        if fingerprint_file(Path(lock["archive_dir"]) / name).sha256 != digest:
            raise ValueError("Bound legacy archive differs")
    return lock


def run_phase(root, phase, workers):
    assert_frozen(root)
    check_free(root)
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("At most two one-thread workers permitted")
    if phase == "main":
        g = read_json(root / "storage-runtime-gate.json")
        if (
            g["status"] != "passed"
            or g["audit_sha256"] != fingerprint_file(root / "preflight-audit.json").sha256
        ):
            raise ValueError("A matching passed audited preflight gate is required")
    directory = root / phase
    directory.mkdir()
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_observation, [(t, str(directory)) for t in tasks(phase)]):
            results.append(result)
            print(json.dumps(result), flush=True)
            check_free(root)
            if retained_bytes(root) > SETTINGS["cap_bytes"]:
                raise ValueError("Approved retained-evidence cap exceeded")
    results.append(legacy_replays(root, phase))
    if sum(r["attempts"] for r in results) != SETTINGS[f"{phase}_fits"]:
        raise ValueError("Frozen workflow accounting mismatch")
    write_json(
        {
            "phase": phase,
            "workers": workers,
            "observations": results,
            "elapsed_seconds": time.monotonic() - start,
        },
        root / f"{phase}-execution.json",
    )


def gate(root, staging_bytes):
    assert_frozen(root)
    if type(staging_bytes) is not int or staging_bytes < 0:
        raise ValueError("Nonnegative measured staging bytes required")
    audit = read_json(root / "preflight-audit.json")
    execution = read_json(root / "preflight-execution.json")
    factor = SETTINGS["main_fits"] / SETTINGS["preflight_fits"]
    projected = int(
        retained_bytes(root / "preflight") * factor * 1.25
        + retained_bytes(root)
        + 2 * staging_bytes
        + 128 * 1024**2
    )
    seconds = sum(r["fit_seconds"] for r in execution["observations"]) * factor * 1.25 / 2
    free = shutil.disk_usage(root).free
    passed = (
        audit["passed"]
        and projected <= SETTINGS["cap_bytes"]
        and seconds <= 7200
        and free >= 8 * 1024**3
    )
    result = {
        "status": "passed" if passed else "failed",
        "projected_bytes": projected,
        "projected_main_seconds_two_workers": seconds,
        "measured_staging_bytes": staging_bytes,
        "staging_copies_counted": 2,
        "free_bytes": free,
        "audit_sha256": fingerprint_file(root / "preflight-audit.json").sha256,
    }
    write_json(result, root / "storage-runtime-gate.json")
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("freeze", "preflight", "audit-preflight", "gate", "main", "audit-main", "report"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--staging-bytes", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    if args.phase == "freeze":
        if args.archive_dir is None:
            parser.error("freeze requires --archive-dir")
        freeze(root, args.archive_dir)
    elif args.phase in {"preflight", "main"}:
        run_phase(root, args.phase, args.workers)
    elif args.phase == "gate":
        if gate(root, args.staging_bytes)["status"] != "passed":
            raise SystemExit(1)
    elif args.phase.startswith("audit-"):
        from super_deli.benchmarking.denoising_audit import audit_phase

        audit_phase(root, args.phase.removeprefix("audit-"))
    else:
        from super_deli.benchmarking.denoising_report import report

        report(root)


if __name__ == "__main__":
    main()
