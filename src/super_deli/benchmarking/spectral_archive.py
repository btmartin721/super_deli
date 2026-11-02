"""Source-bound numeric representations; never retrain or execute archived code."""

import hashlib
import platform
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from super_deli.benchmarking.denoising_study import tasks as source_tasks
from super_deli.benchmarking.neural_analysis import (
    arrays,
    audit_point,
    read_raw,
    reference,
    require,
)
from super_deli.benchmarking.neural_moons_archive import inventory
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.spectral_config import SPACES
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def load_real(folder):
    task = read_json(folder / "task.json")
    ref = arrays(folder / "generating-reference.npz")
    noisy, z, truth = reference(task)
    np.testing.assert_array_equal(z, ref["z"])
    np.testing.assert_array_equal(truth, ref["truth"])
    noise = np.random.default_rng(task["noise_seed"]).normal(0, 0.15, noisy.shape)
    np.testing.assert_array_equal(noise, ref["noise"])
    np.testing.assert_allclose(noisy, ref["clean"] + noise, atol=1e-13)
    spaces = {"source": (z - z.mean(0)) / z.std(0)}
    baselines = {"source": arrays(folder / "source-point" / "point.npz")}
    records = read_json(folder / "fits.json")
    for condition in ("clean", "noisy"):
        ids, x = read_raw(folder / condition / "raw")
        require(ids == [f"invented_{i:04d}" for i in range(task["n"])], "Original identity order")
        np.testing.assert_array_equal(x, ref["clean"] + (noise if condition == "noisy" else 0))
        spaces[condition + "_traits"] = (x - x.mean(0)) / x.std(0)
        for arm in (
            "density",
            "neural_0",
            "neural_101",
            "neural_202",
            "denoising_0",
            "denoising_101",
            "denoising_202",
        ):
            r = next(
                r
                for r in records
                if r["condition"] == condition and r["kind"] == "point" and r["arm"] == arm
            )
            require(
                r["status"] == "complete", "Completed source unexpectedly contains failed point"
            )
            a = audit_point(
                x, folder / condition / "point" / arm, arm.replace("denoising_", "neural_"), r
            )
            spaces[condition + "_" + arm] = a["scores"]
            baselines[condition + "_" + arm] = a
    np.testing.assert_array_equal(baselines["source"]["scores"], spaces["source"])
    return task, ids, truth, spaces, baselines


def bind_archive(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not root.is_relative_to(source) and not source.is_relative_to(root),
        "Separate artifact roots required",
    )
    root.mkdir(exist_ok=True)
    before = inventory(source)
    observations = []
    (root / "coordinates").mkdir()
    if fixture:
        f = read_json(source / "fixture.json")
        require(f["kind"] == "invented_untrained_coordinate_fixture", "Fixture kind mismatch")
        entries = f["observations"]
        sizes = f["sizes"]
        software = None
    else:
        frozen = read_json(source / "frozen.json")
        complete = read_json(source / "completion.json")
        require(
            complete["completed"]
            and complete["head_sha"] == "9fea1e05b9e806a9e21842902d99323a522ebf7b",
            "Expected completed denoising study",
        )
        require(
            read_json(source / "main-audit.json")["passed"] and complete["main_workflows"] == 2618,
            "Source main audit",
        )
        software = frozen["software"]
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
            require(version(name) == software[name], "Archive numerical runtime mismatch: " + name)
        require(
            frozen["python"] == sys.version and frozen["platform"] == platform.platform(),
            "Archive Python/platform differs",
        )
        with tarfile.open(source / "frozen-sources.tar.gz") as bundle:
            hashes = {
                m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                for m in bundle.getmembers()
                if m.isfile()
            }
        require(hashes == frozen["source_sha256"], "Frozen source bundle differs")
        package = Path(__file__).resolve().parents[1]
        for name, digest in hashes.items():
            if name.startswith(
                (
                    "models/",
                    "benchmarking/neural_analysis.py",
                    "benchmarking/denoising_",
                    "benchmarking/neural_study.py",
                )
            ):
                require(
                    fingerprint_file(package / name).sha256 == digest,
                    "Protected source differs: " + name,
                )
        entries = source_tasks("main")
        sizes = [64, 256]
    for entry in entries:
        if fixture:
            data = arrays(safe_path(source, entry["file"]))
            ids = entry["unit_ids"]
            truth = data["truth"]
            spaces = {s: data[s] for s in SPACES}
            baseline = {
                s: {"labels": data[s + "_labels"], "strength": data[s + "_strength"]}
                for s in SPACES
                if not s.endswith("_traits")
            }
            task = entry
        else:
            folder = source / "main" / entry["key"]
            expected = read_json(folder / "checksums.json")
            actual = {
                str(p.relative_to(folder)): fingerprint_file(p).sha256
                for p in folder.rglob("*")
                if p.is_file() and p.name != "checksums.json"
            }
            require(expected == actual, "Complete source observation inventory")
            task, ids, truth, spaces, baseline = load_real(folder)
            require(task == entry, "Source task differs")
        require(set(spaces) == set(SPACES), "Seventeen frozen spaces required")
        n = len(ids)
        require(
            len(set(ids)) == n and np.asarray(truth).shape == (n,), "Unique aligned cohort required"
        )
        for name, x in spaces.items():
            require(
                x.shape == (n, 8 if name.endswith("_traits") else 2) and np.isfinite(x).all(),
                "Coordinate shape/finite guard",
            )
        values = {**spaces, "truth": truth}
        for name, b in baseline.items():
            require(b["labels"].shape == b["strength"].shape == (n,), "Baseline shapes")
            values[name + "_labels"] = b["labels"]
            values[name + "_strength"] = b["strength"]
        path = root / "coordinates" / (task["key"] + ".npz")
        np.savez_compressed(path, **values)
        observations.append(
            {
                "key": task["key"],
                "case": task["case"],
                "n": n,
                "replicate": task["replicate"],
                "unit_ids": ids,
                "coordinate_file": str(path.relative_to(root)),
                "coordinate_sha256": fingerprint_file(path).sha256,
            }
        )
    require(inventory(source) == before, "Source changed while binding")
    result = {
        "schema_version": 1,
        "kind": "fixture" if fixture else "denoising_main_archive",
        "passed": True,
        "source_root": str(source),
        "source_sha256": before,
        "sizes": sizes,
        "observations": observations,
        "source_software": software,
        "neural_training_fits": 0,
        "archived_density_partitions": 15 * len(observations),
    }
    write_json(result, root / "binding.json")
    return result


def load_coordinates(root, entry):
    path = Path(root) / entry["coordinate_file"]
    require(path.resolve().is_relative_to(Path(root).resolve()), "Coordinate path escape")
    require(
        fingerprint_file(path).sha256 == entry["coordinate_sha256"], "Materialized coordinate hash"
    )
    return arrays(path)
