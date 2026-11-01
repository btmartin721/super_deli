"""Bind immutable neural evidence without relaxing its original execution gate."""

import hashlib
import platform
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import (
    arrays,
    audit_point,
    read_raw,
    reference,
    require,
)
from super_deli.benchmarking.neural_moons_config import DENSITY_ARMS
from super_deli.benchmarking.neural_moons_coordinates import load_context, safe_path
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def inventory(root):
    return {
        str(p.relative_to(root)): fingerprint_file(p).sha256
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def check_binding(root, binding):
    require(inventory(root) == binding["source_sha256"], "Archive inventory/hash changed")


def audit_archive(root, *, output_dir, fixture=False):
    import numpy as np

    from super_deli.benchmarking.density_audit import _audit_support
    from super_deli.benchmarking.neural_study import ARMS, SETTINGS, tasks

    require(__debug__, "Archive auditing requires Python assertions enabled")
    root, out = Path(root).resolve(), Path(output_dir).resolve()
    require(
        not out.is_relative_to(root) and not root.is_relative_to(out),
        "Separate evidence roots required",
    )
    out.mkdir(exist_ok=True)
    before = inventory(root)
    if fixture:
        source = read_json(root / "fixture.json")
        require(source["kind"] == "invented_untrained_fixture", "Fixture source kind")
        for entry in source["observations"] + source["replays"]:
            load_context(root, entry)
        binding = {
            **source,
            "passed": True,
            "source_sha256": before,
            "original_workflows": None,
            "archive_failures": 0,
        }
    else:
        freeze = read_json(root / "frozen.json")
        require(freeze["settings"] == SETTINGS, "Archived scientific settings mismatch")
        for package, v in freeze["software"].items():
            if package != "super-deli":
                require(version(package) == v, f"Archival numerical runtime differs: {package}")
        require(
            freeze["python"] == sys.version and freeze["platform"] == platform.platform(),
            "Archival Python/platform differs",
        )
        with tarfile.open(root / "frozen-sources.tar.gz") as archive:
            bundle = {
                m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive.getmembers()
                if m.isfile()
            }
        require(bundle == freeze["source_sha256"], "Frozen source bundle mismatch")
        package_root = Path(__file__).parents[1]
        protected = (
            "models/",
            "benchmarking/neural_analysis.py",
            "benchmarking/neural_study.py",
            "benchmarking/density_audit.py",
            "benchmarking/moons_",
        )
        for name, digest in bundle.items():
            if name.startswith(protected):
                require(
                    fingerprint_file(package_root / name).sha256 == digest,
                    "Protected numerical/audit source changed: " + name,
                )
        observations, replays, total, failed, draw_count = [], [], 0, 0, 0
        for phase in ("preflight", "main"):
            for task in tasks(phase):
                folder = root / phase / task["key"]
                require(read_json(folder / "task.json") == task, "Original task differs")
                checksums = read_json(folder / "checksums.json")
                actual = {
                    str(p.relative_to(folder))
                    for p in folder.rglob("*")
                    if p.is_file() and p.name != "checksums.json"
                }
                require(actual == set(checksums), "Original artifact coverage differs")
                for name, digest in checksums.items():
                    require(
                        fingerprint_file(safe_path(folder, name)).sha256 == digest,
                        "Original artifact hash",
                    )
                ref = arrays(folder / "generating-reference.npz")
                x, z, truth = reference(task)
                np.testing.assert_allclose(ref["x"], x, atol=1e-14)
                np.testing.assert_array_equal(ref["z"], z)
                np.testing.assert_array_equal(ref["truth"], [] if truth is None else truth)
                unit_ids, raw = read_raw(folder / "raw")
                np.testing.assert_array_equal(raw, ref["x"])
                require(
                    unit_ids == [f"invented_{i:04d}" for i in range(len(x))], "Original identities"
                )
                records = read_json(folder / "fits.json")
                seeds = np.random.SeedSequence(task["subsampling_seed"]).spawn(task["draws"])
                draws = [
                    np.sort(
                        np.random.default_rng(s).choice(len(x), int(0.8 * len(x)), replace=False)
                    )
                    for s in seeds
                ]
                expected = (
                    [("point", None, a) for a in ARMS]
                    + [("draw", i, a) for i in range(len(draws)) for a in ARMS]
                    + ([("replay", None, a) for a in ARMS] if task["replay"] else [])
                )
                require(
                    [(r["kind"], r.get("draw"), r["arm"]) for r in records] == expected,
                    "Original fit grid",
                )
                for r in records:
                    total += 1
                    draw_count += r["kind"] == "draw"
                    ids = draws[r["draw"]] if r["kind"] == "draw" else np.arange(len(x))
                    np.testing.assert_array_equal(r["training_indices"], ids)
                    parent = (
                        folder / "draws" / f"draw_{r['draw']:03d}"
                        if r["kind"] == "draw"
                        else folder / r["kind"]
                    )
                    data_dir = parent / "raw" if r["kind"] == "draw" else folder / "raw"
                    subset_ids, subset = read_raw(data_dir)
                    require(subset_ids == [unit_ids[i] for i in ids], "Original draw identities")
                    np.testing.assert_array_equal(subset, ref["x"][ids])
                    fit = parent / r["arm"]
                    if r["status"] == "failed":
                        failed += 1
                        require(
                            bool(r["error"]) and not (fit / "summary.json").exists(),
                            "Missing failure evidence",
                        )
                        continue
                    audit_point(subset, fit, r["arm"], r)
                    require(
                        read_json(fit / "summary.json")["unit_ids"] == subset_ids,
                        "Original fit alignment",
                    )
                    if r["kind"] == "replay":
                        filename = "mixture.npz" if r["arm"] == "gmm" else "point.npz"
                        current, previous = (
                            arrays(fit / filename),
                            arrays(folder / "point" / r["arm"] / filename),
                        )
                        require(current.keys() == previous.keys(), "Original replay keys")
                        for key in current:
                            np.testing.assert_array_equal(current[key], previous[key])
                for arm in ARMS:
                    arm_draws = [r for r in records if r["kind"] == "draw" and r["arm"] == arm]
                    support = arrays(folder / "support" / f"{arm}.npz")
                    if arm_draws:
                        _audit_support(len(x), arm_draws, support)
                    else:
                        require(not support, "Invented original support")
                    require(
                        read_json(folder / "support" / f"{arm}.json")["requested"] == len(draws),
                        "Original support denominator",
                    )
                if phase == "main" and task["case"] != "cline":
                    prefix = str(folder.relative_to(root))
                    entry = {
                        "key": task["key"],
                        "case": task["case"],
                        "n": task["n"],
                        "replicate": task["replicate"],
                        "indices": list(range(len(x))),
                        "unit_ids": unit_ids,
                        "reference": prefix + "/generating-reference.npz",
                        "points": {a: prefix + f"/point/{a}/point.npz" for a in DENSITY_ARMS},
                    }
                    observations.append(entry)
                    if task["case"] == "moons" and task["replicate"] == 0:
                        for arm in DENSITY_ARMS:
                            replays.append(
                                {
                                    **entry,
                                    "key": task["key"] + "_draw0_" + arm,
                                    "indices": draws[0].tolist(),
                                    "unit_ids": [unit_ids[i] for i in draws[0]],
                                    "points": {
                                        a: prefix + f"/draws/draw_000/{a}/point.npz"
                                        for a in DENSITY_ARMS
                                    },
                                    "replay_arm": arm,
                                }
                            )
        require(
            total == 1330 and draw_count == 880 and len(observations) == 60 and len(replays) == 8,
            "Full archive accounting",
        )
        binding = {
            "kind": "neural_study_v1",
            "passed": True,
            "source_sha256": before,
            "original_workflows": total,
            "original_main_draws": 800,
            "archive_failures": failed,
            "observations": observations,
            "replays": replays,
            "source_software": freeze["software"],
            "source_bundle_verified": True,
        }
    require(inventory(root) == before, "Archive changed during its read-only audit")
    write_json(binding, out / "archive-binding.json")
    return {
        "passed": True,
        "kind": binding["kind"],
        "bound_observations": len(binding["observations"]),
        "original_workflows": binding["original_workflows"],
        "archive_failures": binding["archive_failures"],
    }
