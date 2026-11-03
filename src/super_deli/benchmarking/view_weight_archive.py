"""Bind twelve original legacy points, or a separately identified engineering fixture."""

import hashlib
import tarfile
from pathlib import Path

from super_deli.benchmarking.matched_config import StudyConfig, tasks
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, output, *, fixture=False):
    source, out = Path(source).resolve(), Path(output).resolve()
    require(
        not source.is_relative_to(out) and not out.is_relative_to(source),
        "Separate evidence roots required",
    )
    out.mkdir(exist_ok=True)
    files, entries = {}, []
    if fixture:
        record = read_json(source / "fixture.json")
        require(
            record["kind"] == "invented_trained_legacy_fixture",
            "Explicit engineering source required",
        )
        entries = record["replays"]
        folders = sorted({e["path"] for e in entries})
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        complete = read_json(source / "completion.json")
        require(
            complete["completed"]
            and complete["head_sha"] == "547a74035e4f7ebf5fe371f16e16d626fa1a95ae",
            "Expected completed legacy study",
        )
        frozen = read_json(source / "frozen.json")
        from importlib.metadata import version

        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
            require(
                version(name) == frozen["software"][name], "Exact legacy numerical runtime required"
            )
        with tarfile.open(source / "frozen-sources.tar.gz") as archive:
            bundle = {
                m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive.getmembers()
                if m.isfile()
            }
        require(bundle == frozen["source_sha256"], "Original source bundle")
        package = Path(__file__).parents[1]
        for name in (
            "models/tabular_neural.py",
            "models/neural_config.py",
            "models/density_scores.py",
            "models/hdbscan_config.py",
            "models/subsample_stability.py",
            "benchmarking/matched_data.py",
        ):
            require(
                fingerprint_file(package / name).sha256 == bundle[name],
                "Unchanged trainer/density/sampling kernel: " + name,
            )
        folders = []
        for task in tasks(StudyConfig(), "main"):
            if task["case"] not in ("morphology_only", "four_complementary") or task["replicate"]:
                continue
            relative = "main/" + task["key"]
            folders.append(relative)
            for seed in (0, 101, 202):
                entries.append(
                    {
                        "key": "legacy_" + task["key"] + f"_seed{seed}",
                        "path": relative,
                        "task": task,
                        "seed": seed,
                        "arm": f"paired_{seed}",
                    }
                )
        require(len(entries) == 12, "Twelve prespecified numerical replays")
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-inventory.json",
        ):
            files[name] = fingerprint_file(source / name).sha256
    for relative in folders:
        folder = safe_path(source, relative)
        actual = file_hashes(folder)
        if not fixture:
            original = read_json(source / "main-inventory.json")
            prefix = folder.name + "/"
            require(
                actual
                == {k.removeprefix(prefix): v for k, v in original.items() if k.startswith(prefix)},
                "Original source file coverage/content",
            )
        files.update({relative + "/" + k: v for k, v in actual.items()})
    binding = {"passed": True, "fixture": fixture, "replays": entries, "source_sha256": files}
    write_json(binding, out / "replay-binding.json")
    check_binding(source, binding)
    return {"passed": True, "replays": len(entries), "bound_files": len(files)}


def check_binding(source, binding):
    for name, digest in binding["source_sha256"].items():
        require(
            fingerprint_file(safe_path(source, name)).sha256 == digest,
            "Legacy source changed: " + name,
        )
    for relative in {e["path"] for e in binding["replays"]}:
        actual = set(file_hashes(safe_path(source, relative)))
        require(
            actual
            == {
                k.removeprefix(relative + "/")
                for k in binding["source_sha256"]
                if k.startswith(relative + "/")
            },
            "Legacy source inventory changed",
        )


def create_fixture(source):
    import numpy as np

    from super_deli.benchmarking.matched_data import configuration, write_dataset
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.benchmarking.view_weight_config import ViewWeightStudy
    from super_deli.benchmarking.view_weight_config import tasks as new_tasks
    from super_deli.benchmarking.view_weight_data import generate
    from super_deli.benchmarking.view_weight_run import run_arm

    source = Path(source)
    source.mkdir(parents=True, exist_ok=False)
    folder = source / "source_observation"
    folder.mkdir()
    task = {**new_tasks(ViewWeightStudy(fixture=True), "main")[0], "draws": 0}
    data = generate(task)
    write_json(task, folder / "task.json")
    np.savez_compressed(folder / "generating-reference.npz", **data)
    write_dataset(data, folder / "raw")
    write_dataset(data, folder / "canonical", canonical=True)
    result = run_arm(
        task,
        folder / "raw",
        folder / "canonical",
        folder / "paired_0",
        "joint_equal_0",
        legacy_config=configuration("paired_0", 0),
    )
    require(
        result["status"] == "available" and not result["numerical_guard_failed"],
        "Engineering legacy source fit",
    )
    checks = arrays(folder / "paired_0/optimizer-checks.npz")
    np.savez_compressed(
        folder / "paired_0/initial-weights.npz",
        **{
            k.replace("fit_0_initial_", "fit_0_"): v
            for k, v in checks.items()
            if k.startswith("fit_0_initial_")
        },
    )
    write_json(
        {
            "kind": "invented_trained_legacy_fixture",
            "replays": [
                {
                    "key": "legacy_fixture",
                    "path": "source_observation",
                    "task": task,
                    "seed": 0,
                    "arm": "paired_0",
                }
            ],
        },
        source / "fixture.json",
    )
    return source
