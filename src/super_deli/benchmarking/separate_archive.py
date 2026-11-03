"""Bind unchanged equal-weight points from the completed view-weight study."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.benchmarking.view_weight_archive import create_fixture as create_fixture
from super_deli.benchmarking.view_weight_config import ViewWeightStudy, tasks
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, output, *, fixture=False):
    if fixture:
        from super_deli.benchmarking.view_weight_archive import bind as bind_fixture

        return bind_fixture(source, output, fixture=True)
    source, out = Path(source).resolve(), Path(output).resolve()
    require(
        not source.is_relative_to(out) and not out.is_relative_to(source),
        "Separate evidence roots required",
    )
    complete = read_json(source / "completion.json")
    frozen = read_json(source / "frozen.json")
    require(
        complete["completed"]
        and complete["head_sha"] == "d0ba4e5b05d4ecf881180d024b651b91df8cdb8a",
        "Expected completed view-weight archive",
    )
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
    require(bundle == frozen["source_sha256"], "Legacy source bundle")
    package = Path(__file__).parents[1]
    for name in (
        "models/tabular_neural.py",
        "models/neural_config.py",
        "models/density_scores.py",
        "models/hdbscan_config.py",
        "models/subsample_stability.py",
        "benchmarking/matched_data.py",
        "benchmarking/view_weight_data.py",
    ):
        require(
            fingerprint_file(package / name).sha256 == bundle[name],
            "Unchanged numerical/sampling kernel: " + name,
        )
    entries = []
    files = {
        name: fingerprint_file(source / name).sha256
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-inventory.json",
        )
    }
    original = read_json(source / "main-inventory.json")
    for task in tasks(ViewWeightStudy(), "main"):
        if task["replicate"] or task["case"] not in ("morphology_only", "four_complementary"):
            continue
        relative = "main/" + task["key"]
        folder = safe_path(source, relative)
        actual = file_hashes(folder)
        prefix = folder.name + "/"
        require(
            actual
            == {k.removeprefix(prefix): v for k, v in original.items() if k.startswith(prefix)},
            "Original source coverage/content",
        )
        files.update({relative + "/" + k: v for k, v in actual.items()})
        for seed in (0, 101, 202):
            entries.append(
                {
                    "key": "legacy_" + task["key"] + f"_seed{seed}",
                    "path": relative,
                    "task": task,
                    "seed": seed,
                    "arm": f"joint_equal_{seed}",
                }
            )
    require(len(entries) == 12, "Twelve prespecified numerical replays")
    out.mkdir(exist_ok=True)
    binding = {"passed": True, "fixture": False, "replays": entries, "source_sha256": files}
    write_json(binding, out / "replay-binding.json")
    check_binding(source, binding)
    return {"passed": True, "replays": len(entries), "bound_files": len(files)}
