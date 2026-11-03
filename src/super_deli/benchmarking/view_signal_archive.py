"""Content-bound access to the completed matched-view study, never a new cohort."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.matched_config import StudyConfig, tasks
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_moons_coordinates import safe_path
from super_deli.benchmarking.view_signal_config import CASES
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def file_hashes(root):
    return {
        str(p.relative_to(root)): fingerprint_file(p).sha256
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def check_binding(source, binding):
    for name, expected in binding["source_sha256"].items():
        require(
            fingerprint_file(safe_path(source, name)).sha256 == expected,
            "Bound source changed: " + name,
        )
    for entry in binding["observations"]:
        actual = set(file_hashes(safe_path(source, entry["path"])))
        expected = {
            n.removeprefix(entry["path"] + "/")
            for n in binding["source_sha256"]
            if n.startswith(entry["path"] + "/")
        }
        require(actual == expected, "Archive file coverage changed")


def bind(source, output, *, fixture=False):
    from super_deli.benchmarking.matched_audit import audit_sources
    from super_deli.benchmarking.view_signal_coordinates import load_context

    source, out = Path(source).resolve(), Path(output).resolve()
    require(not source.is_relative_to(out) and not out.is_relative_to(source), "Separate roots")
    out.mkdir(exist_ok=True)
    entries, hashes = [], {}
    if fixture:
        manifest = read_json(source / "fixture.json")
        require(manifest["kind"] == "invented_untrained_fixture", "Explicit engineering fixture")
        entries = manifest["observations"]
        hashes["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        completion = read_json(source / "completion.json")
        require(
            completion["completed"]
            and completion["ci_conclusion"] == "success"
            and completion["head_sha"] == "547a74035e4f7ebf5fe371f16e16d626fa1a95ae",
            "Expected completed parent study required",
        )
        freeze = read_json(source / "frozen.json")
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
            require(version(name) == freeze["software"][name], "Archival numerical runtime differs")
        with tarfile.open(source / "frozen-sources.tar.gz") as bundle:
            sources = {
                m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                for m in bundle.getmembers()
                if m.isfile()
            }
        require(sources == freeze["source_sha256"], "Parent source bundle differs")
        package = Path(__file__).parents[1]
        for name, digest in sources.items():
            if name.startswith(
                (
                    "models/",
                    "workflows/",
                    "benchmarking/matched_",
                    "benchmarking/moons_",
                    "benchmarking/unknown_k_reference",
                )
            ):
                require(
                    fingerprint_file(package / name).sha256 == digest,
                    "Unchanged parent numerical/audit code required: " + name,
                )
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "approved-plan.md",
            "main-audit.json",
            "preflight-audit.json",
            "main-inventory.json",
            "preflight-inventory.json",
            "step-main-verification-audit.json",
        ):
            hashes[name] = fingerprint_file(source / name).sha256
        for phase in ("preflight", "main"):
            for task in tasks(StudyConfig(), phase):
                if task["case"] not in CASES:
                    continue
                entries.append(
                    {
                        "path": phase + "/" + task["key"],
                        "phase": phase,
                        "task": task,
                        "fixture": False,
                    }
                )
        require(len(entries) == 66, "Frozen sixty main/six preflight observations")
    for entry in entries:
        folder = safe_path(source, entry["path"])
        actual = file_hashes(folder)
        if not fixture:
            locked = read_json(source / (entry["phase"] + "-inventory.json"))
            prefix = folder.name + "/"
            original = {
                k.removeprefix(prefix): v for k, v in locked.items() if k.startswith(prefix)
            }
            require(actual == original, "Original checksums/coverage")
            require(read_json(folder / "task.json") == entry["task"], "Original task identity")
            audit_sources(folder, entry["task"])
        hashes.update({entry["path"] + "/" + k: v for k, v in actual.items()})
        load_context(source, entry)
    binding = {
        "schema_version": 1,
        "passed": True,
        "kind": "invented_untrained_fixture" if fixture else "completed_matched_archive",
        "observations": entries,
        "source_sha256": hashes,
    }
    write_json(binding, out / "archive-binding.json")
    check_binding(source, binding)
    return {"passed": True, "observations": len(entries), "bound_files": len(hashes)}
