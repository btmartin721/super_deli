"""Bind complete predecessor observations without modifying any source evidence."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.abstention_source import bind_observation_inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import verify_seal
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json

SOURCE_HEAD = "c4fa85e9d758d4c4e3b89d721b132f1337c5e013"


def check_source(source, binding):
    require(type(binding.get("fixture")) is bool, "Explicit source kind required")
    check_binding(source, binding)
    for entry in binding["observations"]:
        relative = entry["path"]
        expected = {
            k.removeprefix(relative + "/"): v
            for k, v in binding["source_sha256"].items()
            if k.startswith(relative + "/")
        }
        require(
            file_hashes(Path(source) / relative) == expected,
            "Complete predecessor observation inventory/content",
        )
        if not binding["fixture"]:
            task = entry["task"]
            require(relative == "main/" + task["key"], "Original observation path/identity")
            sealed = verify_seal(Path(source), "main", dict(kind="fresh", task=task))
            require(not sealed["interrupted"], "Completed original observation seal")


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    files = {}

    def keep(name):
        files[name] = fingerprint_file(source / name).sha256

    if fixture:
        record = read_json(source / "fixture.json")
        require(
            record["kind"] == "invented_local_reference_source_v1",
            "Explicit invented class-local source",
        )
        paths = record["observations"]
        keep("fixture.json")
    else:
        complete = read_json(source / "completion.json")
        require(
            complete["completed"]
            and complete["head_sha"] == SOURCE_HEAD
            and complete["ci_jobs"] == 6,
            "Completed exact predecessor checkpoint",
        )
        frozen = read_json(source / "frozen.json")
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
            require(
                version(name) == frozen["software"][name], "Exact predecessor numerical runtime"
            )
        with tarfile.open(source / "frozen-sources.tar.gz") as archive:
            bundle = {
                m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive.getmembers()
                if m.isfile()
            }
        require(bundle == frozen["source_sha256"], "Original frozen source bundle")
        package = Path(__file__).parents[1]
        for name, digest in bundle.items():
            if name not in ("cli.py", "__init__.py"):
                require(
                    fingerprint_file(package / name).sha256 == digest,
                    "Unchanged predecessor module: " + name,
                )
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-execution.json",
            "main-audit.json",
            "verification-evidence/ci-verification.json",
        ):
            keep(name)
        entries = read_json(source / "main-execution.json")["records"]
        paths = ["main/" + row["key"] for row in entries if row["kind"] == "fresh"]
        require(len(paths) == 80, "All eighty predecessor observations")
    observations = []
    for path in sorted(paths):
        task = read_json(source / path / "task.json")
        observations.append(dict(path=path, task=task))
        bind_observation_inventory(source, path, files)
        if not fixture:
            keep("ledger/main/" + task["key"] + ".json")
    replays = []
    if not fixture:
        for item in observations:
            t = item["task"]
            if t["case"] in ("linear_weak", "combined") and t["replicate"] == 0:
                for arm in ("logistic", "residual_0"):
                    replays.append(
                        dict(key=f"replay_{t['key']}_{arm}", path=item["path"], arm=arm, n=t["n"])
                    )
        require(len(replays) == 8, "Eight frozen prediction replays")
    root.mkdir(exist_ok=True)
    value = dict(
        passed=True,
        fixture=fixture,
        observations=observations,
        replays=replays,
        source_sha256=files,
    )
    check_source(source, value)
    write_json(value, root / "replay-binding.json")
    return dict(
        passed=True, observations=len(observations), replays=len(replays), bound_files=len(files)
    )
