"""Bind complete archived calibrations without changing their inputs or p-values."""

from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(not source.is_relative_to(root) and not root.is_relative_to(source), "Disjoint roots")
    files = {}
    if fixture:
        meta = read_json(source / "fixture.json")
        require(meta["kind"] == "invented_false_split_archive_v1", "Invented archive kind")
        paths = meta["paths"]
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        done, frozen = (read_json(source / name) for name in ("completion.json", "frozen.json"))
        require(
            done["completed"]
            and done["head_sha"] == "54dd00c5710bb26a5d73115696a8b55b92820695"
            and done["ci_jobs"] == 6,
            "Completed published calibration pilot",
        )
        for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
            require(version(name) == frozen["software"][name], "Exact archive runtime: " + name)
        package = Path(__file__).parents[1]
        for name, expected in frozen["source_sha256"].items():
            if name not in ("__init__.py", "cli.py"):
                require(
                    fingerprint_file(package / name).sha256 == expected,
                    "Preserved existing source: " + name,
                )
        paths = ["main/gaussian_independent_0009", "main/gaussian_correlated_0000"]
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-inventory.json",
            "main-audit.json",
        ):
            files[name] = fingerprint_file(source / name).sha256
        inv = read_json(source / "main-inventory.json")
        for path in paths:
            prefix = path.removeprefix("main/") + "/"
            require(
                file_hashes(source / path)
                == {k.removeprefix(prefix): v for k, v in inv.items() if k.startswith(prefix)},
                "Exact archived case inventory",
            )
    entries = []
    for i, path in enumerate(paths):
        task = read_json(source / path / "task.json")
        entries.append(dict(key=f"archived_replay_{i}", path=path, task=task))
        files.update({path + "/" + k: v for k, v in file_hashes(source / path).items()})
    record = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    write_json(record, root / "replay-binding.json")
    check_binding(source, record)
    return dict(passed=True, replays=len(entries), bound_files=len(files))
