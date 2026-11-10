"""Bind two completed source studies without changing any archived scientific result."""

from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import verify_seal
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json

SOURCES = (
    (
        "neural-calibration-20261107",
        "four_complementary_0000",
        "54dd00c5710bb26a5d73115696a8b55b92820695",
    ),
    (
        "neural-false-splits-20261108",
        "gaussian_independent_0000",
        "bf352caba337acef6b00ff0509b79a8f66ad2aed",
    ),
)


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not root.is_relative_to(source) or root.parent == source,
        "Source studies and new output must be disjoint",
    )
    require(root != source and not source.is_relative_to(root), "Disjoint source/output")
    files, entries = {}, []
    if fixture:
        meta = read_json(source / "fixture.json")
        require(meta["kind"] == "invented_power_archive_v1", "Invented archive kind")
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
        paths = meta["paths"]
    else:
        paths = []
        package = Path(__file__).parents[1]
        for study, key, head in SOURCES:
            base = source / study
            require(
                not root.is_relative_to(base) and not base.is_relative_to(root),
                "Archive and output disjoint",
            )
            done, frozen = (read_json(base / name) for name in ("completion.json", "frozen.json"))
            require(
                done["completed"] and done["head_sha"] == head and done["ci_jobs"] == 6,
                "Completed exact published source study",
            )
            for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
                require(version(name) == frozen["software"][name], "Exact archive runtime: " + name)
            for name, expected in frozen["source_sha256"].items():
                if name not in ("__init__.py", "cli.py"):
                    require(
                        fingerprint_file(package / name).sha256 == expected,
                        "Preserved scientific source: " + name,
                    )
            task = read_json(base / "main" / key / "task.json")
            if study == SOURCES[0][0]:
                old = read_json(base / "main-inventory.json")
                require(
                    inventory(base / "main" / key)
                    == {
                        p.removeprefix(key + "/"): v
                        for p, v in old.items()
                        if p.startswith(key + "/")
                    },
                    "Pilot inventory",
                )
                files[study + "/main-inventory.json"] = fingerprint_file(
                    base / "main-inventory.json"
                ).sha256
            else:
                verify_seal(base, "main", dict(kind="fresh", task=task))
                name = "ledger/main/" + key + ".json"
                files[study + "/" + name] = fingerprint_file(base / name).sha256
            for name in (
                "completion.json",
                "frozen.json",
                "frozen-sources.tar.gz",
                "main-audit.json",
                "main-execution.json",
            ):
                files[study + "/" + name] = fingerprint_file(base / name).sha256
            paths.append(study + "/main/" + key)
    for i, path in enumerate(paths):
        require(not root.is_relative_to(source / path), "Output cannot overlap archived case")
        task = read_json(source / path / "task.json")
        entries.append(dict(key=f"archived_replay_{i}", path=path, task=task))
        files.update({path + "/" + k: v for k, v in file_hashes(source / path).items()})
    record = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    write_json(record, root / "replay-binding.json")
    check_binding(source, record)
    return dict(passed=True, replays=len(entries), bound_files=len(files))
