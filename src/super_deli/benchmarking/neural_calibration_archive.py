"""Bind completed previous point results for four exact, inference-free replays."""

from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    files = {}
    entries = []
    if fixture:
        meta = read_json(source / "fixture.json")
        require(meta["kind"] == "invented_neural_calibration_source", "Invented fixture kind")
        entries = meta["replays"]
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        old = source / "separate-gmm-20261106"
        done = read_json(old / "completion.json")
        require(
            done["completed"]
            and done["head_sha"] == "1113e387bf1a57318b7d98553aecbda82b913f55"
            and done["ci_jobs"] == 6,
            "Completed previous checkpoint",
        )
        frozen = read_json(old / "frozen.json")
        for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
            require(version(name) == frozen["software"][name], "Exact replay runtime: " + name)
        package = Path(__file__).parents[1]
        for name in (
            "models/tabular_neural.py",
            "models/neural_config.py",
            "models/separate_neural.py",
            "models/separate_gmm.py",
            "models/separate_gmm_config.py",
            "models/mixture.py",
            "models/empirical_null.py",
            "models/monte_carlo.py",
        ):
            require(
                fingerprint_file(package / name).sha256 == frozen["source_sha256"][name],
                "Preserved kernel: " + name,
            )
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-inventory.json",
        ):
            files[str((old / name).relative_to(source))] = fingerprint_file(old / name).sha256
        inv = read_json(old / "main-inventory.json")
        for case in (
            "gaussian_independent",
            "gaussian_correlated",
            "four_complementary",
            "four_redundant",
        ):
            folder = old / "main" / (case + "_n64_0000")
            actual = file_hashes(folder)
            prefix = folder.name + "/"
            require(
                actual
                == {k.removeprefix(prefix): v for k, v in inv.items() if k.startswith(prefix)},
                "Original inventory",
            )
            entries.append(
                dict(key="replay_" + case, path=str(folder.relative_to(source)), fixture=False)
            )
    for entry in entries:
        p = source / entry["path"]
        require(
            not root.is_relative_to(p) and not p.is_relative_to(root),
            "Disjoint replay/output roots",
        )
        files.update({entry["path"] + "/" + k: v for k, v in file_hashes(p).items()})
    record = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    write_json(record, root / "replay-binding.json")
    check_binding(source, record)
    return dict(passed=True, replays=len(entries), bound_files=len(files))
