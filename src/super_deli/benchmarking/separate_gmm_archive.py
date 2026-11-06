"""Bind exact legacy numerical trajectories and the frozen-coordinate comparison."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, output, *, fixture=False):
    source, output = (Path(source).resolve(), Path(output).resolve())
    files = {}
    entries = []
    require(
        source != output and (not source.is_relative_to(output)), "Separate scoped evidence roots"
    )
    if fixture:
        meta = read_json(source / "fixture.json")
        require(meta["kind"] == "invented_shared_encoder_source", "Fixture kind")
        entries = meta["replays"]
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        old = source / "separate-encoder-20261103"
        diagnostic = source / "coordinate-diagnosis-20261105"
        for root, head in (
            (old, "c4dde4cb9c1524e8c9c182987f3edbfbfc57dd25"),
            (diagnostic, "8544dda15908e7c154f81195d55fe3d1edda40f1"),
        ):
            done = read_json(root / "completion.json")
            require(
                done["completed"] and done["head_sha"] == head and (done["ci_jobs"] == 6),
                "Completed source",
            )
            for name in (
                "completion.json",
                "frozen.json",
                "frozen-sources.tar.gz",
                "main-inventory.json",
            ):
                files[str((root / name).relative_to(source))] = fingerprint_file(root / name).sha256
        frozen = read_json(old / "frozen.json")
        with tarfile.open(old / "frozen-sources.tar.gz") as bundle:
            hashes = {
                m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                for m in bundle.getmembers()
                if m.isfile()
            }
        require(hashes == frozen["source_sha256"], "Bound original numerical source bundle")
        package = Path(__file__).parents[1]
        for name in (
            "models/tabular_neural.py",
            "models/neural_config.py",
            "models/separate_neural.py",
            "models/separate_config.py",
            "models/mixture.py",
            "models/density_scores.py",
            "models/hdbscan_config.py",
            "models/subsample_stability.py",
            "models/paired_subsampling.py",
            "benchmarking/matched_data.py",
            "benchmarking/view_weight_data.py",
        ):
            require(
                fingerprint_file(package / name).sha256 == hashes[name],
                "Unchanged archived numerical kernel: " + name,
            )
        for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
            require(
                version(name) == frozen["software"][name],
                "Exact archived numerical runtime: " + name,
            )
        for case, n in (
            ("morphology_only", 64),
            ("four_complementary", 64),
            ("four_redundant", 256),
            ("gaussian_independent", 64),
        ):
            key = f"{case}_n{n}_0000"
            folder = old / "main" / key
            for seed in (0, 101, 202):
                entries.append(
                    dict(
                        key=f"legacy_{key}_seed{seed}",
                        path=str(folder.relative_to(source)),
                        coordinate_path=str((diagnostic / "main" / key).relative_to(source)),
                        task=read_json(folder / "task.json"),
                        seed=seed,
                        arm=f"separate_{seed}",
                    )
                )
    folders = {e["path"] for e in entries} | {
        e["coordinate_path"] for e in entries if "coordinate_path" in e
    }
    for relative in sorted(folders):
        folder = source / relative
        require(
            not output.is_relative_to(folder) and (not folder.is_relative_to(output)),
            "Protected archive disjoint from outputs",
        )
        if not fixture:
            inventory = read_json(folder.parents[1] / "main-inventory.json")
            actual = file_hashes(folder)
            prefix = folder.name + "/"
            require(
                actual
                == {
                    k.removeprefix(prefix): v for k, v in inventory.items() if k.startswith(prefix)
                },
                "Original inventory coverage",
            )
        files.update({relative + "/" + k: v for k, v in file_hashes(source / relative).items()})
    output.mkdir(exist_ok=True)
    record = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    write_json(record, output / "replay-binding.json")
    check_binding(source, record)
    return dict(passed=True, replays=len(entries), bound_files=len(files))
