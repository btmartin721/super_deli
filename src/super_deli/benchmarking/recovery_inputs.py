"""Pinned, read-only source validation and truth-separated raw dataset staging."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.joint_analysis import point
from super_deli.benchmarking.joint_study import case_policy, simulate_joint
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.recovery_config import variants


def inventory(root):
    return {
        p.relative_to(root).as_posix(): fingerprint_file(p).sha256
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class DirectoryArchive:
    """Read-only subset of the ZipFile interface used by numerical auditors."""

    def __init__(self, root):
        self.root = Path(root)

    def read(self, name):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe artifact path")
        return (self.root / path).read_bytes()


def check_sources(root, config):
    import super_deli

    pins = {
        "study.json": config.source_study_sha256,
        "study-analysis/files.json": config.source_inventory_sha256,
        "runtime_sources.json": config.source_modules_sha256,
    }
    for name, digest in pins.items():
        if fingerprint_file(root / name).sha256 != digest:
            raise ValueError(f"Source fingerprint mismatch: {name}")
    study = JointStudyConfig.from_dict(read_json(root / "study.json"))
    if study.n_datasets != config.n_datasets or study.n_units != config.small_n:
        raise ValueError("Source cohort dimensions disagree with diagnostic declaration")
    variants(study.calibration.model)
    modules = read_json(root / "runtime_sources.json")
    package_root = Path(super_deli.__file__).resolve().parent.parent
    checked = []
    for name, digest in modules.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "super_deli":
            raise ValueError("Unsafe source module path")
        # Version metadata and CLI dispatch may evolve; scientific source must match.
        if name not in {"super_deli/__init__.py", "super_deli/cli.py"}:
            if fingerprint_file(package_root / path).sha256 != digest:
                raise ValueError(f"Scientific source changed since archived study: {name}")
            checked.append(name)
    if "super_deli/benchmarking/joint_study.py" not in checked:
        raise ValueError("Source generator fingerprint is required")
    return study, pins, checked


def stage_generated(values, layout, labels, directory):
    import numpy as np

    from super_deli.models.joint_null import write_raw_joint

    directory.mkdir(exist_ok=False)
    write_raw_joint(values, layout, directory / "raw")
    with (directory / "values.npz").open("xb") as handle:
        np.savez(handle, **values)
    write_json(layout, directory / "layout.json")
    write_json(
        {
            "scope": "Constructed generating partition; evaluation only, not species",
            "unit_ids": layout["unit_ids"],
            "labels": labels.tolist(),
        },
        directory / "truth.json",
    )


def prepare_archived(root, destination, study):
    import numpy as np

    from super_deli.models.joint_null import write_raw_joint

    declared = read_json(root / "study-analysis/files.json")
    shard = root / "study/complementary"
    if JointStudyConfig.from_dict(read_json(shard / "study.json")).to_dict() != study.to_dict():
        raise ValueError("Source shard and study declarations disagree")
    ledger = [json.loads(line) for line in (shard / "datasets.jsonl").read_text().splitlines()]
    if len(ledger) != study.n_datasets:
        raise ValueError("Source ledger has an unexpected number of datasets")
    snapshots = destination / "source_snapshots"
    snapshots.mkdir()
    evidence = []
    for i, row in enumerate(ledger):
        seed = derived_seed(study.random_seed, "complementary", i, "observed_data")
        if (
            row["repetition"] != i
            or row["data_seed"] != seed
            or row["directory"] != f"dataset_{i:04d}"
            or row["case"] != "complementary"
        ):
            raise ValueError("Archived dataset seed or identity mismatch")
        base = shard / row["directory"]
        # Hash all selected source files, including the original 99-draw archive.
        for p in sorted(base.rglob("*")):
            if (
                p.is_file()
                and declared.get(p.relative_to(root / "study").as_posix())
                != fingerprint_file(p).sha256
            ):
                raise ValueError("Archived source file changed")
        values, layout, labels = simulate_joint("complementary", study.n_units, seed)
        with np.load(base / "values.npz", allow_pickle=False) as arrays:
            for key, value in values.items():
                np.testing.assert_array_equal(arrays[key], value)
        np.testing.assert_array_equal(read_json(base / "truth.json")["labels"], labels)
        with TemporaryDirectory(prefix="super-deli-recovery-source-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_joint(values, layout, generated)
            if inventory(generated) != inventory(base / "observed"):
                raise ValueError("Archived raw generator or identity mismatch")
        policy = case_policy(study, "complementary", i)
        record = read_json(base / "calibration/observed.json")
        if record["status"] != "complete":
            raise ValueError("Source baseline must have a complete archived observation")
        snapshot = snapshots / row["directory"]
        snapshot.mkdir()
        with ZipFile(base / "calibration/pipeline_artifacts.zip") as archive:
            point(archive, record, policy, values, layout)
            members = archive.namelist()
            if len(set(members)) != len(members):
                raise ValueError("Duplicate archived member")
            archive_inventory = read_json(base / "calibration/archive_inventory.json")
            for name in members:
                if not name.startswith("observed/"):
                    continue
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source archive member")
                data = archive.read(name)
                if archive_inventory[name] != {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }:
                    raise ValueError("Archived observation member checksum mismatch")
                target = snapshot / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        write_json(record, snapshot / "record.json")
        stage_generated(values, layout, labels, destination / "archived" / row["directory"])
        evidence.append(
            {
                "repetition": i,
                "seed": seed,
                "snapshot": snapshot.relative_to(destination).as_posix(),
                "source_sha256": {
                    p.relative_to(root / "study").as_posix(): fingerprint_file(p).sha256
                    for p in sorted(base.rglob("*"))
                    if p.is_file()
                },
            }
        )
    return evidence
