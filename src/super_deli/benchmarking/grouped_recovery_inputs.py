"""Pinned grouped observations and a truth-blind one-specimen control."""

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.grouped_analysis import point as grouped_point
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.recovery_inputs import inventory
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_config import GroupedStudyConfig
from super_deli.models.grouped_recovery_config import variants


def point(archive, record, policy, values, layout):
    # The existing archive auditor consumes canonical JSON identities (lists).
    return grouped_point(archive, record, policy, values, json.loads(json.dumps(layout)))


def source_pins(config):
    return {
        "study.json": config.source_study_sha256,
        "study-audit/files.json": config.source_inventory_sha256,
        "runtime_sources.json": config.source_modules_sha256,
    }


def check_sources(root, config):
    import super_deli

    pins = source_pins(config)
    for name, digest in pins.items():
        if fingerprint_file(root / name).sha256 != digest:
            raise ValueError(f"Source fingerprint mismatch: {name}")
    study = GroupedStudyConfig.from_dict(read_json(root / "study.json"))
    if study.n_datasets != config.n_datasets or study.n_units != config.small_n:
        raise ValueError("Source cohort dimensions disagree with diagnostic declaration")
    if study.calibration.schema_version != 2:
        raise ValueError("Source requires boundary-capable schema 2")
    variants(study.calibration.model)
    package = Path(super_deli.__file__).resolve().parent
    checked = []
    for name, digest in read_json(root / "runtime_sources.json").items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Unsafe source module path")
        relative = path.relative_to("super_deli") if path.parts[0] == "super_deli" else path
        if relative.as_posix() not in {"__init__.py", "cli.py"}:
            if fingerprint_file(package / relative).sha256 != digest:
                raise ValueError(f"Scientific source changed since archived study: {name}")
            checked.append("super_deli/" + relative.as_posix())
    if "super_deli/benchmarking/grouped_study.py" not in checked:
        raise ValueError("Source generator fingerprint is required")
    return study, pins, checked


def first_specimen(values, layout):
    """Select the first declared morphology row per group, retaining original identities."""
    import numpy as np

    control = copy.deepcopy(layout)
    view = control["views"][0]
    indices = [view["target_ids"].index(u) for u in control["unit_ids"]]
    for key in ("unit_ids", "observation_ids", "target_ids"):
        view[key] = [view[key][i] for i in indices]
    selected = set(view["unit_ids"])
    control["specimens"] = [s for s in control["specimens"] if s["specimen_id"] in selected]
    control["counts"] = [1] * len(control["unit_ids"])
    return {
        "morphological": values["morphological"][indices].copy(),
        "ecological": np.array(values["ecological"], copy=True),
    }, control


def stage_generated(values, layout, labels, directory):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    directory.mkdir(exist_ok=False)
    write_raw_views(values, layout, directory / "raw")
    control, control_layout = first_specimen(values, layout)
    write_raw_views(control, control_layout, directory / "first_raw")
    with (directory / "values.npz").open("xb") as handle:
        np.savez(handle, **values)
    write_json(layout, directory / "layout.json")
    write_json(
        {
            "scope": "Constructed partition; evaluation only, not species",
            "unit_ids": layout["unit_ids"],
            "labels": labels.tolist(),
        },
        directory / "truth.json",
    )


def prepare_archived(root, destination, study):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    declared = read_json(root / "study-audit/files.json")
    shard = root / "study/complementary"
    if GroupedStudyConfig.from_dict(read_json(shard / "study.json")) != study:
        raise ValueError("Source shard and study declarations disagree")
    ledger = [json.loads(line) for line in (shard / "datasets.jsonl").read_text().splitlines()]
    if len(ledger) != study.n_datasets:
        raise ValueError("Unexpected archived cohort size")
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
        source_hashes = inventory(base)
        for relative, digest in source_hashes.items():
            if declared.get(f"complementary/{row['directory']}/{relative}") != digest:
                raise ValueError("Archived source file changed")
        expected = {p for p in declared if p.startswith(f"complementary/{row['directory']}/")}
        if len(expected) != len(source_hashes):
            raise ValueError("Archived source file missing")
        values, layout, labels = simulate_grouped(
            "complementary", study.n_units, seed, target_level=study.calibration.model.target_level
        )
        with np.load(base / "values.npz", allow_pickle=False) as arrays:
            for key, value in values.items():
                np.testing.assert_array_equal(arrays[key], value)
        np.testing.assert_array_equal(read_json(base / "truth.json")["labels"], labels)
        with TemporaryDirectory(prefix="super-deli-grouped-source-") as temporary:
            generated = Path(temporary) / "raw"
            write_raw_views(values, layout, generated)
            if inventory(generated) != inventory(base / "observed"):
                raise ValueError("Archived raw generator or identity mismatch")
        record = read_json(base / "calibration/observed.json")
        if record["status"] != "complete":
            raise ValueError("Source observation must be complete")
        snapshot = snapshots / row["directory"]
        snapshot.mkdir()
        with ZipFile(base / "calibration/pipeline_artifacts.zip") as archive:
            point(archive, record, study.calibration, values, layout)
            members = archive.namelist()
            if len(set(members)) != len(members):
                raise ValueError("Duplicate archived member")
            declared_archive = read_json(base / "calibration/archive_inventory.json")
            for name in members:
                if not name.startswith("observed/"):
                    continue
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source archive member")
                data = archive.read(name)
                if declared_archive[name] != {
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
                "source_sha256": source_hashes,
            }
        )
    return evidence


def replay(root, n_datasets):
    import numpy as np

    result = []
    for i in range(n_datasets):
        old_dir = root / "source_snapshots" / f"dataset_{i:04d}"
        new_dir = root / "archived" / f"dataset_{i:04d}/baseline"
        old, new = read_json(old_dir / "record.json"), read_json(new_dir / "record.json")
        if new["status"] != "complete" or old["selected_k"] != new["selected_k"]:
            result.append({"repetition": i, "status": "failed", "reason": "baseline selection"})
            continue
        for a, b in zip(old["candidates"], new["candidates"], strict=True):
            if {k: v for k, v in a.items() if k != "bic"} != {
                k: v for k, v in b.items() if k != "bic"
            }:
                raise ValueError("Archived candidate status/constraints replay mismatch")
            if a["bic"] is not None:
                np.testing.assert_allclose(a["bic"], b["bic"], rtol=1e-10, atol=1e-7)
        with (
            np.load(old_dir / "observed/run/mixture.npz", allow_pickle=False) as a,
            np.load(new_dir / "run/mixture.npz", allow_pickle=False) as b,
        ):
            np.testing.assert_allclose(a["scores"], b["scores"], rtol=1e-9, atol=1e-11)
            np.testing.assert_array_equal(a["labels"], b["labels"])
        result.append({"repetition": i, "status": "reproduced"})
    return result
