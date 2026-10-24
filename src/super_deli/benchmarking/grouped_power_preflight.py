"""Measured lossless-retention budget before the approved larger study starts."""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from super_deli.benchmarking.grouped_power_study import COUNT_KEYS
from super_deli.benchmarking.local_archives import compact_directory, materialize, retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig, PowerCell


def preflight_config(config, *, random_seed=20261122):
    return replace(
        config,
        cells=tuple(
            PowerCell(f"probe_n{n}", "complementary", n, 1, 3.0)
            for n in sorted({c.n_units for c in config.cells})
        ),
        random_seed=random_seed,
        stability_first=1,
        calibration=replace(config.calibration, random_seed=random_seed),
    )


def storage_preflight(config, *, probe_study, probe_audit, artifact_root, scratch, output_file):
    import json

    import numpy as np

    config = GroupedPowerStudyConfig.from_dict(config.to_dict())
    source, artifact_root, scratch = map(
        lambda p: Path(p).resolve(), (probe_study, artifact_root, scratch)
    )
    probe = GroupedPowerStudyConfig.from_dict(read_json(source / "study.json"))
    expected = preflight_config(config, random_seed=probe.random_seed)
    if expected != probe or probe.random_seed == config.random_seed:
        raise ValueError("Use an independent prespecified probe with identical fitting settings")
    audit = read_json(Path(probe_audit) / "summary.json")
    execution = read_json(source / "summary.json")
    planned = sum(r["planned_fits"] for r in probe.plan())
    if (
        audit["observations"] != len(probe.cells)
        or audit["numerically_audited_observations"] != len(probe.cells)
        or audit["fit_accounting"]["successful_fits"] != planned
        or any(execution[k] != audit["fit_accounting"][k] for k in COUNT_KEYS)
    ):
        raise ValueError(
            "Storage projection requires a fully successful independently audited probe"
        )
    ledger = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    measurements = {}
    scratch.mkdir(parents=True, exist_ok=True)
    for record in ledger:
        n = record["cell_config"]["n_units"]
        with materialize(
            source / record["archive"],
            record,
            scratch=scratch,
            min_free_bytes=config.min_free_bytes,
        ) as base:
            counts = read_json(base / "generating_layout.json")["counts"]
            with TemporaryDirectory(prefix="super-deli-budget-", dir=scratch) as temporary:
                temporary = Path(temporary)
                reduced = temporary / "calibration-only"
                shutil.copytree(
                    base,
                    reduced,
                    ignore=lambda directory, names: [
                        name
                        for name in names
                        if Path(directory) == base and name.startswith("stability")
                    ],
                )
                main = compact_directory(reduced, temporary / "calibration-only.tar.gz")
            measurements[n] = {
                "calibration_archive_bytes": main["size_bytes"],
                "additional_stability_bytes": max(0, record["size_bytes"] - main["size_bytes"]),
                "full_archive_bytes": record["size_bytes"],
                "morphology_rows": sum(counts),
                "source_row_proxy": 4 * n + sum(counts),
            }
    if set(measurements) != {c.n_units for c in config.cells}:
        raise ValueError("Missing sample-size storage probe")
    main_bytes = stability_bytes = 0.0
    cells = {c.name: c for c in config.cells}
    data_seeds = {r["data_seed"] for r in probe.plan()}
    main_seeds = {r["data_seed"] for r in config.plan()}
    if data_seeds & main_seeds:
        raise ValueError("Probe and main observation streams overlap")
    for declaration in config.plan():
        cell = cells[declaration["cell"]]
        measured = measurements[cell.n_units]
        # Counts are non-informative design variables, frozen from their first RNG draw.
        counts = np.random.default_rng(declaration["data_seed"]).integers(1, 5, cell.n_units)
        ratio = (4 * cell.n_units + int(counts.sum())) / measured["source_row_proxy"]
        main_bytes += measured["calibration_archive_bytes"] * ratio
        if declaration["stability"]:
            stability_bytes += measured["additional_stability_bytes"] * max(1.0, ratio)
    existing = retained_bytes(artifact_root)
    reserve = 256 * 1024**2
    projected_new = math.ceil(1.10 * (main_bytes + stability_bytes)) + reserve
    projected_total = existing + projected_new
    free = shutil.disk_usage(artifact_root).free
    passed = (
        projected_total <= config.max_artifact_bytes
        and free >= config.min_free_bytes + projected_new
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "study_sha256": configuration_hash(config.to_dict()),
        "artifact_root": str(artifact_root),
        "probe_study_sha256": configuration_hash(probe.to_dict()),
        "probe_planned_fits": planned,
        "probe_successful_fits": audit["fit_accounting"]["successful_fits"],
        "measurements_by_n": measurements,
        "projected_calibration_bytes": math.ceil(main_bytes),
        "projected_stability_bytes": math.ceil(stability_bytes),
        "growth_multiplier": 1.10,
        "audit_and_reporting_reserve_bytes": reserve,
        "existing_local_evidence_bytes": existing,
        "projected_new_bytes": projected_new,
        "projected_complete_artifact_bytes": projected_total,
        "max_artifact_bytes": config.max_artifact_bytes,
        "current_free_bytes": free,
        "min_free_bytes": config.min_free_bytes,
        "planned_pipeline_fits": sum(r["planned_fits"] for r in config.plan()),
        "policy": (
            "Measured projection only; guards retain evidence and stop on unexpected growth. "
            "No observations or fits may be dropped to meet the cap."
        ),
    }
    write_json(result, output_file)
    if not passed:
        raise RuntimeError(
            "Storage preflight failed; obtain a storage/design revision before full launch"
        )
    return result
