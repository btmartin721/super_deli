"""Measured, audited full-settings storage gate; no evidence thinning on failure."""

import json
import math
import shutil
from pathlib import Path

from super_deli.benchmarking.predictive_study import retained_bytes
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json
from super_deli.models.bootstrap_diagnostic_config import BootstrapDiagnosticConfig


def storage_preflight(c, *, probe_study, probe_audit, probe_check, artifact_root, output_file):
    source = Path(probe_study)
    probe = BootstrapDiagnosticConfig.from_dict(read_json(source / "configuration.json"))
    if c.probe(probe.random_seed) != probe or c.random_seed == probe.random_seed:
        raise ValueError("Separate full-settings one-per-cell probe required")
    audit = read_json(Path(probe_audit) / "summary.json")
    check = read_json(probe_check)
    searches = sum(r["planned_searches"] for r in probe.plan())
    oracles = sum(r["planned_oracles"] for r in probe.plan())
    if (
        audit["status"] != "passed"
        or audit["audited_searches"] != searches
        or audit["audited_oracles"] != oracles
        or audit["unattempted_fits"]
        or audit["unavailable_diagnostics"]
        or check["status"] != "verified"
        or check["audited_searches"] != searches
        or check["audited_oracles"] != oracles
    ):
        raise ValueError("Complete independent numerical and byte/gate probe audits required")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    sizes = {r["cell"]: r["size_bytes"] + (source / r["manifest"]).stat().st_size for r in rows}
    if len(sizes) != len(rows) or set(sizes) != {v.name for v in c.cells}:
        raise ValueError("Exactly one measured archive plus manifest per cell required")
    # Thirty replay workflows; conservatively use the largest per-workflow size.
    replay = math.ceil(30 * max(sizes.values()) / (2 + 2 * c.n_draws + c.extended_draws))
    new = math.ceil(1.25 * (sum(sizes[r["cell"]] for r in c.plan()) + replay)) + 128 * 1024**2
    root = Path(artifact_root).resolve()
    retained, free = retained_bytes(root), shutil.disk_usage(root).free
    total = retained + new
    passed = total <= c.max_artifact_bytes and free >= c.min_free_bytes + new
    result = dict(
        status="passed" if passed else "failed",
        artifact_root=str(root),
        configuration_sha256=configuration_hash(c.to_dict()),
        measured_cell_bytes=sizes,
        current_retained_bytes=retained,
        projected_new_bytes=new,
        projected_complete_bytes=total,
        max_artifact_bytes=c.max_artifact_bytes,
        current_free_bytes=free,
        min_free_bytes=c.min_free_bytes,
        growth_factor=1.25,
        reserve_bytes=128 * 1024**2,
        replay_archive_allowance=replay,
        probe_searches=searches,
        probe_oracles=oracles,
        main_searches_including_replays=sum(r["planned_searches"] for r in c.plan()) + 180,
        main_oracles=sum(r["planned_oracles"] for r in c.plan()),
    )
    write_json(result, output_file)
    if not passed:
        raise RuntimeError("Storage gate failed; preserve evidence and revise budget explicitly")
    return result
