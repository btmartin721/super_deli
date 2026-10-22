"""Shared point fitting, archival and failure denominators for raw-data tests."""

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def archive_tree(archive, directory, prefix, inventory, limit):
    size = sum(item["size_bytes"] for item in inventory.values())
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            fingerprint = fingerprint_file(path)
            size += fingerprint.size_bytes
            if size > limit:
                raise ValueError("Archived uncompressed artifacts exceed max_archive_bytes")
            member = f"{prefix}/{path.relative_to(directory).as_posix()}"
            inventory[member] = {"sha256": fingerprint.sha256, "size_bytes": fingerprint.size_bytes}
            archive.write(path, arcname=member)


def fit_pipeline(staging, declaration, directory, prefix, *, model, runner, validate=None):
    from super_deli.models.gaussian_calibration import bic_improvement

    record = {"archive_prefix": prefix}
    write_json(declaration.to_dict(), staging / "dataset.json")
    try:
        run = runner(
            declaration, model_config=model, base_dir=directory, output_dir=staging / "run"
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        record.update(status="failed", error=str(exc), error_stage="integrative_point_pipeline")
    else:
        if validate is not None:
            validate(run, staging / "run")
        candidates = read_json(staging / "run/candidates.json")
        record.update(
            status="complete",
            statistic=bic_improvement(candidates),
            selected_k=run["selected_k"],
            point_seed=run["point_seed"],
            n_retained=len(run["unit_ids"]),
            candidates=candidates,
            mixture_sha256=run["provenance"]["mixture_sha256"],
            fusion_sha256=run["provenance"]["fusion_sha256"],
        )
    return record


def fit_counts(point, records, requested):
    attempts = (*((point,) if point is not None else ()), *records)
    return {
        "planned_fits": requested + 1,
        "attempted_fits": len(attempts),
        "successful_fits": sum(r["status"] == "complete" for r in attempts),
        "failed_fits": sum(r["status"] == "failed" for r in attempts),
        "unattempted_fits": requested + 1 - len(attempts),
    }
