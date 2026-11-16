"""Attach separately versioned sensitivity evidence to an unchanged native point run."""

import csv
import shutil
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json
from super_deli.io.individual_subsample_source import load_point_source
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig


def run_individual_subsampling(reference_run, *, stability_config, output_dir):
    import numpy as np

    from super_deli.models.individual_subsampling import fit_subsamples, summarize
    from super_deli.workflows.individual_subsample_report import write_report

    config = IndividualSubsampleConfig.from_dict(stability_config.to_dict())
    source = load_point_source(reference_run)
    destination = Path(output_dir).absolute()
    if destination.resolve().is_relative_to(source.root) or source.root.is_relative_to(
        destination.resolve()
    ):
        raise ValueError("Use a new output directory outside the point run")
    n = len(source.units)
    m = int(n * config.fraction)
    if m < source.model.min_cluster_size or 24 * n * n * 8 > source.model.max_pairwise_bytes:
        raise ValueError("Subsample support or dense stability memory limit failed")
    destination.mkdir()
    snapshot = destination / "reference"
    snapshot.mkdir()
    for name, digest in source.files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.root / name, target)
        if fingerprint_file(target).sha256 != digest:
            raise ValueError("Point snapshot checksum mismatch")
    write_json(config.to_dict(), destination / "configuration.json")
    write_json(
        dict(
            schema_version=1,
            source_files=source.files,
            source=str(source.root),
            point_configuration=source.model.to_dict(),
            unit_ids=source.units,
        ),
        destination / "source.json",
    )
    try:
        records = fit_subsamples(source, config, destination / "subsamples")
        summary, arrays, support = summarize(source, records)
        summary.update(
            schema_version=1,
            policy=config.policy,
            source_point_k=source.summary["selected_k"],
            individuals=n,
            subsample_individuals=m,
            fraction=config.fraction,
            source_bootstrap=source.summary["bootstrap"],
            p_value=None,
            species_probability=None,
            point_analysis_changed=False,
            source_assignment_threshold=source.model.assignment_threshold,
            partition_basis=(
                "Candidate GMM argmax partitions; original assignment-threshold "
                "decisions stay in the source run"
            ),
            sampling_frame="Fixed ingested genomic calls/loci and complete matched raw arrays",
            pipeline="Refit preprocessing, GPA, view PCA, profile scaling, full joint PCA and GMM",
            native_minimum_distinct_individuals=source.model.min_cluster_size,
            maximum_feasible_k=min(source.model.max_k, m // source.model.min_cluster_size),
        )
        write_json(records, destination / "subsample_replicates.json")
        write_json(support, destination / "cluster_support.json")
        with (destination / "stability.npz").open("xb") as f:
            np.savez(f, **arrays)
        with (destination / "cluster_support.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(support[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(support)
        source.reverify()
        write_json(summary, destination / "summary.json")
        write_report(summary, support, destination / "report.html")
        return summary
    except Exception as exc:
        write_json(
            dict(status="failed", error=f"{type(exc).__name__}: {exc}"),
            destination / "failure.json",
        )
        raise
