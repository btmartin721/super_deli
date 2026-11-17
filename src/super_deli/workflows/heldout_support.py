"""Attach predictive evidence without changing point or subsampling results."""

import csv
import shutil
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json
from super_deli.io.heldout_support_source import load_subsample_source
from super_deli.models.heldout_support_config import HeldoutSupportConfig


def run_heldout_support(reference_run, *, support_config, output_dir):
    import numpy as np

    from super_deli.models.heldout_predictive import score_draws, summarize_scores
    from super_deli.workflows.heldout_support_report import write_report

    config = HeldoutSupportConfig.from_dict(support_config.to_dict())
    source = load_subsample_source(reference_run)
    output = Path(output_dir).absolute()
    if output.resolve().is_relative_to(source.root) or source.root.is_relative_to(output.resolve()):
        raise ValueError("Use a new output directory outside the source subsampling run")
    output.mkdir()
    snapshot = output / "reference"
    snapshot.mkdir()
    try:
        for name, sha in source.files.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.root / name, target)
            if fingerprint_file(target).sha256 != sha:
                raise ValueError("Source snapshot checksum mismatch")
        write_json(config.to_dict(), output / "configuration.json")
        write_json(
            dict(schema_version=1, source=str(source.root), files=source.files),
            output / "source.json",
        )
        records = score_draws(source, output / "scores")
        summary, arrays, groups = summarize_scores(source, records, output / "scores", config)
        summary.update(
            schema_version=1,
            policy=config.policy,
            minimum_coverage=config.minimum_coverage,
            source_point_k=source.point.summary["selected_k"],
            native_minimum_distinct_individuals=source.point.model.min_cluster_size,
            source_assignment_threshold=source.point.model.assignment_threshold,
        )
        write_json(records, output / "predictive_replicates.json")
        write_json(groups, output / "cluster_support.json")
        with (output / "individual_scores.npz").open("xb") as f:
            np.savez(f, **arrays)
        with (output / "individual_scores.csv").open("x", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["unit_id", "requested", "successful", "omitted_mean_gain"])
            for i, unit in enumerate(source.point.units):
                w.writerow(
                    [
                        unit,
                        int(arrays["requested"][i]),
                        int(arrays["successful"][i]),
                        None if np.isnan(arrays["means"][i]) else float(arrays["means"][i]),
                    ]
                )
        source.reverify()
        write_json(summary, output / "summary.json")
        write_report(summary, groups, output / "report.html")
        return summary
    except Exception as exc:
        write_json(
            dict(status="failed", error=f"{type(exc).__name__}: {exc}"), output / "failure.json"
        )
        raise
