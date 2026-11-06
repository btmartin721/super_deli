"""Shared source-bound I/O for independently versioned paired neural models."""

import csv
import html
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import write_json
from super_deli.io.tabular import ingest_tabular


def run_paired_neural(
    dataset,
    *,
    model_config,
    base_dir,
    output_dir,
    config_type,
    fit_function,
    training,
    render,
    assignment_factory=None,
):
    import numpy as np

    from super_deli.models.tabular_neural import point_arrays

    dataset = config_from_dict(dataset.to_dict())
    config = config_type.from_dict(model_config.to_dict())
    selected = {
        m.name: m for m in dataset.modalities if m.name in {config.morphology, config.ecology}
    }
    if set(selected) != {config.morphology, config.ecology}:
        raise ValueError("Both selected views must be declared")
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        source = {name: resolve_input(m.path, base_dir) for name, m in selected.items()}
        before = {name: fingerprint_file(path).sha256 for name, path in source.items()}
        loaded = ingest_tabular(
            dataset,
            base_dir=base_dir,
            output_dir=out / "ingestion",
            modality_names=[config.morphology, config.ecology],
        )
        point, stability, alignment = fit_function(loaded.tables, config=config)
        write_json(alignment, out / "cohort.json")
        write_json(config.to_dict(), out / "model-config.json")
        np.savez_compressed(out / "point.npz", **point_arrays(point))
        np.savez_compressed(out / "stability.npz", **stability.arrays)
        write_json(stability.records, out / "subsamples.json")
        write_json(point.diagnostics, out / "training.json")
        with (out / "assignments.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            if assignment_factory is not None:
                header, rows = assignment_factory(point, alignment)
                writer.writerow(header)
                writer.writerows(rows)
            else:
                writer.writerow(
                    [
                        "unit_id",
                        "morphology_observation_id",
                        "ecology_observation_id",
                        "cluster",
                        "density_membership_strength",
                        *[f"latent_{i + 1}" for i in range(point.scores.shape[1])],
                    ]
                )
                writer.writerows(
                    zip(
                        alignment["unit_ids"],
                        alignment["views"][0]["observation_ids"],
                        alignment["views"][1]["observation_ids"],
                        point.labels,
                        point.strength,
                        *point.scores.T,
                        strict=True,
                    )
                )
        configuration = {
            "dataset": dataset.to_dict(),
            "model": config.to_dict(),
            "training": training,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": config.model_family,
            "dataset_id": dataset.dataset_id,
            "level": "specimen",
            "unit_ids": alignment["unit_ids"],
            "selected_k": point.selected_k,
            "noise_count": int(sum(point.labels < 0)),
            "assigned_fraction": float(np.mean(point.labels >= 0)),
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "source_sha256": before,
            "ingestion": loaded.summary,
            "model_sha256": fingerprint_file(out / "point.npz").sha256,
            "stability": stability.summary,
            "per_view_standardized_mse": point.diagnostics["per_view_standardized_mse"],
            "species_significance": {
                "status": "not_computed",
                "reason": "No neural null calibrated",
            },
            "interpretation": [
                "transductive groups, no best seed",
                "individual ecology and independence are declared assumptions",
                "view-balanced loss is not equal biological influence",
                "no new-unit cluster prediction or species probability",
            ],
            "software": {
                n: version(n)
                for n in ("super-deli", "numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")
            },
        }
        if config.schema_version == 2:
            summary["view_loss_weights"] = {
                "declared": list(config.view_loss_weights),
                "normalized": list(config.loss_fractions()),
            }
            summary["interpretation"][2] = (
                "prespecified loss fractions are not biological importance"
            )
        if before != {name: fingerprint_file(path).sha256 for name, path in source.items()}:
            raise ValueError("Source changed during paired analysis")
        detail = (
            "<p>Both views are aligned by specimen ID. "
            "Individually measured ecology and independent sampling are assumptions.</p>"
        )
        if config.schema_version == 2:
            fractions = config.loss_fractions()
            detail += (
                f"<p>Prespecified reconstruction emphasis: {html.escape(config.morphology)} "
                f"{fractions[0]:.0%}; {html.escape(config.ecology)} {fractions[1]:.0%}. "
                "These fractions weight each view's average error.</p>"
            )
        detail += (
            "<ul>"
            + "".join(
                f"<li>{html.escape(name)}: standardized reconstruction MSE {error:.6g}</li>"
                for name, error in zip(
                    (config.morphology, config.ecology),
                    summary["per_view_standardized_mse"],
                    strict=True,
                )
            )
            + "</ul>"
        )
        render(
            out,
            summary,
            point,
            alignment["unit_ids"],
            title="Matched morphology–ecology autoencoder and HDBSCAN",
            view_details=detail,
        )
        summary["artifact_sha256"] = {
            str(p.relative_to(out)): fingerprint_file(p).sha256
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        assignment = out / "assignments.csv"
        if assignment.exists():
            assignment.rename(out / "failed-assignments.csv")
        if a := getattr(exc, "arrays", {}):
            np.savez_compressed(out / "failed-training.npz", **a)
        write_json(
            {"status": "failed", "error": str(exc), "diagnostics": getattr(exc, "diagnostics", {})},
            out / "failure.json",
        )
        raise
