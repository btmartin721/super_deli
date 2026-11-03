"""Separate paired workflow, strict numeric reload and source-bound model artifacts."""

import csv
import html
from importlib.metadata import version
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.matched_neural_config import MatchedNeuralConfig
from super_deli.workflows._neural_common import _report


def run_matched_neural(dataset, *, model_config, base_dir, output_dir):
    import numpy as np

    from super_deli.models.matched_neural import fit_matched_neural
    from super_deli.models.neural_config import TRAINING
    from super_deli.models.tabular_neural import point_arrays

    dataset = config_from_dict(dataset.to_dict())
    config = MatchedNeuralConfig.from_dict(model_config.to_dict())
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
        point, stability, alignment = fit_matched_neural(loaded.tables, config=config)
        write_json(alignment, out / "cohort.json")
        write_json(config.to_dict(), out / "model-config.json")
        np.savez_compressed(out / "point.npz", **point_arrays(point))
        np.savez_compressed(out / "stability.npz", **stability.arrays)
        write_json(stability.records, out / "subsamples.json")
        write_json(point.diagnostics, out / "training.json")
        with (out / "assignments.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    "unit_id",
                    "morphology_observation_id",
                    "ecology_observation_id",
                    "cluster",
                    "density_membership_strength",
                    "latent_1",
                    "latent_2",
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
            "training": TRAINING,
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
        _report(
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


def load_matched_neural_model(directory):
    import numpy as np

    from super_deli.models.matched_neural import MatchedNeuralModel

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    config = MatchedNeuralConfig.from_dict(summary["configuration"]["model"])
    if summary.get("model") != config.model_family or summary.get("status") != "complete":
        raise ValueError("Not a completed matched neural model")
    if configuration_hash(summary["configuration"]) != summary["configuration_sha256"]:
        raise ValueError("Matched configuration hash mismatch")
    if fingerprint_file(directory / "point.npz").sha256 != summary["model_sha256"]:
        raise ValueError("Matched model hash mismatch")
    try:
        import zipfile

        with zipfile.ZipFile(directory / "point.npz") as z:
            if sum(i.file_size for i in z.infolist()) > config.max_working_bytes:
                raise ValueError("Numeric model exceeds its declared workspace")
        with np.load(directory / "point.npz", allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        p = len(config.morphology_features) + len(config.ecology_features)
        if not np.array_equal(
            a["selected_view_counts"],
            [len(config.morphology_features), len(config.ecology_features)],
        ):
            raise ValueError("Invalid selected view counts")
        kept = a["kept_indices"]
        if (
            kept.ndim != 1
            or kept.dtype.kind not in "iu"
            or not np.array_equal(kept, np.unique(kept))
            or np.any(kept < 0)
            or np.any(kept >= p)
        ):
            raise ValueError("Invalid kept indices")
        count = np.array(
            [
                sum(kept < len(config.morphology_features)),
                sum(kept >= len(config.morphology_features)),
            ]
        )
        if (
            np.any(count < 3)
            or np.any(count > 32)
            or not np.array_equal(count, a["retained_view_counts"])
        ):
            raise ValueError("Invalid per-view fitted counts")
        from super_deli.models.matched_preprocessing import effective_view_scales

        weights = effective_view_scales(count, config.loss_fractions())
        if config.schema_version == 2 and (
            not np.array_equal(a["declared_view_loss_weights"], config.view_loss_weights)
            or not np.array_equal(a["normalized_view_loss_weights"], config.loss_fractions())
        ):
            raise ValueError("Fitted view-loss weights differ from configuration")
        if config.schema_version == 1 and (
            "declared_view_loss_weights" in a or "normalized_view_loss_weights" in a
        ):
            raise ValueError("Unexpected view-loss fractions in a legacy model")
        if not np.array_equal(weights, a["view_weights"]) or not np.array_equal(
            a["column_weights"], weights[(kept >= len(config.morphology_features)).astype(int)]
        ):
            raise ValueError("Invalid view balance")
        shapes = {
            "means": (p,),
            "scales": (p,),
            "latent_means": (2,),
            "latent_scales": (2,),
            "losses": (501,),
        }
        for side, widths in (
            ("encoder", [len(kept), 16, 8, 2]),
            ("decoder", [2, 8, 16, len(kept)]),
        ):
            for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                shapes[f"{side}_{i}_weight"] = (right, left)
                shapes[f"{side}_{i}_bias"] = (right,)
        for name, shape in shapes.items():
            if (
                a[name].shape != shape
                or a[name].dtype.kind not in "if"
                or not np.isfinite(a[name]).all()
            ):
                raise ValueError("Invalid model array " + name)
        if np.any(a["scales"] <= 0) or np.any(a["latent_scales"] <= 0):
            raise ValueError("Invalid fitted scales")
        if len(set(summary["unit_ids"])) != len(summary["unit_ids"]):
            raise ValueError("Invalid saved identities")
        n = len(summary["unit_ids"])
        if (
            a["scores"].shape != (n, 2)
            or not np.isfinite(a["scores"]).all()
            or a["labels"].shape != (n,)
            or a["strength"].shape != (n,)
        ):
            raise ValueError("Invalid saved point dimensions")
        if (
            a["labels"].dtype.kind not in "iu"
            or np.any(a["labels"] < -1)
            or not np.isfinite(a["strength"]).all()
            or np.any((a["strength"] < 0) | (a["strength"] > 1))
            or np.any(a["strength"][a["labels"] < 0] != 0)
        ):
            raise ValueError("Invalid saved density assignments")
        if len(set(a["labels"]) - {-1}) != summary["selected_k"]:
            raise ValueError("Saved density count mismatch")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed matched model arrays") from exc
    return MatchedNeuralModel(config, a)
