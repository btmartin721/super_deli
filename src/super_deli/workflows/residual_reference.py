"""Source-bound supervised training and strict numeric frozen reference prediction."""

import csv
import html
import shutil
import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input
from super_deli.io.config import read_json, write_json
from super_deli.io.reference_labels import read_reference_labels
from super_deli.io.tabular import ingest_tabular
from super_deli.models.residual_config import FAMILY, TRAINING, ResidualConfig


def _tables(dataset, config, base_dir, out):
    selected = {
        m.name: m for m in dataset.modalities if m.name in {config.morphology, config.ecology}
    }
    if set(selected) != {config.morphology, config.ecology}:
        raise ValueError("Both declared reference-classifier views are required")
    sources = {name: resolve_input(m.path, base_dir) for name, m in selected.items()}
    before = {name: fingerprint_file(p).sha256 for name, p in sources.items()}
    loaded = ingest_tabular(
        dataset,
        base_dir=base_dir,
        output_dir=out / "ingestion",
        modality_names=[config.morphology, config.ecology],
    )
    return loaded, sources, before


def _predictions(out, result, classes):
    with (out / "reference-predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "unit_id",
                "predicted_reference_label",
                "scope",
                *[f"class_{j}_probability" for j in range(len(classes))],
            ]
        )
        writer.writerows(
            (unit, label, "reference_only_novelty_unassessed", *p)
            for unit, label, p in zip(
                result["unit_ids"],
                result["predicted_reference_labels"],
                result["probabilities"],
                strict=True,
            )
        )
    write_json(
        {"class_order": list(classes), "interpretation": TRAINING["scope"]}, out / "classes.json"
    )


def _failure(out, exc):
    import numpy as np

    predictions = out / "reference-predictions.csv"
    if predictions.exists():
        predictions.rename(out / "failed-reference-predictions.csv")
    if arrays := getattr(exc, "arrays", {}):
        np.savez_compressed(out / "failed-training.npz", **arrays)
    write_json(
        {"status": "failed", "error": str(exc), "diagnostics": getattr(exc, "diagnostics", {})},
        out / "failure.json",
    )


def train_matched_residual(
    dataset, *, labels_path, model_config, base_dir, output_dir, _observer=None
):
    import numpy as np

    from super_deli.models.residual_network import fit_residual_reference

    dataset = config_from_dict(dataset.to_dict())
    config = ResidualConfig.from_dict(model_config.to_dict())
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        labels_path = Path(labels_path).resolve()
        label_hash = fingerprint_file(labels_path).sha256
        shutil.copyfile(labels_path, out / "reference-labels.csv")
        labels = read_reference_labels(out / "reference-labels.csv")
        loaded, sources, before = _tables(dataset, config, base_dir, out)
        model, diagnostics, alignment = fit_residual_reference(
            loaded.tables, labels, config=config, observer=_observer
        )
        np.savez_compressed(out / "model.npz", **model.arrays)
        write_json(alignment, out / "cohort.json")
        write_json(diagnostics, out / "training.json")
        _predictions(out, model.predict(loaded.tables), model.classes)
        if label_hash != fingerprint_file(labels_path).sha256 or before != {
            name: fingerprint_file(p).sha256 for name, p in sources.items()
        }:
            raise ValueError("Reference inputs changed during training")
        configuration = {
            "dataset": dataset.to_dict(),
            "model": config.to_dict(),
            "training": TRAINING,
        }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": FAMILY,
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "classes": list(model.classes),
            "unit_ids": alignment["unit_ids"],
            "source_sha256": before,
            "labels_sha256": label_hash,
            "model_sha256": fingerprint_file(out / "model.npz").sha256,
            "training": diagnostics,
            "scope": TRAINING["scope"],
            "species_significance": "not_computed",
            "evaluation": "Training assignments only; generalization has not been evaluated",
        }
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Reference classifier</title>"
            "<h1>Supervised reference assignment</h1><p>"
            + html.escape(TRAINING["scope"])
            + "</p><p>Training specimens: "
            + str(len(alignment["unit_ids"]))
            + ". Reference classes: "
            + str(len(model.classes))
            + ". These training predictions are not an independent evaluation.</p>",
            encoding="utf-8",
        )
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise


def load_residual_reference_model(directory):
    import numpy as np

    from super_deli.models.residual_network import ResidualReferenceModel, initial_weights

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("model") != FAMILY
        or summary.get("schema_version") != 1
    ):
        raise ValueError("Not a completed residual reference model")
    configuration = summary["configuration"]
    config = ResidualConfig.from_dict(configuration["model"])
    classes = summary["classes"]
    if (
        configuration.get("training") != TRAINING
        or configuration_hash(configuration) != summary["configuration_sha256"]
        or fingerprint_file(directory / "model.npz").sha256 != summary["model_sha256"]
    ):
        raise ValueError("Reference model/configuration fingerprint mismatch")
    if (
        not isinstance(classes, list)
        or any(not isinstance(v, str) or not v or v != v.strip() for v in classes)
        or not 2 <= len(classes) <= 32
        or sorted(set(classes)) != classes
    ):
        raise ValueError("Invalid fitted reference class order")
    with zipfile.ZipFile(directory / "model.npz") as zipped:
        if sum(i.file_size for i in zipped.infolist()) > min(67108864, config.max_working_bytes):
            raise ValueError("Reference model expanded size exceeds budget")
    with np.load(directory / "model.npz", allow_pickle=False) as stored:
        a = {k: stored[k] for k in stored.files}
    d = len(config.morphology_features) + len(config.ecology_features)
    prep_shapes = {
        "means": (d,),
        "scales": (d,),
        "selected_view_counts": (2,),
        "retained_view_counts": (2,),
        "view_weights": (2,),
        "losses": (501,),
        "regularization": (501,),
    }
    if "kept_indices" not in a or a["kept_indices"].dtype.kind not in "iu":
        raise ValueError("Invalid fitted feature indices")
    kept = a["kept_indices"]
    if (
        kept.ndim != 1
        or not 6 <= len(kept) <= d
        or not np.array_equal(kept, np.unique(kept))
        or kept[0] < 0
        or kept[-1] >= d
    ):
        raise ValueError("Invalid fitted feature selection")
    weights = initial_weights(len(kept), len(classes), config.random_seed)
    shapes = {
        **prep_shapes,
        "kept_indices": kept.shape,
        "column_weights": kept.shape,
        **{k: v.shape for k, v in weights.items()},
        **{"initial_" + k: v.shape for k, v in weights.items()},
    }
    if a.keys() != shapes.keys() or any(
        v.shape != shapes[k] or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
        for k, v in a.items()
    ):
        raise ValueError("Invalid numeric reference-model schema")
    split = len(config.morphology_features)
    counts = np.array([sum(kept < split), sum(kept >= split)])
    if (
        np.any(counts < 3)
        or np.any(a["scales"] <= 0)
        or np.any(a["losses"] < 0)
        or np.any(a["regularization"] < 0)
    ):
        raise ValueError("Invalid reference scaling/losses")
    np.testing.assert_array_equal(a["selected_view_counts"], [split, d - split])
    np.testing.assert_array_equal(a["retained_view_counts"], counts)
    expected = np.sqrt(len(kept) / (2 * counts))
    np.testing.assert_array_equal(a["view_weights"], expected)
    np.testing.assert_array_equal(a["column_weights"], expected[(kept >= split).astype(int)])
    for k, v in weights.items():
        np.testing.assert_array_equal(a["initial_" + k], v)
    return ResidualReferenceModel(config, tuple(classes), a)


def predict_matched_residual(dataset, *, reference_run, base_dir, output_dir):
    dataset = config_from_dict(dataset.to_dict())
    model = load_residual_reference_model(reference_run)
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        loaded, sources, before = _tables(dataset, model.config, base_dir, out)
        result = model.predict(loaded.tables)
        _predictions(out, result, model.classes)
        if before != {name: fingerprint_file(p).sha256 for name, p in sources.items()}:
            raise ValueError("Prediction inputs changed during inference")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "model": FAMILY,
            "scope": TRAINING["scope"],
            "classes": list(model.classes),
            "unit_ids": result["unit_ids"],
            "source_sha256": before,
            "reference_model_sha256": fingerprint_file(Path(reference_run) / "model.npz").sha256,
            "species_significance": "not_computed",
        }
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise
