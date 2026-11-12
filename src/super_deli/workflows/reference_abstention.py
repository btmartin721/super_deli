"""Additive, source-bound calibration and frozen reference-assignment abstention."""

import csv
import json
import shutil
import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json, write_json
from super_deli.io.reference_labels import read_reference_labels, validate_reference_labels
from super_deli.models.reference_abstention_config import SCOPE, ReferenceAbstentionConfig
from super_deli.workflows.residual_reference import _tables, load_residual_reference_model

FAMILY = "matched_residual_reference_abstention"


def _loaded(dataset, model, base, directory):
    from super_deli.models.reference_projection import aligned_reference_matrix

    loaded, sources, hashes = _tables(
        config_from_dict(dataset.to_dict()), model.config, base, directory
    )
    x, ids = aligned_reference_matrix(loaded.tables, model.config)
    return x, ids, sources, hashes


def _unchanged(sources, before):
    if {name: fingerprint_file(p).sha256 for name, p in sources.items()} != before:
        raise ValueError("Reference-support inputs changed during use")


def _failure(out, exc):
    p = out / "reference-assignments.csv"
    if p.exists():
        p.rename(out / "failed-reference-assignments.csv")
    write_json(dict(status="failed", error=str(exc), scope=SCOPE), out / "failure.json")


def calibrate_reference_abstention(
    training_dataset,
    calibration_dataset,
    *,
    reference_run,
    calibration_labels,
    abstention_config,
    training_base_dir,
    calibration_base_dir,
    output_dir,
):
    import numpy as np

    from super_deli.models.reference_abstention import calibrate, fit_support

    config = ReferenceAbstentionConfig.from_dict(abstention_config.to_dict())
    reference_run, out = Path(reference_run).resolve(), Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        if (reference_run / "failure.json").exists():
            raise ValueError("Reference classifier has a retained failure")
        model = load_residual_reference_model(reference_run)
        original = read_json(reference_run / "summary.json")
        reference_hash = fingerprint_file(reference_run / "summary.json").sha256
        train_out, calibration_out = out / "training", out / "calibration"
        train_out.mkdir()
        calibration_out.mkdir()
        train, train_ids, train_sources, train_hashes = _loaded(
            training_dataset, model, training_base_dir, train_out
        )
        if train_hashes != original["source_sha256"] or train_ids != original["unit_ids"]:
            raise ValueError("Support training specimens/sources differ from original classifier")
        training_labels = reference_run / "reference-labels.csv"
        if fingerprint_file(training_labels).sha256 != original["labels_sha256"]:
            raise ValueError("Original training label fingerprint mismatch")
        classes, y, _ = validate_reference_labels(train_ids, read_reference_labels(training_labels))
        if classes != model.classes:
            raise ValueError("Original reference class order mismatch")
        cal, cal_ids, cal_sources, cal_hashes = _loaded(
            calibration_dataset, model, calibration_base_dir, calibration_out
        )
        if set(train_ids) & set(cal_ids):
            raise ValueError("Training and calibration specimen identities overlap")
        label_hash = fingerprint_file(Path(calibration_labels)).sha256
        shutil.copyfile(calibration_labels, out / "calibration-labels.csv")
        cal_classes, cal_y, counts = validate_reference_labels(
            cal_ids, read_reference_labels(out / "calibration-labels.csv")
        )
        if cal_classes != classes:
            raise ValueError("Calibration requires exactly the original reference classes")
        support = fit_support(train, np.asarray(y), model.arrays, len(classes))
        fitted = calibrate(
            support, cal, np.asarray(cal_y), model.predict_matrix(cal)["probabilities"]
        )
        np.savez_compressed(out / "support.npz", **fitted)
        _unchanged(train_sources, train_hashes)
        _unchanged(cal_sources, cal_hashes)
        if fingerprint_file(Path(calibration_labels)).sha256 != label_hash:
            raise ValueError("Calibration labels changed during fitting")
        if fingerprint_file(reference_run / "summary.json").sha256 != reference_hash:
            raise ValueError("Reference classifier changed during calibration")
        summary = dict(
            schema_version=1,
            status="complete",
            model=FAMILY,
            scope=SCOPE,
            configuration=config.to_dict(),
            configuration_sha256=configuration_hash(config.to_dict()),
            reference_configuration=model.config.to_dict(),
            classes=list(classes),
            training_unit_ids=train_ids,
            calibration_unit_ids=cal_ids,
            training_source_sha256=train_hashes,
            calibration_source_sha256=cal_hashes,
            calibration_labels_sha256=label_hash,
            calibration_class_counts=counts,
            reference_summary_sha256=reference_hash,
            reference_model_sha256=original["model_sha256"],
            support_sha256=fingerprint_file(out / "support.npz").sha256,
            species_significance="not_computed",
            interpretation=(
                "Known-class compatibility under exchangeability; "
                "no unfamiliar-group acceptance guarantee"
            ),
        )
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Reference support</title>"
            "<h1>Reference-assignment support</h1><p>Calibration is complete. "
            "Predictions may be withheld when reference support is insufficient or ambiguous. "
            "Withholding does not establish a new species. The .05 calibration level is not a "
            "guarantee about unfamiliar specimens or accuracy among accepted assignments.</p>",
            encoding="utf-8",
        )
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise


def load_reference_abstention(calibration_run, *, reference_run):
    import numpy as np

    from super_deli.models.reference_abstention import PREPROCESSING, probabilities, rank_thresholds

    root, reference_run = Path(calibration_run), Path(reference_run)
    if (root / "failure.json").exists() or (reference_run / "failure.json").exists():
        raise ValueError("Reference model/calibration has a retained failure")
    meta = read_json(root / "summary.json")
    if (
        meta.get("status") != "complete"
        or meta.get("model") != FAMILY
        or meta.get("schema_version") != 1
    ):
        raise ValueError("Not a completed reference-support calibration")
    config = ReferenceAbstentionConfig.from_dict(meta["configuration"])
    model = load_residual_reference_model(reference_run)
    if (
        configuration_hash(config.to_dict()) != meta["configuration_sha256"]
        or configuration_hash(model.config.to_dict())
        != configuration_hash(meta["reference_configuration"])
        or list(model.classes) != meta["classes"]
        or fingerprint_file(reference_run / "summary.json").sha256
        != meta["reference_summary_sha256"]
        or fingerprint_file(reference_run / "model.npz").sha256 != meta["reference_model_sha256"]
        or fingerprint_file(root / "support.npz").sha256 != meta["support_sha256"]
    ):
        raise ValueError("Reference/support model binding mismatch")
    for name in ("training_unit_ids", "calibration_unit_ids"):
        v = meta[name]
        if (
            not isinstance(v, list)
            or any(not isinstance(u, str) or not u or u != u.strip() for u in v)
            or sorted(set(v)) != v
        ):
            raise ValueError("Invalid retained reference identities")
    if set(meta["training_unit_ids"]) & set(meta["calibration_unit_ids"]):
        raise ValueError("Retained calibration overlaps training")
    label_file = root / "calibration-labels.csv"
    if fingerprint_file(label_file).sha256 != meta["calibration_labels_sha256"]:
        raise ValueError("Calibration label fingerprint mismatch")
    label_classes, label_indices, label_counts = validate_reference_labels(
        meta["calibration_unit_ids"], read_reference_labels(label_file)
    )
    if label_classes != model.classes or label_counts != meta["calibration_class_counts"]:
        raise ValueError("Calibration label/class identity mismatch")
    original = read_json(reference_run / "summary.json")
    if meta["training_unit_ids"] != original["unit_ids"]:
        raise ValueError("Retained support identities differ from classifier training")
    with zipfile.ZipFile(root / "support.npz") as zipped:
        if sum(i.file_size for i in zipped.infolist()) > min(
            67108864, model.config.max_working_bytes
        ):
            raise ValueError("Expanded reference-support artifact exceeds budget")
    with np.load(root / "support.npz", allow_pickle=False) as stored:
        a = {k: stored[k] for k in stored.files}
    c, d, m = (
        len(model.classes),
        len(model.arrays["kept_indices"]),
        len(meta["calibration_unit_ids"]),
    )
    shapes = {k: model.arrays[k].shape for k in PREPROCESSING}
    shapes.update(
        centers=(c, d),
        covariance=(d, d),
        unshrunk_covariance=(d, d),
        cholesky=(d, d),
        training_counts=(c,),
        calibration_counts=(c,),
        calibration_ranks=(c,),
        distance_thresholds=(c,),
        probability_thresholds=(c,),
        calibration_distances=(m, c),
        calibration_probabilities=(m, c),
        calibration_labels=(m,),
    )
    if a.keys() != shapes.keys() or any(
        v.shape != shapes[k] or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
        for k, v in a.items()
    ):
        raise ValueError("Invalid numeric reference-support schema")
    for k in PREPROCESSING:
        np.testing.assert_array_equal(a[k], model.arrays[k])
    if np.any(a["training_counts"] < 2) or a["training_counts"].sum() != len(
        meta["training_unit_ids"]
    ):
        raise ValueError("Invalid training counts")
    for key in ("training_counts", "calibration_counts", "calibration_ranks", "calibration_labels"):
        if a[key].dtype.kind not in "iu":
            raise ValueError("Reference counts/labels must be integers")
    probabilities(a["calibration_probabilities"], n=m, classes=c)
    np.testing.assert_array_equal(a["calibration_labels"], label_indices)
    if np.any(a["calibration_distances"] < 0):
        raise ValueError("Invalid retained calibration distances")
    cov = a["unshrunk_covariance"]
    if np.trace(cov) <= 0:
        raise ValueError("Invalid within-reference covariance")
    if np.linalg.eigvalsh(cov).min() < -1e-12 * np.trace(cov):
        raise ValueError("Invalid within-reference covariance spectrum")
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        a["covariance"], 0.9 * cov + 0.1 * np.trace(cov) / d * np.eye(d), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        a["cholesky"], np.linalg.cholesky(a["covariance"]), rtol=1e-12, atol=1e-12
    )
    for policy, score in (
        ("distance", a["calibration_distances"]),
        ("probability", 1 - a["calibration_probabilities"]),
    ):
        q, counts, ranks = rank_thresholds(score, a["calibration_labels"])
        for key, expected in (
            (policy + "_thresholds", q),
            ("calibration_counts", counts),
            ("calibration_ranks", ranks),
        ):
            np.testing.assert_array_equal(a[key], expected)
    if meta["calibration_class_counts"] != dict(
        zip(model.classes, a["calibration_counts"].tolist(), strict=True)
    ):
        raise ValueError("Calibration class accounting mismatch")
    if (
        dict(zip(model.classes, a["training_counts"].tolist(), strict=True))
        != original["training"]["reference_class_counts"]
    ):
        raise ValueError("Original training class accounting mismatch")
    return model, a, config, meta


def predict_reference_abstention(dataset, *, reference_run, calibration_run, base_dir, output_dir):
    from super_deli.models.reference_abstention import decide, distances

    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        model, support, config, meta = load_reference_abstention(
            calibration_run, reference_run=reference_run
        )
        calibration_hash = fingerprint_file(Path(calibration_run) / "summary.json").sha256
        x, ids, sources, before = _loaded(dataset, model, base_dir, out)
        if set(ids) & (set(meta["training_unit_ids"]) | set(meta["calibration_unit_ids"])):
            raise ValueError("Prediction specimens must be disjoint from training and calibration")
        prediction = model.predict_matrix(x)
        result = decide(
            prediction["probabilities"],
            distances(x, support),
            support[config.policy + "_thresholds"],
            config.policy,
        )
        with (out / "reference-assignments.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "unit_id",
                    "predicted_reference_label",
                    "accepted_reference_label",
                    "decision",
                    "compatible_references",
                    "scope",
                    *[f"class_{j}_probability" for j in range(len(model.classes))],
                ]
            )
            for i, unit in enumerate(ids):
                compatible = [
                    c for c, ok in zip(model.classes, result["compatible"][i], strict=True) if ok
                ]
                writer.writerow(
                    [
                        unit,
                        model.classes[result["predicted_indices"][i]],
                        model.classes[result["accepted_indices"][i]]
                        if result["accepted"][i]
                        else "",
                        result["decisions"][i],
                        json.dumps(compatible),
                        SCOPE,
                        *prediction["probabilities"][i],
                    ]
                )
        _unchanged(sources, before)
        if (
            fingerprint_file(Path(calibration_run) / "summary.json").sha256 != calibration_hash
            or fingerprint_file(Path(reference_run) / "model.npz").sha256
            != meta["reference_model_sha256"]
            or fingerprint_file(Path(reference_run) / "summary.json").sha256
            != meta["reference_summary_sha256"]
            or fingerprint_file(Path(calibration_run) / "support.npz").sha256
            != meta["support_sha256"]
        ):
            raise ValueError("Reference calibration/model changed during prediction")
        summary = dict(
            schema_version=1,
            status="complete",
            model=FAMILY,
            scope=SCOPE,
            policy=config.policy,
            classes=list(model.classes),
            unit_ids=ids,
            source_sha256=before,
            reference_model_sha256=meta["reference_model_sha256"],
            calibration_sha256=calibration_hash,
            decisions={
                s: int(sum(result["decisions"] == s)) for s in sorted(set(result["decisions"]))
            },
            species_significance="not_computed",
        )
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Reference assignments</title>"
            "<h1>Reference assignments</h1><p>Accepted assignments: "
            + str(summary["decisions"].get("assigned", 0))
            + " of "
            + str(len(ids))
            + ". The CSV retains the original classifier prediction and the reason for "
            "each withheld assignment. Withholding does not establish a new species.</p>",
            encoding="utf-8",
        )
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise
