"""Additive known-only density calibration and frozen residual reference prediction."""

import csv
import json
import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.io.reference_labels import read_reference_labels, validate_reference_labels
from super_deli.models.reference_abstention_config import SCOPE, ReferenceAbstentionConfig
from super_deli.models.reference_density_config import ReferenceDensityConfig
from super_deli.workflows.reference_abstention import (
    _failure,
    _loaded,
    _unchanged,
    calibrate_reference_abstention,
    load_reference_abstention,
)

FAMILY = "matched_residual_class_specific_reference_density"


def _hash(path):
    return fingerprint_file(Path(path)).sha256


def _budget(model):
    return min(64 * 1024**2, model.config.max_working_bytes)


def _report(out, title):
    (out / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Reference support</title><h1>"
        + title
        + "</h1><p>Known-only class-specific Gaussian support preserves original classifier "
        "predictions. The .05 rank level concerns true-reference set inclusion under "
        "classwise exchangeability. It does not guarantee accepted accuracy, unfamiliar "
        "rejection or species status. Covariances and thresholds are frozen.</p>",
        encoding="utf-8",
    )


def calibrate_reference_density(
    training_dataset,
    calibration_dataset,
    *,
    reference_run,
    calibration_labels,
    support_config,
    training_base_dir,
    calibration_base_dir,
    output_dir,
):
    import numpy as np

    from super_deli.models.reference_density import calibrate_density_support, fit_density_support

    config = ReferenceDensityConfig.from_dict(support_config.to_dict())
    reference_run = Path(reference_run).resolve()
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        calibrate_reference_abstention(
            training_dataset,
            calibration_dataset,
            reference_run=reference_run,
            calibration_labels=calibration_labels,
            abstention_config=ReferenceAbstentionConfig(config.assume_independent_calibration),
            training_base_dir=training_base_dir,
            calibration_base_dir=calibration_base_dir,
            output_dir=out / "base-calibration",
        )
        model, metric, _, base = load_reference_abstention(
            out / "base-calibration", reference_run=reference_run
        )
        train_out, cal_out = out / "density-training", out / "density-calibration"
        train_out.mkdir()
        cal_out.mkdir()
        x, tids, tsources, thashes = _loaded(training_dataset, model, training_base_dir, train_out)
        xc, cids, csources, chashes = _loaded(
            calibration_dataset, model, calibration_base_dir, cal_out
        )
        if (
            tids != base["training_unit_ids"]
            or cids != base["calibration_unit_ids"]
            or thashes != base["training_source_sha256"]
            or chashes != base["calibration_source_sha256"]
        ):
            raise ValueError("Density sources changed after known-only reference binding")
        classes, y, _ = validate_reference_labels(
            tids, read_reference_labels(reference_run / "reference-labels.csv")
        )
        if classes != model.classes:
            raise ValueError("Density reference class order mismatch")
        fitted = calibrate_density_support(
            fit_density_support(x, np.asarray(y), model.arrays, len(classes)),
            xc,
            metric["calibration_labels"],
            max_working_bytes=_budget(model),
        )
        fitted.update(training_features=x.copy(), training_labels=np.asarray(y, dtype=int))
        if sum(v.nbytes for v in fitted.values()) > _budget(model):
            raise ValueError("Expanded density artifact exceeds memory budget")
        np.savez_compressed(out / "density-support.npz", **fitted)
        _unchanged(tsources, thashes)
        _unchanged(csources, chashes)
        if _hash(calibration_labels) != base["calibration_labels_sha256"]:
            raise ValueError("Known calibration labels changed during density fitting")
        load_reference_abstention(out / "base-calibration", reference_run=reference_run)
        summary = dict(
            schema_version=1,
            status="complete",
            model=FAMILY,
            scope=SCOPE,
            configuration=config.to_dict(),
            configuration_sha256=configuration_hash(config.to_dict()),
            classes=list(model.classes),
            training_unit_ids=tids,
            calibration_unit_ids=cids,
            base_calibration_sha256=_hash(out / "base-calibration/summary.json"),
            support_sha256=_hash(out / "density-support.npz"),
            reference_model_sha256=base["reference_model_sha256"],
            species_significance="not_computed",
            unknown_training_examples=0,
            interpretation=(
                "Known-reference inclusion is not unfamiliar rejection or unique accuracy"
            ),
        )
        _report(out, "Class-specific known-reference support calibrated")
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise


def load_reference_density(calibration_run, *, reference_run):
    import numpy as np

    from super_deli.models.reference_density import calibrate_density_support, fit_density_support

    root = Path(calibration_run)
    reference_run = Path(reference_run)
    if (root / "failure.json").exists():
        raise ValueError("Density calibration has a retained failure")
    meta = read_json(root / "summary.json")
    if (
        type(meta.get("schema_version")) is not int
        or meta.get("schema_version") != 1
        or meta.get("status") != "complete"
        or meta.get("model") != FAMILY
    ):
        raise ValueError("Not a completed class-specific reference calibration")
    config = ReferenceDensityConfig.from_dict(meta["configuration"])
    model, metric, _, base = load_reference_abstention(
        root / "base-calibration", reference_run=reference_run
    )
    if (
        configuration_hash(config.to_dict()) != meta["configuration_sha256"]
        or meta["classes"] != list(model.classes)
        or meta["training_unit_ids"] != base["training_unit_ids"]
        or meta["calibration_unit_ids"] != base["calibration_unit_ids"]
        or meta["base_calibration_sha256"] != _hash(root / "base-calibration/summary.json")
        or meta["support_sha256"] != _hash(root / "density-support.npz")
        or meta["reference_model_sha256"] != base["reference_model_sha256"]
    ):
        raise ValueError("Density reference/source binding mismatch")
    with zipfile.ZipFile(root / "density-support.npz") as archive:
        if sum(i.file_size for i in archive.infolist()) > _budget(model):
            raise ValueError("Expanded density artifact exceeds memory budget")
    with np.load(root / "density-support.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    n, m, c, d, p = (
        len(meta["training_unit_ids"]),
        len(meta["calibration_unit_ids"]),
        len(model.classes),
        len(metric["kept_indices"]),
        len(metric["means"]),
    )
    shapes = dict(
        means=(p,),
        scales=(p,),
        kept_indices=(d,),
        column_weights=(d,),
        centers=(c, d),
        unshrunk_covariances=(c, d, d),
        covariances=(c, d, d),
        cholesky=(c, d, d),
        log_determinants=(c,),
        training_counts=(c,),
        training_features=(n, p),
        training_labels=(n,),
        calibration_features=(m, p),
        calibration_labels=(m,),
        calibration_scores=(m, c),
        thresholds=(c,),
        calibration_counts=(c,),
        calibration_ranks=(c,),
    )
    if a.keys() != shapes.keys() or any(
        v.shape != shapes[k] or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
        for k, v in a.items()
    ):
        raise ValueError("Invalid numeric density-support schema")
    for k in [
        "kept_indices",
        "training_counts",
        "training_labels",
        "calibration_labels",
        "calibration_counts",
        "calibration_ranks",
    ]:
        if a[k].dtype.kind not in "iu":
            raise ValueError("Density identities, counts and ranks must be integers")
    classes, y, _ = validate_reference_labels(
        meta["training_unit_ids"], read_reference_labels(reference_run / "reference-labels.csv")
    )
    if classes != model.classes:
        raise ValueError("Original density training class order mismatch")
    np.testing.assert_array_equal(a["training_labels"], y)
    np.testing.assert_array_equal(a["calibration_labels"], metric["calibration_labels"])
    original = fit_density_support(a["training_features"], a["training_labels"], model.arrays, c)
    expected = calibrate_density_support(
        original,
        a["calibration_features"],
        a["calibration_labels"],
        max_working_bytes=_budget(model),
    )
    for k, v in expected.items():
        np.testing.assert_allclose(a[k], v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a["centers"], metric["centers"], rtol=1e-10, atol=1e-10)
    pooled = np.einsum("c,cij->ij", a["training_counts"], a["unshrunk_covariances"]) / n
    np.testing.assert_allclose(pooled, metric["unshrunk_covariance"], rtol=1e-10, atol=1e-10)
    return model, metric, a, config, meta


def predict_reference_density(dataset, *, reference_run, calibration_run, base_dir, output_dir):
    import numpy as np

    from super_deli.models.reference_abstention import decide
    from super_deli.models.reference_density import density_scores

    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        model, _, support, _, meta = load_reference_density(
            calibration_run, reference_run=reference_run
        )
        calibration_hash = _hash(Path(calibration_run) / "summary.json")
        x, ids, sources, before = _loaded(dataset, model, base_dir, out)
        if set(ids) & (set(meta["training_unit_ids"]) | set(meta["calibration_unit_ids"])):
            raise ValueError("Prediction specimens must be disjoint from training and calibration")
        p = model.predict_matrix(x)["probabilities"]
        scores = density_scores(x, support, max_working_bytes=_budget(model))
        result = decide(p, scores, support["thresholds"], "distance")
        np.savez_compressed(out / "support-predictions.npz", scores=scores, **result)
        with (out / "reference-assignments.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "unit_id",
                    "predicted_reference_label",
                    "accepted_reference_label",
                    "decision",
                    "compatible_references",
                    "compatible_reference_count",
                    "scope",
                    *[f"class_{j}_probability" for j in range(len(model.classes))],
                ]
            )
            for i, u in enumerate(ids):
                compatible = [
                    c for c, ok in zip(model.classes, result["compatible"][i], strict=True) if ok
                ]
                writer.writerow(
                    [
                        u,
                        model.classes[result["predicted_indices"][i]],
                        model.classes[result["accepted_indices"][i]]
                        if result["accepted"][i]
                        else "",
                        result["decisions"][i],
                        json.dumps(compatible),
                        len(compatible),
                        SCOPE,
                        *p[i],
                    ]
                )
        _unchanged(sources, before)
        if calibration_hash != _hash(Path(calibration_run) / "summary.json"):
            raise ValueError("Density calibration changed during prediction")
        load_reference_density(calibration_run, reference_run=reference_run)
        summary = dict(
            schema_version=1,
            status="complete",
            model=FAMILY,
            scope=SCOPE,
            unit_ids=ids,
            classes=list(model.classes),
            source_sha256=before,
            calibration_sha256=calibration_hash,
            reference_model_sha256=meta["reference_model_sha256"],
            species_significance="not_computed",
            decisions={
                s: int(np.sum(result["decisions"] == s)) for s in sorted(set(result["decisions"]))
            },
        )
        _report(out, "Frozen known-reference assignments")
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise
