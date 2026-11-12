"""Additive class-local calibration with the predecessor's strict provenance contract."""

import csv
import json
import zipfile
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.io.reference_labels import read_reference_labels, validate_reference_labels
from super_deli.models.local_reference_config import LocalReferenceSupportConfig
from super_deli.models.reference_abstention_config import SCOPE, ReferenceAbstentionConfig
from super_deli.workflows.reference_abstention import (
    _failure,
    _loaded,
    _unchanged,
    calibrate_reference_abstention,
    load_reference_abstention,
)

FAMILY = "matched_residual_class_local_reference_support"


def _hash(path):
    return fingerprint_file(Path(path)).sha256


def _budget(model):
    return min(64 * 1024**2, model.config.max_working_bytes)


def calibrate_local_reference_support(
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

    from super_deli.models.local_reference import calibrate_local_support, fit_local_support

    config = LocalReferenceSupportConfig.from_dict(support_config.to_dict())
    reference_run, out = Path(reference_run).resolve(), Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        # Compose the tested predecessor binding; never reinterpret its artifact schema.
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
        for name in ("local-training", "local-calibration"):
            (out / name).mkdir()
        train, tids, tsources, thashes = _loaded(
            training_dataset, model, training_base_dir, out / "local-training"
        )
        cal, cids, csources, chashes = _loaded(
            calibration_dataset, model, calibration_base_dir, out / "local-calibration"
        )
        if (
            tids != base["training_unit_ids"]
            or cids != base["calibration_unit_ids"]
            or thashes != base["training_source_sha256"]
            or chashes != base["calibration_source_sha256"]
        ):
            raise ValueError("Class-local sources changed after reference binding")
        classes, y, _ = validate_reference_labels(
            tids, read_reference_labels(reference_run / "reference-labels.csv")
        )
        if classes != model.classes:
            raise ValueError("Class-local training class order mismatch")
        fitted = calibrate_local_support(
            fit_local_support(train, np.asarray(y), metric),
            metric,
            cal,
            metric["calibration_labels"],
            max_working_bytes=_budget(model),
        )
        if sum(v.nbytes for v in fitted.values()) > _budget(model):
            raise ValueError("Expanded class-local support exceeds artifact budget")
        np.savez_compressed(out / "local-support.npz", **fitted)
        _unchanged(tsources, thashes)
        _unchanged(csources, chashes)
        original_labels_hash = read_json(reference_run / "summary.json")["labels_sha256"]
        if (
            _hash(calibration_labels) != base["calibration_labels_sha256"]
            or _hash(reference_run / "reference-labels.csv") != original_labels_hash
        ):
            raise ValueError("Reference labels changed during class-local calibration")
        # Reload also checks copied labels, reference bytes and full rank accounting.
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
            support_sha256=_hash(out / "local-support.npz"),
            reference_model_sha256=base["reference_model_sha256"],
            species_significance="not_computed",
        )
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Class-local reference support</title>"
            "<h1>Class-local reference support</h1><p>Calibration completed. Compatible references "
            "can overlap. The nominal level concerns true-known-reference set inclusion under "
            "the stated sampling assumptions; it does not certify accepted accuracy, unfamiliar "
            "detection or species. Prediction retains the classifier's original label.</p>",
            encoding="utf-8",
        )
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise


def load_local_reference_support(calibration_run, *, reference_run):
    import numpy as np

    from super_deli.models.local_reference import score_whitened
    from super_deli.models.reference_abstention import rank_thresholds

    root, reference_run = Path(calibration_run), Path(reference_run)
    if (root / "failure.json").exists():
        raise ValueError("Class-local calibration has a retained failure")
    meta = read_json(root / "summary.json")
    if (
        meta.get("model") != FAMILY
        or meta.get("status") != "complete"
        or meta.get("schema_version") != 1
    ):
        raise ValueError("Not a completed class-local calibration")
    config = LocalReferenceSupportConfig.from_dict(meta["configuration"])
    model, metric, _, base = load_reference_abstention(
        root / "base-calibration", reference_run=reference_run
    )
    if (
        configuration_hash(config.to_dict()) != meta["configuration_sha256"]
        or meta["classes"] != list(model.classes)
        or meta["training_unit_ids"] != base["training_unit_ids"]
        or meta["calibration_unit_ids"] != base["calibration_unit_ids"]
        or meta["base_calibration_sha256"] != _hash(root / "base-calibration/summary.json")
        or meta["reference_model_sha256"] != base["reference_model_sha256"]
        or meta["support_sha256"] != _hash(root / "local-support.npz")
    ):
        raise ValueError("Class-local reference/support binding mismatch")
    original = read_json(reference_run / "summary.json")
    if _hash(reference_run / "reference-labels.csv") != original["labels_sha256"]:
        raise ValueError("Original training label fingerprint mismatch")
    classes, y, _ = validate_reference_labels(
        meta["training_unit_ids"], read_reference_labels(reference_run / "reference-labels.csv")
    )
    if classes != model.classes:
        raise ValueError("Original class-local training class order mismatch")
    with zipfile.ZipFile(root / "local-support.npz") as archive:
        if sum(i.file_size for i in archive.infolist()) > _budget(model):
            raise ValueError("Expanded class-local support exceeds artifact budget")
    with np.load(root / "local-support.npz", allow_pickle=False) as stored:
        a = {key: stored[key] for key in stored.files}
    n, m, c, d = (
        len(meta["training_unit_ids"]),
        len(meta["calibration_unit_ids"]),
        len(model.classes),
        len(metric["kept_indices"]),
    )
    shapes = dict(
        whitened_training=(n, d),
        training_labels=(n,),
        whitened_calibration=(m, d),
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
        raise ValueError("Invalid numeric class-local support schema")
    for key in ("training_labels", "calibration_labels", "calibration_counts", "calibration_ranks"):
        if a[key].dtype.kind not in "iu":
            raise ValueError("Class-local identities/counts/ranks must be integers")
    np.testing.assert_array_equal(a["training_labels"], y)
    np.testing.assert_array_equal(a["calibration_labels"], metric["calibration_labels"])
    counts = np.bincount(a["training_labels"], minlength=c)
    if np.any(counts < 5):
        raise ValueError("At least five original training specimens per reference required")
    np.testing.assert_array_equal(counts, metric["training_counts"])
    centers = np.stack(
        [a["whitened_training"][a["training_labels"] == j].mean(0) for j in range(c)]
    )
    np.testing.assert_allclose(
        centers, np.linalg.solve(metric["cholesky"], metric["centers"].T).T, rtol=1e-10, atol=1e-10
    )
    residual = a["whitened_training"] - centers[a["training_labels"]]
    inverse = np.linalg.solve(metric["cholesky"], np.eye(d))
    np.testing.assert_allclose(
        residual.T @ residual / n,
        inverse @ metric["unshrunk_covariance"] @ inverse.T,
        rtol=1e-10,
        atol=1e-10,
    )
    scores = score_whitened(a["whitened_calibration"], a, c, max_working_bytes=_budget(model))
    np.testing.assert_allclose(scores, a["calibration_scores"], rtol=1e-12, atol=1e-12)
    q, counts, ranks = rank_thresholds(a["calibration_scores"], a["calibration_labels"])
    for key, value in (
        ("thresholds", q),
        ("calibration_counts", counts),
        ("calibration_ranks", ranks),
    ):
        np.testing.assert_array_equal(a[key], value)
    return model, metric, a, config, meta


def predict_local_reference_support(
    dataset, *, reference_run, calibration_run, base_dir, output_dir
):
    import numpy as np

    from super_deli.models.local_reference import local_distances
    from super_deli.models.reference_abstention import decide

    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    try:
        model, metric, support, _, meta = load_local_reference_support(
            calibration_run, reference_run=reference_run
        )
        calibration_hash = _hash(Path(calibration_run) / "summary.json")
        x, ids, sources, before = _loaded(dataset, model, base_dir, out)
        if set(ids) & (set(meta["training_unit_ids"]) | set(meta["calibration_unit_ids"])):
            raise ValueError("Prediction specimens must be disjoint from training and calibration")
        p = model.predict_matrix(x)["probabilities"]
        scores, neighbors, ties = local_distances(
            x, metric, support, max_working_bytes=_budget(model), provenance=True
        )
        result = decide(p, scores, support["thresholds"], "distance")
        np.savez_compressed(
            out / "support-predictions.npz",
            scores=scores,
            neighbor_indices=neighbors,
            cutoff_tie_counts=ties,
            **result,
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
                    "compatible_reference_count",
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
                        len(compatible),
                        SCOPE,
                        *p[i],
                    ]
                )
        _unchanged(sources, before)
        if calibration_hash != _hash(Path(calibration_run) / "summary.json"):
            raise ValueError("Class-local calibration changed during prediction")
        # Full end-of-use binding check, including reference and numeric support bytes.
        load_local_reference_support(calibration_run, reference_run=reference_run)
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
            decisions={
                s: int(np.sum(result["decisions"] == s)) for s in sorted(set(result["decisions"]))
            },
            species_significance="not_computed",
        )
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Class-local reference assignments</title>"
            "<h1>Reference assignments</h1><p>The CSV preserves the original prediction "
            "and explains "
            "withholding. Multiple compatible references indicate ambiguity; "
            "no compatible reference "
            "indicates insufficient support. Neither establishes a new species.</p>",
            encoding="utf-8",
        )
        write_json(summary, out / "summary.json")
        return summary
    except Exception as exc:
        _failure(out, exc)
        raise
