"""Independent quadratic/order-statistic and source-bound frozen workflow checks."""

# ruff: noqa: E402
import csv
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from super_deli.benchmarking.local_reference_fixture import matrices, reference_fixture
from super_deli.models.local_reference import (
    calibrate_local_support,
    fit_local_support,
    local_distances,
    score_whitened,
)
from super_deli.models.local_reference_config import LocalReferenceSupportConfig
from super_deli.models.reference_abstention import decide, fit_support, transform


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    pytest.importorskip("torch")
    return reference_fixture(tmp_path_factory.mktemp("local_reference_api"))


def setup_support():
    x, y, xc, yc, prep = matrices()
    metric = fit_support(x, y, prep, 4)
    support = fit_local_support(x, y, metric)
    return x, y, xc, yc, metric, support


def test_independent_raw_quadratic_neighbor_and_rank_parity():
    x, y, xc, yc, metric, support = setup_support()
    scores, indices, ties = local_distances(xc, metric, support, provenance=True)
    z, zc = transform(x, metric), transform(xc, metric)
    precision = np.linalg.inv(metric["covariance"])
    for c in range(4):
        train_indices = np.flatnonzero(y == c)
        delta = zc[:, None, :] - z[train_indices][None, :, :]
        direct = np.einsum("nki,ij,nkj->nk", delta, precision, delta)
        order = np.argsort(direct, axis=1, kind="stable")[:, :5]
        np.testing.assert_allclose(
            scores[:, c],
            np.take_along_axis(direct, order[:, -1:], axis=1)[:, 0],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_array_equal(indices[:, c], train_indices[order])
        assert np.all(ties[:, c] >= 1)
    calibrated = calibrate_local_support(support, metric, xc, yc)
    np.testing.assert_allclose(
        calibrated["thresholds"],
        [np.sort(scores[yc == c, c])[-1] for c in range(4)],
        rtol=1e-12,
        atol=1e-12,
    )
    assert calibrated["calibration_ranks"].tolist() == [20] * 4


def test_calibration_and_other_queries_do_not_refit_neighbors():
    _, _, xc, yc, metric, support = setup_support()
    before = {k: v.copy() for k, v in support.items()}
    calibrate_local_support(support, metric, xc + 100, yc)
    for k in support:
        np.testing.assert_array_equal(before[k], support[k])
    expected = local_distances(xc, metric, support)
    np.testing.assert_allclose(
        local_distances(xc[::-1], metric, support)[::-1], expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        local_distances(xc[:1], metric, support)[0], expected[0], rtol=1e-12, atol=1e-12
    )


def test_equal_values_are_distinct_neighbors_and_ties_are_stable():
    lib = dict(whitened_training=np.zeros((12, 2)), training_labels=np.repeat([0, 1], 6))
    score, indices, ties = score_whitened(np.zeros((1, 2)), lib, 2, provenance=True)
    np.testing.assert_array_equal(score, [[0, 0]])
    np.testing.assert_array_equal(indices, [[[0, 1, 2, 3, 4], [6, 7, 8, 9, 10]]])
    np.testing.assert_array_equal(ties, [[6, 6]])
    assert not decide(np.array([[0.9, 0.1]]), score, np.array([0, 0]), "distance")["accepted"][0]


def test_insufficient_library_invalid_coordinates_and_memory_fail_explicitly():
    _, _, xc, _, metric, support = setup_support()
    with pytest.raises(ValueError, match="five"):
        score_whitened(
            np.zeros((1, 12)),
            dict(whitened_training=np.zeros((8, 12)), training_labels=np.repeat([0, 1], 4)),
            2,
        )
    with pytest.raises(ValueError, match="memory"):
        local_distances(xc, metric, support, max_working_bytes=128)
    with pytest.raises(ValueError, match="Finite"):
        score_whitened(np.full((2, 12), np.nan), support, 4)
    with pytest.raises(ValueError, match="integer"):
        score_whitened(np.zeros((1, 12)), support, 4, max_working_bytes=True)


def test_memory_bound_and_chunk_equivalence():
    import tracemalloc

    from threadpoolctl import threadpool_limits

    _, _, xc, _, metric, support = setup_support()
    query = np.tile(xc, (10, 1))
    with threadpool_limits(limits=1):
        expected = local_distances(query, metric, support)
        tracemalloc.start()
        try:
            actual = local_distances(query, metric, support, max_working_bytes=4 * 1024**2)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    np.testing.assert_array_equal(actual, expected)
    assert peak < 4 * 1024**2


@pytest.fixture(scope="module")
def local_calibrated(fitted):
    from super_deli.workflows.local_reference_support import calibrate_local_reference_support

    root, cohorts = fitted
    train, cal = cohorts["train"], cohorts["cal"]
    out = root / "class_local"
    summary = calibrate_local_reference_support(
        train[0],
        cal[0],
        reference_run=root / "model",
        calibration_labels=cal[1] / "labels.csv",
        support_config=LocalReferenceSupportConfig(True),
        training_base_dir=train[1] / "raw",
        calibration_base_dir=cal[1] / "raw",
        output_dir=out,
    )
    assert summary["species_significance"] == "not_computed"
    return root, cohorts, out


def test_api_probabilities_provenance_and_numpy_only_reload(local_calibrated):
    from super_deli.workflows.local_reference_support import (
        load_local_reference_support,
        predict_local_reference_support,
    )

    root, cohorts, out = local_calibrated
    test = cohorts["test"]
    model, _, _, _, _ = load_local_reference_support(out, reference_run=root / "model")
    pred = root / "local_prediction"
    summary = predict_local_reference_support(
        test[0],
        reference_run=root / "model",
        calibration_run=out,
        base_dir=test[1] / "raw",
        output_dir=pred,
    )
    assert summary["status"] == "complete"
    with (pred / "reference-assignments.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    p = model.predict_matrix(matrices()[2] + 2)["probabilities"]
    np.testing.assert_array_equal(
        np.array([[float(r[f"class_{j}_probability"]) for j in range(4)] for r in rows]), p
    )
    assert all(bool(r["accepted_reference_label"]) == (r["decision"] == "assigned") for r in rows)
    code = (
        "import sys;"
        "from super_deli.workflows.local_reference_support import load_local_reference_support;"
        "load_local_reference_support(sys.argv[1],reference_run=sys.argv[2]);"
        'assert "torch" not in sys.modules and "snpio" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code, str(out), str(root / "model")], check=True)


def test_leakage_and_tampering_remain_failures(local_calibrated):
    from super_deli.workflows.local_reference_support import (
        load_local_reference_support,
        predict_local_reference_support,
    )

    root, cohorts, out = local_calibrated
    cal = cohorts["cal"]
    failed = root / "local_leak"
    with pytest.raises(ValueError, match="disjoint"):
        predict_local_reference_support(
            cal[0],
            reference_run=root / "model",
            calibration_run=out,
            base_dir=cal[1] / "raw",
            output_dir=failed,
        )
    assert (failed / "failure.json").exists() and not (failed / "summary.json").exists()
    path = out / "local-support.npz"
    before = path.read_bytes()
    path.write_bytes(before + b"invented corruption")
    try:
        with pytest.raises(ValueError, match="binding"):
            load_local_reference_support(out, reference_run=root / "model")
    finally:
        path.write_bytes(before)


def test_failure_before_final_report_retains_no_completion(fitted, monkeypatch):
    from pathlib import Path

    from super_deli.workflows.local_reference_support import calibrate_local_reference_support

    root, cohorts = fitted
    train, cal = cohorts["train"], cohorts["cal"]
    original = Path.write_text
    out = root / "local_failed_report"

    def fail(path, *args, **kwargs):
        if path == out / "report.html":
            raise OSError("invented report failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="invented"):
        calibrate_local_reference_support(
            train[0],
            cal[0],
            reference_run=root / "model",
            calibration_labels=cal[1] / "labels.csv",
            support_config=LocalReferenceSupportConfig(True),
            training_base_dir=train[1] / "raw",
            calibration_base_dir=cal[1] / "raw",
            output_dir=out,
        )
    assert (out / "failure.json").exists() and not (out / "summary.json").exists()


def test_library_semantics_are_checked_beyond_fingerprints(local_calibrated):
    import json

    from super_deli.core.provenance import fingerprint_file
    from super_deli.workflows.local_reference_support import load_local_reference_support

    root, _, out = local_calibrated
    path = out / "local-support.npz"
    summary_path = out / "summary.json"
    before, meta = path.read_bytes(), summary_path.read_bytes()
    try:
        with np.load(path, allow_pickle=False) as stored:
            a = {k: stored[k] for k in stored.files}
        a["whitened_training"] *= 2
        np.savez_compressed(path, **a)
        summary = json.loads(meta)
        summary["support_sha256"] = fingerprint_file(path).sha256
        summary_path.write_text(json.dumps(summary))
        with pytest.raises(AssertionError):
            load_local_reference_support(out, reference_run=root / "model")
    finally:
        path.write_bytes(before)
        summary_path.write_bytes(meta)


@pytest.mark.parametrize("role", ["training", "calibration"])
def test_label_source_drift_cannot_complete_calibration(fitted, monkeypatch, role):
    from super_deli.workflows import local_reference_support as workflow

    root, cohorts = fitted
    train, cal = cohorts["train"], cohorts["cal"]
    target = root / "model/reference-labels.csv" if role == "training" else cal[1] / "labels.csv"
    original = target.read_bytes()
    real_check = workflow._unchanged
    changed = False

    def inject(sources, before):
        nonlocal changed
        real_check(sources, before)
        if not changed:
            target.write_bytes(original + b"\n")
            changed = True

    monkeypatch.setattr(workflow, "_unchanged", inject)
    out = root / ("label_drift_" + role)
    try:
        with pytest.raises(ValueError, match="labels changed"):
            workflow.calibrate_local_reference_support(
                train[0],
                cal[0],
                reference_run=root / "model",
                calibration_labels=cal[1] / "labels.csv",
                support_config=LocalReferenceSupportConfig(True),
                training_base_dir=train[1] / "raw",
                calibration_base_dir=cal[1] / "raw",
                output_dir=out,
            )
        assert (out / "failure.json").exists() and not (out / "summary.json").exists()
    finally:
        target.write_bytes(original)
