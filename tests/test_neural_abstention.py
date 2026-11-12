"""Independent matrix checks and meaningful source/role/frozen-workflow regression."""

# ruff: noqa: E402
import csv
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from super_deli.models.matched_preprocessing import prepare_views
from super_deli.models.reference_abstention import (
    calibrate,
    decide,
    distances,
    fit_support,
    rank_thresholds,
)
from super_deli.models.reference_abstention_config import DECISIONS, ReferenceAbstentionConfig


def matrices():
    rng = np.random.default_rng(883471)
    y = np.repeat(np.arange(4), 8)
    x = rng.normal(size=(32, 12)) + 2 * y[:, None]
    yc = np.repeat(np.arange(4), 20)
    xc = rng.normal(size=(80, 12)) + 2 * yc[:, None]
    _, prep = prepare_views(x, 4)
    return x, y, xc, yc, prep


def test_independent_quadratic_and_training_only_fit():
    x, y, xc, yc, prep = matrices()
    s = fit_support(x, y, prep, 4)
    z = (x - prep["means"]) / prep["scales"] * prep["column_weights"]
    centers = np.stack([z[y == c].mean(0) for c in range(4)])
    residual = z - centers[y]
    covariance = np.einsum("ni,nj->ij", residual, residual) / len(x)
    shrunk = 0.9 * covariance + 0.1 * np.trace(covariance) / 12 * np.eye(12)
    zz = (xc - prep["means"]) / prep["scales"] * prep["column_weights"]
    delta = zz[:, None, :] - centers
    expected = np.einsum("nci,ij,ncj->nc", delta, np.linalg.inv(shrunk), delta)
    np.testing.assert_allclose(distances(xc, s), expected, rtol=1e-11, atol=1e-11)
    before = {k: v.copy() for k, v in s.items()}
    p = np.full((80, 4), 0.25)
    result = calibrate(s, xc, yc, p)
    calibrate(s, xc + 100, yc, p)
    for k, v in before.items():
        np.testing.assert_array_equal(s[k], v)
    assert result["calibration_counts"].tolist() == [20] * 4


def test_exact_finite_rank_and_conservative_ties():
    scores = np.tile(np.arange(40)[:, None], (4, 4))
    y = np.repeat(np.arange(4), 40)
    q, n, k = rank_thresholds(scores, y)
    np.testing.assert_array_equal(k, [39] * 4)
    np.testing.assert_array_equal(q, [38] * 4)
    result = decide(
        np.array([[0.9, 0.1], [0.9, 0.1]]),
        np.array([[2.0, 3.0], [2.0 + 1e-9, 3.0]]),
        np.array([2.0, 2.0]),
        "distance",
    )
    assert result["accepted"].tolist() == [True, False]


def test_all_decisions_and_no_silent_label_override():
    p = np.array([[0.8, 0.2]] * 4)
    d = np.array([[0.0, 5.0], [5.0, 5.0], [0.0, 0.0], [5.0, 0.0]])
    result = decide(p, d, [1.0, 1.0], "distance")
    assert result["decisions"].tolist() == list(DECISIONS)
    assert result["predicted_indices"].tolist() == [0] * 4
    assert result["accepted_indices"].tolist() == [0, -1, -1, -1]
    np.testing.assert_array_equal(p, result["probabilities"])
    # Identical measurements/probabilities cannot distinguish an unseen identity.
    repeated = decide(
        np.repeat(p[:1], 2, axis=0), np.repeat(d[:1], 2, axis=0), [1.0, 1.0], "distance"
    )
    assert repeated["accepted"].tolist() == [True, True]


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 4), np.nan),
        np.zeros((2, 4)),
        np.ones((2, 4)),
        np.array([[1.1, -0.1], [1.0, 0.0]]),
    ],
)
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        decide(bad, np.zeros_like(bad), np.ones(bad.shape[1]), "distance")


def test_insufficient_calibration_and_singular_support():
    with pytest.raises(ValueError, match="twenty"):
        rank_thresholds(np.zeros((38, 2)), np.repeat([0, 1], 19))
    x, y, _, _, prep = matrices()
    fixed = np.repeat(np.arange(4), 8)[:, None] * np.ones((1, 12))
    _, fixed_prep = prepare_views(fixed, 4)
    with pytest.raises(ValueError, match="covariance"):
        fit_support(fixed, y, fixed_prep, 4)
    with pytest.raises(ValueError):
        fit_support(x, np.full(32, -1), prep, 4)


def test_zero_accepted_denominators_are_undefined():
    from super_deli.benchmarking.abstention_metrics import metrics, target

    p = np.array([[0.8, 0.2], [0.2, 0.8]])
    result = decide(p, np.ones_like(p) * 10, np.ones(2), "distance")
    m = metrics(np.array([0, 1]), result, known=True)
    assert m["accepted_accuracy"] is None and m["accepted_fraction"] == 0
    assert not target(dict(iid=m, near=dict(accepted_fraction=0), far=dict(accepted_fraction=0)))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    pytest.importorskip("torch")
    from super_deli.benchmarking.residual_config import model_for
    from super_deli.benchmarking.residual_data import write_cohort, write_labels
    from super_deli.workflows.residual_reference import train_matched_residual

    root = tmp_path_factory.mktemp("abstention_api")
    x, y, xc, yc, _ = matrices()
    cohorts = {}
    for name, data, truth in (("train", x, y), ("cal", xc, yc), ("test", xc + 2, yc)):
        folder = root / name
        folder.mkdir()
        ds, ids = write_cohort(
            dict(morphology=data[:, :4], ecology=data[:, 4:], ecology_order=np.arange(len(data))),
            folder / "raw",
            tag=name,
        )
        write_labels(ids, truth, folder / "labels.csv")
        cohorts[name] = (ds, folder, ids)
    train_matched_residual(
        cohorts["train"][0],
        labels_path=cohorts["train"][1] / "labels.csv",
        model_config=model_for("residual", 0),
        base_dir=cohorts["train"][1] / "raw",
        output_dir=root / "model",
    )
    return root, cohorts


@pytest.mark.parametrize("policy", ["distance", "probability"])
def test_source_bound_api_and_frozen_reload(fitted, policy):
    from super_deli.workflows.reference_abstention import (
        calibrate_reference_abstention,
        load_reference_abstention,
        predict_reference_abstention,
    )

    root, cohorts = fitted
    train, cal, test = [cohorts[n] for n in ("train", "cal", "test")]
    config = ReferenceAbstentionConfig(True, policy=policy)
    c = root / ("support_" + policy)
    summary = calibrate_reference_abstention(
        train[0],
        cal[0],
        reference_run=root / "model",
        calibration_labels=cal[1] / "labels.csv",
        abstention_config=config,
        training_base_dir=train[1] / "raw",
        calibration_base_dir=cal[1] / "raw",
        output_dir=c,
    )
    assert summary["species_significance"] == "not_computed"
    model, a, _, meta = load_reference_abstention(c, reference_run=root / "model")
    out = root / ("predictions_" + policy)
    predict_reference_abstention(
        test[0],
        reference_run=root / "model",
        calibration_run=c,
        base_dir=test[1] / "raw",
        output_dir=out,
    )
    with (out / "reference-assignments.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert all(bool(r["accepted_reference_label"]) == (r["decision"] == "assigned") for r in rows)
    raw = np.array([r for r in matrices()[2]]) + 2
    expected = model.predict_matrix(raw)["probabilities"]
    np.testing.assert_array_equal(
        [[float(r[f"class_{j}_probability"]) for j in range(4)] for r in rows], expected
    )
    code = (
        "import sys;from super_deli.workflows.reference_abstention "
        "import load_reference_abstention;"
        "load_reference_abstention(sys.argv[1],reference_run=sys.argv[2]);"
        'assert "torch" not in sys.modules and "snpio" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code, str(c), str(root / "model")], check=True)
    with pytest.raises(ValueError, match="disjoint"):
        predict_reference_abstention(
            cal[0],
            reference_run=root / "model",
            calibration_run=c,
            base_dir=cal[1] / "raw",
            output_dir=root / ("leaked_" + policy),
        )
    assert (root / ("leaked_" + policy) / "failure.json").exists()
    # Tampering is not turned into a successful outside-support decision.
    p = c / "support.npz"
    payload = p.read_bytes()
    p.write_bytes(payload + b"invented tampering")
    with pytest.raises(ValueError, match="binding"):
        load_reference_abstention(c, reference_run=root / "model")
    p.write_bytes(payload)
    labels = c / "calibration-labels.csv"
    original = labels.read_bytes()
    labels.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="label fingerprint"):
        load_reference_abstention(c, reference_run=root / "model")
    labels.write_bytes(original)


def test_training_source_mismatch_is_retained(fitted):
    from dataclasses import replace

    from super_deli.workflows.reference_abstention import calibrate_reference_abstention

    root, cohorts = fitted
    train, cal = cohorts["train"], cohorts["cal"]
    changed = replace(train[0], dataset_id="changed_metadata")
    # Same raw sources and identities remain valid; different source bytes do not.
    raw = train[1] / "raw/morphology.tsv"
    before = raw.read_bytes()
    raw.write_bytes(before + b"\n")
    try:
        with pytest.raises(ValueError):
            calibrate_reference_abstention(
                changed,
                cal[0],
                reference_run=root / "model",
                calibration_labels=cal[1] / "labels.csv",
                abstention_config=ReferenceAbstentionConfig(True),
                training_base_dir=train[1] / "raw",
                calibration_base_dir=cal[1] / "raw",
                output_dir=root / "changed_source",
            )
        assert (root / "changed_source/failure.json").exists()
    finally:
        raw.write_bytes(before)


def test_training_calibration_overlap_is_retained(fitted):
    from super_deli.workflows.reference_abstention import calibrate_reference_abstention

    root, cohorts = fitted
    train = cohorts["train"]
    with pytest.raises(ValueError, match="overlap"):
        calibrate_reference_abstention(
            train[0],
            train[0],
            reference_run=root / "model",
            calibration_labels=train[1] / "labels.csv",
            abstention_config=ReferenceAbstentionConfig(True),
            training_base_dir=train[1] / "raw",
            calibration_base_dir=train[1] / "raw",
            output_dir=root / "overlap",
        )
    assert (root / "overlap/failure.json").exists()


def test_incomplete_calibration_has_no_completion_marker(fitted, monkeypatch):
    from pathlib import Path

    from super_deli.workflows.reference_abstention import calibrate_reference_abstention

    root, cohorts = fitted
    train, cal = cohorts["train"], cohorts["cal"]
    original = Path.write_text

    def fail_report(path, *args, **kwargs):
        if path.name == "report.html":
            raise OSError("invented report write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report)
    out = root / "incomplete_report"
    with pytest.raises(OSError, match="invented"):
        calibrate_reference_abstention(
            train[0],
            cal[0],
            reference_run=root / "model",
            calibration_labels=cal[1] / "labels.csv",
            abstention_config=ReferenceAbstentionConfig(True),
            training_base_dir=train[1] / "raw",
            calibration_base_dir=cal[1] / "raw",
            output_dir=out,
        )
    assert (out / "failure.json").exists() and not (out / "summary.json").exists()


def test_large_projection_has_explicit_shuffled_identity_alignment(tmp_path):
    from super_deli.benchmarking.residual_config import model_for
    from super_deli.benchmarking.residual_data import write_cohort
    from super_deli.io.tabular import ingest_tabular
    from super_deli.models.reference_projection import aligned_reference_matrix

    rng = np.random.default_rng(810322)
    x = rng.normal(size=(2048, 12))
    dataset, ids = write_cohort(
        dict(morphology=x[:, :4], ecology=x[:, 4:], ecology_order=rng.permutation(len(x))),
        tmp_path / "raw",
        tag="large",
    )
    loaded = ingest_tabular(dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "loaded")
    matrix, actual = aligned_reference_matrix(loaded.tables, model_for("residual", 0))
    assert actual == ids
    np.testing.assert_array_equal(matrix, x)


def test_reference_distance_memory_stays_within_projection_envelope():
    import tracemalloc

    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(810323)
    x = rng.normal(size=(256, 64))
    _, prep = prepare_views(x, 32)
    support = fit_support(x, np.repeat(np.arange(32), 8), prep, 32)
    query = rng.normal(size=(2048, 64))
    with threadpool_limits(limits=1):
        tracemalloc.start()
        try:
            actual = distances(query, support)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert actual.shape == (2048, 32) and np.isfinite(actual).all()
    assert peak < 64 * 1024**2
