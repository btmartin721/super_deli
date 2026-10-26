"""Numerical density contracts, training-only subsampling, failure accounting, and CLI."""

# ruff: noqa: E402
import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from super_deli.benchmarking.density_audit import _audit_fit, _audit_support
from super_deli.benchmarking.density_study import (
    DensityStudyConfig,
    generate_raw,
    recovery_metrics,
    run_density_observation,
    write_raw,
)
from super_deli.cli import main
from super_deli.core.data import Feature
from super_deli.core.tabular import TabularTable
from super_deli.models import tabular_hdbscan
from super_deli.models.calibration_config import GaussianCalibrationConfig
from super_deli.models.hdbscan_config import HDBSCANConfig
from super_deli.models.subsample_stability import summarize_subsamples
from super_deli.models.tabular_hdbscan import fit_tabular_hdbscan
from super_deli.workflows.score_calibration import _source_scores, run_integrative_calibration
from super_deli.workflows.tabular_hdbscan import run_tabular_hdbscan

pytestmark = pytest.mark.baseline


def table(x, kind="morphological", level="specimen"):
    return TabularTable(
        "traits",
        kind,
        level,
        tuple(Feature(f"v{i}", "continuous") for i in range(x.shape[1])),
        tuple(f"o{i}" for i in range(len(x))),
        tuple(f"s{i}" for i in range(len(x))),
        tuple(tuple(None if not np.isfinite(v) else float(v) for v in row) for row in x),
    )


def data():
    rng = np.random.default_rng(72)
    return np.concatenate([rng.normal(-5, 0.2, (30, 2)), rng.normal(5, 0.2, (30, 2))])


@pytest.mark.parametrize("kind,level", [("morphological", "specimen"), ("ecological", "locality")])
def test_known_groups_and_independent_geometry(kind, level):
    x = data()
    fit = fit_tabular_hdbscan(
        table(x, kind, level), config=HDBSCANConfig(True, n_subsamples=3, random_seed=89)
    )
    assert fit.point.selected_k == 2
    z = StandardScaler().fit_transform(x)
    np.testing.assert_allclose(pdist(z), pdist(fit.point.scores), atol=1e-12)
    ref = HDBSCAN(min_cluster_size=5, min_samples=5, allow_single_cluster=True).fit(
        fit.point.scores
    )
    np.testing.assert_array_equal(ref.labels_, fit.point.labels)
    np.testing.assert_allclose(ref.probabilities_, fit.point.strength)
    assert fit.stability.summary["status"] == "complete"


def test_all_noise_is_success_and_noise_never_coassigned():
    fit = fit_tabular_hdbscan(
        table(data()),
        config=HDBSCANConfig(
            True, min_cluster_size=100, allow_single_cluster=False, n_subsamples=3, random_seed=8
        ),
    )
    assert fit.point.selected_k == 0 and np.all(fit.point.labels == -1)
    assert np.all(fit.point.strength == 0)
    assert fit.stability.summary["k_frequencies"] == {"0": 3}
    assert np.all(fit.stability.arrays["pair_coassigned"] == 0)
    finite = np.isfinite(fit.stability.arrays["noise_frequency"])
    assert np.all(fit.stability.arrays["noise_frequency"][finite] == 1)


def test_noise_denominators_and_failed_draw_bounds_independently():
    records = [
        {"training_indices": [0, 1, 2], "status": "complete", "labels": [0, 0, -1]},
        {"training_indices": [1, 2, 3], "status": "complete", "labels": [-1, -1, 1]},
        {"training_indices": [0, 2, 3], "status": "failed", "error": "retained"},
    ]
    fit = summarize_subsamples(5, records)
    _audit_support(5, records, fit.arrays)
    assert fit.summary["status"] == "incomplete"
    assert fit.arrays["coassignment"][0, 1] == 1
    assert fit.arrays["coassignment"][1, 2] == 0
    assert np.isnan(fit.arrays["coassignment"][0, 3])
    assert fit.arrays["coassignment_lower"][0, 3] == 0
    assert fit.arrays["coassignment_upper"][0, 3] == 1
    assert np.isnan(fit.arrays["noise_frequency"][4])
    assert fit.arrays["noise_lower"][2] == 2 / 3
    assert fit.arrays["noise_upper"][2] == 1
    assert fit.arrays["noise_frequency"][1] == 0.5
    assert fit.arrays["noise_mc_se"][1] == np.sqrt(0.5 * 0.5 / 2)


def test_failed_draws_are_not_replaced_or_relabelled(monkeypatch):
    original = tabular_hdbscan.fit_density_once
    calls = []

    def wrapped(x, *args):
        calls.append(x.copy())
        if len(calls) == 2:
            raise ValueError("injected failure")
        return original(x, *args)

    monkeypatch.setattr(tabular_hdbscan, "fit_density_once", wrapped)
    fit = fit_tabular_hdbscan(table(data()), config=HDBSCANConfig(True, n_subsamples=3))
    assert len(calls) == 4
    assert fit.stability.summary["failed"] == 1
    assert fit.stability.records[0]["error"] == "injected failure"
    assert "labels" not in fit.stability.records[0]


def test_preprocessing_is_refitted_on_included_raw_units(monkeypatch):
    original = tabular_hdbscan.fit_density_once
    seen = []

    def wrapped(x, *args):
        fit = original(x, *args)
        np.testing.assert_allclose(fit.model.means, x.mean(axis=0))
        seen.append(x.copy())
        return fit

    monkeypatch.setattr(tabular_hdbscan, "fit_density_once", wrapped)
    x = data()
    fit = fit_tabular_hdbscan(table(x), config=HDBSCANConfig(True, n_subsamples=4, random_seed=35))
    for raw, record in zip(seen[1:], fit.stability.records, strict=True):
        ids = record["training_indices"]
        assert len(ids) == 48 and len(set(ids)) == 48
        np.testing.assert_array_equal(raw, x[ids])


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda t: replace(t, kind="genomic"), "morphological"),
        (
            lambda t: replace(t, features=(Feature("v0", "count"), t.features[1])),
            "Count|continuous",
        ),
        (lambda t: replace(t, unit_ids=(t.unit_ids[0],) * len(t.unit_ids)), "Repeated"),
        (lambda t: replace(t, values=((None, 1.0),) + t.values[1:]), "complete"),
        (
            lambda t: replace(
                t, features=(Feature("v0", "continuous", role="covariate"), t.features[1])
            ),
            "Too few",
        ),
    ],
)
def test_input_scope_is_enforced(mutation, message):
    with pytest.raises(ValueError, match=message):
        fit_tabular_hdbscan(mutation(table(data())), config=HDBSCANConfig(True))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"max_working_bytes": 1}, "workspace"),
        ({"max_pairwise_bytes": 1, "n_subsamples": 3}, "pairwise"),
        ({"max_working_bytes": 300000, "n_subsamples": 100}, "Retained"),
    ],
)
def test_budgets_before_allocating_support(changes, message):
    with pytest.raises(ValueError, match=message):
        fit_tabular_hdbscan(table(data()), config=HDBSCANConfig(True, **changes))


def test_rank_deficiency_and_all_failed_subsamples():
    x = data()
    x[:, 1] = x[:, 0]
    with pytest.raises(ValueError, match="rank deficient"):
        fit_tabular_hdbscan(table(x), config=HDBSCANConfig(True))
    fit = fit_tabular_hdbscan(
        table(data()), config=HDBSCANConfig(True, min_samples=55, n_subsamples=2)
    )
    assert fit.stability.summary["status"] == "unavailable"
    assert fit.stability.summary["failed"] == 2


def test_end_to_end_artifacts_and_no_calibration(tmp_path, capsys):
    raw = tmp_path / "raw"
    dataset = write_raw(data(), raw)
    config = HDBSCANConfig(True, n_subsamples=2)
    (raw / "model.json").write_text(json.dumps(config.to_dict()))
    output = tmp_path / "fit"
    assert (
        main(
            [
                "cluster-tabular-density",
                str(raw / "dataset.json"),
                "--model-config",
                str(raw / "model.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected_k"] == 2
    assert summary["inference"]["p_value"] is None
    assert len(summary["artifacts"]) == 5
    with pytest.raises(ValueError, match="supported integrative|HDBSCAN"):
        _source_scores(output, GaussianCalibrationConfig(True))
    with pytest.raises(ValueError, match="HDBSCAN significance"):
        run_integrative_calibration(
            output, config=GaussianCalibrationConfig(True), output_dir=tmp_path / "calibration"
        )
    with pytest.raises(FileExistsError):
        run_tabular_hdbscan(dataset, model_config=config, base_dir=raw, output_dir=output)
    # Tampering is independently detectable; no pickle/object arrays are used.
    with np.load(output / "point.npz", allow_pickle=False) as arrays:
        assert len(arrays["labels"]) == 60
    records = json.loads((output / "subsamples.json").read_text())
    assert len(records) == 2


def test_noise_metrics_do_not_inflate_agreement_by_grouping_noise():
    truth = np.array([0, 0, 1, 1])
    result = recovery_metrics(np.full(4, -1), truth)
    assert result["selected_k"] == 0 and result["ari"] == 0
    assert result["assigned_ari"] is None
    assert result["core_assigned_fraction"] == 0


@pytest.mark.parametrize(
    "case", ["gaussian", "four_spherical", "four_unequal", "moons", "background", "cline"]
)
def test_simulator_truth_and_raw_shape(case):
    x, y = generate_raw(case, 64, [93709, 94217, 0, 64, 0])
    assert x.shape == (64, 2) and np.isfinite(x).all()
    assert (y is None) == (case == "cline")
    if y is not None:
        assert len(y) == 64
        if case == "background":
            assert np.sum(y == -1) == 8


def test_paired_worker_and_independent_audits(tmp_path):
    c = DensityStudyConfig(n_datasets=1, stability_datasets=1, n_subsamples=2, sample_sizes=(64,))
    plan = c.plan("preflight")[1]
    result = run_density_observation((str(tmp_path), c.to_dict(), plan))
    path = tmp_path / plan["key"]
    x, _ = generate_raw(plan["case"], 64, plan["data_seed"])
    for family in ("hdbscan", "gmm"):
        record = json.loads((path / f"{family}.json").read_text())
        with np.load(path / f"{family}.npz", allow_pickle=False) as f:
            arrays = dict(f)
        _audit_fit(x, arrays, "point", record["point"], family)
        _audit_support(64, record["draws"], arrays)
        assert result["families"][family]["successful"] == 3


def test_small_study_audit_storage_gate_and_tampering(tmp_path):
    from super_deli.benchmarking.density_audit import audit_density_study
    from super_deli.benchmarking.density_study import storage_gate
    from super_deli.io.config import write_json
    from super_deli.models.hdbscan_config import HDBSCANConfig

    c = DensityStudyConfig(
        n_datasets=1, stability_datasets=1, n_subsamples=1, sample_sizes=(32,), min_free_bytes=1
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    model = HDBSCANConfig(True)
    write_json(
        {
            "configuration": c.to_dict(),
            "phase": "preflight",
            "plan": c.plan("preflight"),
            "density_model": model.to_dict(),
            "gmm_model": replace(
                model.preprocessing_config(), min_cluster_size=5, n_init=10
            ).to_dict(),
        },
        probe / "frozen-plan.json",
    )
    for row in c.plan("preflight"):
        run_density_observation((str(probe), c.to_dict(), row))
    audit = tmp_path / "audit"
    result = audit_density_study(probe, output_dir=audit)
    assert result["status"] == "passed" and len(result["primary_rates"]) == 10
    gate = storage_gate(
        c,
        probe_dir=probe,
        audit_file=audit / "summary.json",
        artifact_root=tmp_path,
        output_file=tmp_path / "gate.json",
    )
    assert gate["status"] == "passed"
    target = probe / c.plan("preflight")[0]["key"] / "raw/traits.tsv"
    target.write_text(target.read_text() + "corruption")
    with pytest.raises(AssertionError):
        audit_density_study(probe, output_dir=tmp_path / "corrupt-audit")
