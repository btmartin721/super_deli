"""Projection, parameter counts, candidate observations and incomplete-search semantics."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")


def test_centroid_retention_and_recovery_meanings():
    from super_deli.benchmarking.signal_loss_metrics import parameters, projected_energy, recovery

    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    labels = np.arange(4)
    assert projected_energy(x, labels, np.eye(2)) == 1
    assert projected_energy(x, labels, np.array([[1.0, 0]])) == 0.5
    assert recovery(labels, labels, "continuous_control")["correct_recovery"] is None
    assert recovery(labels, labels // 2, "genomic_only")["correct_recovery"]
    assert recovery(labels, labels, "shape_only")["correct_recovery"]
    assert recovery(labels, labels, "conflicting")["target_k"] is None
    assert parameters(4, 8, "tied") == 71 and parameters(4, 8, "full") == 179


@pytest.mark.parametrize("covariance", ["diag", "full", "tied"])
def test_read_only_observer_and_independent_densities(tmp_path, covariance):
    from super_deli.benchmarking.signal_loss_fit import fit_scores
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    rng = np.random.default_rng(37)
    x = rng.normal(size=(64, 2))
    c = TabularClusterConfig(True, n_bootstrap=0, n_init=10, random_seed=301)
    a = fit_scores(x, c, 123, covariance, tmp_path / "first")
    b = fit_scores(x, c, 123, covariance, tmp_path / "second")
    assert a == b and a["status"] == "complete" and a["audited_candidates"] == 4
    with (
        np.load(tmp_path / "first/point.npz") as left,
        np.load(tmp_path / "second/point.npz") as right,
    ):
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


def test_failed_candidate_is_not_k1(tmp_path, monkeypatch):
    from sklearn.mixture import GaussianMixture

    from super_deli.benchmarking.signal_loss_fit import fit_scores
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    native = GaussianMixture.fit

    def fail(model, x, *args, **kwargs):
        if model.n_components == 3:
            raise ValueError("invented failure")
        return native(model, x, *args, **kwargs)

    monkeypatch.setattr(GaussianMixture, "fit", fail)
    result = fit_scores(
        np.random.default_rng(3).normal(size=(64, 2)),
        TabularClusterConfig(True, n_bootstrap=0),
        123,
        "tied",
        tmp_path / "failed",
    )
    assert result["status"] == "failed" and result["selected_k"] is None
    rows = json.loads((tmp_path / "failed/candidates/index.json").read_text())
    assert len(rows) == 3 and rows[-1]["fit_status"] == "failed"
    assert not (tmp_path / "failed/point.npz").exists()


def test_preserved_dimension_guard(tmp_path):
    from super_deli.benchmarking.signal_loss_fit import fit_scores
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    with pytest.raises(ValueError):
        fit_scores(
            np.zeros((64, 8)),
            TabularClusterConfig(True, n_bootstrap=0),
            123,
            "tied",
            tmp_path / "invalid",
        )
