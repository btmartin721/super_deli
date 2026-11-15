"""Public estimator parity, failure propagation and native pipeline contracts."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from super_deli.models.full_retention import full_retention_model
from super_deli.models.initialization import fit_diverse_batches, truth_free_model
from super_deli.models.integrative import fit_integrative, project_integrative

pytestmark = pytest.mark.baseline


def test_fixed_batches_equal_independent_public_estimator_fits():
    x = np.random.default_rng(74401).normal(size=(64, 4))
    template = GaussianMixture(
        2, covariance_type="tied", n_init=100, random_state=391, max_iter=500
    )
    with threadpool_limits(limits=1):
        fitted, record = fit_diverse_batches(template, x)
        independent = []
        for i, (method, starts) in enumerate(
            (("kmeans", 34), ("k-means++", 33), ("random_from_data", 33))
        ):
            seed = (
                391
                if i == 0
                else int(np.random.SeedSequence([391, i, 20261115]).generate_state(1)[0])
            )
            other = GaussianMixture(
                2,
                covariance_type="tied",
                n_init=starts,
                init_params=method,
                random_state=seed,
                max_iter=500,
            ).fit(x)
            independent.append(other)
        winner = max(range(3), key=lambda i: (independent[i].lower_bound_, -i))
        assert (
            record["selected_batch"] == winner
            and sum(r["starts"] for r in record["batches"]) == 100
        )
        np.testing.assert_array_equal(fitted.predict_proba(x), independent[winner].predict_proba(x))
    assert template.n_init == 100 and not hasattr(template, "weights_")


def test_failed_batch_is_not_silently_replaced(monkeypatch):
    called = []

    def failed(self, x):
        called.append(self.init_params)
        raise ValueError("retained failure")

    monkeypatch.setattr(GaussianMixture, "fit", failed)
    with pytest.raises(ValueError, match="retained failure"):
        fit_diverse_batches(GaussianMixture(n_init=100, random_state=1), np.ones((20, 2)))
    assert called == ["kmeans"]


def test_native_projection_and_paired_bootstrap_keep_training_geometry():
    spec = importlib.util.spec_from_file_location(
        "init_fixture", Path(__file__).with_name("test_integrative_model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs, base, _ = module.simulate(all_views=True)
    old = full_retention_model(
        replace(base, covariance_type="tied", n_bootstrap=2), min_cluster_size=9
    )
    first = fit_integrative(inputs, config=old)
    second = fit_integrative(inputs, config=truth_free_model(old))
    np.testing.assert_array_equal(first.pipeline.scores, second.pipeline.scores)
    assert [r["training_indices"] for r in first.stability.replicates] == [
        r["training_indices"] for r in second.stability.replicates
    ]
    np.testing.assert_allclose(
        project_integrative(second.inputs, second.unit_ids, second.pipeline),
        second.pipeline.scores,
        atol=1e-10,
    )
    assert all(
        r.get("initialization", {}).get("total_starts") == 100
        for r in second.point.candidates
        if r["status"] != "insufficient_samples"
    )
