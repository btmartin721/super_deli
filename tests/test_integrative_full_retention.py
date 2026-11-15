"""Full-space geometry, frozen projection and unchanged native refits."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from scipy.spatial.distance import pdist

from super_deli.models.full_retention import full_retention_model
from super_deli.models.integrative import fit_integrative, project_integrative

spec = importlib.util.spec_from_file_location(
    "full_retention_invented_fixture", Path(__file__).with_name("test_integrative_model.py")
)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
simulate = fixture_module.simulate

pytestmark = pytest.mark.baseline


@pytest.mark.parametrize("all_views", [False, True])
def test_all_coordinates_match_explicit_native_pca_and_bootstrap(all_views):
    inputs, base, _ = simulate(all_views=all_views)
    base = replace(base, covariance_type="tied", n_bootstrap=2)
    tagged = full_retention_model(base, min_cluster_size=9)
    explicit = replace(base, n_pcs=sum(v.n_pcs for v in base.views), min_cluster_size=9)
    first = fit_integrative(inputs, config=explicit)
    second = fit_integrative(inputs, config=tagged)
    np.testing.assert_array_equal(first.pipeline.scores, second.pipeline.scores)
    np.testing.assert_array_equal(first.point.membership, second.point.membership)
    assert first.stability.replicates == second.stability.replicates
    for key in first.stability.arrays:
        np.testing.assert_array_equal(first.stability.arrays[key], second.stability.arrays[key])
    np.testing.assert_allclose(
        pdist(second.pipeline.fused), pdist(second.pipeline.scores), atol=1e-10
    )
    # Project frozen training transforms, then project a separately perturbed cohort.
    projected = project_integrative(second.inputs, second.unit_ids, second.pipeline)
    np.testing.assert_allclose(projected, second.pipeline.scores, atol=1e-10)
    changed = list(second.inputs)
    changed[0] = replace(changed[0], values=changed[0].values + 0.01)
    before = second.pipeline.pca.components_.copy()
    held_out = project_integrative(changed, second.unit_ids, second.pipeline)
    assert held_out.shape == projected.shape and np.isfinite(held_out).all()
    np.testing.assert_array_equal(before, second.pipeline.pca.components_)


def test_rank_deficiency_is_not_repaired_or_truncated():
    inputs, base, _ = simulate()
    # Identical duplicated view scores cannot support four full-rank joint axes.
    second = replace(inputs[0], name=inputs[1].name, kind=inputs[1].kind)
    model = full_retention_model(
        replace(base, covariance_type="tied", n_bootstrap=0), min_cluster_size=5
    )
    with pytest.raises(ValueError, match="rank deficient"):
        fit_integrative((inputs[0], second), config=model)
