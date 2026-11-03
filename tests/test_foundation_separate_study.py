"""Frozen design counts, independent seed streams and resource ceilings."""

import pytest

from super_deli.benchmarking.separate_config import ARMS, NATIVE, SeparateStudy, counts, tasks


def test_research_budget():
    c = SeparateStudy()
    assert len(ARMS) == 17 and len(NATIVE) == 9
    assert counts(c, "main") == {
        "observations": 80,
        "point_workflows": 1360,
        "subsample_refits": 108,
        "legacy_replays": 12,
        "workflows": 1480,
        "neural_fits": 1596,
        "density_fits": 1400,
        "gmm_searches": 80,
    }
    assert (
        counts(c, "preflight")["workflows"] == 380 and counts(c, "preflight")["neural_fits"] == 432
    )
    assert sum(counts(c, p)["neural_fits"] for p in ("main", "preflight")) * 500 == 1014000
    for key in (
        "data_seed",
        "noise_seed",
        "permutation_seed",
        "ecology_order_seed",
        "subsampling_path",
    ):
        assert not {tuple(t[key]) for t in tasks(c, "main")} & {
            tuple(t[key]) for t in tasks(c, "preflight")
        }


@pytest.mark.parametrize(
    "value",
    [
        {"max_workers": 3},
        {"artifact_cap_bytes": 3 * 1024**3},
        {"max_projected_seconds": 3601},
        {"min_free_bytes": 1},
        {"fixture": 1},
        {"other": 0},
    ],
)
def test_frozen_resource_limits(value):
    with pytest.raises(ValueError):
        SeparateStudy.from_dict(value)
