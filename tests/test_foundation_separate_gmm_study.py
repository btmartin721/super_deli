"""Frozen independent-observation, head and training budgets."""

import pytest

from super_deli.benchmarking.separate_gmm_config import SharedEncoderStudy, counts, tasks


def test_frozen_study_counts_and_control_priority():
    c = SharedEncoderStudy()
    grid = tasks(c, "main")
    assert len(grid) == 520
    assert sum(t["case"].startswith("gaussian_") for t in grid) == 400
    assert counts(c, "main")["workflows"] == 7516
    assert counts(c, "main")["neural_fits"] == 3624
    assert counts(c, "preflight")["workflows"] == 568
    assert counts(c, "preflight")["neural_fits"] == 336
    assert len({tuple(t["data_seed"]) for t in grid}) == 520
    assert {tuple(t["data_seed"]) for t in grid}.isdisjoint(
        tuple(t["data_seed"]) for t in tasks(c, "preflight")
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(max_workers=3),
        dict(artifact_cap_bytes=5 * 1024**3),
        dict(max_projected_seconds=7201),
        dict(min_free_bytes=1),
    ],
)
def test_ceilings_do_not_silently_expand(change):
    with pytest.raises(ValueError):
        SharedEncoderStudy(**change)
