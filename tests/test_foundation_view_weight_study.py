"""Dependency-light design/serialization contracts for the approved checkpoint."""

import json
import subprocess
import sys

import pytest

from super_deli.benchmarking.view_weight_config import ARMS, JOINT, ViewWeightStudy, counts, tasks


def test_frozen_research_budget():
    config = ViewWeightStudy()
    assert counts(config, "main") == {
        "observations": 60,
        "point_workflows": 960,
        "subsample_refits": 108,
        "legacy_replays": 12,
        "workflows": 1080,
        "neural_fits": 1020,
        "gmm_searches": 60,
    }
    assert counts(config, "preflight")["workflows"] == 300
    assert len(ARMS) == 16 and len(JOINT) == 9
    main, preflight = tasks(config, "main"), tasks(config, "preflight")
    for kind in (
        "data_seed",
        "noise_seed",
        "permutation_seed",
        "ecology_order_seed",
        "subsampling_path",
    ):
        assert not {tuple(t[kind]) for t in main} & {tuple(t[kind]) for t in preflight}
    assert sum(t["draws"] > 0 for t in main) == 4


@pytest.mark.parametrize(
    "value",
    [
        {"artifact_cap_bytes": 3 * 1024**3},
        {"min_free_bytes": 1},
        {"max_workers": 3},
        {"max_projected_seconds": 3601},
        {"fixture": 1},
        {"unknown": 0},
    ],
)
def test_budget_boundaries(value):
    with pytest.raises(ValueError):
        ViewWeightStudy.from_dict(value)


def test_lazy_help_without_optional_imports():
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','scipy','sklearn','snpio'}:
            raise RuntimeError('unexpected optional import: '+fullname)
sys.meta_path.insert(0,Block())
from super_deli.cli import main
main(['benchmark-view-weights','--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "audit-preflight" in result.stdout


def test_config_json_roundtrip():
    c = ViewWeightStudy(fixture=True)
    assert ViewWeightStudy.from_dict(json.loads(json.dumps(c.to_dict()))) == c
