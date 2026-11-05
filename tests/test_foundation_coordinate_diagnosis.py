"""Pure contracts and lazy CLI remain usable without scientific dependencies."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.benchmarking.coordinate_config import (
    SPACES,
    CoordinateStudy,
    MixtureSettings,
    counts,
    tasks,
)


def test_frozen_counts_and_archive_grid():
    cfg = CoordinateStudy()
    assert len(SPACES) == 10
    assert counts(cfg, "main")["workflows"] == 2400
    assert counts(cfg, "preflight")["workflows"] == 480
    assert counts(cfg, "main")["gmm_searches"] == 1600
    assert counts(cfg, "main")["density_fits"] == 800
    assert all(counts(cfg, p)["neural_fits"] == 0 for p in ("preflight", "main"))
    assert {t["key"] for t in tasks(cfg, "preflight")} < {t["key"] for t in tasks(cfg, "main")}


@pytest.mark.parametrize(
    "value",
    [
        {"artifact_cap_bytes": 2 * 1024**3},
        {"max_workers": 3},
        {"max_projected_seconds": 3601},
        {"min_free_bytes": 1},
        {"fixture": 1},
        {"unknown": 0},
    ],
)
def test_resource_limits(value):
    with pytest.raises(ValueError):
        CoordinateStudy.from_dict(value)


def test_mixture_family_and_settings_are_fixed():
    assert MixtureSettings("tied").random_seed == 71323
    with pytest.raises(ValueError):
        MixtureSettings("diag")
    with pytest.raises(ValueError):
        replace(MixtureSettings("full"), n_init=100)


def test_cli_help_imports_no_optional_stack():
    code = """
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'numpy','scipy','sklearn','torch','snpio'}:
   raise RuntimeError('Unexpected optional import: '+name)
sys.meta_path.insert(0,Block())
from super_deli.cli import main
main(['diagnose-neural-coordinates','--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy()
    )
    assert result.returncode == 0, result.stderr
    assert "audit-main" in result.stdout
