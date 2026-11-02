"""Strict diagnostic scope and optional dependency isolation."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from super_deli.benchmarking.spectral_config import SPACES, SpectralConfig, density_jobs, graph_jobs


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("fixture", 1),
        ("max_workers", 3),
        ("artifact_cap_bytes", 3 * 1024**3),
        ("min_free_bytes", 1),
        ("max_projected_seconds", 3601),
        ("max_working_bytes", False),
    ],
)
def test_invalid_scope(field, value):
    with pytest.raises(ValueError):
        replace(SpectralConfig(), **{field: value})


def test_frozen_grid():
    assert len(SPACES) == 17
    binding = {
        "sizes": [64, 256],
        "observations": [
            {"key": f"{case}-{n}-{rep}", "case": case, "n": n, "replicate": rep}
            for case in ("gaussian", "four_spherical", "moons")
            for n in (64, 256)
            for rep in range(10)
        ],
    }
    for phase, expected in (("preflight", 504), ("main", 4628)):
        assert 2 * len(graph_jobs(binding, phase)) + len(density_jobs(binding, phase)) == expected
    assert SpectralConfig.from_dict(SpectralConfig().to_dict()) == SpectralConfig()
    with pytest.raises(ValueError):
        SpectralConfig.from_dict({"choose_k": True})
    with pytest.raises(ValueError):
        graph_jobs(binding, "other")


def test_help_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from super_deli.cli import main
try:
    main(["diagnose-spectral-graphs","--help"])
except SystemExit as e:
    assert e.code==0
assert not {"numpy","scipy","sklearn","torch"}.intersection(sys.modules)
""",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
