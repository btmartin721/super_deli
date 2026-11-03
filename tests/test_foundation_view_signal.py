"""Dependency-free scope and lazy-command contracts."""

import subprocess
import sys

import pytest

from super_deli.benchmarking.view_signal_config import REPRESENTATIONS, ViewSignalConfig


def test_view_signal_scope():
    assert len(REPRESENTATIONS) == len(set(REPRESENTATIONS)) == 21
    assert ViewSignalConfig.from_dict(ViewSignalConfig().to_dict()) == ViewSignalConfig()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"fixture": 1},
        {"artifact_cap_bytes": 512 * 1024**2 + 1},
        {"max_workers": 3},
        {"max_workers": True},
        {"min_free_bytes": 1},
        {"max_projected_seconds": 3601},
        {"new_setting": True},
    ],
)
def test_view_signal_invalid_config(change):
    with pytest.raises(ValueError):
        ViewSignalConfig.from_dict({**ViewSignalConfig().to_dict(), **change})


def test_view_signal_explicit_schema():
    with pytest.raises(ValueError, match="schema"):
        ViewSignalConfig.from_dict({})


def test_view_signal_help_is_lazy():
    code = """import sys
from super_deli.cli import main
try:
    main(['diagnose-matched-view-signal', '--help'])
except SystemExit as e:
    assert e.code == 0
assert not {'numpy', 'scipy', 'sklearn', 'torch'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
