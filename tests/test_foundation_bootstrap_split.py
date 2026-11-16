"""Dependency-free bounds, fixed roster and lazy command contracts."""

from dataclasses import replace

import pytest

from super_deli.benchmarking.bootstrap_split_config import BootstrapSplitConfig, tasks


def test_exact_bounded_configuration():
    c = BootstrapSplitConfig("a" * 64, "b" * 64)
    assert BootstrapSplitConfig.from_dict(c.to_dict()) == c
    for kw in (
        {"min_free_bytes": 1},
        {"artifact_cap_bytes": 9 * 1024**3},
        {"fixture": 1},
        {"max_main_seconds": 5401},
        {"source_completion_sha256": "A" * 64},
    ):
        with pytest.raises(ValueError):
            replace(c, **kw)
    with pytest.raises(ValueError):
        BootstrapSplitConfig.from_dict({**c.to_dict(), "unknown": 1})


@pytest.mark.parametrize("phase,count,base", [("preflight", 4, 9100), ("main", 24, 9200)])
def test_disjoint_roster(phase, count, base):
    roster = tasks(phase)
    assert len(roster) == count
    assert [t["seed"] for t in roster] == list(range(base, base + count))
    assert {(t["n"], t["law"]) for t in roster} == {
        (n, law) for n in (64, 256) for law in ("continuous_control", "shape_only")
    }
    with pytest.raises(ValueError):
        tasks("invented")


def test_lazy_command(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as e:
        main(["diagnose-bootstrap-splitting", "--help"])
    assert e.value.code == 0 and "--phase" in capsys.readouterr().out
