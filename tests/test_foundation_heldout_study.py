"""Frozen total-N/abundance design and source/resource boundaries."""

import hashlib
import json
from dataclasses import replace

import pytest

from super_deli.benchmarking.heldout_config import HeldoutStudyConfig, tasks
from super_deli.benchmarking.heldout_source import HeldoutArchive


def test_frozen_roster_and_caps():
    assert len(tasks("main")) == 200 and len(tasks("preflight")) == 10
    assert {t["seed"] for t in tasks("main")}.isdisjoint(t["seed"] for t in tasks("preflight"))
    assert all(sum(t["counts"]) == t["n"] for t in tasks("main"))
    assert all(t["counts"][-1] == 10 for t in tasks("main") if t["design"] == "fixed_ten")
    assert [t["counts"] for t in tasks("preflight") if t["design"] == "rare_six"] == [
        [29, 16, 13, 6]
    ]
    c = HeldoutStudyConfig("a" * 64, "b" * 64)
    assert HeldoutStudyConfig.from_dict(c.to_dict()) == c
    for kw in (
        {"artifact_cap_bytes": 13 * 1024**3},
        {"min_free_bytes": 1},
        {"max_main_seconds": 7201},
        {"fixture": 1},
    ):
        with pytest.raises(ValueError):
            replace(c, **kw)


def test_invented_seals_cannot_authorize_scientific_fits(tmp_path):
    (tmp_path / "completion.json").write_text(json.dumps(dict(completed=True, fixture=True)))
    (tmp_path / "study-manifest.json").write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    c = HeldoutStudyConfig(
        sha(tmp_path / "completion.json"), sha(tmp_path / "study-manifest.json"), fixture=True
    )
    source = HeldoutArchive(tmp_path, c)
    with pytest.raises(ValueError, match="Invented"):
        source.case(0)
    with pytest.raises(ValueError, match="inside"):
        source.path("../escape")
