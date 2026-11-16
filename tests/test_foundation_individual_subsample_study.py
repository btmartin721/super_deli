"""Bounded scientific roster and source-gate contracts without numerical imports."""

import hashlib
import json
from dataclasses import replace

import pytest

from super_deli.benchmarking.individual_subsample_config import (
    IndividualSubsampleStudyConfig,
    tasks,
)
from super_deli.benchmarking.individual_subsample_source import IndividualSubsampleArchive


def test_roster_and_resource_limits():
    c = IndividualSubsampleStudyConfig("a" * 64, "b" * 64)
    assert IndividualSubsampleStudyConfig.from_dict(c.to_dict()) == c
    assert len(tasks("main")) == 48 and len(tasks("preflight")) == 6
    assert {t["seed"] for t in tasks("main")}.isdisjoint(t["seed"] for t in tasks("preflight"))
    for kw in (
        {"fixture": 1},
        {"min_free_bytes": 1},
        {"max_main_seconds": 5401},
        {"artifact_cap_bytes": 9 * 1024**3},
        {"source_manifest_sha256": "A" * 64},
    ):
        with pytest.raises(ValueError):
            replace(c, **kw)
    with pytest.raises(ValueError):
        tasks("unknown")


def test_invented_seals_do_not_supply_scientific_replays(tmp_path):
    (tmp_path / "completion.json").write_text(json.dumps(dict(completed=True, fixture=True)))
    (tmp_path / "study-manifest.json").write_text("{}")

    def sha(p):
        return hashlib.file_digest(p.open("rb"), "sha256").hexdigest()

    c = IndividualSubsampleStudyConfig(
        sha(tmp_path / "completion.json"), sha(tmp_path / "study-manifest.json"), fixture=True
    )
    archive = IndividualSubsampleArchive(tmp_path, c)
    with pytest.raises(ValueError, match="Invented"):
        archive.stage(0, tmp_path / "out")
    with pytest.raises(ValueError, match="inside"):
        archive.path("../outside")
