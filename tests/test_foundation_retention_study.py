"""Frozen roster, source seals and invented-archive inference boundaries."""

import hashlib

import pytest

from super_deli.benchmarking.full_retention_config import FullRetentionStudyConfig, tasks
from super_deli.benchmarking.full_retention_source import RetentionArchive
from super_deli.io.config import write_json


def test_independent_rosters_and_fixed_stability_parents():
    main, pre = tasks("main"), tasks("preflight")
    assert len(main) == 160 and len(pre) == 8
    assert len({t["seed"] for t in main + pre}) == 168
    assert sum(t["stability"] for t in main) == 24
    assert sum(3 * (1 + 25 * t["stability"]) for t in main) == 2280
    for cell in range(8):
        subset = [t for t in main if t["cell"] == cell]
        assert len(subset) == 20 and [t["repeat"] for t in subset if t["stability"]] == [0, 1, 2]


@pytest.mark.parametrize(
    "options",
    [
        dict(schema_version=True),
        dict(fixture=1),
        dict(artifact_cap_bytes=7 * 1024**3),
        dict(min_free_bytes=1024),
        dict(max_main_seconds=1801),
        dict(source_manifest_sha256="unknown"),
    ],
)
def test_resource_and_source_contract_rejects_unapproved_expansion(options):
    args = dict(source_completion_sha256="a" * 64, source_manifest_sha256="b" * 64)
    with pytest.raises(ValueError):
        FullRetentionStudyConfig(**{**args, **options})


def test_invented_archive_cannot_authorize_scientific_staging(tmp_path):
    write_json(dict(completed=True, fixture=True), tmp_path / "completion.json")
    item = tmp_path / "invented.json"
    item.write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    write_json({"invented.json": dict(bytes=2, sha256=sha(item))}, tmp_path / "study-manifest.json")
    c = FullRetentionStudyConfig(
        sha(tmp_path / "completion.json"), sha(tmp_path / "study-manifest.json"), fixture=True
    )
    a = RetentionArchive(tmp_path, c)
    assert a.json("invented.json") == {}
    with pytest.raises(ValueError, match="cannot supply scientific"):
        a.stage(0, tmp_path / "staged")
    with pytest.raises(ValueError):
        a.path("../invented.json")
    item.write_text('{"tampered":true}')
    with pytest.raises(ValueError):
        a.json("invented.json")
