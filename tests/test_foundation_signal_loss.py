"""Frozen diagnostic scope and source seals without numerical dependencies."""

import hashlib
from dataclasses import replace

import pytest

from super_deli.benchmarking.signal_loss_config import SignalLossConfig, fresh_tasks
from super_deli.benchmarking.signal_loss_source import SealedStudy


def config():
    return SignalLossConfig("a" * 64, "b" * 64)


def test_roundtrip_and_budgeted_grid():
    c = config()
    assert SignalLossConfig.from_dict(c.to_dict()) == c
    assert len(fresh_tasks("main")) == 48 and len(fresh_tasks("preflight")) == 4
    assert {x["n"] for x in fresh_tasks("main")} == {64, 256}
    assert len({x["seed"] for x in fresh_tasks("main")}) == 48
    assert not (
        {x["seed"] for x in fresh_tasks("main")} & {x["seed"] for x in fresh_tasks("preflight")}
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"fixture": 1},
        {"source_completion_sha256": "bad"},
        {"artifact_cap_bytes": 3 * 1024**3},
        {"min_free_bytes": 1},
        {"max_projected_seconds": 1801},
    ],
)
def test_contract_bounds(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


def test_source_tamper_and_fixture_cannot_impersonate_production(tmp_path):
    from super_deli.io.config import write_json

    write_json(dict(completed=True, fixture=True), tmp_path / "completion.json")
    (tmp_path / "record.json").write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    write_json(
        {"record.json": dict(bytes=2, sha256=sha(tmp_path / "record.json"))},
        tmp_path / "study-manifest.json",
    )
    c = SignalLossConfig(
        sha(tmp_path / "completion.json"), sha(tmp_path / "study-manifest.json"), fixture=True
    )
    source = SealedStudy(tmp_path, c)
    assert source.json("record.json") == {}
    with pytest.raises(ValueError):
        SealedStudy(tmp_path, replace(c, fixture=False))
    with pytest.raises(ValueError):
        source.path("../record.json")
    (tmp_path / "record.json").write_text("{ }")
    with pytest.raises(ValueError):
        source.reverify()


def test_lazy_help(capsys):
    from super_deli.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["diagnose-multimodal-signal", "--help"])
    assert exc.value.code == 0 and "--source" in capsys.readouterr().out
