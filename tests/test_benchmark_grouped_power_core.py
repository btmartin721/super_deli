"""Dependency-free configuration, lossless evidence and archive safety checks."""

import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.benchmarking.local_archives import compact_directory, materialize, write_progress
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig, PowerCell

ROOT = Path(__file__).resolve().parents[1]


def configuration(name="study.json"):
    return GroupedPowerStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-power-recovery" / name)
    )


def test_approved_design_fit_budget_and_json_compatibility():
    config = configuration()
    assert GroupedPowerStudyConfig.from_dict(config.to_dict()) == config
    plan = config.plan()
    assert len(plan) == 400 and sum(r["planned_fits"] for r in plan) == 40780
    assert sum(r["stability"] for r in plan) == 30
    assert len({c.name for c in config.cells}) == 10
    assert sum(c.n_datasets for c in config.cells if c.has_groups) == 300
    assert config.calibration.n_simulations == 99 and config.calibration.alpha == 0.05
    assert config.calibration.model.random_seed == 71323
    assert config.calibration.model.n_init == 10
    assert config.calibration.model.n_pcs == 4
    assert config.calibration.model.covariance_type == "full"
    assert config.calibration.model.selection_criterion == "aic"
    assert config.calibration.model.n_bootstrap == 0
    assert config.calibration.model.views[0].pca_fit == "target_means"


def test_seed_streams_survive_reordering_without_duplicates():
    config = configuration()
    original = {(r["cell"], r["repetition"]): r for r in config.plan()}
    reordered = {
        (r["cell"], r["repetition"]): r
        for r in replace(config, cells=tuple(reversed(config.cells))).plan()
    }
    assert reordered == original
    streams = [r[k] for r in original.values() for k in ("data_seed", "null_stream_seed")]
    assert len(streams) == len(set(streams)) == 800


@pytest.mark.parametrize(
    "updates",
    [
        {"name": "../escape"},
        {"name": 1},
        {"law": "unknown"},
        {"law": []},
        {"n_units": 63},
        {"n_units": True},
        {"n_datasets": 0},
        {"signal": float("nan")},
        {"signal": -1},
        {"signal": True},
    ],
)
def test_invalid_cells_are_rejected(updates):
    with pytest.raises(ValueError):
        replace(PowerCell("four", "complementary", 64, 50, 3), **updates)


def test_control_signal_unknown_fields_and_stability_overlap_rejected():
    with pytest.raises(ValueError):
        PowerCell("null", "independent_gaussian", 64, 25, 3)
    data = configuration().to_dict()
    data["unknown"] = 1
    with pytest.raises(ValueError):
        GroupedPowerStudyConfig.from_dict(data)
    with pytest.raises(ValueError):
        replace(configuration(), cells=(configuration().cells[0],) * 2)
    with pytest.raises(ValueError):
        replace(configuration(), stability_first=51)
    with pytest.raises(ValueError):
        replace(configuration(), stability_draws=0)


def test_atomic_progress_does_not_change_exclusive_evidence_writer(tmp_path):
    path = tmp_path / "progress.json"
    write_progress({"count": 1}, path)
    write_progress({"count": 2}, path)
    assert read_json(path) == {"count": 2}
    evidence = tmp_path / "evidence.json"
    write_json({"count": 1}, evidence)
    with pytest.raises(FileExistsError):
        write_json({"count": 2}, evidence)


def evidence(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.bin").write_bytes(bytes(range(256)) * 5)
    (source / "empty.log").write_bytes(b"")
    archive = tmp_path / "evidence.tar.gz"
    return source, archive, compact_directory(source, archive)


def test_lossless_archive_preserves_every_byte_and_original(tmp_path):
    source, archive, record = evidence(tmp_path)
    expected = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert record["files"] == 2 and source.exists()
    with materialize(archive, record, scratch=tmp_path / "scratch") as directory:
        assert {
            p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()
        } == expected
        extracted = directory
    assert not extracted.exists()
    with pytest.raises(ValueError):
        compact_directory(source, archive)


@pytest.mark.parametrize("kind", ["bytes", "manifest", "traversal", "duplicate", "symlink"])
def test_corruption_or_unsafe_archive_is_rejected(tmp_path, kind):
    source, archive, record = evidence(tmp_path)
    if kind == "bytes":
        archive.write_bytes(archive.read_bytes() + b"altered")
    elif kind == "manifest":
        path = archive.with_name(record["manifest"])
        path.write_text(path.read_text() + "\n")
    else:
        with tarfile.open(archive, "w:gz") as target:
            if kind == "traversal":
                member = tarfile.TarInfo("../../escaped")
                target.addfile(member, io.BytesIO())
            elif kind == "symlink":
                member = tarfile.TarInfo("nested/data.bin")
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/outside"
                target.addfile(member)
            else:
                for _ in range(2):
                    member = tarfile.TarInfo("nested/data.bin")
                    member.size = 1280
                    target.addfile(member, io.BytesIO((source / member.name).read_bytes()))
        record["archive_sha256"] = fingerprint_file(archive).sha256
        record["size_bytes"] = archive.stat().st_size
    with pytest.raises(ValueError):
        with materialize(archive, record, scratch=tmp_path / "scratch"):
            pass
    assert (source / "nested/data.bin").read_bytes() == bytes(range(256)) * 5


def test_unverified_compaction_retains_source_and_partial(tmp_path, monkeypatch):
    from super_deli.benchmarking import local_archives

    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_text("irreplaceable evidence")

    def reject(*args, **kwargs):
        raise ValueError("deliberate archive verification failure")

    monkeypatch.setattr(local_archives, "_verify", reject)
    with pytest.raises(ValueError, match="deliberate"):
        compact_directory(source, tmp_path / "result.tar.gz")
    assert (source / "data").read_text() == "irreplaceable evidence"
    assert (tmp_path / "result.tar.gz.partial").is_file()
    assert not (tmp_path / "result.tar.gz").exists()
