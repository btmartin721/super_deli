"""Sampling/reference provenance cannot silently become fitting authorization."""

import builtins
import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.cli import main
from super_deli.core.empirical import EmpiricalProfile
from super_deli.core.empirical_evidence import EmpiricalEvidence
from super_deli.core.empirical_protocol import EmpiricalProtocol
from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json
from super_deli.workflows.empirical_assessment import assess_empirical
from super_deli.workflows.empirical_curation import curate_empirical

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "examples/empirical-foundation/fixture"
E = ROOT / "examples/empirical-gates/fixture"


def settings():
    return (
        EmpiricalProfile.from_dict(read_json(F / "profile.json")),
        EmpiricalProtocol.from_dict(read_json(F / "protocol.json")),
        EmpiricalEvidence.from_dict(read_json(E / "evidence.json")),
    )


def curated(tmp_path):
    p, c, _ = settings()
    root = tmp_path / "curated"
    curate_empirical(F / "source.csv", profile=p, protocol=c, output_dir=root)
    return root


def assess(tmp_path, directory, **kwargs):
    p, c, e = settings()
    options = dict(
        profile=p,
        protocol=c,
        evidence=e,
        curated_dir=directory,
        evidence_dir=E,
        output_dir=tmp_path / "assessment",
    )
    options.update(kwargs)
    return assess_empirical(F / "source.csv", **options)


def refresh(directory):
    hashes = {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    (directory / "checksums.json").write_text(json.dumps(hashes))


def test_valid_fit_free_assessment_and_deterministic_integrity(tmp_path):
    directory = curated(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    s = assess(tmp_path, directory)
    assert s["source_rows"] == 12 and s["included_rows"] == 10
    assert s["empirical_fits"] == s["stability_draws"] == s["significance_draws"] == 0
    assert s["readiness"]["ingestion"]["status"] == "available"
    assert s["readiness"]["published_reference_description"]["status"] == "available"
    for n in (
        "independent_unit_point_fitting",
        "resampling",
        "significance",
        "independent_species_boundary_validation",
    ):
        assert s["readiness"][n]["status"] == "blocked"
    assert s["readiness"]["grouped_predictive_screen"]["status"] == "inapplicable"
    assert all(p.read_bytes() == b for p, b in before.items())
    out = tmp_path / "assessment"
    for n, h in read_json(out / "checksums.json").items():
        assert hashlib.sha256((out / n).read_bytes()).hexdigest() == h
    second = tmp_path / "second"
    assert assess(tmp_path, directory, output_dir=second) == s
    for p in out.iterdir():
        assert p.read_bytes() == (second / p.name).read_bytes()
    assert read_json(out / "source-facts.json")["exact_numeric_tokens_checked"] == 60


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(phase="fitting"),
        lambda d: d.update(fitting_authorized=True),
        lambda d: d.update(assumptions=["All records are independent"]),
        lambda d: d.update(contradictions=[]),
        lambda d: d["claims"].pop(),
        lambda d: d["claims"].append(d["claims"][0]),
        lambda d: d["claims"][0].update(status="verified"),
        lambda d: d["claims"][0].update(source_ids=["unbound"]),
        lambda d: d["claims"][0].update(source_ids="synthetic_review"),
        lambda d: d["claims"][0].update(limitation=""),
        lambda d: d["sources"][0].update(role="analyst_assertion"),
        lambda d: d["sources"][0].update(role="issue_report"),
        lambda d: d["sources"][0].update(access="unavailable", failure_reason="challenge"),
        lambda d: d["sources"][0].update(url="http://example.invalid/evidence"),
        lambda d: d["sources"][0].update(source_id="../escape"),
        lambda d: d["sources"][0].update(size_bytes=True),
        lambda d: d["sources"].append(d["sources"][0]),
        lambda d: d.update(protocol_sha256="wrong"),
    ],
)
def test_invalid_claims_and_evidence_promotion_rejected(mutation):
    d = settings()[2].to_dict()
    mutation(d)
    with pytest.raises(ValueError):
        EmpiricalEvidence.from_dict(d)


@pytest.mark.parametrize("name", ["source_sha256", "profile_sha256", "protocol_sha256"])
def test_all_bindings_enforced(tmp_path, name):
    directory = curated(tmp_path)
    p, c, e = settings()
    with pytest.raises(ValueError, match="exact source"):
        assess(tmp_path, directory, evidence=replace(e, **{name: "0" * 64}))
    assert not (tmp_path / "assessment").exists()
    assert p == EmpiricalProfile.from_dict(p.to_dict())
    assert c == EmpiricalProtocol.from_dict(c.to_dict())


@pytest.mark.parametrize(
    "filename",
    [
        "analysis/morphology.csv",
        "analysis/ecology.csv",
        "evaluation/references.csv",
        "evaluation/metadata.csv",
        "audit/source_index.csv",
        "audit/row_accounting.json",
        "audit/profile.json",
        "audit/protocol.json",
        "analysis/dataset.json",
        "summary.json",
    ],
)
def test_semantics_reconstructed_even_after_checksums_are_rewritten(tmp_path, filename):
    directory = curated(tmp_path)
    path = directory / filename
    if path.suffix == ".csv":
        path.write_text(path.read_text().replace("unit_", "bad_", 1))
    else:
        obj = read_json(path)
        if filename == "audit/row_accounting.json":
            obj[0]["included"] = not obj[0]["included"]
        elif filename == "summary.json":
            obj["included_rows"] = 999
        elif filename == "analysis/dataset.json":
            obj["modalities"][0]["features"][0]["role"] = "covariate"
        else:
            obj["dataset_id"] = "altered"
        path.write_text(json.dumps(obj))
    refresh(directory)
    with pytest.raises(ValueError):
        assess(tmp_path, directory)
    assert not (tmp_path / "assessment").exists()


def test_corrupt_snapshot_and_missing_curated_artifact_rejected(tmp_path):
    directory = curated(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "synthetic_review.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="snapshot"):
        assess(tmp_path, directory, evidence_dir=snapshots)
    (directory / "analysis/ecology.csv").unlink()
    with pytest.raises(ValueError):
        assess(tmp_path, directory)


def test_unlisted_file_and_symlink_cannot_hide_in_curated_manifest(tmp_path):
    directory = curated(tmp_path)
    other = directory / "extra.txt"
    other.write_text("undeclared")
    with pytest.raises(ValueError, match="Unlisted"):
        assess(tmp_path, directory)
    other.unlink()
    other.symlink_to(F / "source.csv")
    with pytest.raises(ValueError, match="symbolic"):
        assess(tmp_path, directory)


def test_new_outputs_only_and_input_overlap_rejected(tmp_path):
    directory = curated(tmp_path)
    assess(tmp_path, directory)
    with pytest.raises(FileExistsError):
        assess(tmp_path, directory)
    with pytest.raises(ValueError, match="separate"):
        assess(tmp_path, directory, output_dir=directory / "assessment")


def test_late_failure_is_retained_without_completed_summary(tmp_path, monkeypatch):
    directory = curated(tmp_path)
    import super_deli.workflows.empirical_assessment as module

    original = module.verify_curated
    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("input mutation during assessment")
        return original(path)

    monkeypatch.setattr(module, "verify_curated", changed)
    with pytest.raises(ValueError, match="input mutation"):
        assess(tmp_path, directory)
    assert read_json(tmp_path / "assessment/failure.json")["status"] == "failed"
    assert not (tmp_path / "assessment/summary.json").exists()


def test_reference_mutation_changes_coverage_but_never_independence_or_inputs(tmp_path):
    p, c, e = settings()
    rows = list(csv.DictReader(io.StringIO((F / "source.csv").read_text())))
    for row in rows:
        row["Species"] = "NA"
    b = io.StringIO(newline="")
    writer = csv.DictWriter(b, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = b.getvalue().encode()
    source = tmp_path / "relabeled.csv"
    source.write_bytes(raw)
    lock = replace(
        p.source,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    )
    p = replace(p, source=lock)
    c = replace(c, source_sha256=lock.sha256, profile_sha256=configuration_hash(p.to_dict()))
    e = replace(
        e,
        source_sha256=lock.sha256,
        profile_sha256=configuration_hash(p.to_dict()),
        protocol_sha256=configuration_hash(c.to_dict()),
    )
    directory = tmp_path / "curated"
    curate_empirical(source, profile=p, protocol=c, output_dir=directory)
    s = assess_empirical(
        source,
        profile=p,
        protocol=c,
        evidence=e,
        curated_dir=directory,
        evidence_dir=E,
        output_dir=tmp_path / "assessment",
    )
    assert s["reference_coverage"] == 0
    assert s["readiness"]["published_reference_description"]["status"] == "blocked"
    assert s["readiness"]["independent_unit_point_fitting"]["status"] == "blocked"
    original = tmp_path / "original"
    original.mkdir()
    baseline = curated(original)
    for path in (baseline / "analysis").iterdir():
        assert path.read_bytes() == (directory / "analysis" / path.name).read_bytes()


def test_cli_does_not_import_estimation_or_network_libraries(tmp_path, capsys, monkeypatch):
    directory = curated(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "numpy",
            "scipy",
            "sklearn",
            "snpio",
            "requests",
        } or (name.startswith("super_deli.models.") and not name.endswith("_config")):
            raise AssertionError("Scientific/network imports forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        main(
            [
                "assess-empirical",
                str(F / "profile.json"),
                "--source",
                str(F / "source.csv"),
                "--protocol",
                str(F / "protocol.json"),
                "--evidence",
                str(E / "evidence.json"),
                "--curated-dir",
                str(directory),
                "--evidence-dir",
                str(E),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["empirical_fits"] == 0


def test_public_real_evidence_is_bound_and_preserves_legacy_protocol():
    f = ROOT / "examples/empirical-foundation"
    p = EmpiricalProfile.from_dict(read_json(f / "profile.json"))
    c = EmpiricalProtocol.from_dict(read_json(f / "protocol.json"))
    e = EmpiricalEvidence.from_dict(read_json(ROOT / "examples/empirical-gates/evidence.json"))
    e.check_bindings(p, c)
    assert not c.fitting_authorized and not c.independent_units_confirmed
    assert any(s.access == "unavailable" and s.size_bytes > 0 for s in e.sources)
    assert e.assumptions == ()
