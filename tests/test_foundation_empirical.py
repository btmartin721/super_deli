"""Integrity, role separation and fit-free applicability boundaries."""

import builtins
import csv
import hashlib
import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.cli import main
from super_deli.core.empirical import EmpiricalProfile
from super_deli.core.empirical_protocol import EmpiricalProtocol
from super_deli.core.provenance import configuration_hash
from super_deli.core.validation import validate_dataset
from super_deli.io.config import load_config, read_json
from super_deli.workflows.empirical_curation import curate_empirical

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/empirical-foundation/fixture"


def settings():
    return (
        EmpiricalProfile.from_dict(read_json(FIXTURE / "profile.json")),
        EmpiricalProtocol.from_dict(read_json(FIXTURE / "protocol.json")),
    )


def changed(tmp_path, mutation):
    p, protocol = settings()
    with (FIXTURE / "source.csv").open(newline="") as h:
        reader = csv.DictReader(h)
        columns, rows = reader.fieldnames, list(reader)
    mutation(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    source = tmp_path / "modified.csv"
    source.write_bytes(raw)
    lock = replace(
        p.source,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    )
    p = replace(p, source=lock)
    return (
        source,
        p,
        replace(
            protocol, source_sha256=lock.sha256, profile_sha256=configuration_hash(p.to_dict())
        ),
    )


def run(tmp_path, *, source=None, profile=None, protocol=None, name="curated"):
    p, c = settings()
    return curate_empirical(
        source or FIXTURE / "source.csv",
        profile=profile or p,
        protocol=protocol or c,
        output_dir=tmp_path / name,
    )


def test_valid_manifest_exact_values_and_complete_exclusion_accounting(tmp_path):
    before = (FIXTURE / "source.csv").read_bytes()
    result = run(tmp_path)
    assert result["source_rows"] == 12 and result["included_rows"] == 10
    assert result["excluded_rows"] == 2 and result["reference_coverage"] == 10
    assert result["readiness"]["statistical_fitting"] is False
    assert result["readiness"]["bootstrap"] == "not_authorized"
    out = tmp_path / "curated"
    config = load_config(out / "analysis/dataset.json")
    assert validate_dataset(config, base_dir=out / "analysis").valid
    assert not config.localities and not config.populations
    assert all(s.locality_id is None and s.population_id is None for s in config.specimens)
    with io.StringIO(before.decode()) as f:
        source = {(r["studyName"], r["Individual ID"]): r for r in csv.DictReader(f)}
    with (out / "audit/source_index.csv").open(newline="") as f:
        index = {
            r["specimen_id"]: source[r["studyName"], r["Individual ID"]] for r in csv.DictReader(f)
        }
    p, _ = settings()
    for m in config.modalities:
        with (out / "analysis" / m.path).open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["observation_id", *(x.name for x in m.features)]
            links = {x.observation_id: x.unit_id for x in m.observations}
            for row in reader:
                original = index[links[row["observation_id"]]]
                for feature in p.features:
                    if feature.kind == m.kind:
                        assert row[feature.name] == original[feature.source_column]
    accounting = read_json(out / "audit/row_accounting.json")
    assert sum(not r["included"] for r in accounting) == 2
    assert sum(len(r["missing_features"]) for r in accounting) == 2
    for name, digest in read_json(out / "checksums.json").items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert (FIXTURE / "source.csv").read_bytes() == before


def test_reference_metadata_and_source_order_cannot_change_analytical_inputs(tmp_path):
    run(tmp_path, name="original")

    def mutate(rows):
        for row in rows:
            row["Species"] = "withheld_reference"
            row["Sex"] = "changed_covariate"
            row["Island"] = "changed_site"
            row["Comments"] = "unused free text"
        rows[0]["Species"] = ""
        rows.reverse()

    source, p, c = changed(tmp_path, mutate)
    result = run(tmp_path, source=source, profile=p, protocol=c, name="mutated")
    assert result["included_rows"] == 10 and result["reference_coverage"] == 9
    for path in (tmp_path / "original/analysis").iterdir():
        assert path.read_bytes() == (tmp_path / "mutated/analysis" / path.name).read_bytes()
    assert (tmp_path / "original/evaluation/references.csv").read_bytes() != (
        tmp_path / "mutated/evaluation/references.csv"
    ).read_bytes()


@pytest.mark.parametrize("value", ["", "NA", " padded "])
def test_unknown_identity_cannot_be_silently_removed(tmp_path, value):
    source, p, c = changed(tmp_path, lambda rows: rows[0].update({"Individual ID": value}))
    with pytest.raises(ValueError):
        run(tmp_path, source=source, profile=p, protocol=c)
    assert not (tmp_path / "curated").exists()


def test_duplicate_identity_is_rejected_without_deduplication(tmp_path):
    def mutate(rows):
        rows[1]["Individual ID"] = rows[0]["Individual ID"]

    source, p, c = changed(tmp_path, mutate)
    with pytest.raises(ValueError, match="duplicate source identity"):
        run(tmp_path, source=source, profile=p, protocol=c)


@pytest.mark.parametrize(
    "column,value",
    [("Body Mass (g)", "0"), ("Delta 13 C (o/oo)", "NaN"), ("Culmen Length (mm)", " 12")],
)
def test_invalid_measurement_is_not_imputed_or_silently_excluded(tmp_path, column, value):
    source, p, c = changed(tmp_path, lambda rows: rows[0].update({column: value}))
    with pytest.raises(ValueError):
        run(tmp_path, source=source, profile=p, protocol=c)


@pytest.mark.parametrize(
    "field,value", [("sha256", "0" * 64), ("size_bytes", 1), ("git_blob_sha1", "0" * 40)]
)
def test_source_fingerprints_are_all_enforced(tmp_path, field, value):
    p, c = settings()
    p = replace(p, source=replace(p.source, **{field: value}))
    c = replace(c, source_sha256=p.source.sha256, profile_sha256=configuration_hash(p.to_dict()))
    with pytest.raises(ValueError):
        run(tmp_path, profile=p, protocol=c)
    assert not (tmp_path / "curated").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra="undeclared"),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(identity_columns=["Species"]),
        lambda d: d["features"][0].update(source_column="Species"),
        lambda d: d["features"][0].update(name="observation_id"),
        lambda d: d["features"][0].update(kind=[]),
        lambda d: d["source"].update(url="http://example.invalid/source.csv"),
        lambda d: d["source"].update(size_bytes=True),
        lambda d: d.update(identity_grain="independent_individual"),
    ],
)
def test_invalid_or_leaking_roles_rejected(mutation):
    raw = settings()[0].to_dict()
    mutation(raw)
    with pytest.raises(ValueError):
        EmpiricalProfile.from_dict(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fitting_authorized", True),
        ("independent_units_confirmed", True),
        ("reference_independence_confirmed", True),
        ("stability_draws", 99),
        ("significance_draws", 99),
        ("target_level", "locality"),
        ("n_init", 100),
        ("max_k", 3),
        ("scaling", "none"),
        ("view_weights", "optimized"),
        ("n_threads", 4),
        ("selection_criterion", "aic"),
        ("cohort_rule", "known_species_only"),
        ("schema_version", True),
    ],
)
def test_frozen_protocol_cannot_enable_fits_or_outcome_tuning(field, value):
    p = settings()[1].to_dict()
    p[field] = value
    with pytest.raises(ValueError):
        EmpiricalProtocol.from_dict(p)


def test_protocol_must_bind_exact_source_and_dataset(tmp_path):
    p, c = settings()
    with pytest.raises(ValueError, match="exact dataset"):
        run(tmp_path, protocol=replace(c, source_sha256="0" * 64))
    with pytest.raises(ValueError, match="exact dataset"):
        run(tmp_path, protocol=replace(c, dataset_id="other_dataset"))
    assert p == EmpiricalProfile.from_dict(p.to_dict())
    assert c == EmpiricalProtocol.from_dict(c.to_dict())


def test_cli_is_fit_free_and_does_not_import_scientific_libraries(tmp_path, capsys, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "scipy", "sklearn", "snpio"}:
            raise AssertionError("Scientific imports forbidden in curation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        main(
            [
                "curate-empirical",
                str(FIXTURE / "profile.json"),
                "--source",
                str(FIXTURE / "source.csv"),
                "--protocol",
                str(FIXTURE / "protocol.json"),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["included_rows"] == 10


def test_new_outputs_only_and_late_failure_preserved(tmp_path, monkeypatch):
    run(tmp_path)
    original = (tmp_path / "curated/checksums.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert (tmp_path / "curated/checksums.json").read_bytes() == original
    import super_deli.workflows.empirical_curation as module

    def fail(*args, **kwargs):
        raise ValueError("retained ingestion failure")

    monkeypatch.setattr(module, "ingest_tabular", fail)
    with pytest.raises(ValueError, match="retained ingestion failure"):
        run(tmp_path, name="failed")
    assert read_json(tmp_path / "failed/failure.json")["status"] == "failed"
    assert not (tmp_path / "failed/summary.json").exists()


def downloader():
    spec = importlib.util.spec_from_file_location(
        "empirical_download", ROOT / "examples/empirical-foundation/download.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["valid", "truncated", "extra", "corrupt", "redirect"])
def test_download_bounded_integrity_and_no_replacement(tmp_path, monkeypatch, mode):
    module = downloader()
    p, _ = settings()
    body = (FIXTURE / "source.csv").read_bytes()
    if mode == "truncated":
        body = body[:-1]
    if mode == "extra":
        body += b"x"
    if mode == "corrupt":
        body = b"x" + body[1:]
    response = io.BytesIO(body)
    response.url = "https://example.invalid/unlocked" if mode == "redirect" else p.source.url
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: response)
    out = tmp_path / "download"
    if mode == "valid":
        assert module.download(FIXTURE / "profile.json", out).read_bytes() == body
        with pytest.raises(FileExistsError):
            module.download(FIXTURE / "profile.json", out)
    else:
        with pytest.raises(ValueError):
            module.download(FIXTURE / "profile.json", out)
        assert read_json(out / "failure.json")["status"] == "failed"
        assert not (out / "source_001.csv").exists()


def test_protocol_binds_feature_units_and_identity_policy(tmp_path):
    p, c = settings()
    altered = replace(p, features=(replace(p.features[0], unit="cm"), *p.features[1:]))
    with pytest.raises(ValueError, match="feature/identity declarations"):
        run(tmp_path, profile=altered, protocol=c)
    assert not (tmp_path / "curated").exists()


def test_integrity_write_failure_cannot_publish_completed_summary(tmp_path, monkeypatch):
    import super_deli.workflows.empirical_curation as module

    original = module.write_json

    def fail_checksums(value, path):
        if Path(path).name == "checksums.json":
            raise OSError("retained checksum write failure")
        return original(value, path)

    monkeypatch.setattr(module, "write_json", fail_checksums)
    with pytest.raises(OSError, match="checksum write failure"):
        run(tmp_path)
    assert not (tmp_path / "curated/summary.json").exists()
    assert read_json(tmp_path / "curated/failure.json")["status"] == "failed"
    assert (tmp_path / "curated/summary.pending.json").exists()
