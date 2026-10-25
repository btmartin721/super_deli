"""Fit-free empirical source integrity, role separation and evidence boundaries."""

import builtins
import csv
import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.cli import main
from super_deli.core.replacement_audit import ReplacementAuditPlan
from super_deli.io.config import read_json
from super_deli.io.specimen_audit import genbank_references, inspect_tps
from super_deli.workflows.replacement_audit import ARCHIVE_NAMES, audit_replacement

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "examples/replacement-empirical/fixture"


def settings():
    return ReplacementAuditPlan.from_dict(read_json(F / "plan.json"))


def run(tmp_path, *, plan=None, source=F, name="audit"):
    return audit_replacement(plan or settings(), source_dir=source, output_dir=tmp_path / name)


def changed(tmp_path, role, mutate):
    root = tmp_path / "sources"
    shutil.copytree(F, root)
    p = settings()
    locks = {s.role: s for s in p.sources}
    path = root / locks[role].filename
    mutate(path)
    # Rebind deliberate semantic mutants, keeping the archive's own inventory consistent.
    invpath = root / locks["archive_files"].filename
    inventory = read_json(invpath)
    for entry in inventory["_embedded"]["stash:files"]:
        r = next(r for r, name in ARCHIVE_NAMES.items() if name == entry["path"])
        raw = (root / locks[r].filename).read_bytes()
        entry.update(size=len(raw), digest=hashlib.sha256(raw).hexdigest())
    invpath.write_text(json.dumps(inventory))
    docpath = root / locks["archive_metadata"].filename
    doc = read_json(docpath)
    doc["storageSize"] = sum(x["size"] for x in inventory["_embedded"]["stash:files"])
    docpath.write_text(json.dumps(doc))
    sources = tuple(
        replace(
            s,
            size_bytes=(root / s.filename).stat().st_size,
            sha256=hashlib.sha256((root / s.filename).read_bytes()).hexdigest(),
        )
        for s in p.sources
    )
    return root, replace(p, sources=sources)


def test_full_audit_preserves_missingness_scales_and_separate_references(tmp_path):
    result = run(tmp_path)
    assert result["metadata_records"] == 4 and result["reference_records"] == 4
    assert result["reference_link_counts"] == {"candidate_namespace": 3, "unmatched": 1}
    assert result["literal_authoritative_references"] == 0
    assert result["readiness"]["independent_unit_fitting"].startswith("blocked")
    assert result["readiness"]["independent_reference_concordance"].startswith("conditional")
    assert (
        result["empirical_fits"]
        == result["pvalues_generated"]
        == result["stability_draws"]
        == result["significance_draws"]
        == 0
    )
    out = tmp_path / "audit"
    dataset = read_json(out / "analysis/dataset.json")
    assert (
        len(dataset["specimens"]) == 4 and not dataset["populations"] and not dataset["localities"]
    )
    assert len(dataset["modalities"]) == 2
    assert {m["name"] for m in dataset["modalities"]} == {"full_disc", "colour"}
    assert "species" not in (out / "analysis/colour.csv").read_text()
    assert ",0,0,90,-,-" in (out / "analysis/colour.csv").read_text()
    assert read_json(out / "colour-audit.json")["missing_by_column"]["mean_grey_band"] == 1
    tps = list((out / "analysis/full_disc_tps").glob("*.tps"))
    assert len(tps) == 3
    assert all(
        p.read_text().count("LM=9") == 1
        and "SCALE=0.010000" in p.read_text()
        and "VARIABLES" not in p.read_text()
        for p in tps
    )
    assert read_json(out / "geometry-audit.json")["anterior"][0]["curves"]
    assert read_json(out / "manifest-audit.json")["scale_unit"] is None
    assert result["archive"]["curationStatus"] == "Published"
    assert result["archive"]["versionStatus"] == "submitted"
    for n, h in read_json(out / "checksums.json").items():
        assert hashlib.sha256((out / n).read_bytes()).hexdigest() == h
    second = run(tmp_path, name="second")
    assert second == result
    for p in out.rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (tmp_path / "second" / p.relative_to(out)).read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(phase="fit"),
        lambda d: d.update(fitting_authorized=True),
        lambda d: d.update(reference_policy="authoritative_prefixes"),
        lambda d: d.update(sampling_policy="all_independent"),
        lambda d: d.update(scale_policy="convert_mm"),
        lambda d: d.update(voucher_prefixes=["LAB-", "LAB-A-"]),
        lambda d: d.update(review_limitations=[]),
        lambda d: d["sources"].pop(),
        lambda d: d["sources"][0].update(sha256="wrong"),
        lambda d: d["sources"][0].update(size_bytes=True),
        lambda d: d["sources"][0].update(filename="../escape"),
        lambda d: d["sources"][0].update(url="http://example.invalid/source"),
        lambda d: d.update(taxon_names=[["alpha", "Species a"], ["alpha", "Species b"]]),
    ],
)
def test_invalid_or_overpromoted_specifications_rejected(mutation):
    d = settings().to_dict()
    mutation(d)
    with pytest.raises(ValueError):
        ReplacementAuditPlan.from_dict(d)


@pytest.mark.parametrize(
    "role,old,new",
    [
        ("metadata", "U02,20", "U01,20"),
        ("metadata", "U01,10", "U01,0"),
        ("metadata", "length", "Length"),
        ("colour", "Id,", "ID,"),
        ("colour", "U02,200", "U01,200"),
        ("colour", "U01,100", "U01,NaN"),
        ("colour", "100,50,50", "256,50,50"),
        ("full_disc", "LM=9", "LM=8"),
        ("full_disc", "IMAGE=U02", "IMAGE=U01"),
        ("full_disc", "SCALE=0.010000", "SCALE=-1"),
        ("full_disc", "VARIABLES=OrigNum=0", "UNSUPPORTED=1"),
        ("anterior", "POINTS=8", "POINTS=7"),
        ("rostrum", "CURVES=2", "CURVES=1"),
    ],
)
def test_semantically_invalid_sources_leave_failure_without_summary(tmp_path, role, old, new):
    source, p = changed(
        tmp_path, role, lambda path: path.write_text(path.read_text().replace(old, new, 1))
    )
    with pytest.raises(ValueError):
        run(tmp_path, plan=p, source=source)
    assert read_json(tmp_path / "audit/failure.json")["status"] == "failed"
    assert not (tmp_path / "audit/summary.json").exists()


def test_hash_corruption_is_rejected_before_output_creation(tmp_path):
    root = tmp_path / "sources"
    shutil.copytree(F, root)
    (root / "metadata.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, source=root)
    assert not (tmp_path / "audit").exists()


def test_reference_labels_and_metadata_order_do_not_change_analytical_inputs(tmp_path):
    run(tmp_path, name="original")

    def mutate(path):
        rows = list(csv.DictReader(path.open(newline="")))
        for row in rows:
            row["species"] = "withheld"
            row["sex"] = "unknown"
        b = io.StringIO(newline="")
        wr = csv.DictWriter(b, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(reversed(rows))
        path.write_text(b.getvalue())

    source, p = changed(tmp_path, "metadata", mutate)
    result = run(tmp_path, source=source, plan=p, name="relabeled")
    assert result["candidate_organism_agreements"] == 0
    for path in (tmp_path / "original/analysis").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (
                    tmp_path
                    / "relabeled/analysis"
                    / path.relative_to(tmp_path / "original/analysis")
                ).read_bytes()
            )
    assert result["readiness"]["independent_unit_fitting"].startswith("blocked")


def test_ambiguous_molecular_candidate_never_becomes_an_authoritative_reference(tmp_path):
    def mutate(path):
        tree = ET.parse(path)
        root = tree.getroot()
        copy = ET.fromstring(ET.tostring(root[0]))
        copy.find("GBSeq_primary-accession").text = "ZZ000001"
        copy.find("GBSeq_accession-version").text = "ZZ000001.1"
        root.append(copy)
        tree.write(path)

    source, p = changed(tmp_path, "references", mutate)
    result = run(tmp_path, source=source, plan=p)
    assert result["reference_link_counts"]["ambiguous"] == 1
    assert result["literal_authoritative_references"] == 0
    assert result["readiness"]["independent_reference_concordance"].startswith("conditional")


def test_unmatched_images_are_accounted_for_without_inferring_identity(tmp_path):
    source, p = changed(
        tmp_path,
        "full_disc",
        lambda path: path.write_text(path.read_text().replace("IMAGE=U01", "IMAGE=U01.jpg")),
    )
    result = run(tmp_path, source=source, plan=p)
    assert result["full_disc_records"] == 3
    m = read_json(tmp_path / "audit/membership-audit.json")
    assert m["geometry"]["full_disc"]["literal_metadata_matches"] == 2
    assert m["geometry"]["full_disc"]["candidate_image_stem_matches"] == 3
    assert read_json(tmp_path / "audit/manifest-audit.json")["full_disc_observations"] == 2


def test_missing_scale_is_retained_without_physical_unit_or_imputation(tmp_path):
    source, p = changed(
        tmp_path,
        "full_disc",
        lambda path: path.write_text(path.read_text().replace("SCALE=0.010000\n", "", 1)),
    )
    result = run(tmp_path, source=source, plan=p)
    m = read_json(tmp_path / "audit/membership-audit.json")
    assert m["geometry"]["full_disc"]["absent_scales"] == 1
    assert result["readiness"]["physical_scale"].startswith("blocked")


def test_bad_genbank_sequence_or_duplicate_accession_is_not_silently_accepted(tmp_path):
    p = tmp_path / "reference.xml"
    p.write_text((F / "reference_records.xml").read_text().replace("acgtacgt", "acgt"))
    with pytest.raises(ValueError, match="sequence"):
        genbank_references(p)
    p.write_text((F / "reference_records.xml").read_text().replace("XY000001", "XY000000"))
    with pytest.raises(ValueError, match="duplicate"):
        genbank_references(p)


def test_curve_source_cannot_be_read_as_a_fixed_only_source():
    with pytest.raises(ValueError):
        inspect_tps(F / "anterior.tps", landmarks=5, curves=0)


def test_new_outputs_only_and_late_completion_failure(tmp_path, monkeypatch):
    run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    import super_deli.workflows.replacement_audit as module

    original = module.write_json

    def fail(value, path):
        if Path(path).name == "checksums.json":
            raise OSError("retained late completion failure")
        return original(value, path)

    monkeypatch.setattr(module, "write_json", fail)
    with pytest.raises(OSError):
        run(tmp_path, name="failed")
    assert read_json(tmp_path / "failed/failure.json")["status"] == "failed"
    assert not (tmp_path / "failed/summary.json").exists()


def test_source_overlap_and_symlinks_rejected(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        audit_replacement(settings(), source_dir=F, output_dir=F / "nested")
    source = tmp_path / "sources"
    shutil.copytree(F, source)
    p = source / "metadata.csv"
    p.unlink()
    p.symlink_to(F / "metadata.csv")
    with pytest.raises(ValueError, match="symbolic"):
        run(tmp_path, source=source)


def test_offline_cli_rejects_numerical_estimation_and_network_imports(
    tmp_path, monkeypatch, capsys
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "scipy", "sklearn", "snpio", "requests", "torch"}:
            raise AssertionError("Estimation/network import forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        main(
            [
                "audit-replacement-empirical",
                str(F / "plan.json"),
                "--source-dir",
                str(F),
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["empirical_fits"] == 0


def test_source_mutation_after_derivatives_withholds_completed_result(tmp_path, monkeypatch):
    source = tmp_path / "sources"
    shutil.copytree(F, source)
    import super_deli.workflows.replacement_audit as module

    original = module.write_manifests

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = source / "metadata.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(module, "write_manifests", mutate)
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, source=source)
    assert not (tmp_path / "audit/summary.json").exists()
    assert read_json(tmp_path / "audit/failure.json")["status"] == "failed"


def test_no_fixed_shape_join_does_not_claim_shape_readiness(tmp_path):
    source, p = changed(
        tmp_path,
        "full_disc",
        lambda path: path.write_text(
            path.read_text()
            .replace("IMAGE=U01", "IMAGE=missing01")
            .replace("IMAGE=U02", "IMAGE=missing02")
            .replace("IMAGE=U04", "IMAGE=missing04")
        ),
    )
    summary = run(tmp_path, source=source, plan=p)
    assert summary["readiness"]["fixed_shape"] == "blocked_no_literal_metadata_join"
    assert len(read_json(tmp_path / "audit/analysis/dataset.json")["modalities"]) == 1
