"""Native parity, source identity, incomplete-data and failure semantics."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.workflows.genomic_analysis import run_genomic_analysis
from super_deli.workflows.genomic_analysis_adapters import read_genomic_arm, run_genomic_arm

np = pytest.importorskip("numpy")
pytest.importorskip("snpio")
pytest.importorskip("sklearn")


def invented(tmp_path, draws=0):
    path = Path(__file__).resolve().parents[1] / "examples/matched-genomic/smoke.py"
    spec = importlib.util.spec_from_file_location("invented_genomic_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.invented(tmp_path / "source", draws=draws)


def test_native_parity_ordering_and_stability(tmp_path):
    from super_deli.core.schema import config_from_dict

    d, c = invented(tmp_path, draws=2)
    out = tmp_path / "comparison"
    s = run_genomic_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)
    assert s["status"] == "complete"
    assert len(s["comparisons"]) == 10 and s["inference"]["p_value"] is None
    frozen = config_from_dict(json.loads((out / "inputs/dataset.json").read_text()))
    cohort = json.loads((out / "inputs/cohort.json").read_text())
    for a, row in zip(c.arms, s["models"], strict=True):
        assert row["stability"]["requested"] == 2
        run_genomic_arm(a, frozen, out / "inputs", tmp_path / ("replay-" + a.name))
        x = read_genomic_arm(a, out / "arms" / a.name, cohort, frozen)
        y = read_genomic_arm(a, tmp_path / ("replay-" + a.name), cohort, frozen)
        assert x == y
    assert s["models"][0]["evidence_type"] != s["models"][-1]["evidence_type"]
    text = (out / "report.html").read_text()
    assert "https://" not in text and "no automatic winner" in text.lower()
    with pytest.raises(FileExistsError):
        run_genomic_analysis(d, analysis_config=c, base_dir=tmp_path / "source", output_dir=out)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_genotype",
        "missing_trait",
        "duplicate_specimen",
        "block_length",
        "locality_ecology",
        "extra_specimen",
    ],
)
def test_prefit_guards(tmp_path, fault):
    from super_deli.core.data import Specimen

    d, c = invented(tmp_path)
    if fault == "missing_genotype":
        p = tmp_path / "source/calls.phy"
        lines = p.read_text().splitlines()
        left, right = lines[1].split()
        lines[1] = left + " N" + right[1:]
        p.write_text("\n".join(lines) + "\n")
    elif fault == "missing_trait":
        p = tmp_path / "source/traits.csv"
        lines = p.read_text().splitlines()
        r = lines[1].split(",")
        r[1] = ""
        lines[1] = ",".join(r)
        p.write_text("\n".join(lines) + "\n")
    elif fault == "duplicate_specimen":
        m = d.modalities[1]
        obs = list(m.observations)
        obs[1] = replace(obs[1], unit_id=obs[0].unit_id)
        d = replace(
            d, modalities=(d.modalities[0], replace(m, observations=tuple(obs)), d.modalities[2])
        )
    elif fault == "block_length":
        c = replace(c, arms=(replace(c.arms[0], block_ids=("a", "b")), *c.arms[1:]))
    elif fault == "locality_ecology":
        d = replace(d, modalities=(*d.modalities[:2], replace(d.modalities[2], level="locality")))
    else:
        d = replace(d, specimens=(*d.specimens, Specimen("extra")))
    with pytest.raises((ValueError, RuntimeError)):
        run_genomic_analysis(
            d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "bad"
        )
    assert (tmp_path / "bad/failure.json").is_file()
    assert not (tmp_path / "bad/arms").exists()


def test_failed_arm_is_not_k1(tmp_path, monkeypatch):
    from super_deli.workflows import genomic_analysis as workflow

    d, c = invented(tmp_path)
    native = workflow.run_genomic_arm

    def dispatch(arm, *args):
        if arm.name == "morphology":
            raise RuntimeError("invented fit failure")
        return native(arm, *args)

    monkeypatch.setattr(workflow, "run_genomic_arm", dispatch)
    s = workflow.run_genomic_analysis(
        d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "partial"
    )
    assert s["status"] == "partial"
    failed = next(r for r in s["models"] if r["name"] == "morphology")
    assert failed["status"] == "failed" and failed["selected_k"] is None
    assert all(
        r["status"] == "unavailable"
        for r in s["comparisons"]
        if "morphology" in (r["left"], r["right"])
    )


def test_snapshot_tampering_aborts(tmp_path, monkeypatch):
    from super_deli.workflows import genomic_analysis as workflow

    d, c = invented(tmp_path)
    native = workflow.run_genomic_arm

    def dispatch(arm, dataset, base, directory):
        value = native(arm, dataset, base, directory)
        (base / dataset.modalities[0].path).write_text("tampered")
        return value

    monkeypatch.setattr(workflow, "run_genomic_arm", dispatch)
    with pytest.raises(ValueError, match="hash mismatch"):
        workflow.run_genomic_analysis(
            d, analysis_config=c, base_dir=tmp_path / "source", output_dir=tmp_path / "tampered"
        )


@pytest.mark.parametrize("kind", ["phylip", "vcf", "structure", "genepop"])
@pytest.mark.parametrize("popmap", [False, True])
def test_four_snpio_formats_one_phenotype_and_optional_popmap(tmp_path, kind, popmap):
    from super_deli.core.data import GenomicReaderOptions

    d, c = invented(tmp_path)
    source = tmp_path / "source"
    lines = (source / "calls.phy").read_text().splitlines()[1:]
    calls = [(row.split()[0], ["AMC".index(v) for v in row.split()[1]]) for row in lines]
    options = None
    if kind == "phylip":
        text = (source / "calls.phy").read_text()
    elif kind == "vcf":
        text = (
            "##fileformat=VCFv4.2\n##contig=<ID=1,length=1000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(name for name, _ in calls)
            + "\n"
        )
        for j in range(64):
            text += (
                f"1\t{j + 1}\tlocus{j}\tA\tC\t.\tPASS\t.\tGT\t"
                + "\t".join(("0/0", "0/1", "1/1")[raw[j]] for _, raw in calls)
                + "\n"
            )
    elif kind == "structure":
        text = "".join(
            name + " " + " ".join(("1 1", "1 2", "2 2")[v] for v in row) + "\n"
            for name, row in calls
        )
        options = GenomicReaderOptions(allele_start_col=1, allele_encoding=(("1", "A"), ("2", "C")))
    else:
        text = "Invented calls\n" + "".join(f"locus{j}\n" for j in range(64)) + "Pop\n"
        text += "".join(
            name + ", " + " ".join(("0101", "0102", "0202")[v] for v in row) + "\n"
            for name, row in calls
        )
        options = GenomicReaderOptions(allele_encoding=(("01", "A"), ("02", "C")))
    name = "converted." + dict(phylip="phy", vcf="vcf", structure="str", genepop="gen")[kind]
    (source / name).write_text(text)
    if popmap:
        (source / "popmap.txt").write_text("".join(name + " groupA\n" for name, _ in calls))
    g = replace(
        d.modalities[0],
        path=name,
        format=kind,
        reader_options=options,
        popmapfile="popmap.txt" if popmap else None,
    )
    d = replace(d, modalities=(g, d.modalities[1]))
    arm = c.arms[-1]
    model = replace(arm.model, views=arm.model.views[:2])
    c = replace(
        c,
        arms=(*c.arms[:2], replace(arm, modalities=("snp", "traits"), model=model)),
        ecology_measurement_scope=None,
    )
    s = run_genomic_analysis(
        d, analysis_config=c, base_dir=source, output_dir=tmp_path / "analysis"
    )
    assert s["status"] == "complete" and len(s["models"]) == 3
    manifest = json.loads((tmp_path / "analysis/manifest.json").read_text())
    assert len(manifest["sources"]) == 2 + popmap
    assert g.popmapfile == ("popmap.txt" if popmap else None)
