"""Real SNPio reader checks against hand-authored cross-format call truth."""

import importlib.util
import json
import logging
import shutil
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pytest

from super_deli import GenomicReaderOptions, load_config
from super_deli.cli import main
from super_deli.io.genomic import GenomicIngestionError, ingest_genomic

pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(
        importlib.util.find_spec("snpio") is None, reason="Install the genomics extra"
    ),
]
FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "genomic"
CASES = [
    ("phylip", "tiny.phy", None),
    ("vcf", "tiny.vcf", None),
    ("genepop", "tiny.gen", None),
    ("structure", "tiny_one_row.str", None),
    ("structure", "tiny_two_row.str", None),
    (
        "structure",
        "tiny_with_popids.str",
        GenomicReaderOptions(has_popids=True, has_marker_names=True),
    ),
]


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURES, raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw, load_config(raw / "dataset.json")


def snapshot(directory):
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("format,filename,options", CASES)
@pytest.mark.parametrize("with_popmap", [False, True])
def test_formats_preserve_calls_missingness_sample_order_and_originals(
    source, tmp_path, format, filename, options, with_popmap
):
    import numpy as np
    from snpio.read_input.genotype_data import GenotypeData

    raw, config = source
    modality = replace(
        config.modalities[0],
        format=format,
        path=filename,
        reader_options=options,
        popmapfile="tiny.popmap" if with_popmap else None,
    )
    config = replace(config, modalities=(modality,))
    before = snapshot(raw)
    expected = json.loads((raw / "expected.json").read_text())
    loaded = ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")
    assert isinstance(loaded.genotype_data, GenotypeData)
    assert loaded.sample_ids == tuple(expected["sample_ids"])
    assert loaded.specimen_ids == tuple(expected["specimen_ids"])
    np.testing.assert_array_equal(loaded.genotype_data.snp_data, expected["iupac_calls"])
    np.testing.assert_array_equal(loaded.missing_mask, expected["missing_mask"])
    np.testing.assert_array_equal(
        np.load(loaded.output_dir / "genotypes.npy", allow_pickle=False), expected["iupac_calls"]
    )
    assert not loaded.missing_mask.flags.writeable
    assert loaded.summary["sample_count"] == 2
    assert loaded.summary["locus_count"] == 4
    assert loaded.summary["missing_call_count"] == 1
    assert loaded.summary["provenance"]["software"]["snpio"] == version("snpio")
    assert len(loaded.summary["provenance"]["inputs"]) == (2 if with_popmap else 1)
    if with_popmap:
        assert loaded.summary["population_metadata"]["source"] == "popmap"
        assert loaded.summary["population_metadata"]["labels_in_sample_order"] == ("p1", "p2")
    elif format == "genepop":
        assert loaded.genotype_data.popmapfile is None
        assert loaded.summary["population_metadata"]["labels_in_sample_order"] == ("Pop1", "Pop1")
        assert loaded.genotype_data.popmap_inverse == {"Pop1": ["sample_b", "sample_a"]}
    elif options is None:
        assert loaded.genotype_data.popmapfile is None
        assert loaded.summary["population_metadata"]["labels_in_sample_order"] is None
    assert snapshot(raw) == before
    assert not list(Path.cwd().iterdir())


def test_existing_output_is_preserved(source, tmp_path):
    raw, config = source
    loaded = ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")
    before = snapshot(loaded.output_dir)
    with pytest.raises(FileExistsError):
        ingest_genomic(config, base_dir=raw, output_dir=loaded.output_dir)
    assert snapshot(loaded.output_dir) == before


def test_unmatched_samples_fail_without_silent_subset_or_positional_join(source, tmp_path):
    raw, config = source
    modality = replace(config.modalities[0], observations=config.modalities[0].observations[:1])
    config = replace(config, modalities=(modality,))
    with pytest.raises(GenomicIngestionError, match="unexpected samples"):
        ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")
    failure = json.loads((tmp_path / "run" / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert not (tmp_path / "run" / "summary.json").exists()


def test_population_conflicts_are_rejected(source, tmp_path):
    raw, config = source
    specimens = tuple(replace(s, population_id="different") for s in config.specimens)
    modality = replace(config.modalities[0], popmapfile="tiny.popmap")
    config = replace(
        config, populations=("different",), specimens=specimens, modalities=(modality,)
    )
    with pytest.raises(GenomicIngestionError, match="Population metadata conflict"):
        ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")


def test_repeated_source_samples_link_to_one_specimen_without_aggregation(source, tmp_path):
    raw, config = source
    observations = tuple(
        replace(o, unit_id="specimen_a") for o in config.modalities[0].observations
    )
    modality = replace(config.modalities[0], observations=observations)
    loaded = ingest_genomic(
        replace(config, modalities=(modality,)), base_dir=raw, output_dir=tmp_path / "run"
    )
    assert loaded.specimen_ids == ("specimen_a", "specimen_a")
    assert loaded.summary["sample_count"] == 2
    assert loaded.summary["unique_specimen_count"] == 1


@pytest.mark.parametrize(
    "format,filename,encoding",
    [
        (
            "structure",
            "tiny_one_row.str",
            (("1", "A"), ("2", "C"), ("3", "G"), ("4", "T"), ("-9", "N")),
        ),
        ("genepop", "tiny.gen", (("01", "A"), ("02", "C"), ("03", "G"), ("04", "T"))),
    ],
)
def test_explicit_allele_encodings_reach_readers(source, tmp_path, format, filename, encoding):
    import numpy as np

    raw, config = source
    modality = replace(
        config.modalities[0],
        format=format,
        path=filename,
        reader_options=GenomicReaderOptions(allele_encoding=encoding),
    )
    loaded = ingest_genomic(
        replace(config, modalities=(modality,)), base_dir=raw, output_dir=tmp_path / "run"
    )
    expected = json.loads((raw / "expected.json").read_text())
    np.testing.assert_array_equal(loaded.genotype_data.snp_data, expected["iupac_calls"])


def test_multiple_genomic_sources_require_explicit_selection(source, tmp_path):
    raw, config = source
    config = replace(
        config, modalities=(*config.modalities, replace(config.modalities[0], name="other"))
    )
    with pytest.raises(GenomicIngestionError, match="exactly one"):
        ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()
    loaded = ingest_genomic(
        config, base_dir=raw, output_dir=tmp_path / "run", modality_name="other"
    )
    assert loaded.summary["modality"] == "other"


def test_cli_emits_clean_json_and_restores_callers_log_handlers(source, tmp_path, capsys):
    raw, _ = source
    logger = logging.getLogger("snpio.io.phylip_reader")
    handler = logging.NullHandler()
    original = tuple(logger.handlers)
    logger.addHandler(handler)
    try:
        assert (
            main(
                ["ingest-genomic", str(raw / "dataset.json"), "--output-dir", str(tmp_path / "run")]
            )
            == 0
        )
        summary = json.loads(capsys.readouterr().out)
        assert summary["status"] == "complete"
        assert tuple(logger.handlers) == (*original, handler)
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize(
    "contents",
    [
        "sample_b p1\nsample_b p2\nsample_a p2\n",
        "sample_b p1\n",
        "sample_b p1\nsample_a p2\nextra p3\n",
    ],
)
def test_invalid_popmaps_cannot_silently_overwrite_or_subset_samples(source, tmp_path, contents):
    raw, config = source
    (raw / "tiny.popmap").write_text(contents)
    config = replace(config, modalities=(replace(config.modalities[0], popmapfile="tiny.popmap"),))
    with pytest.raises(GenomicIngestionError, match="popmap|Popmap"):
        ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")


def test_bgzipped_vcf_indexes_and_metadata_stay_inside_new_run(source, tmp_path):
    import numpy as np
    import pysam

    raw, config = source
    pysam.tabix_compress(str(raw / "tiny.vcf"), str(raw / "tiny.vcf.gz"))
    before = snapshot(raw)
    modality = replace(config.modalities[0], format="vcf", path="tiny.vcf.gz")
    loaded = ingest_genomic(
        replace(config, modalities=(modality,)), base_dir=raw, output_dir=tmp_path / "run"
    )
    np.testing.assert_array_equal(
        loaded.genotype_data.snp_data,
        json.loads((raw / "expected.json").read_text())["iupac_calls"],
    )
    assert (
        Path(loaded.genotype_data.vcf_attributes_fn)
        .resolve()
        .is_relative_to(loaded.output_dir.resolve())
    )
    assert snapshot(raw) == before


def test_duplicate_source_sample_ids_are_rejected(source, tmp_path):
    raw, config = source
    path = raw / "tiny.phy"
    path.write_text(path.read_text().replace("sample_a", "sample_b"))
    with pytest.raises(GenomicIngestionError):
        ingest_genomic(config, base_dir=raw, output_dir=tmp_path / "run")
    assert not (tmp_path / "run" / "summary.json").exists()
