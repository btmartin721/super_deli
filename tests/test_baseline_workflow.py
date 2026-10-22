"""End-to-end SNPio encoding, artifact semantics, and failure isolation."""

# ruff: noqa: E402 -- skip optional scientific imports before importing the workflow.

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("snpio")

from super_deli import load_config
from super_deli.cli import main
from super_deli.core.provenance import configuration_hash
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.io.genomic import ingest_genomic
from super_deli.io.genomic_encoding import encode_diploid
from super_deli.models.config import BaselineConfig
from super_deli.models.pca_gmm import BaselineError
from super_deli.workflows.genomic_baseline import _result, parse_blocks, run_genomic_baseline

pytestmark = [pytest.mark.baseline, pytest.mark.genomic]
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    shutil.copytree(EXAMPLES / "baseline", raw)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return raw, load_config(raw / "dataset.json")


def snapshot(directory):
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_api_outputs_truth_separation_parameters_and_original_preservation(source, tmp_path):
    raw, dataset = source
    before = snapshot(raw)
    config = replace(BaselineConfig.from_dict(read_json(raw / "model.json")), n_bootstrap=4)
    blocks = parse_blocks(read_json(raw / "blocks.json"))
    output = tmp_path / "run"
    summary = run_genomic_baseline(
        dataset, model_config=config, base_dir=raw, output_dir=output, block_ids=blocks
    )
    assert summary["status"] == "complete" and summary["selected_k"] == 2
    assert summary["inference_scope"] == "candidate_genetic_clusters"
    assert summary["stability"]["requested"] == 4
    assert summary["encoding"]["exclusions"] == []
    assert summary["preprocessing"]["imputed_call_count"] > 0
    result = read_json(output / "result.json")
    digest = configuration_hash(summary["provenance"]["configuration"])
    assert digest == result["configuration_sha256"]
    assert digest == summary["provenance"]["configuration_sha256"]
    assert "truth.json" not in json.dumps(summary)
    assert snapshot(raw) == before and not list(Path.cwd().iterdir())
    assert not (output / "failure.json").exists()
    assert not list(output.rglob("*.pkl"))
    pvalues = [e for e in result["evidence"] if e["kind"] == "p_value"]
    assert len(pvalues) == 1 and pvalues[0]["status"] == "not_computed"
    assert all(e["value"] is None for e in pvalues)
    probabilities = np.load(output / "membership.npy", allow_pickle=False)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    with np.load(output / "preprocessing.npz", allow_pickle=False) as pre:
        x = np.load(output / "dosages.npy", allow_pickle=False)[:, pre["retained_locus_indices"]]
        x = np.where(np.isnan(x), pre["means"], x)
        scores = ((x - pre["means"]) / pre["scales"] - pre["pca_mean"]) @ pre["pca_components"].T
        np.testing.assert_allclose(scores, np.load(output / "pca_scores.npy"), atol=1e-12)
    with np.load(output / "mixture.npz", allow_pickle=False) as parameters:
        logp = -0.5 * np.sum(
            np.log(2 * np.pi * parameters["variances"])[None, :, :]
            + (scores[:, None, :] - parameters["means"]) ** 2 / parameters["variances"],
            axis=2,
        ) + np.log(parameters["weights"])
        logp -= logp.max(axis=1, keepdims=True)
        weights = np.exp(logp)
        np.testing.assert_allclose(weights / weights.sum(axis=1, keepdims=True), probabilities)
    before_output = snapshot(output)
    with pytest.raises(FileExistsError):
        run_genomic_baseline(
            dataset, model_config=config, base_dir=raw, output_dir=output, block_ids=blocks
        )
    assert snapshot(output) == before_output


def test_cli_json_and_failed_run_diagnostics(source, tmp_path, capsys):
    raw, _ = source
    output = tmp_path / "cli"
    args = [
        "baseline-genomic",
        str(raw / "dataset.json"),
        "--model-config",
        str(raw / "model.json"),
        "--blocks",
        str(raw / "blocks.json"),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    # Missing calls with the error policy fail transparently after isolated ingestion.
    model = read_json(raw / "model.json")
    model["missing_policy"] = "error"
    (raw / "error.json").write_text(json.dumps(model))
    args[args.index(str(raw / "model.json"))] = str(raw / "error.json")
    args[-1] = str(tmp_path / "failure")
    assert main(args) == 2
    assert "mean-imputation" in capsys.readouterr().err
    assert read_json(tmp_path / "failure" / "failure.json")["status"] == "failed"
    assert not (tmp_path / "failure" / "summary.json").exists()


@pytest.mark.parametrize(
    "format,name",
    [
        ("vcf", "tiny.vcf"),
        ("phylip", "tiny.phy"),
        ("structure", "tiny_one_row.str"),
        ("genepop", "tiny.gen"),
    ],
)
def test_encoding_real_formats_preserves_source_object_and_rng(tmp_path, format, name):
    dataset = load_config(EXAMPLES / "genomic" / "dataset.json")
    dataset = replace(
        dataset, modalities=(replace(dataset.modalities[0], format=format, path=name),)
    )
    loaded = ingest_genomic(dataset, base_dir=EXAMPLES / "genomic", output_dir=tmp_path / "loaded")
    before = loaded.genotype_data.snp_data.copy()
    original_prefix = loaded.genotype_data.prefix
    np.random.seed(123)
    state = np.random.get_state()
    logger = logging.getLogger("snpio.analysis.genotype_encoder")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        dosage, report = encode_diploid(loaded, output_prefix=tmp_path / "encoded", random_seed=73)
        assert handler in logger.handlers
    finally:
        logger.removeHandler(handler)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    np.testing.assert_array_equal(before, loaded.genotype_data.snp_data)
    assert loaded.genotype_data.prefix == original_prefix
    assert dosage.shape == (2, 4)
    np.testing.assert_array_equal(np.isnan(dosage), loaded.missing_mask)
    assert report["exclusions"] == []
    second, _ = encode_diploid(loaded, output_prefix=tmp_path / "second", random_seed=73)
    np.testing.assert_array_equal(dosage, second)


def test_triallelic_and_ambiguous_loci_are_never_collapsed(tmp_path):
    (tmp_path / "calls.phy").write_text("4 5\na AARMA\nb CCBMC\nc GGAAM\nd AAAAA\n")
    dataset = config_from_dict(
        {
            "schema_version": 1,
            "dataset_id": "three_alleles",
            "specimens": [{"specimen_id": i} for i in "abcd"],
            "modalities": [
                {
                    "name": "snps",
                    "kind": "genomic",
                    "format": "phylip",
                    "level": "specimen",
                    "path": "calls.phy",
                    "observations": [{"observation_id": i, "unit_id": i} for i in "abcd"],
                }
            ],
        }
    )
    loaded = ingest_genomic(dataset, base_dir=tmp_path, output_dir=tmp_path / "loaded")
    encoded, report = encode_diploid(loaded, output_prefix=tmp_path / "encoded", random_seed=3)
    assert np.isnan(encoded[:, :3]).all()
    assert not np.isnan(encoded[:, 3:]).any()
    assert len(report["exclusions"]) == 3


def test_vcf_ploidy_partial_calls_and_metadata(tmp_path):
    dataset = load_config(EXAMPLES / "genomic" / "dataset.json")
    raw = (EXAMPLES / "genomic" / "tiny.vcf").read_text()
    # Original fixture has a called diploid 0/0; replace it with a haploid GT.
    (tmp_path / "haploid.vcf").write_text(raw.replace("0/0", "0", 1))
    modality = replace(dataset.modalities[0], format="vcf", path="haploid.vcf")
    loaded = ingest_genomic(
        replace(dataset, modalities=(modality,)), base_dir=tmp_path, output_dir=tmp_path / "haploid"
    )
    with pytest.raises(ValueError, match="Non-diploid"):
        encode_diploid(loaded, output_prefix=tmp_path / "encoding", random_seed=3)
    (tmp_path / "partial.vcf").write_text(raw.replace("0/0", "0/.", 1))
    modality = replace(modality, path="partial.vcf")
    loaded = ingest_genomic(
        replace(dataset, modalities=(modality,)), base_dir=tmp_path, output_dir=tmp_path / "partial"
    )
    encoded, report = encode_diploid(loaded, output_prefix=tmp_path / "encoding2", random_seed=3)
    assert report["partially_missing_vcf_calls"] == 1
    assert np.isnan(encoded).sum() == 2


def test_repeat_specimens_rejected_before_ingestion(source, tmp_path):
    raw, dataset = source
    modality = dataset.modalities[0]
    observations = tuple(replace(o, unit_id="specimen_00") for o in modality.observations)
    dataset = replace(dataset, modalities=(replace(modality, observations=observations),))
    with pytest.raises(ValueError, match="Repeated"):
        run_genomic_baseline(
            dataset,
            model_config=BaselineConfig(2, "mean", True),
            base_dir=raw,
            output_dir=tmp_path / "repeat",
        )
    assert not (tmp_path / "repeat").exists()


def test_vcf_multiallelic_metadata_is_excluded_even_with_two_observed_alleles(tmp_path):
    dataset = load_config(EXAMPLES / "genomic" / "dataset.json")
    raw = (EXAMPLES / "genomic" / "tiny.vcf").read_text()
    (tmp_path / "multiallelic.vcf").write_text(raw.replace("locus1\tA\tC\t", "locus1\tA\tC,G\t"))
    modality = replace(dataset.modalities[0], format="vcf", path="multiallelic.vcf")
    loaded = ingest_genomic(
        replace(dataset, modalities=(modality,)), base_dir=tmp_path, output_dir=tmp_path / "loaded"
    )
    encoded, report = encode_diploid(loaded, output_prefix=tmp_path / "encoded", random_seed=3)
    assert np.isnan(encoded[:, 0]).all()
    assert report["exclusions"] == [{"locus_index": 0, "reason": "non_biallelic_snp_metadata"}]


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"schema_version": True, "block_ids": ["a"]},
        {"schema_version": 1, "block_ids": []},
        {"schema_version": 1, "block_ids": [" "]},
    ],
)
def test_bad_block_documents(value):
    with pytest.raises(ValueError):
        parse_blocks(value)


def test_unavailable_evidence_is_not_reported_as_a_point_estimate(monkeypatch):
    from super_deli.models import pca_gmm

    rng = np.random.default_rng(1729)
    x = rng.binomial(2, 0.5, (20, 32))
    cfg = BaselineConfig(2, "error", True, max_k=1, n_bootstrap=2, assume_unlinked=True)
    original = pca_gmm._fit_once
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise BaselineError("intentional failure")
        return original(*args)

    monkeypatch.setattr(pca_gmm, "_fit_once", fail)
    fit = pca_gmm.fit_pca_gmm(x, [str(i) for i in range(20)], config=cfg)
    result = _result("failure", fit, "0" * 64)
    support = [e for e in result.evidence if e.kind == "bootstrap_frequency"]
    assert all(e.status == "unavailable" and e.value is None for e in support)
    monkeypatch.setattr(pca_gmm, "_fit_once", original)
    result = _result(
        "no_bootstrap",
        pca_gmm.fit_pca_gmm(
            x,
            [str(i) for i in range(20)],
            config=replace(cfg, n_bootstrap=0, assignment_threshold=1),
        ),
        "0" * 64,
    )
    assert any(
        e.status == "not_computed" and e.kind == "bootstrap_frequency" for e in result.evidence
    )
