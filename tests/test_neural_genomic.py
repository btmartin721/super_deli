"""Categorical derivatives, SNPio identities, full refits and frozen models."""

import json
from dataclasses import replace

import pytest

from super_deli.models.genomic_neural_config import GenomicComparisonConfig, GenomicNeuralConfig

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


def config(**changes):
    return GenomicNeuralConfig(2, "error", True, **changes)


def invented():
    return (
        np.random.default_rng(81)
        .binomial(2, np.linspace(0.1, 0.8, 12), size=(24, 12))
        .astype(float)
    )


def test_categorical_finite_difference_and_actual_adam():
    from super_deli.benchmarking.genomic_neural_steps import gradients, observe_training
    from super_deli.models import genomic_autoencoder as neural
    from super_deli.models.pca_gmm import _preprocess

    raw = invented()
    x, pre = _preprocess(raw, config().baseline_config())
    target = raw[:, pre.kept_indices].astype(int)
    with observe_training(x, target) as checked:
        weights, losses = neural._train(x, target, 0)
    assert len(checked.errors) == len(checked.hashes) == 500
    assert losses.shape == (501,) and losses[-1] < losses[0]
    gradient = gradients(x, target, checked.initial)
    for key, value in checked.initial.items():
        for index in (0, value.size - 1):
            plus = {k: v.copy() for k, v in checked.initial.items()}
            minus = {k: v.copy() for k, v in checked.initial.items()}
            plus[key].flat[index] += 1e-6
            minus[key].flat[index] -= 1e-6
            fd = (
                neural.cross_entropy(x, target, plus) - neural.cross_entropy(x, target, minus)
            ) / 2e-6
            np.testing.assert_allclose(fd, gradient[key].flat[index], rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(neural.cross_entropy(x, target, weights), losses[-1], atol=1e-12)


def test_numeric_projection_and_checksum(tmp_path):
    from super_deli.models.genomic_neural import fit_genomic_neural_point, load_genomic_model
    from super_deli.workflows.genomic_neural import _save_point

    raw = invented()
    raw[:, 0] = 0
    point = fit_genomic_neural_point(raw, config=config())
    assert 0 not in point.arrays["kept_indices"]
    _save_point(point, tmp_path / "model", [f"u{i}" for i in range(len(raw))])
    model = load_genomic_model(tmp_path / "model")
    predicted = model.predict(raw, locus_keys=model.locus_keys)
    np.testing.assert_allclose(predicted["scores"], point.arrays["scores"], atol=1e-12)
    np.testing.assert_allclose(predicted["membership"], point.arrays["membership"], atol=1e-12)
    probabilities = model.reconstruct_probabilities(raw, locus_keys=model.locus_keys)
    np.testing.assert_allclose(probabilities.sum(2), 1, atol=1e-12)
    with pytest.raises(ValueError):
        model.encode(raw, locus_keys=model.locus_keys[::-1])
    with pytest.raises(ValueError):
        model.encode(np.where(raw == 1, np.nan, raw), locus_keys=model.locus_keys)
    path = tmp_path / "model" / "model.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_genomic_model(tmp_path / "model")


@pytest.mark.parametrize(
    "kind", ["nan", "fraction", "boolean", "small", "wide", "workspace", "pairwise"]
)
def test_guards(kind):
    from super_deli.models.genomic_neural import validate_dosages

    x = invented()
    model = config()
    if kind == "nan":
        x[0, 0] = np.nan
    elif kind == "fraction":
        x[0, 0] = 0.5
    elif kind == "boolean":
        x = x.astype(bool)
    elif kind == "small":
        x = x[:9]
    elif kind == "wide":
        model = replace(model, max_loci=10)
    elif kind == "workspace":
        model = replace(model, max_working_bytes=1)
    elif kind == "pairwise":
        model = replace(model, n_subsamples=1, max_pairwise_bytes=1)
    with pytest.raises(ValueError):
        validate_dosages(x, model)


def fixture(raw, kind, path, popmap=False):
    from super_deli.core.schema import config_from_dict

    path.mkdir()
    ids = [f"s{i:03d}" for i in range(len(raw))]
    alleles = np.asarray(["A", "R", "G"])[raw.astype(int)]
    options = None
    if kind == "phylip":
        text = f"{len(raw)} {raw.shape[1]}\n" + "".join(
            s + " " + "".join(row) + "\n" for s, row in zip(ids, alleles, strict=True)
        )
    elif kind == "vcf":
        text = (
            '##fileformat=VCFv4.2\n##contig=<ID=1,length=100000>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
            + "\t".join(ids)
            + "\n"
        )
        for j, row in enumerate(raw.T):
            text += (
                f"1\t{j + 1}\tloc{j}\tA\tG\t.\tPASS\t.\tGT\t"
                + "\t".join(("0/0", "0/1", "1/1")[int(g)] for g in row)
                + "\n"
            )
    elif kind == "structure":
        text = "".join(
            s + " " + " ".join(("1 1", "1 3", "3 3")[int(g)] for g in row) + "\n"
            for s, row in zip(ids, raw, strict=True)
        )
        options = dict(
            has_popids=False,
            has_marker_names=False,
            allele_start_col=1,
            allele_encoding={"1": "A", "3": "G"},
        )
    else:
        text = (
            "Invented genotype calls\n"
            + "".join(f"locus{j}\n" for j in range(raw.shape[1]))
            + "Pop\n"
        )
        text += "".join(
            s + ", " + " ".join(("0101", "0103", "0303")[int(g)] for g in row) + "\n"
            for s, row in zip(ids, raw, strict=True)
        )
        options = dict(allele_encoding={"01": "A", "03": "G"})
    filename = "invented." + dict(phylip="phy", vcf="vcf", structure="str", genepop="gen")[kind]
    (path / filename).write_text(text)
    m = dict(
        name="genomic",
        kind="genomic",
        level="specimen",
        format=kind,
        path=filename,
        observations=[dict(observation_id=s, unit_id="u" + s) for s in ids],
    )
    if options:
        m["reader_options"] = options
    if popmap:
        (path / "popmap.txt").write_text("".join(s + "\tgroupA\n" for s in ids))
        m["popmapfile"] = "popmap.txt"
    dataset = config_from_dict(
        dict(
            schema_version=1,
            dataset_id="invented_genomic",
            specimens=[dict(specimen_id="u" + s) for s in ids],
            modalities=[m],
        )
    )
    return dataset


@pytest.mark.parametrize("kind", ["phylip", "vcf", "structure", "genepop"])
@pytest.mark.parametrize("popmap", [False, True])
def test_snpio_formats_and_optional_popmap(kind, popmap, tmp_path):
    from super_deli.io.genomic_neural import prepare_genotypes

    raw = invented()
    dataset = fixture(raw, kind, tmp_path / "input", popmap)
    encoded, cohort, sources = prepare_genotypes(
        dataset, config(), tmp_path / "input", tmp_path / "prepared"
    )
    assert cohort["unit_ids"] == ["u" + s for s in cohort["sample_ids"]]
    assert not cohort["populations_used_as_predictors"]
    for j in range(raw.shape[1]):
        assert np.array_equal(encoded[:, j], raw[:, j]) or np.array_equal(
            encoded[:, j], 2 - raw[:, j]
        )
    assert len(sources) == 1 + popmap


def test_workflow_pairing_native_parity_and_failures(tmp_path, monkeypatch):
    from super_deli.models.genomic_neural import load_genomic_model
    from super_deli.workflows import genomic_neural as workflow
    from super_deli.workflows.genomic_baseline import run_genomic_baseline

    raw = invented()
    dataset = fixture(raw, "phylip", tmp_path / "input")
    model = config(n_subsamples=1)
    analysis = GenomicComparisonConfig(model, (0,))
    summary = workflow.run_genomic_comparison(
        dataset,
        analysis_config=analysis,
        base_dir=tmp_path / "input",
        output_dir=tmp_path / "analysis",
    )
    assert summary["status"] == "complete"
    assert summary["inference"]["p_value"] is None
    assert len(summary["models"]) == 3
    records = json.loads((tmp_path / "analysis/subsamples/draw-0000/records.json").read_text())
    assert len({tuple(r["training_indices"]) for r in records.values()}) == 1
    assert all(r["status"] == "complete" for r in records.values())
    native = run_genomic_baseline(
        dataset,
        model_config=model.baseline_config(),
        base_dir=tmp_path / "input",
        output_dir=tmp_path / "legacy",
    )
    assert native["selected_k"] == summary["models"][0]["selected_k"]
    fitted = load_genomic_model(tmp_path / "analysis/arms/pca_diagonal")
    for new, old in (("scores", "pca_scores"), ("labels", "labels"), ("membership", "membership")):
        np.testing.assert_array_equal(
            fitted.arrays[new], np.load(tmp_path / "legacy" / (old + ".npy"))
        )
    original = workflow._point

    def failure(family, *args):
        if family == "categorical_ae":
            raise RuntimeError("invented training failure")
        return original(family, *args)

    monkeypatch.setattr(workflow, "_point", failure)
    partial = workflow.run_genomic_comparison(
        dataset,
        analysis_config=analysis,
        base_dir=tmp_path / "input",
        output_dir=tmp_path / "partial",
    )
    assert partial["status"] == "partial" and partial["models"][-1]["selected_k"] is None
    assert partial["models"][-1]["stability"]["failed"] == 1
    with pytest.raises(FileExistsError):
        workflow.run_genomic_comparison(
            dataset,
            analysis_config=analysis,
            base_dir=tmp_path / "input",
            output_dir=tmp_path / "partial",
        )


def test_frozen_laws_and_independent_density_audit(tmp_path):
    from super_deli.benchmarking.genomic_neural_study import LAWS, audit_model, simulate
    from super_deli.models.genomic_neural import (
        fit_genomic_neural_point,
        fit_genomic_pca_shared_point,
    )
    from super_deli.workflows.genomic_neural import _save_point

    for law in LAWS:
        values, labels, reference = simulate(law, 64, 8101)
        again = simulate(law, 64, 8101)
        np.testing.assert_array_equal(values, again[0])
        assert values.shape == (64, 128) and np.isfinite(values).all()
        assert len(np.unique(labels)) == (4 if law == "four_groups" else 1)
        assert reference["probability"].shape == values.shape
        if law == "perfect_ld":
            np.testing.assert_array_equal(values[:, ::8], values[:, 7::8])
    raw = invented()
    for i, function in enumerate((fit_genomic_pca_shared_point, fit_genomic_neural_point)):
        point = function(raw, config=config())
        folder = tmp_path / f"model{i}"
        _save_point(point, folder, [f"u{j}" for j in range(len(raw))])
        assert audit_model(folder, raw)["passed"]
