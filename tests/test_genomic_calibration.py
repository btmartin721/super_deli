"""Independent likelihoods, exact missingness, supported readers and full refits."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from scipy.stats import binom

from super_deli.benchmarking.genomic_analysis import point, verify_model
from super_deli.benchmarking.genomic_study import simulate_genomic
from super_deli.io.config import read_json
from super_deli.models.genomic_config import GenomicCalibrationConfig
from super_deli.models.genomic_null import fit_null, simulate_null, write_raw_genotypes
from super_deli.workflows.genomic_calibration import run_genomic_calibration

pytestmark = pytest.mark.baseline
ROOT = Path(__file__).resolve().parents[1]
SNP = pytest.mark.skipif(
    importlib.util.find_spec("snpio") is None, reason="Install the genomics extra"
)


def policy():
    c = GenomicCalibrationConfig.from_dict(
        read_json(ROOT / "examples/genomic-calibration/calibration.json")
    )
    return replace(c, n_simulations=2, model=replace(c.model, n_init=1, max_k=2))


def source(tmp_path, case="rare_missing_hwe"):
    x, layout, _ = simulate_genomic(case, 24, 193)
    directory = tmp_path / "raw"
    return write_raw_genotypes(x, layout, directory), x, layout, directory


def test_raw_mle_matches_binomial_likelihood_and_keeps_boundaries(tmp_path):
    data, x, layout, _ = source(tmp_path)
    x[:, 0], x[:, 1] = 0, 2
    loaded = SimpleNamespace(
        sample_ids=tuple(layout["sample_ids"]), specimen_ids=tuple(layout["unit_ids"])
    )
    null = fit_null(data, loaded, x, {"exclusions": []}, policy())
    expected = np.nansum(x, axis=0) / (2 * np.sum(~np.isnan(x), axis=0))
    np.testing.assert_array_equal(null.frequencies, expected)
    likelihood = binom.logpmf(np.nan_to_num(x), 2, expected)
    assert null.layout["fit"]["negative_log_likelihood"] == pytest.approx(
        -np.where(np.isnan(x), 0, likelihood).sum()
    )
    assert null.frequencies[0] == 0 and null.frequencies[1] == 1
    draws = np.stack([simulate_null(null, seed) for seed in range(100)])
    np.testing.assert_array_equal(np.isnan(draws), np.broadcast_to(np.isnan(x), draws.shape))
    assert np.all(draws[:, :, 0] == 0) and np.all(draws[:, :, 1] == 2)
    np.testing.assert_allclose(np.nanmean(draws, axis=(0, 1)), 2 * expected, atol=0.06)
    with pytest.raises(FileExistsError):
        write_raw_genotypes(x, layout, tmp_path / "raw")
    x[:, 2] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        fit_null(data, loaded, x, {"exclusions": []}, policy())


@SNP
@pytest.mark.genomic
def test_fresh_snpio_pipeline_refits_and_independent_numeric_audit(tmp_path, monkeypatch):
    from super_deli.workflows import genomic_baseline

    data, x, layout, directory = source(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    original, calls = genomic_baseline.run_genomic_baseline, []

    def run(*args, **kwargs):
        calls.append(kwargs["model_config"])
        return original(*args, **kwargs)

    monkeypatch.setattr(genomic_baseline, "run_genomic_baseline", run)
    c, out = policy(), tmp_path / "fit"
    result = run_genomic_calibration(data, config=c, base_dir=directory, output_dir=out)
    assert result["successful_fits"] == 3 and len(calls) == 3
    assert all(m == c.model for m in calls)
    assert result["inference"]["p_value"] >= 1 / 3
    assert "Approximate" in result["inference"]["p_value_kind"]
    fitted, draw, *_ = verify_model(tmp_path, out, x, layout, c)
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        point(archive, result["point"], c, x, fitted)
        for record in read_json(out / "null_replicates.json"):
            point(archive, record, c, draw(record["simulation_seed"]), fitted)
        assert all("truth" not in name for name in archive.namelist())
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        run_genomic_calibration(data, config=c, base_dir=directory, output_dir=out)


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_point_or_draw_withholds_p_and_never_redraws(tmp_path, monkeypatch, failure_call):
    from super_deli.workflows import genomic_baseline

    data, _, _, directory = source(tmp_path)
    original, calls = genomic_baseline.run_genomic_baseline, []

    def run(*args, **kwargs):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("injected genomic fit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(genomic_baseline, "run_genomic_baseline", run)
    out = tmp_path / "fit"
    result = run_genomic_calibration(data, config=policy(), base_dir=directory, output_dir=out)
    assert result["inference"]["p_value"] is None
    assert result["failed_fits"] == 1
    assert len(calls) == (1 if failure_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failure_call == 1 else 0)


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("format", ["phylip", "vcf", "structure", "genepop"])
@pytest.mark.parametrize("popmap", [False, True])
def test_all_readers_with_optional_popmaps_preserve_identity(tmp_path, format, popmap):
    data, x, layout, directory = source(tmp_path, "rare_missing_hwe")
    samples = layout["sample_ids"]
    path = directory / "source"
    if format == "phylip":
        path = directory / "genotypes.phy"
    elif format == "vcf":
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=1>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "\t".join(
                ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]
            ),
        ]
        for j in range(x.shape[1]):
            lines.append(
                "\t".join(
                    [
                        "1",
                        str(j + 1),
                        f"locus_{j}",
                        "A",
                        "G",
                        ".",
                        "PASS",
                        ".",
                        "GT",
                        *["./." if np.isnan(g) else ("0/0", "0/1", "1/1")[int(g)] for g in x[:, j]],
                    ]
                )
            )
        path.write_text("\n".join(lines) + "\n")
    elif format == "structure":
        path.write_text(
            "\n".join(
                s
                + " "
                + " ".join("-9 -9" if np.isnan(g) else ("1 1", "1 3", "3 3")[int(g)] for g in row)
                for s, row in zip(samples, x, strict=True)
            )
            + "\n"
        )
    else:
        lines = ["fixed SNP panel", *[f"locus_{j}" for j in range(x.shape[1])], "Pop"]
        lines += [
            s
            + ", "
            + " ".join("0000" if np.isnan(g) else ("0101", "0103", "0303")[int(g)] for g in row)
            for s, row in zip(samples, x, strict=True)
        ]
        path.write_text("\n".join(lines) + "\n")
    if popmap:
        (directory / "source.popmap").write_text(
            "\n".join(s + " population" for s in samples) + "\n"
        )
    m = replace(
        data.modalities[0],
        format=format,
        path=path.name,
        popmapfile="source.popmap" if popmap else None,
        observations=tuple(reversed(data.modalities[0].observations)),
    )
    data = replace(data, modalities=(m,))
    c = replace(policy(), n_simulations=1)
    out = tmp_path / "fit"
    result = run_genomic_calibration(data, config=c, base_dir=directory, output_dir=out)
    assert result["successful_fits"] == 2
    fitted = read_json(out / "layout.json")
    assert fitted["sample_ids"] == samples and fitted["unit_ids"] == layout["unit_ids"]
    verify_model(tmp_path, out, x, layout, c)


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("stage", ["null_fit", "archive"])
def test_failed_null_fit_and_archive_keep_attempt_counts(tmp_path, stage):
    data, x, layout, directory = source(tmp_path)
    c, out = policy(), tmp_path / "fit"
    if stage == "null_fit":
        x[:, 0] = np.nan
        data = write_raw_genotypes(x, layout, tmp_path / "all_missing")
        directory = tmp_path / "all_missing"
    else:
        c = replace(c, max_archive_bytes=1)
    with pytest.raises(ValueError):
        run_genomic_calibration(data, config=c, base_dir=directory, output_dir=out)
    failure = read_json(out / "failure.json")
    expected = 0 if stage == "null_fit" else 1
    assert failure["attempted_fits"] == failure["successful_fits"] == expected
    assert failure["unattempted_fits"] == 3 - expected
    assert not (out / "summary.json").exists()


@SNP
@pytest.mark.genomic
def test_independent_audit_rejects_changed_mle_even_with_valid_npz(tmp_path):
    data, x, layout, directory = source(tmp_path)
    c, out = policy(), tmp_path / "fit"
    run_genomic_calibration(data, config=c, base_dir=directory, output_dir=out)
    path = out / "null_model.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["frequencies"][0] += 0.01
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(AssertionError):
        verify_model(tmp_path, out, x, layout, c)
