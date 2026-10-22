"""Whole-block categorical likelihoods, independent donors and complete SNPio refits."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.linked_analysis import point, verify_model
from super_deli.benchmarking.linked_study import simulate_linked
from super_deli.io.config import read_json
from super_deli.models.linked_config import LinkedCalibrationConfig
from super_deli.models.linked_null import fit_null, simulate_null, write_raw_linked
from super_deli.workflows.linked_calibration import run_linked_calibration

ROOT = Path(__file__).resolve().parents[1]
SNP = pytest.mark.skipif(
    importlib.util.find_spec("snpio") is None, reason="Install the genomics extra"
)
pytestmark = pytest.mark.baseline


def policy():
    c = LinkedCalibrationConfig.from_dict(
        read_json(ROOT / "examples/linked-calibration/calibration.json")
    )
    return replace(c, n_simulations=2, model=replace(c.model, n_init=1, max_k=2))


def source(tmp_path, case="moderate_ld"):
    x, layout, _ = simulate_linked(case, 24, 193)
    directory = tmp_path / "raw"
    data = write_raw_linked(x, layout, directory)
    return data, x, layout, directory


def test_categorical_mle_and_resampling_preserve_within_block_but_break_cross_block_dependence(
    tmp_path,
):
    data, x, layout, _ = source(tmp_path)
    x[:] = np.resize(np.repeat([0.0, 1.0, 2.0], 8), (24, 1))
    # Every block is perfectly concordant in the source; null blocks draw independently.
    loaded = SimpleNamespace(
        sample_ids=tuple(layout["sample_ids"]), specimen_ids=tuple(layout["unit_ids"])
    )
    null = fit_null(data, loaded, x, {"exclusions": []}, policy())
    assert null.layout["fit"]["negative_log_likelihood"] == pytest.approx(24 * 16 * np.log(3))
    for b in null.layout["blocks"]:
        assert b["n_profiles"] == 3 and b["maximum_profile_probability"] == pytest.approx(1 / 3)
    draws = np.concatenate([simulate_null(null, s)["genotypes"] for s in range(100)])
    np.testing.assert_array_equal(draws[:, 0], draws[:, 7])
    assert abs(np.corrcoef(draws[:, 0], draws[:, 8])[0, 1]) < 0.08
    np.testing.assert_allclose(
        [np.mean(draws[:, 0] == j) for j in range(3)], [1 / 3] * 3, atol=0.04
    )
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing calls"):
        fit_null(data, loaded, x, {"exclusions": []}, policy())


@SNP
@pytest.mark.genomic
def test_full_refits_donor_artifacts_and_numeric_audit(tmp_path, monkeypatch):
    from super_deli.workflows import genomic_baseline

    data, x, layout, directory = source(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    original, calls = genomic_baseline.run_genomic_baseline, []

    def run(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    monkeypatch.setattr(genomic_baseline, "run_genomic_baseline", run)
    c, out = policy(), tmp_path / "fit"
    result = run_linked_calibration(data, config=c, base_dir=directory, output_dir=out)
    assert result["successful_fits"] == len(calls) == 3
    assert all(k["block_ids"] == c.block_ids and k["model_config"] == c.model for k in calls)
    fitted, draw, *_ = verify_model(tmp_path, out, x, layout, c)
    with ZipFile(out / "pipeline_artifacts.zip") as archive:
        point(archive, result["point"], c, x, fitted)
        for record in read_json(out / "null_replicates.json"):
            point(archive, record, c, draw(record["simulation_seed"]), fitted)
        assert not any("truth" in name for name in archive.namelist())
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(FileExistsError):
        run_linked_calibration(data, config=c, base_dir=directory, output_dir=out)


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_point_or_draw_withholds_p_without_redraw(tmp_path, monkeypatch, failure_call):
    from super_deli.workflows import genomic_baseline

    data, _, _, directory = source(tmp_path)
    original, calls = genomic_baseline.run_genomic_baseline, []

    def run(*a, **kw):
        calls.append(None)
        if len(calls) == failure_call:
            raise ValueError("injected linked failure")
        return original(*a, **kw)

    monkeypatch.setattr(genomic_baseline, "run_genomic_baseline", run)
    result = run_linked_calibration(
        data, config=policy(), base_dir=directory, output_dir=tmp_path / "fit"
    )
    assert result["inference"]["p_value"] is None and result["failed_fits"] == 1
    assert result["attempted_fits"] == (1 if failure_call == 1 else 3)
    assert result["unattempted_fits"] == (2 if failure_call == 1 else 0)


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("bad", ["missing", "map", "budget", "archive"])
def test_invalid_raw_null_and_archive_limits_retain_failure_counts(tmp_path, bad):
    data, x, layout, directory = source(tmp_path)
    c = policy()
    if bad == "missing":
        x[0, 0] = np.nan
        directory = tmp_path / "missing"
        data = write_raw_linked(x, layout, directory)
    elif bad == "map":
        c = replace(c, block_ids=c.block_ids[:-1])
    elif bad == "budget":
        c = replace(c, max_working_bytes=1)
    else:
        c = replace(c, max_archive_bytes=1)
    out = tmp_path / "fit"
    with pytest.raises(ValueError):
        run_linked_calibration(data, config=c, base_dir=directory, output_dir=out)
    failure = read_json(out / "failure.json")
    assert failure["successful_fits"] == failure["attempted_fits"] == (1 if bad == "archive" else 0)
    assert not (out / "summary.json").exists()


@SNP
@pytest.mark.genomic
@pytest.mark.parametrize("format", ["phylip", "vcf", "structure", "genepop"])
@pytest.mark.parametrize("popmap", [False, True])
def test_readers_with_optional_popmaps_keep_identity_and_block_profiles(tmp_path, format, popmap):
    data, x, layout, directory = source(tmp_path)
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
        lines += [
            "\t".join(
                [
                    "1",
                    str(j + 1),
                    ".",
                    "A",
                    "G",
                    ".",
                    "PASS",
                    ".",
                    "GT",
                    *[("0/0", "0/1", "1/1")[int(g)] for g in x[:, j]],
                ]
            )
            for j in range(128)
        ]
        path.write_text("\n".join(lines) + "\n")
    elif format == "structure":
        path.write_text(
            "\n".join(
                s + " " + " ".join(("1 1", "1 3", "3 3")[int(g)] for g in row)
                for s, row in zip(samples, x, strict=True)
            )
            + "\n"
        )
    else:
        lines = ["block genotypes", *[f"locus_{j}" for j in range(128)], "Pop"]
        lines += [
            s + ", " + " ".join(("0101", "0103", "0303")[int(g)] for g in row)
            for s, row in zip(samples, x, strict=True)
        ]
        path.write_text("\n".join(lines) + "\n")
    if popmap:
        (directory / "source.popmap").write_text(
            "\n".join(s + " population" for s in samples) + "\n"
        )
    modality = replace(
        data.modalities[0],
        format=format,
        path=path.name,
        popmapfile="source.popmap" if popmap else None,
        observations=tuple(reversed(data.modalities[0].observations)),
    )
    data = replace(data, modalities=(modality,))
    out = tmp_path / "fit"
    c = replace(policy(), n_simulations=1)
    result = run_linked_calibration(data, config=c, base_dir=directory, output_dir=out)
    assert result["successful_fits"] == 2
    verify_model(tmp_path, out, x, layout, c)


@SNP
@pytest.mark.genomic
def test_independent_categorical_audit_rejects_changed_profile_counts(tmp_path):
    data, x, layout, directory = source(tmp_path)
    c, out = policy(), tmp_path / "fit"
    run_linked_calibration(data, config=c, base_dir=directory, output_dir=out)
    path = out / "null_model.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["block_0000_counts"][0] += 1
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(AssertionError):
        verify_model(tmp_path, out, x, layout, c)


def test_constant_heterozygous_block_is_preserved_without_hwe_randomization(tmp_path):
    data, x, layout, _ = source(tmp_path)
    x[:, :8] = 1
    loaded = SimpleNamespace(
        sample_ids=tuple(layout["sample_ids"]), specimen_ids=tuple(layout["unit_ids"])
    )
    null = fit_null(data, loaded, x, {"exclusions": []}, policy())
    assert null.layout["blocks"][0]["n_profiles"] == 1
    assert null.layout["blocks"][0]["maximum_profile_probability"] == 1
    for seed in range(10):
        np.testing.assert_array_equal(
            simulate_null(null, seed)["genotypes"][:, :8], np.ones((24, 8))
        )
