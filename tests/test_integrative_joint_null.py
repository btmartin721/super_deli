"""Joint likelihood, conditional simulation and unsupported-data boundaries."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from scipy.stats import multivariate_normal

from super_deli.benchmarking.joint_study import simulate_joint
from super_deli.io.config import read_json
from super_deli.io.tabular import ingest_tabular
from super_deli.models.joint_config import JointCalibrationConfig
from super_deli.models.joint_null import fit_null, simulate_null, write_raw_joint

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, case="linear_association", transform=None, family="full"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = JointCalibrationConfig.from_dict(
        read_json(ROOT / "examples/joint-calibration/calibration.json")
    )
    c = replace(c, model=replace(c.model, covariance_type=family))
    values, layout, _ = simulate_joint(case, 64, 191)
    if transform:
        transform(values, layout)
    dataset = write_raw_joint(values, layout, tmp_path / "raw")
    tables = ingest_tabular(
        dataset,
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "ingested",
        modality_names=["morphological", "ecological"],
    ).tables
    loaded = SimpleNamespace(
        sample_ids=tuple(layout["genomic"]["sample_ids"]), specimen_ids=tuple(layout["unit_ids"])
    )
    null = fit_null(dataset, loaded, values["genotypes"], {"exclusions": []}, tables, c)
    return null, c, values


def test_independent_gaussian_likelihood_and_conditional_draws(tmp_path):
    null, c, values = fixture(tmp_path)
    x = null.arrays["design"]
    beta = null.arrays["coefficients"]
    y = values["phenotypes"]
    nll = -sum(
        multivariate_normal.logpdf(y[i], mean=x[i] @ beta, cov=null.arrays["residual_covariance"])
        for i in range(len(x))
    )
    assert nll == pytest.approx(null.layout["fit"]["gaussian_negative_log_likelihood"])
    np.testing.assert_allclose(x.T @ (y - x @ beta), 0, atol=1e-10)
    assert null.layout["fit"]["residual_df"] == 59
    draw = simulate_null(null, 8)
    again = simulate_null(null, 8)
    for key in draw:
        np.testing.assert_array_equal(draw[key], again[key])
    for b, block in enumerate(null.genomic.layout["blocks"]):
        indices = block["locus_indices"]
        np.testing.assert_array_equal(
            draw["genotypes"][:, indices],
            values["genotypes"][draw["donor_indices"][:, b]][:, indices],
        )
    assert not np.array_equal(draw["phenotypes"], y)
    with pytest.raises(FileExistsError):
        write_raw_joint(draw, null.layout, tmp_path / "raw")


@pytest.mark.parametrize("kind", ["predictor", "residual", "missing"])
def test_unidentifiable_or_missing_fits_rejected(tmp_path, kind):
    def change(v, layout):
        if kind == "predictor":
            v["genotypes"][:, :8] = 0
        if kind == "residual":
            v["phenotypes"][:, 1] = v["phenotypes"][:, 0]
        if kind == "missing":
            v["genotypes"][0, 3] = np.nan

    with pytest.raises(ValueError, match="rank deficient|singular|Missing calls"):
        fixture(tmp_path, transform=change)


def test_raw_units_and_reordered_phenotype_rows(tmp_path):
    null, c, values = fixture(tmp_path / "a")

    def change(v, layout):
        for view in layout["views"]:
            view["unit_ids"] = list(reversed(view["unit_ids"]))
            view["observation_ids"] = list(reversed(view["observation_ids"]))
        v["phenotypes"] *= np.array([0.001, 10, 1, 2, 0.1, 100, 1, 3])

    other, _, _ = fixture(tmp_path / "b", transform=change)
    np.testing.assert_allclose(
        other.values[:, 128:], values["phenotypes"] * np.array([0.001, 10, 1, 2, 0.1, 100, 1, 3])
    )


def test_predictor_polarity_does_not_follow_snpio_major_allele(tmp_path):
    from super_deli.workflows.joint_calibration import canonical_dosages

    raw = np.array([["A", "A", "R"], ["R", "A", "R"], ["G", "A", "R"]])
    loaded = SimpleNamespace(genotype_data=SimpleNamespace(snp_data=raw))
    canonical = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]], dtype=float)
    got, metadata = canonical_dosages(loaded, 2 - canonical, {"exclusions": []})
    np.testing.assert_array_equal(got, canonical)
    assert metadata["joint_predictor_coding"]["snpio_reversed_loci"] == [0, 1]


def test_raw_null_and_draws_do_not_depend_on_clustering_covariance(tmp_path):
    full, _, _ = fixture(tmp_path / "full")
    tied, _, _ = fixture(tmp_path / "tied", family="tied")
    assert full.layout == tied.layout
    for key, value in full.model_arrays().items():
        np.testing.assert_array_equal(value, tied.model_arrays()[key])
    full_draw, tied_draw = simulate_null(full, 119), simulate_null(tied, 119)
    for key in full_draw:
        np.testing.assert_array_equal(full_draw[key], tied_draw[key])
