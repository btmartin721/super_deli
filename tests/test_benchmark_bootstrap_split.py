"""Density, failure policy and paired native resampling contracts on invented arrays."""

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.bootstrap_split_audit import audit, logdensity
from super_deli.benchmarking.bootstrap_split_fit import run
from super_deli.benchmarking.bootstrap_split_study import seeds
from super_deli.io.config import read_json
from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig

pytestmark = pytest.mark.baseline


def model():
    return IntegrativeConfig(
        assume_independent_units=True,
        views=(
            ViewConfig("traits", "morphological", n_pcs=4),
            ViewConfig("habitat", "ecological", n_pcs=4),
        ),
        n_pcs=8,
        min_cluster_size=9,
        n_init=100,
        covariance_type="tied",
        random_seed=301,
        n_bootstrap=0,
    )


def test_density_known_gaussian():
    x = np.array([[0.0, 0.0], [1.0, -1.0]])
    ll, p, y = logdensity(
        x, dict(weights=np.ones(1), means=np.zeros((1, 2)), covariances=np.eye(2))
    )
    np.testing.assert_allclose(ll, -np.log(2 * np.pi) - np.sum(x * x, 1) / 2)
    np.testing.assert_array_equal(p, np.ones((2, 1)))
    assert not y.any()


def test_paired_unique_id_draws():
    ss = seeds(301, 64, 25)
    assert len(ss) == 26 and len(set(s for d, s in ss)) == 26
    assert any(len(set(d)) < len(d) for d, s in ss[1:])
    np.testing.assert_array_equal(ss[0][0], np.arange(64))


def test_numerical_withhold_and_provenance_stop(tmp_path):
    x = np.random.default_rng(8901).normal(size=(64, 8))
    e = np.random.default_rng(8902).normal(size=(128, 8))
    out = tmp_path / "fit"
    run(out, x, x, e, model(), head=True, draw=np.arange(64), seed=301)
    assert audit(out, x, e, head=True)["withheld"] == 0
    (out / "diagnostic-audit.json").unlink()
    path = out / "candidates/index.json"
    records = read_json(path)
    record = records[0]
    arr = out / "candidates" / record["arrays"]
    with np.load(arr, allow_pickle=False) as f:
        values = {k: f[k].copy() for k in f.files}
    values["weights"] *= 0.9
    with arr.open("wb") as f:
        np.savez(f, **values)
    assert audit(out, x, e, head=True)["rows"][0]["status"] == "withheld_numerical_audit"
    (out / "diagnostic-audit.json").unlink()
    records[0]["scores_sha256"] = "0" * 64
    path.write_text(json.dumps(records))
    with pytest.raises(AssertionError):
        audit(out, x, e, head=True)


def test_raw_parent_invented(tmp_path):
    pytest.importorskip("snpio")
    from super_deli.benchmarking.bootstrap_split_study import parent

    generator = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    task = dict(n=64, law="shape_only", seed=8903, index=0)
    r = parent(task, tmp_path / "raw", generator, draws=1)
    assert r["searches"] == 4 and len(r["arms"]["ordinary"]) == 2
    assert len(r["arms"]["unique"]) == len(r["arms"]["reference"]) == 1
    assert all(a["status"] == "complete" for rows in r["arms"].values() for a in rows)
    assert r["arms"]["unique"][0]["training_rows"] == r["arms"]["unique"][0]["unique_originals"]
