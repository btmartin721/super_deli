"""Raw-law parity, exact truth accounting and failure-inclusive reporting."""

# ruff: noqa: E402
from io import BytesIO
from zipfile import ZipFile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, cells, tasks
from super_deli.benchmarking.neural_power_data import generate, reference


@pytest.mark.parametrize("index", range(11))
def test_independent_raw_law_and_standard_parity(index):
    from super_deli.benchmarking.neural_calibration_data import generate as old

    task = tasks(NeuralPowerStudy(), "preflight")[index]
    task = {
        **task,
        **{
            name: [900, index, j]
            for j, name in enumerate(
                (
                    "data_seed",
                    "noise_seed",
                    "permutation_seed",
                    "ecology_order_seed",
                    "null_seed_path",
                )
            )
        },
    }
    a, b = generate(task), reference(task)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
    if task["case"] == "four_complementary":
        assert np.bincount(a["truth"]).tolist() == [task["n"] // 4] * 4
        original = old(task)
        if task["signal_multiplier"] == 1:
            for key in a:
                np.testing.assert_array_equal(a[key], original[key])
        else:
            np.testing.assert_array_equal(a["noise"], original["noise"])
            np.testing.assert_array_equal(a["truth"], original["truth"])
    if task["case"] == "continuous_gradient":
        assert "truth" not in a and np.all(np.abs(a["gradient"]) <= 3)


@pytest.mark.parametrize("pvalue,k", [(0.01, 2), (0.2, 4), (None, 4)])
def test_rejection_is_not_group_recovery(tmp_path, pvalue, k):
    from super_deli.benchmarking.neural_power_report import assess
    from super_deli.io.config import write_json

    c = NeuralPowerStudy(fixture=True)
    t = tasks(c, "main")[0]
    (tmp_path / "ledger/main").mkdir(parents=True)
    write_json(dict(interrupted=False), tmp_path / "ledger/main" / (t["key"] + ".json"))
    base = tmp_path / "main" / t["key"]
    base.mkdir(parents=True)
    truth = np.repeat(np.arange(4), 8)
    np.savez(base / "generating-reference.npz", truth=truth)
    for arm in ("neural", "pca4"):
        out = base / arm / "calibration"
        out.mkdir(parents=True)
        write_json(dict(status="complete", selected_k=k), out / "observed.json")
        write_json(
            dict(
                inference=dict(
                    p_value=pvalue,
                    reject_joint_raw_gaussian=None if pvalue is None else pvalue <= 0.05,
                )
            ),
            out / "summary.json",
        )
        data = BytesIO()
        np.savez(data, labels=truth if k == 4 else truth // 2)
        with ZipFile(out / "pipeline_artifacts.zip", "w") as z:
            z.writestr("observed/run/point.npz", data.getvalue())
    rows = assess(tmp_path, c)
    assert all(r["recovered"] == (k == 4) for r in rows)
    assert all(r["joint_rejection_recovery"] == (None if pvalue is None else False) for r in rows)


def test_primary_grid_and_missing_paired_bounds():
    from super_deli.benchmarking.neural_power_report import aggregate, paired

    rows = []
    for cell, (case, n, s, _) in enumerate(cells(NeuralPowerStudy())[:6]):
        for i in range(10):
            for arm in ("neural", "pca4"):
                rows.append(
                    dict(
                        cell=cell,
                        case=case,
                        n=n,
                        signal_multiplier=s,
                        key=str(i),
                        arm=arm,
                        selected_k=4,
                        point_available=True,
                        ari=1.0,
                        p_value=0.01,
                        failed_draws=0,
                        ordinary_split=True,
                        global_rejection=True,
                        selected_search_boundary=True,
                        correct_k=True,
                        underselected=False,
                        recovered=True,
                        joint_rejection_recovery=True,
                    )
                )
    groups, primary, pairs = aggregate(rows)
    assert len(groups) == 12 and len(primary) == 36 and len(pairs) == 18
    assert all(r["count"] == r["requested"] == 10 for r in primary)
    assert all(r["simultaneous_95_interval"][0] < 0.7 for r in primary)
    data = [dict(key="x", arm=a, event=None) for a in ("neural", "pca4")]
    value = paired(data, "event")
    assert value["unknown_pairs"] == 1 and value["simultaneous_failure_interval"] == [-1, 1]


def test_projection_rejects_missing_sample_size(tmp_path, monkeypatch):
    from super_deli.benchmarking import neural_power_budget as b
    from super_deli.io.config import write_json

    c = NeuralPowerStudy(fixture=True)
    t = tasks(c, "preflight")[0]
    for name in ("frozen.json", "preflight-execution.json", "preflight-audit.json"):
        write_json({}, tmp_path / name)
    (tmp_path / "preflight" / t["key"]).mkdir(parents=True)
    e = dict(
        completed=True,
        unknown_counter_units=0,
        records=[dict(key=t["key"], seconds=1)],
        attempted=6,
        neural_fits=6,
    )
    a = dict(records=[], seconds=1, verified_optimizer_updates=3000)
    monkeypatch.setattr(
        b.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 16 * 1024**3})()
    )
    jobs = {"preflight": [dict(task=t)], "main": [dict(task={**t, "n": 256})]}
    value = b.project(
        tmp_path,
        {"counts": {"preflight": {"workflows": 6, "neural_fits": 6, "verified_updates": 3000}}},
        c,
        e,
        a,
        jobs,
        staging_bytes=0,
    )
    assert not value["passed"] and len(value["missing_strata"]) == 1
