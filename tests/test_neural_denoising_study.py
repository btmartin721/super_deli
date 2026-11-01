"""Development-only streams exercise the frozen comparison and independent audits."""

# ruff: noqa: E402
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from super_deli.benchmarking.denoising_audit import audit_density, audit_inventory
from super_deli.benchmarking.denoising_numerics import audit_training
from super_deli.benchmarking.denoising_report import interval, noise_ari, paired, rate
from super_deli.benchmarking.denoising_study import (
    ARMS,
    generate,
    run_arm,
    seed32,
    tasks,
)
from super_deli.benchmarking.neural_analysis import audit_point, reference
from super_deli.benchmarking.neural_study import write_dataset
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import write_json


def test_frozen_accounting_and_pairing():
    namespaces = []
    for phase, expected, nn in (("preflight", 266, 222), ("main", 2618, 2190)):
        grid = tasks(phase)
        count = 8 + sum(
            2 * 7 * (1 + t["draws"] + int(t["replay"])) + 1 + int(t["replay"]) for t in grid
        )
        assert count == expected
        assert 6 + sum(2 * 6 * (1 + t["draws"] + int(t["replay"])) for t in grid) == nn
        assert len({t["key"] for t in grid}) == len(grid)
        namespaces.append({tuple(t["data_seed"]) for t in grid})
    assert not namespaces[0].intersection(namespaces[1])
    for case in ("gaussian", "four_spherical", "moons"):
        task = {"case": case, "n": 32, "data_seed": [781, 1], "noise_seed": [782, 2]}
        clean, noise, z, truth = generate(task)
        noisy, rz, rt = reference(task)
        np.testing.assert_allclose(clean + noise, noisy, atol=1e-13)
        np.testing.assert_array_equal(z, rz)
        np.testing.assert_array_equal(truth, rt)
        assert seed32([1, 2, 3]) == seed32([1, 2, 3])


def test_small_saved_fit_and_corruption_audits(tmp_path):
    raw = np.random.default_rng(783).normal(size=(24, 8))
    dataset = write_dataset(raw, tmp_path / "raw")
    for arm in (ARMS[0], ARMS[1], ARMS[4]):
        result = run_arm(dataset, tmp_path / "raw", tmp_path / arm, arm, 784)
        assert result["status"] == "complete"
        a = audit_point(raw, tmp_path / arm, arm.replace("denoising_", "neural_"), result)
        audit_density(a, result)
        if arm.startswith("denoising_"):
            audit_training((raw - raw.mean(0)) / raw.std(0), a, 784)
        damaged = dict(a, labels=np.zeros(len(raw), dtype=int) + 999)
        with pytest.raises(AssertionError):
            audit_density(damaged, result)
    write_json(
        {
            str(p.relative_to(tmp_path)): fingerprint_file(p).sha256
            for p in tmp_path.rglob("*")
            if p.is_file()
        },
        tmp_path / "checksums.json",
    )
    audit_inventory(tmp_path)
    (tmp_path / "unexpected.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        audit_inventory(tmp_path)


def test_simultaneous_and_paired_failure_bounds():
    rows = [
        {"status": "complete", "correct_k": True, "ari": 0.5, "coverage": 0.8, "selected_k": 4},
        {"status": "failed"},
    ]
    r = rate(rows)
    assert r["failure_bounds"] == [0.5, 1]
    assert r["simultaneous_95_interval"][1] == 1
    no = {**rows[0], "correct_k": False}
    p = paired([(rows[0], no), (rows[1], no)])
    assert p["failure_bounds"] == [0, 1]
    assert p["resolved_difference"] == 1
    assert p["simultaneous_95_difference_interval"][0] < 0
    assert interval(0, 10)[0] == 0 and interval(10, 10)[1] == 1
    assert noise_ari([-1, -1, 0, 0], [-1, -1, 3, 3]) == 1
    assert noise_ari([-1, -1, 0, 0], [4, 4, 3, 3]) < 1


def test_resource_gate_and_main_refusal(tmp_path, monkeypatch):
    from super_deli.benchmarking import denoising_study as module

    monkeypatch.setattr(module, "assert_frozen", lambda root: {})
    write_json({"passed": True}, tmp_path / "preflight-audit.json")
    write_json({"observations": [{"fit_seconds": 1}]}, tmp_path / "preflight-execution.json")
    (tmp_path / "preflight").mkdir()
    result = module.gate(tmp_path, 2 * 1024**3)
    assert result["status"] == "failed"
    with pytest.raises(ValueError, match="passed audited"):
        module.run_phase(tmp_path, "main", 1)
    assert not (tmp_path / "main").exists()


def test_complete_observation_orchestration_and_replays(tmp_path):
    from super_deli.benchmarking.denoising_study import run_observation
    from super_deli.benchmarking.density_audit import _audit_support
    from super_deli.benchmarking.neural_analysis import arrays
    from super_deli.io.config import read_json

    task = {
        "case": "gaussian",
        "n": 24,
        "replicate": 0,
        "key": "invented-development",
        "data_seed": [787, 1],
        "noise_seed": [788, 2],
        "subsampling_seed": [789, 3],
        "corruption_seed": [790, 4],
        "draws": 1,
        "replay": True,
    }
    result = run_observation((task, str(tmp_path)))
    assert result["attempts"] == 44
    out = tmp_path / task["key"]
    audit_inventory(out)
    rows = read_json(out / "fits.json")
    for condition in ("clean", "noisy"):
        for arm in ARMS:
            point = next(
                r
                for r in rows
                if r["kind"] == "point" and r["condition"] == condition and r["arm"] == arm
            )
            replay = next(
                r
                for r in rows
                if r["kind"] == "replay" and r["condition"] == condition and r["arm"] == arm
            )
            assert point["status"] == replay["status"] == "complete"
            for key, value in arrays(out / point["path"] / "point.npz").items():
                np.testing.assert_array_equal(
                    value, arrays(out / replay["path"] / "point.npz")[key]
                )
            draws = [
                r
                for r in rows
                if r["kind"] == "draw" and r["condition"] == condition and r["arm"] == arm
            ]
            _audit_support(24, draws, arrays(out / condition / f"{arm}-support.npz"))
