"""Small development fixtures verify orchestration without using study seeds."""

# ruff: noqa: E402
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from super_deli.benchmarking.neural_analysis import ari, audit_point, reference
from super_deli.benchmarking.neural_study import (
    ARMS,
    generate,
    run_arm,
    tasks,
    write_dataset,
)


def test_frozen_grid_and_independent_generating_laws():
    for phase, expected in (("preflight", 120), ("main", 1210)):
        grid = tasks(phase)
        assert sum(5 * (1 + t["draws"] + int(t["replay"])) for t in grid) == expected
        assert len({t["key"] for t in grid}) == len(grid)
    for i, case in enumerate(("gaussian", "four_spherical", "moons", "cline")):
        # Development streams are disjoint from every approved observation.
        task = {"case": case, "n": 32, "data_seed": [61, i], "noise_seed": [62, i]}
        a = generate(task)
        b = reference(task)
        np.testing.assert_allclose(a[0], b[0], atol=1e-14)
        np.testing.assert_array_equal(a[1], b[1])
        assert a[0].shape == (32, 8)
        if case == "cline":
            assert a[2] is b[2] is None
        else:
            np.testing.assert_array_equal(a[2], b[2])


def test_saved_fit_audits_for_all_model_families(tmp_path):
    raw = np.random.default_rng(63).normal(size=(32, 8))
    dataset = write_dataset(raw, tmp_path / "raw")
    for arm in ARMS[:3]:
        result = run_arm(dataset, tmp_path / "raw", tmp_path / arm, arm)
        assert result["status"] == "complete"
        audit_point(raw, tmp_path / arm, arm, result)
        with pytest.raises(AssertionError):
            audit_point(raw, tmp_path / arm, arm, {**result, "labels": [999] * len(raw)})


def test_ari_matches_independent_library():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(64)
    for n in (2, 10, 32):
        for _ in range(5):
            a, b = rng.integers(0, 4, size=(2, n))
            assert ari(a.tolist(), b.tolist()) == pytest.approx(adjusted_rand_score(a, b))


def test_shared_density_extraction_preserves_parent_implementation(tmp_path):
    import importlib.util
    import subprocess
    import sys

    from super_deli.io.tabular import ingest_tabular
    from super_deli.models.hdbscan_config import HDBSCANConfig
    from super_deli.models.tabular_hdbscan import (
        fit_density_once,
        point_arrays,
        prepare_density_table,
    )

    root = Path(__file__).resolve().parents[1]
    try:
        source = subprocess.check_output(
            [
                "git",
                "show",
                "13a4f6106b2a34817979f98695885ee35141056d:src/super_deli/models/tabular_hdbscan.py",
            ],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Parent Git history unavailable in installed source archives")
    path = tmp_path / "parent_density.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("parent_density", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        raw = np.random.default_rng(65).normal(size=(40, 8))
        dataset = write_dataset(raw, tmp_path / "raw")
        table = ingest_tabular(
            dataset, base_dir=tmp_path / "raw", output_dir=tmp_path / "ingest"
        ).tables[0]
        config = HDBSCANConfig(True)
        x, names, types = prepare_density_table(table, config)
        old = module.point_arrays(module.fit_density_once(x, names, types, config))
        new = point_arrays(fit_density_once(x, names, types, config))
        for key, value in old.items():
            np.testing.assert_array_equal(value, new[key])
    finally:
        del sys.modules[spec.name]
