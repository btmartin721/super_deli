"""Independent retained-evidence audit must detect altered resampling identities."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.full_retention_audit import audit_run
from super_deli.benchmarking.full_retention_run import run_native
from super_deli.io.config import read_json
from super_deli.models.full_retention import full_retention_model


def test_raw_draw_audit_detects_changed_training_members(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("retention_audit_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "inputs"
    d, c = module.invented(source, law="continuous_control", seed=74103, draws=0)
    d, c = module.select_views(d, c, {"traits", "habitat", "outline"})
    model = full_retention_model(replace(c.arms[-1].model, n_bootstrap=1), min_cluster_size=7)
    out = tmp_path / "native"
    row = run_native(
        d,
        model,
        source,
        out,
        law="continuous_control",
        truth=read_json(source / "truth-evaluation-only.json"),
    )
    assert row["status"] == "complete" and row["audit"]["draws"] == 1
    target = out / "observed-pipeline/event_001.npz"
    original = target.read_bytes()
    with np.load(target, allow_pickle=False) as f:
        changed = {k: f[k].copy() for k in f.files}
    changed["v0_indices"][0] = (changed["v0_indices"][0] + 1) % 64
    with target.open("wb") as f:
        np.savez(f, **changed)
    with pytest.raises(AssertionError):
        audit_run(out)
    target.write_bytes(original)
    assert audit_run(out) == row["audit"]
    from super_deli.benchmarking.full_retention_diagnostics import truth_initialized

    native_bytes = (out / "mixture.npz").read_bytes()
    result = truth_initialized(
        out, read_json(source / "truth-evaluation-only.json"), tmp_path / "truth-diagnostic"
    )
    assert result["selected"] is False and result["truth_informed"] is True
    assert (out / "mixture.npz").read_bytes() == native_bytes
    attempts = read_json(tmp_path / "truth-diagnostic/candidates/index.json")
    assert len(attempts) == 1 and attempts[0]["n_init"] == 1


@pytest.mark.parametrize("phase,searches,truth_fits", [("preflight", 637, 2), ("main", 2332, 22)])
def test_mocked_orchestration_honors_fixed_fit_budgets(
    tmp_path, monkeypatch, phase, searches, truth_fits
):
    # Mock-only scheduling contract: no generated scientific data or numerical fits.
    from super_deli.benchmarking import full_retention_study as module
    from super_deli.benchmarking.full_retention_config import FullRetentionStudyConfig
    from super_deli.io.config import write_json
    from super_deli.models.config import BaselineConfig
    from super_deli.models.integrative_config import IntegrativeConfig, ViewConfig
    from super_deli.models.multimodal_analysis_config import (
        MultimodalAnalysisArm,
        MultimodalAnalysisConfig,
    )
    from super_deli.models.shape_cluster_config import ShapeClusterConfig
    from super_deli.models.shape_config import ShapeConfig
    from super_deli.models.tabular_cluster_config import TabularClusterConfig

    names = ("a", "b", "c", "d")
    views = (
        ViewConfig("snp", "genomic", ploidy=2),
        ViewConfig("traits", "morphological", feature_names=names),
        ViewConfig("habitat", "ecological", feature_names=names),
        ViewConfig("outline", "morphometric", shape=ShapeConfig()),
    )
    m = IntegrativeConfig(True, views=views, covariance_type="tied", random_seed=301, n_bootstrap=0)
    a = MultimodalAnalysisConfig(
        (
            MultimodalAnalysisArm(
                "genomic", "genomic", ("snp",), BaselineConfig(2, "error", True, n_bootstrap=0)
            ),
            MultimodalAnalysisArm(
                "traits",
                "tabular",
                ("traits",),
                TabularClusterConfig(True, feature_names=names, n_bootstrap=0),
            ),
            MultimodalAnalysisArm(
                "habitat",
                "tabular",
                ("habitat",),
                TabularClusterConfig(True, feature_names=names, n_bootstrap=0),
            ),
            MultimodalAnalysisArm(
                "shape", "shape", ("outline",), ShapeClusterConfig(True, n_bootstrap=0)
            ),
            MultimodalAnalysisArm("combined", "integrative", tuple(v.modality for v in views), m),
        ),
        "individual",
    )

    def generate(path, *, law="shape_only", **kwargs):
        path.mkdir()
        write_json(dict(unit_ids=[], labels=[], law=law), path / "truth-evaluation-only.json")
        return None, a

    class Archive:
        root = tmp_path

        def stage(self, index, path):
            return generate(path)

        def baseline(self, index):
            return Path("invented")

        def verify_numeric(self, prefix):
            pass

        def reverify(self):
            pass

        consumed = {}

    calls = []

    def native(dataset, model, base, path, law, truth):
        path.mkdir()
        row = dict(
            status="complete",
            configuration=model.to_dict(),
            seconds=0.01,
            bytes=1024,
            path=str(path),
            planned_searches=1 + model.n_bootstrap,
        )
        write_json(row, path / "validation.json")
        calls.append(model)
        return row

    def known(point, truth, output):
        output.mkdir()
        return dict(status="complete", selected=False)

    monkeypatch.setattr(module, "load_generator", lambda p: generate)
    monkeypatch.setattr(module, "_native", native)
    monkeypatch.setattr(module, "truth_initialized", known)
    monkeypatch.setattr(module, "equal_point_arrays", lambda a, b: 1)
    from super_deli.benchmarking import full_retention_validation

    monkeypatch.setattr(
        full_retention_validation, "audit_phase", lambda *args: dict(passed=True, seconds=0.01)
    )
    generator = tmp_path / "examples/multimodal/synthetic.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# invented scheduling fixture")
    other = tmp_path / "examples/matched-genomic/smoke.py"
    other.parent.mkdir()
    other.write_text("# invented scheduling fixture")
    output = tmp_path / "study"
    if phase == "main":
        output.mkdir()
        write_json(dict(passed=True), output / "resource-gate.json")
    result = module.study(
        Archive(), FullRetentionStudyConfig("a" * 64, "b" * 64), output, phase, generator
    )
    assert result["candidate_searches"] == searches
    assert len(result["diagnostics"]) == truth_fits
    assert sum(1 + c.n_bootstrap for c in calls) == searches
    assert sum(c.n_init == 100 for c in calls) == truth_fits
