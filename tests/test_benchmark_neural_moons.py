"""Numerical, corruption and gate checks using invented, untrained fixtures."""

# ruff: noqa: E402
import copy
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from super_deli.benchmarking.neural_moons_archive import audit_archive, check_binding, inventory
from super_deli.benchmarking.neural_moons_audit import audit_context, audit_fit
from super_deli.benchmarking.neural_moons_config import NeuralMoonsConfig
from super_deli.benchmarking.neural_moons_coordinates import load_context, safe_path, standardize
from super_deli.benchmarking.neural_moons_fit import fit_coordinates, recovery
from super_deli.benchmarking.neural_moons_fixture import create_fixture
from super_deli.benchmarking.neural_moons_report import report
from super_deli.benchmarking.neural_moons_study import audit_phase, freeze, gate, run_phase
from super_deli.io.config import read_json


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return create_fixture(tmp_path_factory.mktemp("neural-moons") / "archive")


@pytest.fixture
def entry_context(archive):
    entry = read_json(archive / "fixture.json")["observations"][1]
    context, diagnostics = load_context(archive, entry)
    return entry, context, diagnostics


def test_coordinate_construction_and_independent_reconstruction(archive, entry_context):
    entry, context, diagnostics = entry_context
    audit_context(context, entry, archive, diagnostics)
    assert context["traits_standard"].shape[1] == 8
    assert context["neural_0_standard"].shape[1] == 2
    bad = {**entry, "unit_ids": list(reversed(entry["unit_ids"]))}
    with pytest.raises(AssertionError):
        audit_context(context, bad, archive)
    corrupted = copy.deepcopy(diagnostics)
    corrupted["neural_0"]["mse_by_feature"][0] += 1
    with pytest.raises(AssertionError):
        audit_context(context, entry, archive, corrupted)


@pytest.mark.parametrize("indices", [[0, 0], [True] * 10, [0.1] * 10, [-1] * 10, [999] * 10])
def test_invalid_or_repeated_source_indices(archive, entry_context, indices):
    with pytest.raises(ValueError):
        load_context(archive, {**entry_context[0], "indices": indices})


def test_path_and_information_guards(archive):
    for path in ("../outside", "/absolute"):
        with pytest.raises(ValueError):
            safe_path(archive, path)
    for values in (np.ones((12, 2)), np.ones(12), np.full((12, 2), np.nan), np.ones((2, 2))):
        with pytest.raises(ValueError):
            standardize(values)


@pytest.mark.parametrize(
    "space",
    ["source_standard", "traits_standard", "pca_native", "neural_0_raw", "neural_0_standard"],
)
def test_independent_graph_policy_and_projection_audit(entry_context, space):
    _, context, _ = entry_context
    record, arrays = fit_coordinates(context[space], context, config=NeuralMoonsConfig())
    assert record["hierarchy_status"] == "available"
    assert set(record["policies"]) == {"baseline", "no_root", "leaf"}
    audit_fit(context, space, record, arrays)
    broken = {**arrays, "core_distances": arrays["core_distances"] + 0.1}
    with pytest.raises(AssertionError):
        audit_fit(context, space, record, broken)
    broken_record = copy.deepcopy(record)
    broken_record["policies"]["baseline"]["selection"]["nodes"][0]["own_stability"] += 0.1
    with pytest.raises(AssertionError):
        audit_fit(context, space, broken_record, arrays)


def test_one_graph_fit_for_three_policies(entry_context, monkeypatch):
    from super_deli.benchmarking import neural_moons_fit as module

    _, context, _ = entry_context
    original, calls = module.HDBSCAN, []

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "HDBSCAN", counted)
    record, _ = module.fit_coordinates(
        context["neural_0_standard"], context, config=NeuralMoonsConfig()
    )
    assert len(calls) == 1 and len(record["policies"]) == 3


def test_unavailable_hierarchy_is_not_a_selected_group(entry_context, monkeypatch):
    from super_deli.benchmarking import neural_moons_fit as module

    _, context, _ = entry_context

    def unavailable(*args):
        raise ValueError("Unsupported test hierarchy")

    monkeypatch.setattr(module, "retained_hierarchy", unavailable)
    record, arrays = module.fit_coordinates(
        context["neural_0_standard"], context, config=NeuralMoonsConfig()
    )
    assert record["hierarchy_status"] == "unavailable"
    assert record["policies"]["no_root"]["status"] == "unavailable"
    assert "selected_k" not in record["policies"]["no_root"]
    audit_fit(context, "neural_0_standard", record, arrays)


def test_all_noise_and_workspace_failure(entry_context):
    _, context, _ = entry_context
    result = recovery(np.full(12, -1), np.repeat([0, 1], 6))
    assert result["selected_k"] == 0 and result["coverage"] == 0 and result["assigned_ari"] is None
    with pytest.raises(ValueError, match="workspace"):
        fit_coordinates(
            context["traits_standard"], context, config=NeuralMoonsConfig(max_working_bytes=1)
        )


def test_known_separated_groups_and_tied_distances():
    rng = np.random.default_rng(4507)
    scores = np.r_[rng.normal(-4, 0.02, (20, 2)), rng.normal(4, 0.02, (20, 2))]
    context = {
        "truth": np.repeat([0, 1], 20),
        "source_standard": standardize(scores),
        "traits_standard": standardize(np.tile(scores, (1, 4))),
        "example": scores,
    }
    record, data = fit_coordinates(scores, context, config=NeuralMoonsConfig())
    assert record["policies"]["baseline"]["selected_k"] == 2
    assert record["policies"]["baseline"]["ari"] == 1
    assert record["graph"]["mutual_threshold_separable"]
    audit_fit(context, "example", record, data)
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    context = {
        "truth": (grid[:, 0] >= 2).astype(int),
        "source_standard": standardize(grid),
        "traits_standard": standardize(np.tile(grid, (1, 4))),
        "example": grid,
    }
    record, data = fit_coordinates(grid, context, config=NeuralMoonsConfig())
    assert record["neighborhood"]["cutoff_tie_rows"] > 0
    audit_fit(context, "example", record, data)


def test_failed_attempts_remain_visible_and_block_the_preflight_gate(tmp_path, archive):
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented failed-attempt engineering test.")
    audit_archive(archive, output_dir=root, fixture=True)
    freeze(root, NeuralMoonsConfig(fixture=True, max_working_bytes=1))
    result = run_phase(root, archive, "preflight")
    assert result["failed"] == result["attempted"] == 20
    audit = audit_phase(root, archive, "preflight")
    assert audit["failed"] == 20
    assert gate(root, staging_bytes=0)["passed"] is False
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, archive, "main")


def test_corrupt_archive_and_weights_are_rejected(tmp_path, archive, entry_context):
    target = tmp_path / "source"
    shutil.copytree(archive, target)
    entry = entry_context[0]
    file = target / entry["points"]["neural_0"]
    with np.load(file, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    data["encoder_0_weight"] = np.ones((2, 2))
    with file.open("wb") as f:
        np.savez_compressed(f, **data)
    with pytest.raises(ValueError, match="weights"):
        load_context(target, entry)
    original = inventory(archive)
    with pytest.raises(ValueError, match="inventory"):
        check_binding(target, {"source_sha256": original})


def test_end_to_end_gate_failure_and_source_preservation(tmp_path, archive):
    before = inventory(archive)
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture; no scientific inference.")
    audit_archive(archive, output_dir=root, fixture=True)
    with pytest.raises(ValueError, match="source kind"):
        freeze(root, NeuralMoonsConfig())
    result = freeze(root, NeuralMoonsConfig(fixture=True))
    assert result["fits"] == {"preflight": 20, "main": 26}
    with pytest.raises(FileNotFoundError):
        run_phase(root, archive, "main")
    run_phase(root, archive, "preflight")
    audit = audit_phase(root, archive, "preflight")
    assert audit["attempted"] == 20 and audit["archive_replays"] == 10
    assert gate(root, staging_bytes=2 * 1024**3)["passed"] is False
    with pytest.raises(ValueError, match="gate"):
        run_phase(root, archive, "main")
    assert inventory(archive) == before


def test_complete_workflow_and_corrupt_output_detection(tmp_path, archive):
    root = tmp_path / "study"
    root.mkdir()
    (root / "approved-plan.md").write_text("Invented engineering fixture; no scientific inference.")
    audit_archive(archive, output_dir=root, fixture=True)
    freeze(root, NeuralMoonsConfig(fixture=True))
    run_phase(root, archive, "preflight")
    audit_phase(root, archive, "preflight")
    assert gate(root, staging_bytes=0)["passed"]
    run_phase(root, archive, "main", workers=2)
    audit = audit_phase(root, archive, "main")
    assert audit["attempted"] == 26 and audit["archive_replays"] == 16
    assert audit["selection_evaluations"] == 62
    assert report(root)["fixture"]
    with pytest.raises(FileExistsError):
        run_phase(root, archive, "main")
    file = next((root / "main").glob("*/neural_0_raw/result.json"))
    file.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        audit_phase(root, archive, "main")
