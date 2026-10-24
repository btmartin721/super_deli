"""Independent generating-law, numerical-audit and failure-retention regression checks."""

# ruff: noqa: E402

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from super_deli.benchmarking.grouped_power_analysis import summarize
from super_deli.benchmarking.grouped_power_audit import audit_observation, independent_values
from super_deli.benchmarking.grouped_power_study import _observation, run_study, simulate
from super_deli.benchmarking.grouped_study import simulate_grouped
from super_deli.benchmarking.local_archives import compact_directory, materialize
from super_deli.io.config import read_json, write_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig, PowerCell

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return GroupedPowerStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-power-recovery/smoke.json")
    )


@pytest.mark.parametrize("n", [64, 256])
def test_signal_three_preserves_original_generator_exactly(n):
    actual = simulate(PowerCell("baseline", "complementary", n, 1, 3.0), 82515)
    expected = simulate_grouped("complementary", n, 82515)
    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[2], expected[2])
    for name in actual[0]:
        np.testing.assert_array_equal(actual[0][name], expected[0][name])


@pytest.mark.parametrize(
    "cell",
    [
        PowerCell("low", "complementary", 64, 1, 1.5),
        PowerCell("high", "complementary", 256, 1, 4.5),
        PowerCell("null", "independent_gaussian", 64, 1),
        PowerCell("heavy", "heavy_tailed", 64, 1),
        PowerCell("cline", "cline", 256, 1),
    ],
)
def test_direct_law_counts_and_truth_agree(cell):
    values, layout, labels = simulate(cell, 73914)
    expected, counts, truth = independent_values(cell, 73914)
    np.testing.assert_array_equal(labels, truth)
    np.testing.assert_array_equal(layout["counts"], counts)
    for name in values:
        np.testing.assert_allclose(values[name], expected[name], atol=2e-14, rtol=2e-14)


@pytest.fixture(scope="module")
def complete_observation(tmp_path_factory):
    root = tmp_path_factory.mktemp("power-observation")
    config = configuration()
    config = replace(config, cells=(config.cells[0],), stability_draws=2)
    result = run_study(config, output_dir=root / "study", workers=1)
    assert result["successful_fits"] == 5
    record = json.loads((root / "study/observations.jsonl").read_text().splitlines()[0])
    return root / "study", config, record


def audit_task(source, config, record, scratch):
    return str(source), config.plan()[0], record, config.to_dict(), str(scratch)


def test_compact_numerical_and_whole_group_audit_never_refits(
    complete_observation, tmp_path, monkeypatch
):
    from super_deli.models import grouped_null, integrative

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit attempted a scientific fit")

    monkeypatch.setattr(integrative, "_fit_once", forbidden)
    monkeypatch.setattr(grouped_null, "fit_null", forbidden)
    source, config, record = complete_observation
    result = audit_observation(audit_task(source, config, record, tmp_path / "scratch"))
    assert result["audit_status"] == "complete_numerical_audit"
    assert result["stability"]["point_replay_verified"]
    assert result["stability"]["requested"] == result["stability"]["successful"] == 2
    assert sum(result["stability"]["selected_k_counts"].values()) == 2
    assert len(result["pipelines"]) == 2


@pytest.mark.parametrize("part", ["source", "stability_seed"])
def test_updated_outer_fingerprints_do_not_hide_semantic_tampering(
    complete_observation, tmp_path, part
):
    source, config, record = complete_observation
    with materialize(source / record["archive"], record, scratch=tmp_path / "materialized") as base:
        if part == "source":
            path = base / "observed/view_000.csv"
            path.write_text(path.read_text() + "\n")
        else:
            path = base / "stability/bootstrap_replicates.json"
            rows = read_json(path)
            rows[0]["training_indices"][0] = (rows[0]["training_indices"][0] + 1) % config.cells[
                0
            ].n_units
            path.write_text(json.dumps(rows))
        replacement = tmp_path / "tampered" / record["archive"]
        packed = compact_directory(base, replacement)
    changed = {**record, **packed, "archive": record["archive"], "manifest": record["manifest"]}
    with pytest.raises((ValueError, AssertionError)):
        audit_observation(audit_task(tmp_path / "tampered", config, changed, tmp_path / "scratch"))


def test_rejected_raw_null_is_retained_without_replacement_or_pvalue(tmp_path, monkeypatch):
    from super_deli.workflows import grouped_likelihood

    def refuse(dataset, *, config, base_dir, output_dir):
        output_dir.mkdir()
        write_json(config.to_dict(), output_dir / "configuration.json")
        write_json({"error": "deliberate raw null refusal"}, output_dir / "failure.json")
        raise ValueError("deliberate raw null refusal")

    monkeypatch.setattr(grouped_likelihood, "run_grouped_likelihood", refuse)
    config = configuration()
    config = replace(config, cells=(config.cells[0],), stability_first=0)
    source = tmp_path / "study"
    source.mkdir()
    record = _observation((str(source), config.to_dict(), config.plan()[0]))
    assert record["calibration"]["p_value"] is None
    assert record["planned_fits"] == record["unattempted_fits"] == 2
    assert record["attempted_fits"] == 0
    result = audit_observation(audit_task(source, config, record, tmp_path / "scratch"))
    assert result["likelihood_p"] is None and result["aic_k"] is None
    assert result["audit_status"] == "retained_outer_failure_incomplete_numerical_audit"


def test_primary_endpoint_denominators_keep_point_and_inference_availability_separate():
    config = GroupedPowerStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-power-recovery/study.json")
    )
    config = replace(config, stability_first=0)
    cells = {c.name: c for c in config.cells}
    rows = []
    for declaration in config.plan():
        cell, i = cells[declaration["cell"]], declaration["repetition"]
        rows.append(
            {
                "cell": cell.name,
                "repetition": i,
                "likelihood_reject": i < 10 if i else None,
                "bic_reject": False if i else None,
                "aic_k": 4 if i % 2 == 0 else 2,
                "bic_k": 2,
                "aic_ari": 0.8 if cell.has_groups else None,
                "bic_ari": 0.4 if cell.has_groups else None,
                "stability": None,
                "audit_status": "complete_numerical_audit",
                "archive_members": 0,
                "accounting": {
                    "planned_fits": 100,
                    "attempted_fits": 100,
                    "successful_fits": 100,
                    "failed_fits": 0,
                    "unattempted_fits": 0,
                },
            }
        )
    result = summarize(rows, config)
    assert len(result["primary_rates"]) == result["primary_family_size"] == 18
    first = {r["endpoint"]: r for r in result["primary_rates"] if r["cell"] == config.cells[0].name}
    assert first["likelihood_rejection"]["events"] == 9
    assert first["aic_k4"]["events"] == 25 and first["aic_k4"]["available"] == 50
    joint = first["joint_rejection_and_aic_k4"]
    assert joint["events"] == 4 and joint["available"] == 49 and joint["requested"] == 50
    assert joint["withheld_outcome_bounds"] == [4 / 50, 5 / 50]
    assert joint["bonferroni_familywise_95"][0] <= joint["exact_interval_95"][0]
    assert joint["bonferroni_familywise_95"][1] >= joint["exact_interval_95"][1]
    with pytest.raises(ValueError, match="observation identities"):
        summarize(rows + [rows[0]], config)


def test_full_study_requires_storage_preflight_before_creating_outputs(tmp_path):
    config = GroupedPowerStudyConfig.from_dict(
        read_json(ROOT / "examples/grouped-power-recovery/study.json")
    )
    with pytest.raises(ValueError, match="preflight"):
        run_study(config, output_dir=tmp_path / "unapproved", workers=4)
    assert not (tmp_path / "unapproved").exists()
