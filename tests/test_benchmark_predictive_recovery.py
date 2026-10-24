"""Numerical reuse, repeated grouping, oracle separation and audit integrity."""

from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from super_deli.benchmarking.local_archives import materialize  # noqa: E402
from super_deli.benchmarking.predictive_recovery_analysis import audit_observation  # noqa: E402
from super_deli.benchmarking.predictive_recovery_study import _observation  # noqa: E402
from super_deli.io.config import read_json  # noqa: E402
from super_deli.models.predictive_recovery_config import PredictiveRecoveryConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = PredictiveRecoveryConfig.from_dict(
        read_json(ROOT / "examples/predictive-recovery/smoke.json")
    )
    return replace(
        c,
        study=replace(
            c.study, cells=(replace(c.study.cells[1], n_units=32),), min_free_bytes=1024**2
        ),
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("predictive-recovery")
    c = configuration()
    declaration = c.plan()[0]
    row = _observation((str(root), c.to_dict(), declaration))
    assert row["successful_searches"] == 17 and row["successful_oracles"] == 6 and not row["errors"]
    return root, c, declaration, row


def test_all_partitions_keep_every_locality_member_and_share_full_fit(fitted, tmp_path):
    root, c, declaration, row = fitted
    with materialize(root / row["archive"], row, scratch=tmp_path) as base:
        layout = read_json(base / "layout.json")
        assert row["parts"][1]["shared_full_search"] and not row["parts"][2]["shared_full_search"]
        for p in row["parts"]:
            workflow = base / p["name"] / "workflow"
            seen = []
            for f in read_json(workflow / "fold_assignments.json"):
                seen += f["validation_indices"]
                expected = set(f["training_unit_ids"])
                trained = read_json(workflow / f"fold_{f['fold']:03d}/training_layout.json")
                assert set(trained["unit_ids"]) == expected
                assert all(s["locality_id"] in expected for s in trained["specimens"])
                assert sum(trained["counts"]) == len(trained["specimens"])
            assert sorted(seen) == list(range(32))
        first = base / "part_000/workflow/full/run/gmm_candidates.npz"
        assert (
            first.read_bytes()
            == (base / "part_001/workflow/full/run/gmm_candidates.npz").read_bytes()
        )
        assert read_json(base / "part_000/workflow/fold_assignments.json") == read_json(
            base / "part_extended/workflow/fold_assignments.json"
        )
        for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
            for i in range(2):
                with (
                    np.load(
                        base / f"part_000/workflow/{prefix}/run/view_{i:03d}/parameters/pca.npz"
                    ) as a,
                    np.load(
                        base
                        / f"part_extended/workflow/{prefix}/run/view_{i:03d}/parameters/pca.npz"
                    ) as b,
                ):
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
        assert set(layout["unit_ids"]).isdisjoint(
            read_json(base / "external_layout.json")["unit_ids"]
        )


def test_independent_audit_verifies_every_space_and_oracle(fitted, tmp_path):
    root, c, declaration, row = fitted
    audited = audit_observation((str(root), declaration, row, c.to_dict(), str(tmp_path)))
    assert len(audited["oracles"]) == 6 and all(
        o["role"].endswith("never selection") for o in audited["oracles"]
    )


def test_external_data_mutation_cannot_change_fitted_selector(fitted, tmp_path):
    from super_deli.benchmarking.predictive_study import evaluate
    from super_deli.models.predictive_study_config import PredictiveCell

    root, c, declaration, row = fitted
    with materialize(root / row["archive"], row, scratch=tmp_path) as base:
        workflow = base / "part_000/workflow"
        before = (workflow / "selected_model.npz").read_bytes()
        with np.load(base / "values.npz") as p:
            values = {k: p[k] for k in p.files}
        with np.load(base / "external_values.npz") as p:
            changed = {k: p[k] + 1000 for k in p.files}
        with np.load(base / "truth.npz") as p:
            truth = p["training"]
        part = base / "evaluation-only"
        part.mkdir()
        import shutil

        shutil.copytree(workflow, part / "workflow")
        result = evaluate(
            part,
            values,
            read_json(base / "layout.json"),
            truth,
            PredictiveCell.from_dict(declaration["cell_config"]),
            changed,
            read_json(base / "external_layout.json"),
            row["parts"][0]["selection"],
        )
        assert result["screened"]["k"] == row["parts"][0]["selection"]["screened_k"]
        assert before == (part / "workflow/selected_model.npz").read_bytes()


def test_oracle_failure_does_not_enter_unsupervised_selection(fitted, tmp_path, monkeypatch):
    import super_deli.benchmarking.full_grouped_oracle as oracle

    root, c, declaration, baseline = fitted
    original = oracle.fit_oracle
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("deliberate oracle-only failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(oracle, "fit_oracle", fail_once)
    row = _observation((str(tmp_path), c.to_dict(), declaration))
    assert (
        row["failed_oracles"] == 1
        and row["successful_oracles"] == 5
        and row["successful_searches"] == 17
    )
    assert [p["selection"] for p in row["parts"]] == [p["selection"] for p in baseline["parts"]]
    audit_observation((str(tmp_path), declaration, row, c.to_dict(), str(tmp_path / "scratch")))


def test_real_fold_failure_is_withheld_and_budget_is_not_replaced(tmp_path, monkeypatch):
    import super_deli.models.integrative as integration

    c = configuration()
    original = integration.select_gaussian_mixture
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("deliberate failed first partition fold")
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "select_gaussian_mixture", fail_once)
    declaration = c.plan()[0]
    row = _observation((str(tmp_path), c.to_dict(), declaration))
    assert row["attempted_searches"] == 17 and row["failed_searches"] == 1
    assert (
        row["parts"][0]["selection"]["status"] == "withheld"
        and row["parts"][0]["selection"]["screened_k"] is None
    )
    assert row["unattempted_oracles"] == 1 and row["attempted_oracles"] == 5
    audit_observation((str(tmp_path), declaration, row, c.to_dict(), str(tmp_path / "scratch")))


def test_stdlib_checker_keeps_ordinary_result_after_fold_failure(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    import super_deli.models.integrative as integration
    from super_deli.benchmarking.predictive_recovery_analysis import summarize
    from super_deli.io.config import write_json

    c = configuration()
    original = integration.select_gaussian_mixture
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("deliberate first fold failure for standalone checker")
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "select_gaussian_mixture", fail_once)
    row = _observation((str(tmp_path), c.to_dict(), c.plan()[0]))
    assert row["parts"][0]["performance"]["ordinary"] is not None
    write_json(c.to_dict(), tmp_path / "configuration.json")
    write_json(c.plan(), tmp_path / "plan.json")
    (tmp_path / "observations.jsonl").write_text(json.dumps(row) + "\n")
    audit = tmp_path / "audit"
    audit.mkdir()
    write_json(summarize([row], c), audit / "summary.json")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/predictive-recovery/check_recovery.py"),
            "--study-root",
            str(tmp_path),
            "--audit-root",
            str(audit),
            "--output",
            str(tmp_path / "independent.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(tmp_path / "independent.json")["failed_searches"] == 1


def test_external_density_near_boundary_uses_stable_solve():
    """A 70-digit independent calculation pins this synthetic regression value."""
    from super_deli.benchmarking.predictive_recovery_audit import external_densities

    weights = np.array(
        [0.07843095657741345, 0.23529455280685665, 0.5098039005926602, 0.17647059002306964]
    )
    means = np.array(
        [
            [0.7304953690931273, -0.7256163813015114, 0.36333082292732916, 0.005854028905433215],
            [0.8097574389246062, 0.11689353488247849, -0.11324798713209339, -0.03486768682261954],
            [-0.46002927305831154, 0.3769883301141415, 0.06833052234409111, 0.01977436552550333],
            [-0.07536806701594564, -0.9224412196462262, -0.2078811904209224, -0.013237384115327131],
        ]
    )
    covariances = np.array(
        [
            [
                [
                    0.1002052570701064,
                    0.03569880489482599,
                    0.020008766215880745,
                    0.011635750206716582,
                ],
                [
                    0.035698804894825985,
                    0.18801872703648911,
                    -0.015963735139138593,
                    -0.040170401291445224,
                ],
                [
                    0.020008766215880745,
                    -0.015963735139138593,
                    0.03955404613944273,
                    -0.038335006998625555,
                ],
                [
                    0.011635750206716585,
                    -0.040170401291445224,
                    -0.038335006998625555,
                    0.07904246990422466,
                ],
            ],
            [
                [
                    0.06757414616460071,
                    0.009191293479148535,
                    0.01727737149590688,
                    0.04287186289693126,
                ],
                [
                    0.009191293479148535,
                    0.023383978813881667,
                    -0.010997786352701606,
                    0.0272599192670204,
                ],
                [
                    0.017277371495906878,
                    -0.010997786352701605,
                    0.03531149620609731,
                    -0.0003218368440673345,
                ],
                [
                    0.04287186289693126,
                    0.0272599192670204,
                    -0.00032183684406733484,
                    0.058799215476139165,
                ],
            ],
            [
                [
                    0.19248476856504865,
                    0.1746773323516336,
                    0.013290326229362808,
                    -0.0067465253816523605,
                ],
                [
                    0.1746773323516336,
                    0.2636672247176938,
                    -0.03316010207491686,
                    -0.02853796889159029,
                ],
                [
                    0.013290326229362808,
                    -0.03316010207491686,
                    0.03343868716929157,
                    0.0043474988825903714,
                ],
                [
                    -0.006746525381652361,
                    -0.02853796889159029,
                    0.0043474988825903714,
                    0.041100949899603445,
                ],
            ],
            [
                [
                    0.0884766205459369,
                    0.008113597804059037,
                    0.009129685377628295,
                    0.018183193020750894,
                ],
                [
                    0.008113597804059037,
                    0.06570374588480922,
                    -0.013798040650204,
                    0.03752559717215998,
                ],
                [
                    0.009129685377628295,
                    -0.013798040650204,
                    0.006093247149942004,
                    -0.007958124777292302,
                ],
                [
                    0.018183193020750894,
                    0.03752559717215998,
                    -0.007958124777292302,
                    0.057030845576749443,
                ],
            ],
        ]
    )
    scores = np.array(
        [[0.4359424868027319, -0.8866339682227851, 0.1453285934464591, 0.2376303075632571]]
    )
    expected = np.array([-23.146037003745445])
    pool = {"k4_weights": weights, "k4_means": means, "k4_covariances": covariances}
    held = {"scores": scores, "k4": expected}
    candidates = [{"k": 4, "status": "small_component"}]
    actual = external_densities(pool, candidates, held)
    np.testing.assert_allclose(actual["4"], expected, atol=1e-9, rtol=0)
    held["k4"] = expected + 1e-4
    with pytest.raises(AssertionError):
        external_densities(pool, candidates, held)
