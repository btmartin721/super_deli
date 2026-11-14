"""Contracts, cohort guards, native parity and failure preservation for matched analyses."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from super_deli.io.config import load_config, read_json
from super_deli.models.matched_analysis_config import MatchedAnalysisArm, MatchedAnalysisConfig

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return MatchedAnalysisConfig.from_dict(
        read_json(ROOT / "examples/matched-analysis/analysis.json")
    )


def test_roundtrip_and_native_serialization():
    c = configuration()
    assert MatchedAnalysisConfig.from_dict(c.to_dict()) == c
    assert c.arms[0].model.covariance_type == "tied"
    assert c.arms[0].model.to_dict() == c.to_dict()["arms"][0]["model"]


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"ecology_measurement_scope": "locality"},
        {"extra": True},
        {"arms": []},
    ],
)
def test_analysis_guards(change):
    with pytest.raises(ValueError):
        MatchedAnalysisConfig.from_dict({**configuration().to_dict(), **change})


def test_reject_unsafe_names_and_feature_disagreement():
    c = configuration()
    with pytest.raises(ValueError):
        replace(c, arms=(replace(c.arms[0], name="../bad"),))
    with pytest.raises(ValueError):
        replace(
            c,
            arms=(
                c.arms[0],
                replace(
                    c.arms[1], model=replace(c.arms[1].model, ecology_features=("e2", "e1", "e3"))
                ),
            ),
        )
    with pytest.raises(ValueError):
        replace(c, arms=(c.arms[0], c.arms[0]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_level", "locality"),
        ("missing_views", "complete_cases"),
        ("covariance_type", "full"),
        ("assignment_threshold", 0.9),
    ],
)
def test_pca_boundary(field, value):
    c = configuration()
    model = replace(c.arms[0].model, **{field: value})
    with pytest.raises(ValueError):
        MatchedAnalysisArm("bad", "pca_gmm", model)


def source_copy(tmp_path):
    import shutil

    source = ROOT / "examples/matched-analysis"
    destination = tmp_path / "source"
    shutil.copytree(source, destination)
    return destination, load_config(destination / "dataset.json")


def test_pca_only_does_not_import_torch(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from super_deli.workflows.matched_analysis import run_matched_analysis

    source, dataset = source_copy(tmp_path)
    c = configuration()
    c = replace(c, arms=(c.arms[0],))
    before = "torch" in sys.modules
    summary = run_matched_analysis(
        dataset, analysis_config=c, base_dir=source, output_dir=tmp_path / "result"
    )
    assert summary["status"] == "complete" and ("torch" in sys.modules) == before
    assert summary["comparisons"] == []


@pytest.mark.parametrize(
    "violation",
    ["missing_unit", "extra_specimen", "missing_cell", "count_feature", "locality_ecology"],
)
def test_cohort_guard_precedes_any_fit(tmp_path, monkeypatch, violation):
    pytest.importorskip("numpy")
    from super_deli.core.schema import config_from_dict
    from super_deli.workflows import matched_analysis as w

    source, dataset = source_copy(tmp_path)
    d = dataset.to_dict()
    if violation == "missing_unit":
        d["modalities"][1]["observations"].pop()
    elif violation == "extra_specimen":
        d["specimens"].append(dict(specimen_id="unobserved"))
    elif violation == "count_feature":
        d["modalities"][0]["features"][0]["dtype"] = "count"
    elif violation == "locality_ecology":
        d["modalities"][1]["level"] = "locality"
    else:
        p = source / d["modalities"][0]["path"]
        s = p.read_text().splitlines()
        parts = s[1].split("\t")
        parts[1] = "NA"
        s[1] = "\t".join(parts)
        p.write_text("\n".join(s) + "\n")
    called = []
    monkeypatch.setattr(w, "run_matched_arm", lambda *a: called.append(a))
    out = tmp_path / "result"
    with pytest.raises((ValueError, RuntimeError)):
        w.run_matched_analysis(
            config_from_dict(d), analysis_config=configuration(), base_dir=source, output_dir=out
        )
    assert not called


def test_failed_arm_is_not_k1_and_other_arm_is_retained(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from super_deli.workflows import matched_analysis as w

    source, dataset = source_copy(tmp_path)
    native = w.run_matched_arm

    def run(arm, *args):
        if arm.family == "separate_gmm":
            raise ValueError("injected native failure")
        return native(arm, *args)

    monkeypatch.setattr(w, "run_matched_arm", run)
    out = tmp_path / "result"
    summary = w.run_matched_analysis(
        dataset, analysis_config=configuration(), base_dir=source, output_dir=out
    )
    assert summary["status"] == "partial"
    assert summary["models"][0]["status"] == "complete"
    assert summary["models"][1]["selected_k"] is None
    assert summary["comparisons"][0]["status"] == "unavailable"
    assert "Point fit unavailable" in (out / "report.html").read_text()
    with pytest.raises(FileExistsError):
        w.run_matched_analysis(
            dataset, analysis_config=configuration(), base_dir=source, output_dir=out
        )


def test_frozen_tampering_aborts_remaining_arms(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from super_deli.workflows import matched_analysis as w

    source, dataset = source_copy(tmp_path)
    native = w.run_matched_arm
    called = []

    def run(arm, dataset, base, out):
        called.append(arm.name)
        result = native(arm, dataset, base, out)
        with (base / dataset.modalities[0].path).open("a") as f:
            f.write("\n")
        return result

    monkeypatch.setattr(w, "run_matched_arm", run)
    with pytest.raises(ValueError, match="hash mismatch"):
        w.run_matched_analysis(
            dataset,
            analysis_config=configuration(),
            base_dir=source,
            output_dir=tmp_path / "result",
        )
    assert called == ["pca"]


def test_neural_native_parity_and_canonical_assignments(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from super_deli.workflows.matched_analysis import run_matched_analysis
    from super_deli.workflows.matched_analysis_adapters import run_matched_arm

    source, dataset = source_copy(tmp_path)
    c = configuration()
    out = tmp_path / "result"
    summary = run_matched_analysis(dataset, analysis_config=c, base_dir=source, output_dir=out)
    assert summary["status"] == "complete"
    frozen = load_config(out / "inputs/dataset.json")
    for arm in c.arms:
        other = tmp_path / ("native-" + arm.name)
        run_matched_arm(arm, frozen, out / "inputs", other)
        name = "mixture.npz" if arm.family == "pca_gmm" else "point.npz"
        with (
            np.load(out / "arms" / arm.name / name, allow_pickle=False) as a,
            np.load(other / name, allow_pickle=False) as b,
        ):
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    assert summary["unit_ids"] == sorted(summary["unit_ids"])
    assert summary["comparisons"][0]["joint_coverage"] == 1
    assert summary["inference"] == dict(status="not_computed", p_value=None)


def test_failed_native_resample_retains_point_and_denominator(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from super_deli.models import separate_gmm
    from super_deli.workflows.matched_analysis import run_matched_analysis

    source, dataset = source_copy(tmp_path)
    c = configuration()
    arm = replace(c.arms[1], model=replace(c.arms[1].model, n_subsamples=1))
    c = replace(c, arms=(c.arms[0], arm))
    original = separate_gmm.fit_shared_point
    calls = []

    def fit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("injected resampling failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(separate_gmm, "fit_shared_point", fit)
    summary = run_matched_analysis(
        dataset, analysis_config=c, base_dir=source, output_dir=tmp_path / "result"
    )
    row = summary["models"][1]
    assert summary["status"] == "partial" and row["selected_k"] is not None
    assert row["stability"]["requested"] == row["stability"]["failed"] == 1
    assert row["stability_status"] == "unavailable"
    assert summary["comparisons"][0]["status"] == "available"


def test_cli_partial_exit_code(tmp_path, monkeypatch, capsys):
    from super_deli.cli import main
    from super_deli.workflows import matched_analysis_cli

    monkeypatch.setattr(matched_analysis_cli, "dispatch", lambda args: dict(status="partial"))
    assert (
        main(
            [
                "analyze-matched-tabular",
                "unused.json",
                "--analysis-config",
                "unused.json",
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
        == 1
    )
    assert "partial" in capsys.readouterr().out


def test_native_fingerprint_corruption_is_withheld(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from super_deli.workflows.matched_analysis import run_matched_analysis
    from super_deli.workflows.matched_analysis_adapters import read_matched_arm

    source, dataset = source_copy(tmp_path)
    c = configuration()
    c = replace(c, arms=(c.arms[0],))
    out = tmp_path / "result"
    run_matched_analysis(dataset, analysis_config=c, base_dir=source, output_dir=out)
    child = out / "arms/pca"
    with (child / "mixture.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="fingerprint"):
        read_matched_arm(
            c.arms[0],
            child,
            read_json(out / "inputs/cohort.json"),
            load_config(out / "inputs/dataset.json"),
        )


def test_report_escapes_dataset_and_failure_text(tmp_path):
    from super_deli.workflows.matched_analysis_report import write_matched_report

    bad = "<script>alert(1)</script>"
    summary = dict(
        dataset_id=bad,
        status="failed",
        models=[dict(name="safe", family="pca_gmm", error=bad)],
        comparisons=[],
        limitations=[bad],
    )
    target = tmp_path / "report.html"
    write_matched_report(summary, {}, target)
    text = target.read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
