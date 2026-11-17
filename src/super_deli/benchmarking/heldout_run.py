"""Unchanged native fits with separate predictive reporting and evaluation."""

import time
from functools import partial
from pathlib import Path

import numpy as np

from super_deli.benchmarking.bootstrap_split_audit import audit
from super_deli.benchmarking.full_retention_observer import observe_pipeline
from super_deli.benchmarking.full_retention_run import run_native
from super_deli.benchmarking.heldout_audit import audit_report
from super_deli.benchmarking.heldout_metrics import events, external_reference, recovery
from super_deli.benchmarking.heldout_simulation import generate
from super_deli.benchmarking.individual_subsample_audit import audit_subsampling
from super_deli.benchmarking.individual_subsample_observer import predictions
from super_deli.benchmarking.individual_subsample_simulation import audit_law, load_evaluation
from super_deli.benchmarking.initialization_audit import audit_initialization
from super_deli.benchmarking.initialization_study import size
from super_deli.benchmarking.signal_loss_fit import observe_candidates
from super_deli.io.config import read_json, write_json
from super_deli.io.individual_subsample_source import load_point_source
from super_deli.models.heldout_support_config import HeldoutSupportConfig
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig
from super_deli.workflows.heldout_support import run_heldout_support
from super_deli.workflows.individual_subsampling import run_individual_subsampling


def subsample(point, output, evaluation, draws):
    output = Path(output)
    captures = output.parent / (output.name + "-captures")
    captures.mkdir()
    with predictions(output, evaluation):
        with observe_candidates(
            captures / "observed-candidates", initialization=True
        ) as candidates:
            with observe_pipeline(captures / "observed-pipeline", candidates):
                summary = run_individual_subsampling(
                    point,
                    stability_config=IndividualSubsampleConfig(draws, 1601),
                    output_dir=output,
                )
    for child in captures.iterdir():
        child.rename(output / child.name)
    captures.rmdir()
    a = audit_subsampling(output, evaluation)
    return summary, a


def evaluate_case(case, task, point, subsummary, subaudit, pointaudit, *, native_searches, seconds):
    case = Path(case)
    prediction = run_heldout_support(
        case / "subsample", support_config=HeldoutSupportConfig(), output_dir=case / "predictive"
    )
    report_audit = audit_report(case / "predictive")
    external = external_reference(case / "subsample", case / "predictive")
    known = read_json(case / "observed/truth-evaluation-only.json")
    lookup = dict(zip(known["unit_ids"], known["labels"], strict=True))
    y = np.array([lookup[u] for u in point.units])
    point_metrics = recovery(y, point.labels)
    numerical_withheld = subaudit["withheld"] + pointaudit.get("withheld", 0)
    primary = events(prediction, external, point_metrics, len(task["counts"]) == 1)
    if numerical_withheld:
        primary = {key: None for key in primary}
    result = dict(
        task=task,
        predictive=prediction,
        external=external,
        point=point_metrics,
        primary=primary,
        subsampling=subsummary,
        numerical_withheld=numerical_withheld,
        predictive_unavailable_draws=subsummary["requested"] - prediction["successful_draws"],
        external_unavailable_draws=sum(r["status"] != "complete" for r in external["rows"]),
        source_audit=subaudit,
        predictive_audit=report_audit,
        native_searches=native_searches,
        baseline_searches=prediction["baseline_searches"],
        planned_searches=native_searches + subsummary["requested"],
        attempted_searches=native_searches + prediction["baseline_searches"],
    )
    result["seconds"] = time.monotonic() - seconds
    result["bytes"] = size(case)
    write_json(result, case / "summary.json")
    return result


def parent(task, directory, generator, *, draws=25):
    directory = Path(directory)
    directory.mkdir()
    started = time.monotonic()
    innovations = [
        int(c.generate_state(1)[0])
        for c in np.random.SeedSequence([task["seed"], 20261117]).spawn(2)
    ]
    ds, model, truth = generate(directory / "observed", task, innovations[0], "o", generator)
    ed, em, _ = generate(
        directory / "evaluation", task, innovations[1], "e", generator, evaluation=True
    )
    audit_law(directory / "observed")
    audit_law(directory / "evaluation")
    evaluation = load_evaluation(
        ed, em, directory / "evaluation", directory / "evaluation-ingestion"
    )
    original = directory / "point"
    with predictions(original, evaluation):
        native = run_native(
            ds,
            model,
            directory / "observed",
            original,
            law="continuous_control" if len(task["counts"]) == 1 else "shape_only",
            truth=truth,
            auditor=audit_initialization,
            observer=partial(observe_candidates, initialization=True),
        )
    if native["status"] != "complete":
        result = dict(
            task=task,
            status="unavailable_point",
            point=None,
            predictive=None,
            external=None,
            primary={k: None for k in ("qualified", "positive", "agreement", "joint")},
            native_searches=1,
            baseline_searches=0,
            planned_searches=1 + 2 * draws,
            attempted_searches=1,
            numerical_withheld=1,
            error=native["error"],
            seconds=time.monotonic() - started,
            bytes=size(directory),
        )
        write_json(result, directory / "summary.json")
        return result
    point = load_point_source(original)
    with np.load(directory / "observed/known-raw.npz", allow_pickle=False) as raw:
        ids = {u: i for i, u in enumerate(truth["unit_ids"])}
        for view in point.inputs:
            np.testing.assert_array_equal(
                view.values, raw[view.kind][[ids[u] for u in view.unit_ids]]
            )
    pa = audit(
        original,
        (point.inputs, point.units),
        evaluation,
        candidate_directory="observed-candidates",
        pipeline_directory="observed-pipeline",
    )
    ss, sa = subsample(original, directory / "subsample", evaluation, draws)
    write_json(
        dict(task=task, innovations=innovations, configuration=model.to_dict(), draws=draws),
        directory / "freeze.json",
    )
    return evaluate_case(
        directory, task, point, ss, sa, pa, native_searches=1 + draws, seconds=started
    )
