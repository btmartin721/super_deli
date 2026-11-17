"""Fit-free omitted/external scoring from all sealed retained native candidates."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.heldout_audit import audit_report
from super_deli.benchmarking.heldout_metrics import events, external_reference, recovery
from super_deli.io.config import read_json, write_json
from super_deli.io.heldout_support_source import load_subsample_source
from super_deli.models.heldout_predictive import (
    gain_values,
    logdensity,
    source_arrays,
    summarize_scores,
)
from super_deli.models.heldout_support_config import HeldoutSupportConfig


def archive(source, output):
    if source.config.fixture:
        raise ValueError("Invented seals cannot authorize a scientific archive audit")
    output = Path(output)
    output.mkdir()
    parents = []
    for index in range(48):
        case = source.case(index)
        native = load_subsample_source(case / "subsample")
        for name in native.files:
            source.path(f"validation/main/fresh-{index:03d}/subsample/{name}")
        directory = output / f"case-{index:03d}"
        directory.mkdir()
        scores = directory / "scores"
        scores.mkdir()
        # The new diagnostic is fit-free here: use the actual saved native K=1 candidate.
        candidates = read_json(case / "subsample/observed-candidates/index.json")
        pipeline = read_json(case / "subsample/observed-pipeline/events.json")
        records = []
        for r, event in zip(native.records, pipeline, strict=True):
            row = dict(
                replicate=r["replicate"],
                omitted_indices=r["omitted_indices"],
                fit_seed=r["fit_seed"],
                source_selected_k=r.get("selected_k"),
                status="unavailable_native",
                baseline_search_attempted=False,
            )
            if r["status"] == "complete":
                a, chosen = source_arrays(native, r)
                c = next(
                    c
                    for c in candidates[event["candidate_start"] : event["candidate_stop"]]
                    if c["k"] == 1
                )
                with np.load(
                    case / "subsample/observed-candidates" / c["arrays"], allow_pickle=False
                ) as f:
                    base_parameters = {k: f[k].copy() for k in ("weights", "means", "covariances")}
                    np.testing.assert_array_equal(f["scores"], a["training_scores"])
                baseline, _, _ = logdensity(a["projected_scores"], base_parameters)
                raw_gain, gain = gain_values(a, chosen, baseline, base_parameters, r["selected_k"])
                name = f"draw_{r['replicate']:04d}.npz"
                with (scores / name).open("xb") as f:
                    np.savez(
                        f,
                        **base_parameters,
                        selected_logdensity=chosen,
                        baseline_logdensity=baseline,
                        raw_gain=raw_gain,
                        gain=gain,
                    )
                native_k1 = next(c for c in r["candidates"] if c["k"] == 1)
                row.update(
                    status="complete",
                    arrays=name,
                    baseline_candidate=native_k1,
                    baseline_origin="Retained native source; zero new fits",
                )
            records.append(row)
        config = HeldoutSupportConfig()
        summary, arrays, groups = summarize_scores(native, records, scores, config)
        summary.update(
            schema_version=1,
            policy=config.policy,
            source_point_k=native.point.summary["selected_k"],
        )
        # Bind the original source instead of duplicating its private arrays in the archive audit.
        summary_path = directory / "summary.json"
        write_json(summary, summary_path)
        write_json(config.to_dict(), directory / "configuration.json")
        write_json(dict(files=native.files, source=str(native.root)), directory / "source.json")
        write_json(records, directory / "predictive_replicates.json")
        with (directory / "individual_scores.npz").open("xb") as f:
            np.savez(f, **arrays)
        write_json(groups, directory / "cluster_support.json")
        numerical = audit_report(directory, source=native)
        external = external_reference(case / "subsample", directory)
        truth = read_json(case / "observed/truth-evaluation-only.json")
        lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
        point = recovery([lookup[u] for u in native.point.units], native.point.labels)
        expected = read_json(case / "summary.json")
        task = expected["task"]
        row = dict(
            index=index,
            task=task,
            predictive=summary,
            external=external,
            point=point,
            primary=events(summary, external, point, task["law"] == "continuous_control"),
            audit=numerical,
            new_fits=0,
        )
        parents.append(row)
        native.reverify()
    source.reverify()
    result = dict(passed=True, parents=parents, observations=48, draws=1200, new_fits=0)
    write_json(result, output / "summary.json")
    return result
