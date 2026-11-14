"""Read-only diagnosis of archived genomic neural maps; zero estimator fits."""

import time
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from super_deli import __version__
from super_deli.benchmarking.genomic_diagnosis_archive import Archive, digest
from super_deli.benchmarking.genomic_diagnosis_config import GenomicDiagnosisConfig
from super_deli.benchmarking.genomic_diagnosis_math import evaluate, require
from super_deli.benchmarking.genomic_diagnosis_summary import adjusted_rand, summarize
from super_deli.io.config import write_json

ARMS = {
    "pca_diagonal": "pca_diagonal",
    "pca_shared": "pca_shared",
    "neural_seed_0": "categorical_ae",
    "neural_seed_1": "categorical_ae",
    "neural_seed_2": "categorical_ae",
}


def read_case(archive, phase, case, out):
    prefix = f"{phase}/{case['case_id']}"
    analysis = prefix + "/analysis"
    require(archive.json(prefix + "/case.json") == case, "Frozen case declaration")
    done = archive.json(prefix + "/completion.json")
    require(
        done["status"] == "complete" and done["audit"]["passed"] and done["steps"]["passed"],
        "Completed source audit required",
    )
    native = archive.json(analysis + "/summary.json")
    manifest = archive.json(analysis + "/manifest.json")
    require(
        native["status"] == "complete"
        and native["inference"] == dict(status="not_computed", p_value=None),
        "Preserved source inference status",
    )
    require(
        {r["name"]: r["family"] for r in native["models"]} == ARMS, "All five original point arms"
    )
    cohort = archive.json(analysis + "/inputs/cohort.json")
    raw = archive.array(analysis + "/inputs/dosages.npy")
    source = archive.arrays(prefix + "/generating-law.npz")
    require(raw.shape == (case["n"], source["values"].shape[1]), "Source case dimensions")
    ids = cohort["unit_ids"]
    require(
        len(ids) == len(set(ids)) == len(raw)
        and ids == native["unit_ids"]
        and len(set(cohort["sample_ids"])) == len(raw),
        "Original source identities",
    )
    require(
        len(cohort["locus_keys"]) == raw.shape[1]
        and cohort["original_locus_indices"] == list(range(raw.shape[1])),
        "Original locus identities",
    )
    for j in range(raw.shape[1]):
        require(
            np.array_equal(raw[:, j], source["values"][:, j])
            or np.array_equal(2 - raw[:, j], source["values"][:, j]),
            "SNPio original dosage/allele orientation",
        )
    alleles = archive.array(analysis + "/inputs/source-alleles.npy")
    require(alleles.shape == raw.shape, "Source allele dimensions")
    for j in range(raw.shape[1]):
        for allele in np.unique(alleles[:, j]):
            states = np.unique(raw[alleles[:, j] == allele, j])
            require(
                len(states) == 1 and states[0] in ([1] if allele == "R" else [0, 2]),
                "Observed allele coding consistency",
            )
    sources = archive.json(analysis + "/inputs/sources.json")
    for item in sources:
        path = archive.path(analysis + "/inputs/" + item["snapshot"])
        require(digest(path) == item["source"]["sha256"], "Frozen source fingerprint")
    require(
        cohort["locus_keys"]
        == [f"{sources[0]['source']['sha256']}:{j}" for j in range(raw.shape[1])],
        "Locus keys bind source hash/order",
    )
    if archive.config.source_role == "completed_study":
        maxima = []
        for replicate in range(1 + case["n_subsamples"]):
            for offset in (2, 3, 4):
                trace = archive.arrays(
                    prefix + f"/step-audits/training-{5 * replicate + offset:04d}.npz"
                )
                require(
                    trace["errors"].shape == (500, 4) and trace["step_sha256"].shape == (500,),
                    "Original actual-update audit schema",
                )
                require(
                    np.isfinite(trace["errors"]).all() and np.all(trace["errors"] >= 0),
                    "Finite original update errors",
                )
                require(
                    all(len(str(value)) == 64 for value in trace["step_sha256"]),
                    "Original update hashes",
                )
                maxima.append(trace["errors"].max(0))
        require(
            done["steps"]["trainings"] == len(maxima)
            and done["steps"]["optimizer_updates"] == 500 * len(maxima),
            "Original audited update accounting",
        )
        np.testing.assert_array_equal(np.max(maxima, axis=0), done["steps"]["maximum_errors"])
    records = []
    minimum = manifest["model"]["min_cluster_size"]

    def model(relative, values, arm, draw):
        meta = archive.json(relative + "/model.json")
        arrays = archive.arrays(relative + "/model.npz")
        require(
            meta["family"] == ARMS[arm]
            and meta["locus_keys"] == cohort["locus_keys"]
            and meta["model_sha256"] == digest(archive.path(relative + "/model.npz")),
            "Source numeric model identity",
        )
        require(
            all(a.dtype.kind in "biuf" and np.isfinite(a).all() for a in arrays.values()),
            "Finite numeric model arrays only",
        )
        result, details = evaluate(values, arrays, meta, minimum)
        key = f"{case['case_id']}__{arm}__" + ("point" if draw is None else f"draw_{draw:04d}")
        result.update(
            case_id=case["case_id"],
            law=case["law"],
            n=case["n"],
            training_n=len(values),
            phase=phase,
            arm=arm,
            family=ARMS[arm],
            draw=draw,
            ari=adjusted_rand(
                source["truth"] if draw is None else source["truth"][draw_indices], arrays["labels"]
            )
            if case["law"] == "four_groups"
            else None,
        )
        np.savez_compressed(out / "records" / (key + ".npz"), **details)
        write_json(result, out / "records" / (key + ".json"))
        records.append(result)
        return arrays

    for arm in ARMS:
        model(analysis + "/arms/" + arm, raw, arm, None)
    subsamples = {arm: archive.json(analysis + "/arms/" + arm + "/subsamples.json") for arm in ARMS}
    for _arm, rows in subsamples.items():
        require(len(rows) == case["n_subsamples"], "Requested archived draws")
    pairs = {arm: np.zeros((len(raw), len(raw)), dtype=np.int64) for arm in ARMS}
    attempts = np.zeros((len(raw), len(raw)), dtype=np.int64)
    for draw in range(case["n_subsamples"]):
        folder = analysis + f"/subsamples/draw-{draw:04d}"
        bundle = archive.json(folder + "/records.json")
        indices = None
        for arm in ARMS:
            record = bundle[arm]
            require(
                record == subsamples[arm][draw] and record["status"] == "complete",
                "Archived draw status/accounting",
            )
            draw_indices = np.asarray(record["training_indices"], dtype=int)
            require(
                len(draw_indices) == int(0.8 * len(raw))
                and np.array_equal(np.unique(draw_indices), draw_indices)
                and draw_indices.min() >= 0
                and draw_indices.max() < len(raw),
                "Unique valid archived training identities",
            )
            require(
                record["unit_ids"] == [ids[i] for i in draw_indices], "Original-unit draw linkage"
            )
            if indices is None:
                indices = draw_indices.copy()
                attempts[np.ix_(indices, indices)] += 1
            else:
                np.testing.assert_array_equal(draw_indices, indices)
            arrays = model(folder + "/" + arm, raw[draw_indices], arm, draw)
            np.testing.assert_array_equal(arrays["labels"], record["labels"])
            require(
                record["selected_k"] == arrays["membership"].shape[1], "Archived draw selected K"
            )
            pairs[arm][np.ix_(indices, indices)] += (
                arrays["labels"][:, None] == arrays["labels"][None]
            )
    if case["n_subsamples"]:
        for arm in ARMS:
            a = archive.arrays(analysis + "/arms/" + arm + "/stability.npz")
            np.testing.assert_array_equal(a["pair_included"], attempts)
            np.testing.assert_array_equal(a["pair_successful"], attempts)
            np.testing.assert_array_equal(a["pair_coassigned"], pairs[arm])
    return records


def run_genomic_diagnosis(archive_dir, *, config, output_dir, case_ids=None):
    config = GenomicDiagnosisConfig.from_dict(config.to_dict())
    out = Path(output_dir).absolute()
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        archive = Archive(archive_dir, config)
        frozen = archive.json("study-freeze.json")
        cases = [
            (phase, case) for phase in ("preflight", "main") for case in frozen["cases"][phase]
        ]
        require(
            len(cases) == config.expected_observations
            and len({c["case_id"] for p, c in cases}) == len(cases),
            "Expected source observation accounting",
        )
        require(
            sum(5 for p, c in cases) == config.expected_point_models
            and sum(5 * c["n_subsamples"] for p, c in cases) == config.expected_subsample_models,
            "Expected archived map counts",
        )
        if config.source_role == "completed_study":
            prior = archive.json("full-independent-audit.json")
            require(
                prior["point_searches"] == 240
                and prior["subsample_searches"] == 360
                and prior["optimizer_updates"] == 180000
                and prior["retained_failed_searches"] == 0,
                "Original independent audit receipt",
            )
            require(
                archive.json("legacy-replay-audit.json")["searches"] == 6,
                "Original native replay receipt",
            )
            require(all(c["n"] in (64, 128) for p, c in cases), "Original sample sizes")
        if case_ids is not None:
            require(
                len(case_ids) == len(set(case_ids))
                and set(case_ids) <= {c["case_id"] for p, c in cases},
                "Explicit unique archive case selection",
            )
            cases = [(p, c) for p, c in cases if c["case_id"] in case_ids]
        require(cases, "At least one saved observation required")
        (out / "records").mkdir()
        records = []
        with threadpool_limits(limits=1):
            for phase, case in cases:
                records.extend(read_case(archive, phase, case, out))
        summary = summarize(records, config.source_role)
        if case_ids is not None:
            summary["decision"]["status"] = "preflight_subset_no_development_decision"
        archive.reverify()
        write_json(
            dict(
                config=config.to_dict(),
                source_root=str(archive.root),
                consumed_sha256=archive.consumed,
            ),
            out / "source-binding.json",
        )
        result = dict(
            schema_version=1,
            software=dict(super_deli=__version__, numpy=np.__version__, scipy=scipy.__version__),
            status="complete",
            observations=len(cases),
            point_models=sum(r["draw"] is None for r in records),
            subsample_models=sum(r["draw"] is not None for r in records),
            archived_neural_maps=sum(r["family"] == "categorical_ae" for r in records),
            new_training_fits=0,
            new_gmm_searches=0,
            new_random_draws=0,
            pvalues=dict(status="not_computed", historical_values_changed=False),
            **summary,
            seconds=time.monotonic() - started,
        )
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        require(size < config.max_output_bytes, "Diagnostic output cap")
        write_json(result, out / "summary.json")
        write_json(
            dict(
                completed=True,
                source_manifest_sha256=config.source_manifest_sha256,
                artifact_sha256={
                    p.relative_to(out).as_posix(): digest(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                },
            ),
            out / "completion.json",
        )
        return result
    except Exception as exc:
        write_json(
            dict(status="withheld", decision="unavailable", error=f"{type(exc).__name__}: {exc}"),
            out / "failure.json",
        )
        raise ValueError("Archived evidence diagnosis withheld; see failure.json") from exc
