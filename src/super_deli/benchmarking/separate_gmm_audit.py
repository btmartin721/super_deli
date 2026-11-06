"""Independent raw, optimizer, probability, selection, projection and support audits."""

import time

import numpy as np

from super_deli.benchmarking.coordinate_audit import audit_mixture
from super_deli.benchmarking.matched_audit import audit_sources, synthetic_tables
from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.separate_audit import audit_pca
from super_deli.benchmarking.separate_gmm_config import ARMS, HEADS, NATIVE, model_for
from super_deli.benchmarking.separate_gmm_data import reference
from super_deli.benchmarking.separate_gmm_observer import evaluation_path
from super_deli.benchmarking.separate_gmm_run import replay_equal
from super_deli.benchmarking.view_weight_audit import audit_density, audit_trajectory
from super_deli.io.config import read_json
from super_deli.models.separate_neural import unpack_branch
from super_deli.models.subsample_stability import subsample_indices, summarize_subsamples
from super_deli.workflows.separate_gmm import load_separate_gmm_model


def audit_arm(folder, task, data, row, cache):
    started = time.monotonic()
    arm = row["arm"]
    native = arm in NATIVE
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    cfg = model_for(arm, seed, task["draws"] if native else 0)
    from super_deli.core.provenance import configuration_hash

    require(
        configuration_hash(row["configuration"]) == configuration_hash(cfg.to_dict()),
        "Frozen arm configuration",
    )
    require(not row["numerical_guard_failed"], "Unresolved optimizer guard")
    events = read_json(folder / "optimizer-checks.json") if native else []
    evidence = arrays(folder / "optimizer-checks.npz") if native else {}
    subs = read_json(folder / "subsamples.json") if (folder / "subsamples.json").exists() else []
    expected = subsample_indices(task["n"], task["draws"], 0.8, seed) if native else ()
    for record, ids in zip(subs, expected, strict=False):
        np.testing.assert_array_equal(record["training_indices"], ids)
        require(record["unit_ids"] == [f"unit_{i:04d}" for i in ids], "Whole-ID matched sample")
    require(len(subs) <= len(expected), "No replacement/excess draws")
    fitrows = read_json(folder / "evaluations.json")
    used = set()
    verified = 0
    candidate_count = 0
    density_count = 0
    nslots = 1 + row["draws_requested"]
    require(
        [(r["index"], r["head"]) for r in fitrows]
        == [
            (i, h)
            for i in range(nslots)
            for h in (("gmm_shared",) if arm == "raw_shared" else HEADS)
        ],
        "Complete head/draw accounting",
    )
    for index in range(nslots):
        ids = np.arange(task["n"]) if index == 0 else expected[index - 1]
        path = (
            evaluation_path(folder, index, "coordinates") / "point.npz"
            if native
            else folder / "point.npz"
        )
        if not path.exists():
            require(
                all(r["status"] == "unattempted" for r in fitrows if r["index"] == index),
                "Unavailable draw explicitly retained",
            )
            continue
        a = arrays(path)
        scores = None
        if native:
            parts = []
            for role in ("morphology", "ecology"):
                b = unpack_branch(a, role)
                x = data[role][ids]
                if "scores" not in b:
                    continue
                kept = b["kept_indices"]
                z = (x[:, kept] - b["means"][kept]) / b["scales"][kept]
                matches = [
                    e
                    for e in events
                    if e["fit"] not in used
                    and e["input_sha256"] == digest_arrays({"input": z})
                    and (e["status"] == "complete")
                ]
                require(bool(matches), "Complete branch trajectory linked to raw IDs")
                event = matches[0]
                used.add(event["fit"])
                audit_trajectory(
                    x, b, row["configuration"], event, evidence, event["fit"], cache, paired=False
                )
                verified += 500
                parts.append(b["scores"])
            if len(parts) == 2:
                scores = np.column_stack(parts)
                if "scores" in a:
                    np.testing.assert_array_equal(scores, a["scores"])
        elif arm == "pca4":
            for role in ("morphology", "ecology"):
                audit_pca(data[role], unpack_branch(a, role))
            scores = np.column_stack(
                [unpack_branch(a, role)["scores"] for role in ("morphology", "ecology")]
            )
            np.testing.assert_array_equal(scores, a["scores"])
        else:
            x = np.column_stack((data["morphology"], data["ecology"]))
            np.testing.assert_allclose(a["means"], x.mean(0), atol=1e-12)
            np.testing.assert_allclose(a["scales"], x.std(0), atol=1e-12)
            scores = (x - x.mean(0)) / x.std(0)
            np.testing.assert_array_equal(scores, a["scores"])
        for record in (r for r in fitrows if r["index"] == index):
            head = record["head"]
            where = evaluation_path(folder, index, head)
            saved = arrays(where / "fit.npz")
            meta = read_json(where / "execution.json")
            require(all(meta[k] == record[k] for k in meta), "Head metadata bound to evaluations")
            if record["status"] == "unattempted":
                require(scores is None, "No silently skipped available representation")
                continue
            require(scores is not None, "Fitted head has complete coordinates")
            if head == "hdbscan":
                if record["status"] == "available":
                    audit_density(
                        {**saved, "scores": scores},
                        {"fit_0_linkage": saved["linkage"], "fit_0_condensed": saved["condensed"]},
                        0,
                    )
                    density_count += 1
                else:
                    require(bool(record.get("error")), "Retained density failure")
            else:
                checked, _ = audit_mixture(scores, {**record, "clusterer": head}, saved)
                candidate_count += checked
            if native and head == "gmm_shared" and (record["status"] == "available"):
                np.testing.assert_array_equal(a["labels"], saved["labels"])
                np.testing.assert_array_equal(a["membership"], saved["membership"])
    for e in events:
        if e["fit"] in used:
            continue
        i = e["fit"]
        errs = evidence[f"fit_{i}_errors"]
        ratios = evidence[f"fit_{i}_normalized_errors"][: len(errs)]
        require(
            e["status"] == "failed" or row["status"] == "failed",
            "Only failed partial outputs may lack latent coordinates",
        )
        require(
            np.isfinite(errs).all() and np.all(np.isfinite(ratios)) and np.all(ratios <= 1),
            "Retained partial live checks",
        )
        require(len(errs) == e.get("updates_checked", 0) <= 500, "Partial update accounting")
        verified += len(errs)
    require(row["neural_fits_attempted"] == len(events), "Attempted training denominator")
    if native and row["status"] == "available":
        model = load_separate_gmm_model(folder)
        p = model.predict(synthetic_tables(data))
        np.testing.assert_array_equal(p["scores"], model.arrays["scores"])
        np.testing.assert_allclose(
            p["membership"], model.arrays["membership"], rtol=1e-08, atol=1e-08
        )
        stable = arrays(folder / "stability.npz")
        expected_stable = summarize_subsamples(task["n"], subs)
        for k, v in expected_stable.arrays.items():
            np.testing.assert_allclose(stable[k], v, equal_nan=True, atol=1e-12)
        for i, record in enumerate(subs):
            if record["status"] == "complete":
                saved = arrays(evaluation_path(folder, i + 1, "coordinates") / "point.npz")
                np.testing.assert_array_equal(saved["labels"], record["labels"])
                for k, v in saved.items():
                    np.testing.assert_array_equal(stable[f"draw_{i}_{k}"], v)
    require(
        row["attempted"] == sum(r["status"] != "unattempted" for r in fitrows), "Attempted heads"
    )
    require(
        row["failed_points"] == sum(r["status"] != "available" for r in fitrows),
        "Unavailable heads",
    )
    return dict(
        arm=arm,
        n=task["n"],
        draws=row["draws_requested"],
        key=task["key"],
        neural_fits=len(events),
        verified_updates=verified,
        candidate_densities_checked=candidate_count,
        density_fits=density_count,
        seconds=time.monotonic() - started,
    )


def audit_dataset(folder, task, *, cache):
    data = audit_sources(folder, task, reference_function=reference)
    rows = read_json(folder / "fits.json")
    require([r["arm"] for r in rows] == list(ARMS), "Ordered representation accounting")
    return [audit_arm(folder / r["arm"], task, data, r, cache) for r in rows]


def audit_replay(out, source, entry, cache):
    source = source.resolve()
    data = arrays(source / entry["path"] / "generating-reference.npz")
    require(read_json(out / "replay.json") == entry, "Prespecified replay roster")
    row = read_json(out / "fits.json")[0]
    require(row.get("exact_numeric_replay"), "Exact legacy numerical replay")
    replay_equal(source, out, entry)
    return [audit_arm(out / row["arm"], {**entry["task"], "draws": 0}, data, row, cache)]
