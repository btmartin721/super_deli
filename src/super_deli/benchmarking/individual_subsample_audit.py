"""Independent sampling/denominator/support and native parameter export checks."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.bootstrap_split_audit import audit, logdensity
from super_deli.benchmarking.full_retention_audit import close
from super_deli.io.config import read_json, write_json
from super_deli.io.individual_subsample_source import load_point_source


def audit_subsampling(directory, evaluation, *, write=True):
    directory = Path(directory)
    decl = read_json(directory / "configuration.json")
    source = load_point_source(directory / "reference")
    records = read_json(directory / "subsample_replicates.json")
    events = read_json(directory / "observed-pipeline/events.json")
    numerical = audit(
        directory,
        (source.inputs, source.units),
        evaluation,
        candidate_directory="observed-candidates",
        pipeline_directory="observed-pipeline",
        write=False,
    )
    assert len(records) == decl["n_subsamples"] == len(events)
    n = len(source.units)
    m = int(np.floor(0.8 * n))
    draw_children = np.random.SeedSequence(decl["random_seed"]).spawn(len(records))
    fit_children = np.random.SeedSequence(source.model.random_seed).spawn(len(records) + 1)[1:]
    pair_attempt, pair_success, pair_same, omitted_attempt, omitted_success, omitted_same = [
        np.zeros((n, n), dtype=np.uint64) for _ in range(6)
    ]
    for _i, (r, e, draw_child, fit_child) in enumerate(
        zip(records, events, draw_children, fit_children, strict=True)
    ):
        draw = np.sort(np.random.default_rng(draw_child).choice(n, m, replace=False))
        absent = np.array(sorted(set(range(n)) - set(draw)))
        np.testing.assert_array_equal(draw, r["training_indices"])
        np.testing.assert_array_equal(draw, e["draw"])
        np.testing.assert_array_equal(absent, r["omitted_indices"])
        assert len(set(draw)) == r["distinct_training_units"] == m and r["training_unit_ids"] == [
            source.units[j] for j in draw
        ]
        assert (
            r["fit_seed"] == e["seed"] == int(fit_child.generate_state(2)[1])
            and r["status"] == e["status"]
        )
        yes = np.zeros(n, dtype=bool)
        yes[draw] = True
        inc = np.logical_and.outer(yes, yes)
        out = np.logical_and.outer(~yes, ~yes)
        pair_attempt += inc
        omitted_attempt += out
        if r["status"] != "complete":
            continue
        with np.load(directory / "observed-pipeline" / e["arrays"], allow_pickle=False) as a:
            labels = a["projected_labels"].copy()
            np.testing.assert_array_equal(labels, r["projected_labels"])
            np.testing.assert_array_equal(labels[draw], r["labels"])
            # Production exports are compared to independently audited native snapshots.
            root = directory / "subsamples" / r["parameters"]
            with np.load(root / "fusion.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(f["fused"], a["fused"])
                np.testing.assert_array_equal(f["components"], a["components"])
                np.testing.assert_array_equal(f["mean"], a["pca_mean"])
            for j, _data in enumerate(source.inputs):
                p = root / f"view_{j:03d}/parameters"
                metadata = read_json(p / "summary.json")
                assert metadata["training_source_indices"] == a[f"v{j}_indices"].tolist()
                assert [v["target_id"] for v in metadata["members"]] == [
                    source.units[k] for k in draw
                ]
                with np.load(p / "profiles.npz", allow_pickle=False) as f:
                    for key in ("profiles", "mean", "scale"):
                        close(f[key], a[f"v{j}_{key}"])
                    close(f["normalized_weight"], a[f"v{j}_weight"])
                    close(
                        f["normalized_weight"],
                        source.model.views[j].weight / sum(v.weight for v in source.model.views),
                    )
            with np.load(root / "fit.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(f["training_scores"], a["scores"])
                np.testing.assert_array_equal(f["projected_labels"], labels)
                if numerical["rows"][_i]["status"] == "complete":
                    ll, prob, independent_labels = logdensity(f["projected_scores"], f)
                    close(prob, f["projected_membership"])
                    np.testing.assert_array_equal(labels, independent_labels)
        same = np.equal.outer(labels, labels)
        pair_success += inc
        pair_same += same & inc
        omitted_success += out
        omitted_same += same & out
    with np.load(directory / "stability.npz", allow_pickle=False) as a:
        for prefix, attempt, success, same in (
            ("included", pair_attempt, pair_success, pair_same),
            ("omitted", omitted_attempt, omitted_success, omitted_same),
        ):
            if prefix == "included":
                np.testing.assert_array_equal(a["included_pair_included"], attempt)
                np.testing.assert_array_equal(a["included_pair_successful"], success)
                np.testing.assert_array_equal(a["included_pair_coassigned"], same)
            else:
                np.testing.assert_array_equal(a["omitted_requested"], attempt)
                np.testing.assert_array_equal(a["omitted_successful"], success)
                np.testing.assert_array_equal(a["omitted_coassignment_counts"], same)
            low = np.full((n, n), np.nan)
            high = low.copy()
            np.divide(same, attempt, out=low, where=attempt > 0)
            np.divide(same + attempt - success, attempt, out=high, where=attempt > 0)
            close(a[prefix + "_coassignment_lower"], low)
            close(a[prefix + "_coassignment_upper"], high)
            # Included native accumulator reports successful-inclusion frequency;
            # omitted predictions with a failed attempt remain unavailable.
            expected = np.full((n, n), np.nan)
            if prefix == "included":
                np.divide(same, success, out=expected, where=success > 0)
            else:
                expected = np.where(attempt == success, low, np.nan)
            close(a[prefix + "_coassignment"], expected)
    support = read_json(directory / "cluster_support.json")
    for k, row in enumerate(support):
        counts = [sum(source.labels[j] == k for j in r["training_indices"]) for r in records]
        assert (
            row["point_individuals"] == int(np.sum(source.labels == k))
            and row["minimum_included"] == min(counts)
            and row["maximum_included"] == max(counts)
        )
        assert row["below_declared_minimum"] == sum(
            c < source.model.min_cluster_size for c in counts
        ) and row["absent"] == sum(c == 0 for c in counts)
        scores = []
        represented = failures = exact = splits = merges = 0
        g = set(np.flatnonzero(source.labels == k))
        for r in records:
            ids = np.array(r["training_indices"])
            c = g & set(ids)
            if not c:
                continue
            represented += 1
            if r["status"] != "complete":
                failures += 1
                continue
            y = np.array(r["labels"])
            parts = [set(ids[y == j]) for j in set(y)]
            scores.append(max(len(c & p) / len(c | p) for p in parts))
            exact += c in parts
            splits += sum(bool(c & p) for p in parts) > 1
            merges += any(c < p for p in parts)
        assert (
            row["represented_attempts"] == represented
            and row["represented_failed"] == failures
            and row["exact_included_recovery_count"] == exact
            and row["split_count"] == splits
            and row["merge_count"] == merges
        )
        if represented:
            close(row["best_jaccard_lower"], sum(scores) / represented)
            close(row["best_jaccard_upper"], (sum(scores) + failures) / represented)
    summary = read_json(directory / "summary.json")
    done = [r for r in records if r["status"] == "complete"]
    assert (
        summary["requested"] == len(records)
        and summary["successful"] == len(done)
        and summary["failed"] == len(records) - len(done)
    )
    for r in summary["selected_k_frequencies"]:
        count = sum(x["selected_k"] == r["k"] for x in done)
        assert r["count"] == count
        close(r["lower"], count / len(records))
        close(r["upper"], (count + len(records) - len(done)) / len(records))
    for row in summary["candidate_eligibility"]:
        statuses = [next(c["status"] for c in r["candidates"] if c["k"] == row["k"]) for r in done]
        assert (
            row["eligible"] == statuses.count("eligible")
            and row["small_component"] == statuses.count("small_component")
            and row["insufficient_samples"] == statuses.count("insufficient_samples")
            and row["unavailable"] == len(records) - len(done)
        )
    assert (
        summary["p_value"] is None
        and summary["species_probability"] is None
        and summary["point_analysis_changed"] is False
    )
    source.reverify()
    result = dict(
        passed=True,
        draws=len(records),
        density_checks=numerical["density_checks"],
        withheld=numerical["withheld"],
        numerical=numerical,
        new_fits=0,
    )
    if write:
        write_json(result, directory / "independent-audit.json")
    return result
