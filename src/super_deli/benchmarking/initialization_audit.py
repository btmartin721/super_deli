"""Independent batch selection and native preprocessing/bootstrap accounting."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.full_retention_audit import _search, audit_run, close
from super_deli.io.config import read_json


def search(event, candidates, directory, draw):
    c = event["configuration"]
    if c.get("initialization_policy", "legacy") == "legacy":
        return _search(event, candidates, directory, draw)
    assert c["initialization_policy"] == "diverse_v1" and c["n_init"] == 100
    available = candidates[event["candidate_start"] : event["candidate_stop"]]
    offset = 0
    eligible = []
    observed = {}
    for row in event["candidates"]:
        k = row["k"]
        if len(set(draw)) < k * c["min_cluster_size"]:
            assert row["status"] == "insufficient_samples"
            continue
        fits = available[offset : offset + 3]
        offset += 3
        assert len(fits) == 3
        declaration = row["initialization"]
        assert declaration["policy"] == "diverse_v1" and declaration["total_starts"] == 100
        seed = int(np.random.SeedSequence([event["seed"], k]).generate_state(1)[0])
        for i, (fit, record, (method, starts)) in enumerate(
            zip(
                fits,
                declaration["batches"],
                (("kmeans", 34), ("k-means++", 33), ("random_from_data", 33)),
                strict=True,
            )
        ):
            expected = (
                seed
                if i == 0
                else int(np.random.SeedSequence([seed, i, 20261115]).generate_state(1)[0])
            )
            assert fit["k"] == k and fit["fit_status"] == "complete" and fit["converged"]
            assert fit["seed"] == record["seed"] == expected
            assert fit["n_init"] == record["starts"] == starts
            assert fit["init_params"] == record["method"] == method and record["batch"] == i
            assert record["converged"] and record["iterations"] == fit["iterations"]
            close(record["log_likelihood"], fit["log_likelihood"])
            close(record["lower_bound"], fit["lower_bound"])
        winner = max(range(3), key=lambda i: (fits[i]["lower_bound"], -i))
        assert declaration["selected_batch"] == winner
        fit = fits[winner]
        observed[k] = fit
        with np.load(directory / fit["arrays"], allow_pickle=False) as a:
            labels = a["labels"]
            np.testing.assert_array_equal(
                np.bincount(labels, minlength=k), row["hard_cluster_sizes"]
            )
            distinct = [len(set(np.asarray(draw)[labels == j])) for j in range(k)]
            assert row["distinct_unit_sizes"] == distinct
        expected = "eligible" if min(distinct) >= c["min_cluster_size"] else "small_component"
        assert row["status"] == expected
        close(row["bic"], fit["bic"])
        if expected == "eligible":
            eligible.append((row["bic"], k))
    assert offset == len(available)
    assert min(eligible)[1] == event["selected_k"]
    return observed[event["selected_k"]]


def audit_initialization(directory):
    """Every used batch has strict labels; unused labels cannot select models."""
    from super_deli.benchmarking.signal_loss_fit import audit_candidates

    directory = Path(directory)
    if not (directory / "observed-pipeline/events.json").exists():
        return audit_run(directory, search_auditor=search)
    records = read_json(directory / "observed-candidates/index.json")
    required = set()
    for event in read_json(directory / "observed-pipeline/events.json"):
        if event["status"] != "complete":
            required.update(
                r["arrays"] for r in records[event["candidate_start"] : event["candidate_stop"]]
            )
            continue
        offset = event["candidate_start"]
        for row in event["candidates"]:
            if row["status"] == "insufficient_samples":
                continue
            declaration = row.get("initialization")
            size = 3 if declaration else 1
            winner = declaration["selected_batch"] if declaration else 0
            required.add(records[offset + winner]["arrays"])
            offset += size
        assert offset == event["candidate_stop"]
    ambiguities = []

    def density_audit(path):
        return audit_candidates(path, required_label_arrays=required, label_ambiguities=ambiguities)

    result = audit_run(directory, search_auditor=search, candidate_auditor=density_audit)
    receipt = dict(
        passed=True,
        strict_used_batch_labels=len(required),
        unused_ambiguous_batches=ambiguities,
        probability_likelihood_bic_tolerances="unchanged",
        saved_labels_checked_for_every_batch=True,
    )
    target = directory / "initialization-label-audit.json"
    if target.exists():
        assert read_json(target) == receipt
    else:
        from super_deli.io.config import write_json

        write_json(receipt, target)
    return result


def audit_phase(root, records):
    audits = []
    for path in sorted(Path(root).rglob("validation.json")):
        row = read_json(path)
        audit = audit_initialization(path.parent)
        assert audit == row["audit"]
        audits.append(audit)
    for record in records:
        arms = record["arms"]
        if record["origin"] == "fresh":
            from super_deli.core.partition_comparison import compare_partitions

            task = record["task"]
            truth = read_json(
                Path(root) / f"fresh-{task['index']:03d}-inputs" / "truth-evaluation-only.json"
            )
            lookup = dict(zip(truth["unit_ids"], truth["labels"], strict=True))
            for row in arms.values():
                if row["status"] != "complete":
                    continue
                native = read_json(Path(row["path"]) / "summary.json")
                y = np.array([lookup[u] for u in native["unit_ids"]])
                if task["law"] in {"continuous_control", "conflicting"}:
                    assert row["ari"] is None and row["target_k"] is None
                    assert row["correct_recovery"] is None
                    assert row["control_split"] == (
                        (row["selected_k"] > 1) if task["law"] == "continuous_control" else None
                    )
                else:
                    if task["law"] == "genomic_only":
                        y = y // 2
                    expected = compare_partitions(
                        y.tolist(), row["labels"], [True] * len(y), [True] * len(y)
                    )["adjusted_rand"]
                    close(expected, row["ari"])
                    assert row["target_k"] == len(set(y))
                    assert row["correct_recovery"] == (
                        row["selected_k"] == len(set(y)) and expected >= 0.8
                    )
        complete = [r for r in arms.values() if r["status"] == "complete"]
        if len(complete) < 2:
            continue
        paths = [Path(r["path"]) for r in complete]
        with np.load(paths[0] / "mixture.npz", allow_pickle=False) as a:
            scores = a["scores"].copy()
        reps = [read_json(p / "bootstrap_replicates.json") for p in paths]
        events = [read_json(p / "observed-pipeline/events.json") for p in paths]
        for i, path in enumerate(paths[1:], 1):
            with np.load(path / "mixture.npz", allow_pickle=False) as a:
                np.testing.assert_array_equal(scores, a["scores"])
            assert [r["training_indices"] for r in reps[0]] == [
                r["training_indices"] for r in reps[i]
            ]
            for left, right in zip(events[0], events[i], strict=True):
                assert left["draw"] == right["draw"]
                assert left["attempted_views"] == right["attempted_views"]
                if left["status"] == right["status"] == "complete":
                    with (
                        np.load(
                            paths[0] / "observed-pipeline" / left["arrays"], allow_pickle=False
                        ) as a,
                        np.load(
                            path / "observed-pipeline" / right["arrays"], allow_pickle=False
                        ) as b,
                    ):
                        np.testing.assert_array_equal(a["scores"], b["scores"])
    return dict(
        passed=True,
        runs=len(audits),
        component_checks=sum(a["component_checks"] for a in audits),
        withheld=sum(a["point_withheld"] for a in audits),
        failed_draws=sum(a.get("failed_draws", 0) for a in audits),
        retained_space_checks=sum(a.get("retained_space_checks", 0) for a in audits),
        shape_refit_checks=sum(a.get("shape_complete_checks", 0) for a in audits),
        attempted_refits=sum(max(1, a["events"]) for a in audits),
        complete_refits=sum(
            0 if a["point_withheld"] else 1 + a.get("draws", 0) - a.get("failed_draws", 0)
            for a in audits
        ),
    )
