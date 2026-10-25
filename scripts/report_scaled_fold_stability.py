"""Descriptive paired summaries; original datasets, not draws/folds, are independent."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def event(count, successful, requested):
    from scipy.stats import binomtest

    interval = binomtest(count, successful).proportion_ci(method="exact") if successful else None
    return dict(
        count=count,
        successful=successful,
        requested=requested,
        frequency=count / requested if successful == requested else None,
        conditional_frequency=count / successful if successful else None,
        conditional_exact_mc_interval=[interval.low, interval.high] if interval else None,
        failure_bounds=[count / requested, (count + requested - successful) / requested],
    )


def stability(summary, raw, requested):
    import numpy as np

    if summary is None or raw is None:
        return dict(
            status="unavailable",
            requested=requested,
            clusters=[],
            error="Own/common reference unavailable",
        )
    clusters = []
    with np.load(io.BytesIO(raw), allow_pickle=False) as p:
        n = p["all_counts"].shape[0]
        off = np.triu(np.ones((n, n), dtype=bool), 1)
        covered = (p["oob_successful"] >= 10) & off
        coverage = dict(
            covered_pairs=int(covered.sum()),
            total_pairs=int(off.sum()),
            fraction=float(covered.sum() / off.sum()),
        )
        for c in summary["clusters"]:
            group = np.zeros(n, dtype=bool)
            group[c["unit_indices"]] = True
            within = np.logical_and.outer(group, group) & ~np.eye(n, dtype=bool)
            cross = np.logical_and.outer(group, ~group)
            values = dict(
                point_cluster=c["point_cluster"],
                n_localities=len(c["unit_indices"]),
                exact_recovery=event(c["exact_recovery_count"], summary["successful"], requested),
            )
            for prefix in ("all", "oob"):
                for label, mask in (("within_cohesion", within), ("cross_merging", cross)):
                    key = prefix + "_" + label
                    values[key] = c[key + "_conditional"]
                    values[key + "_covered_pairs"] = c[key + "_covered_pairs"]
                    values[key + "_total_pairs"] = c[key + "_total_pairs"]
                    finite = np.isfinite(p[prefix + "_failure_lower"]) & mask
                    values[key + "_failure_bounds"] = (
                        [
                            float(p[prefix + "_failure_" + side][finite].mean())
                            for side in ("lower", "upper")
                        ]
                        if finite.any()
                        else None
                    )
            clusters.append(values)
    return dict(
        status=summary["status"],
        requested=requested,
        successful=summary["successful"],
        failed=summary["failed"],
        oob_pair_coverage=coverage,
        clusters=clusters,
        interpretation=(
            "Conditional on original locality folds and optimizer; "
            "all-original includes resubstitution; OOB values use covered pairs"
        ),
    )


def dataset_rows(base, row, config):
    desired = {
        "evaluation.json",
        "common_stability.json",
        "common_stability.npz",
        "legacy/summary.json",
        "legacy/stability.npz",
        "scaled/summary.json",
        "scaled/stability.npz",
    }
    files = {}
    with tarfile.open(base / row["archive"], "r|gz") as archive:
        for member in archive:
            if member.name in desired:
                files[member.name] = archive.extractfile(member).read()

    def load(name):
        return json.loads(files[name])

    evaluation = load("evaluation.json")
    result = []
    requested = config["n_bootstrap"]
    for arm in ("legacy", "scaled"):
        model = load(arm + "/summary.json")
        values = evaluation[arm]
        draws = values["draws"]
        complete = [d for d in draws if d["status"] == "complete"]
        counts = Counter(str(d["k"]) for d in complete)
        own_k = values["point_k"]
        common_k = evaluation["legacy"]["point_k"]
        k4 = [d["k4_gate"] for d in complete if d["k4_gate"] is not None]
        gates = dict(
            full_eligibility=sum(v["full_status"] != "eligible" for v in k4),
            fold_eligibility=sum(any(s != "eligible" for s in v["fold_statuses"]) for v in k4),
            nonpositive_gain=sum(
                v["pooled_validation_gain"] is not None and v["pooled_validation_gain"] <= 0
                for v in k4
            ),
            passing=sum(v["passes"] for v in k4),
            available=len(k4),
        )
        own = stability(model.get("stability"), files.get(arm + "/stability.npz"), requested)
        common = (
            own
            if arm == "legacy"
            else stability(
                load("common_stability.json"), files.get("common_stability.npz"), requested
            )
        )
        result.append(
            dict(
                cell=row["cell"],
                repetition=row["repetition"],
                law=row["cell_config"]["law"],
                role=row["cell_config"]["role"],
                n_localities=row["cell_config"]["n_units"],
                arm=arm,
                point_k=own_k,
                point_truth_ari=values["point_truth_ari"],
                point_common_assignment_changes=values["point_common_assignment_changes"],
                successful=len(complete),
                failed=sum(d["status"] != "complete" for d in draws),
                unattempted=requested - len(draws),
                requested=requested,
                selected_k_counts=dict(counts),
                k4=event(counts.get("4", 0), len(complete), requested),
                k_gt1=event(sum(d["k"] > 1 for d in complete), len(complete), requested),
                same_own_k=event(sum(d["k"] == own_k for d in complete), len(complete), requested),
                same_common_k=event(
                    sum(d["k"] == common_k for d in complete), len(complete), requested
                ),
                mean_truth_ari=mean([d["truth_ari"] for d in complete]),
                mean_oob_truth_ari=mean([d["oob_truth_ari"] for d in complete]),
                mean_own_assignment_changes=mean([d["own_assignment_changes"] for d in complete]),
                mean_common_assignment_changes=mean(
                    [d["common_assignment_changes"] for d in complete]
                ),
                oob_counts=[d["oob_count"] for d in complete],
                k4_gates=gates,
                own_stability=own,
                common_stability=common,
            )
        )
    return result


def report(study, output_dir):
    source, output = Path(study), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "configuration.json").read_text())
    outer = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    datasets = [
        v
        for row in sorted(outer, key=lambda r: (r["cell"], r["repetition"]))
        for v in dataset_rows(source, row, config)
    ]
    grouped = defaultdict(list)
    for row in datasets:
        grouped[row["cell"], row["arm"]].append(row)
    cells = []
    for (cell, arm), rows in sorted(grouped.items()):
        frequencies = [
            r["k4"]["conditional_frequency"]
            for r in rows
            if r["k4"]["conditional_frequency"] is not None
        ]
        cells.append(
            dict(
                cell=cell,
                arm=arm,
                datasets=len(rows),
                datasets_with_available_draws=len(frequencies),
                point_k4=sum(r["point_k"] == 4 for r in rows),
                point_extra=sum(r["point_k"] is not None and r["point_k"] > 1 for r in rows),
                mean_dataset_k4=mean(frequencies),
                dataset_k4_range=[min(frequencies), max(frequencies)] if frequencies else None,
                mean_dataset_extra=mean([r["k_gt1"]["conditional_frequency"] for r in rows]),
                mean_dataset_k4_failure_bounds=[
                    mean([r["k4"]["failure_bounds"][i] for r in rows]) for i in (0, 1)
                ],
                mean_truth_ari=mean([r["mean_truth_ari"] for r in rows]),
                mean_oob_truth_ari=mean([r["mean_oob_truth_ari"] for r in rows]),
                mean_common_assignment_changes=mean(
                    [r["mean_common_assignment_changes"] for r in rows]
                ),
                mean_own_assignment_changes=mean([r["mean_own_assignment_changes"] for r in rows]),
                mean_same_own_k=mean([r["same_own_k"]["conditional_frequency"] for r in rows]),
                mean_same_common_k=mean(
                    [r["same_common_k"]["conditional_frequency"] for r in rows]
                ),
                mean_oob_pair_coverage=mean(
                    [r["own_stability"].get("oob_pair_coverage", {}).get("fraction") for r in rows]
                ),
                successful=sum(r["successful"] for r in rows),
                failed=sum(r["failed"] for r in rows),
                unattempted=sum(r["unattempted"] for r in rows),
                k4_gate_counts={
                    k: sum(r["k4_gates"][k] for r in rows) for k in rows[0]["k4_gates"]
                },
            )
        )
    lookup = {(r["cell"], r["repetition"], r["arm"]): r for r in datasets}
    paired = []
    for (cell, rep, arm), legacy in sorted(lookup.items()):
        if arm != "legacy":
            continue
        new = lookup[cell, rep, "scaled"]
        values = dict(cell=cell, repetition=rep)
        for name, key in (
            ("k4", "k4"),
            ("extra", "k_gt1"),
            ("same_own_k", "same_own_k"),
            ("same_common_k", "same_common_k"),
        ):
            a, b = legacy[key]["conditional_frequency"], new[key]["conditional_frequency"]
            values[name + "_difference"] = b - a if a is not None and b is not None else None
        for key in (
            "mean_truth_ari",
            "mean_oob_truth_ari",
            "mean_common_assignment_changes",
            "mean_own_assignment_changes",
        ):
            a, b = legacy[key], new[key]
            values[key + "_difference"] = b - a if a is not None and b is not None else None
        paired.append(values)
    paired_cells = []
    for cell in sorted({r["cell"] for r in paired}):
        rows = [r for r in paired if r["cell"] == cell]
        metrics = {}
        for key in rows[0]:
            if not key.endswith("_difference"):
                continue
            values = [r[key] for r in rows if r[key] is not None]
            metrics[key] = dict(
                datasets=len(values),
                mean=mean(values),
                range=[min(values), max(values)] if values else None,
            )
        paired_cells.append(dict(cell=cell, datasets=len(rows), metrics=metrics))
    result = dict(
        datasets=datasets,
        cells=cells,
        paired=paired,
        paired_cells=paired_cells,
        interpretation=(
            "Five independent datasets/cell; draws and folds are repeated measurements. "
            "Effects/ranges are descriptive. Event intervals describe unadjusted conditional "
            "Monte Carlo "
            "precision only. Common-reference and own-point stability answer different questions; "
            "stable K=1 is not recovery of separate groups. No general performance, type-I error "
            "or species-boundary guarantee."
        ),
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, rows in (("cells", cells), ("paired", paired), ("paired_cells", paired_cells)):
        with (output / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return dict(
        status="complete",
        dataset_summaries=len(datasets),
        cell_summaries=len(cells),
        paired=len(paired),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(report(args.study, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
