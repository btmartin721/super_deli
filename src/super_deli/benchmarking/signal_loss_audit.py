"""Independent retrospective audits of fixed-score PCA and every retained candidate."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.signal_loss_fit import audit_candidates
from super_deli.benchmarking.signal_loss_run import point_seed
from super_deli.core.partition_comparison import compare_partitions
from super_deli.io.config import read_json


def audit_case(directory):
    directory = Path(directory)
    result = read_json(directory / "summary.json")
    checked = 0
    if not (directory / "context.npz").exists():
        if result["status"] != "failed":
            raise ValueError("Missing nonfailed diagnostic context")
        return dict(passed=True, withheld=True, candidates=0)
    with np.load(directory / "context.npz", allow_pickle=False) as a:
        labels = a["labels"]
        fused = a["fused"]
        n = len(labels)
    names = ("snp", "traits", "habitat", "outline")
    onehot = np.eye(4)[labels]
    grouping = onehot @ np.diag(1 / onehot.sum(0)) @ onehot.T - np.ones((n, n)) / n
    rows = {r["name"]: r for r in result["variants"]}
    for root in sorted(directory.glob("native-candidates-*")):
        checked += audit_candidates(root)
    for root in sorted((directory / "diagnostics").iterdir()):
        if not (root / "summary.json").exists():
            continue
        s = read_json(root / "summary.json")
        index = read_json(root / "candidates/index.json")
        checked += audit_candidates(root / "candidates")
        if s["status"] != "complete":
            continue
        with np.load(root / "point.npz", allow_pickle=False) as a:
            predicted = a["labels"]
        minimum = s["minimum"]
        eligible = []
        for r in index:
            if r["fit_status"] != "complete":
                continue
            if r["n_init"] != 10 or r["seed"] != int(
                np.random.SeedSequence([point_seed(), r["k"]]).generate_state(1)[0]
            ):
                raise ValueError("Unexpected optimizer seed/start count")
            with np.load(root / "candidates" / r["arrays"], allow_pickle=False) as a:
                hard = np.bincount(a["labels"], minlength=r["k"])
            record = next(c for c in s["candidates"] if c["k"] == r["k"])
            if record["hard_cluster_sizes"] != hard.tolist():
                raise ValueError("Candidate membership count mismatch")
            expected = "eligible" if min(hard) >= minimum else "small_component"
            if record["status"] != expected:
                raise ValueError("Candidate eligibility mismatch")
            np.testing.assert_allclose(r["bic"], record["bic"], rtol=1e-8, atol=1e-7)
            if expected == "eligible":
                eligible.append((r["bic"], r["k"]))
        if min(eligible)[1] != s["selected_k"]:
            raise ValueError("Incorrect within-space BIC selection")
        row = rows[root.name]
        if row.get("correct_recovery") is not None:
            target = labels // 2 if result["law"] == "genomic_only" else labels
            ari = compare_partitions(target.tolist(), predicted.tolist(), [True] * n, [True] * n)[
                "adjusted_rand"
            ]
            np.testing.assert_allclose(ari, row["ari"], atol=1e-12)
            if row["correct_recovery"] != (len(set(predicted)) == len(set(target)) and ari >= 0.8):
                raise ValueError("Incorrect recovery event")
        if (root / "representation.npz").exists():
            with np.load(root / "representation.npz", allow_pickle=False) as a:
                x, pc, components = a["fused"], a["scores"], a["components"]
                chosen = a["selected_views"].tolist()
            expected = np.concatenate(
                [
                    fused[:, 2 * names.index(name) : 2 * names.index(name) + 2]
                    * np.sqrt(4 / len(chosen))
                    for name in chosen
                ],
                axis=1,
            )
            np.testing.assert_allclose(x, expected, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose((x - x.mean(0)) @ components.T, pc, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(
                components @ components.T, np.eye(len(components)), atol=1e-9
            )
            eigenvalues = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
            np.testing.assert_allclose(
                np.var(pc, axis=0, ddof=1), eigenvalues[: pc.shape[1]], rtol=1e-8, atol=1e-9
            )
            if "outline" in chosen:
                z = np.zeros_like(x)
                j = 2 * chosen.index("outline")
                z[:, j : j + 2] = x[:, j : j + 2]
                between = grouping @ z
                fraction = np.sum((between @ components.T) ** 2) / np.sum(between**2)
                np.testing.assert_allclose(fraction, row["shape_energy_retained"], atol=1e-9)
    return dict(
        passed=True, withheld=False, candidates=checked, searches=result["planned_searches"]
    )


def audit_phase(root, summary):
    root = Path(root)
    records = [audit_case(root / r["path"]) for r in summary["records"]]
    return dict(
        passed=True,
        observations=len(records),
        withheld=sum(r["withheld"] for r in records),
        candidate_checks=sum(r["candidates"] for r in records),
    )
