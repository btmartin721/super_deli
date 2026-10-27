"""Independent final-export/selected-parameter checks, without fitting imports."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(root, phase):
    import numpy as np
    from scipy.stats import beta
    from sklearn.metrics import adjusted_rand_score

    root = Path(root)
    count = complete = 0
    for folder in sorted((root / phase).glob("*/*")):
        if not folder.is_dir():
            continue
        result = json.loads((folder / "summary.json").read_text())
        count += 1
        if result["status"] != "complete":
            continue
        complete += 1
        with np.load(folder / "fit.npz", allow_pickle=False) as a:
            k = result["selected_k"]
            raw_labels = a[f"k{k}_membership"].argmax(axis=1)
            order = sorted(range(k), key=lambda c: int(np.flatnonzero(raw_labels == c)[0]))
            for name in ("weights", "means", "covariances"):
                np.testing.assert_array_equal(a[name], a[f"k{k}_{name}"][order])
            for candidate in result["candidates"]:
                if "active_bound_eigenvalues" in candidate:
                    eigen = np.linalg.eigvalsh(a[f"k{candidate['k']}_covariances"])
                    assert candidate["active_bound_eigenvalues"] == int(
                        (eigen <= 1e-4 * (1 + 1e-7)).sum()
                    )
            entropy = float(
                (-a["membership"] * np.log(np.maximum(a["membership"], np.finfo(float).tiny)))
                .sum(axis=1)
                .mean()
            )
            np.testing.assert_allclose(result["entropy"], entropy)
            if (folder / "comparison.json").exists():
                reference = folder.parent / f"{result['mask']}_reference"
                with np.load(reference / "fit.npz", allow_pickle=False) as b:
                    ids = np.asarray(result["source_indices"])
                    common, i, j = np.intersect1d(ids, b["kept_indices"], return_indices=True)
                    comparison = json.loads((folder / "comparison.json").read_text())
                    assert comparison["reference_common_n"] == len(common)
                    np.testing.assert_allclose(
                        comparison["reference_common_ari"],
                        adjusted_rand_score(a["labels"][i], b["labels"][j]),
                    )
    assert count == (144 if phase == "preflight" else 1224)
    preserved_file = root / "preflight-preserved-before-resume.json"
    preserved = json.loads(preserved_file.read_text()) if preserved_file.exists() else {}
    for name, digest in preserved.items():
        assert hashlib.sha256((root / "preflight" / name).read_bytes()).hexdigest() == digest
    result = dict(
        status="passed",
        phase=phase,
        searches=count,
        complete=complete,
        original_preflight_files_preserved=len(preserved),
        selected_parameters_verified=True,
    )
    if phase == "main" and (root / "aggregate.json").exists():
        aggregate = json.loads((root / "aggregate.json").read_text())
        assert len(aggregate["cells"]) == 120 and len(aggregate["observed_minus_mean"]) == 30
        summaries = [
            json.loads(p.read_text())
            for p in (root / "main").glob("*/*/summary.json")
            if not p.parent.name.startswith("replay_")
        ]
        for cell in aggregate["cells"]:
            rows = [
                r
                for r in summaries
                if r["task"]["case"] == cell["case"]
                and r["task"]["n"] == cell["n"]
                and r["mask"] == cell["mask"]
                and r["arm"] == cell["arm"]
            ]
            success = [r for r in rows if r["status"] == "complete"]
            assert len(rows) == 10 and cell["successful"] == len(success)
            for name, target in (
                ("correct_k", 4 if cell["case"] == "four_groups" else 1),
                ("k_gt_1", None),
            ):
                n = len(success)
                c = sum(
                    r["selected_k"] == target if target else r["selected_k"] > 1 for r in success
                )
                rate = cell["rates"][name]
                assert rate["count"] == c and rate["successful"] == n
                assert rate["failure_bounds"] == [c / 10, (c + 10 - n) / 10]
                expected = (
                    None
                    if n == 0
                    else [
                        0.0 if c == 0 else float(beta.ppf(0.025, c, n - c + 1)),
                        1.0 if c == n else float(beta.ppf(0.975, c + 1, n - c)),
                    ]
                )
                if expected is None:
                    assert rate["conditional_exact95"] is None
                else:
                    np.testing.assert_allclose(rate["conditional_exact95"], expected, atol=1e-11)
        result["aggregate_cells_verified"] = 120
    output = root / f"{phase}-report-audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=["preflight", "main"], required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir, args.phase), indent=2))
