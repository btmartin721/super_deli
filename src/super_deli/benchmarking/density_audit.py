"""Read-only numerical/denominator audits independent of fit and support helpers.

HDBSCAN itself is not independently reimplemented. Its labels are checked against
invariants and known-case tests; distance audits verify the declared input space.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json, write_json


def _audit_fit(x, arrays, prefix, record, family):
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp

    a = {k[len(prefix) + 1 :]: v for k, v in arrays.items() if k.startswith(prefix + "_")}
    kept = a["kept_indices"]
    mean, scale = x[:, kept].mean(axis=0), x[:, kept].std(axis=0)
    np.testing.assert_allclose(a["means"], mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(a["scales"], scale, rtol=1e-11, atol=1e-12)
    z = (x[:, kept] - mean) / scale
    components, scores = a["components"], a["scores"]
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-11)
    np.testing.assert_allclose(scores, (z - a["pca_mean"]) @ components.T, atol=1e-11)
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1]
    np.testing.assert_allclose(a["explained_variance"], eig[:2], atol=1e-11)
    np.testing.assert_array_equal(a["labels"], record["labels"])
    distance = cdist(scores, scores)
    # Two PCs retain all two-dimensional standardized geometry in this study.
    np.testing.assert_allclose(distance, cdist(z, z), atol=1e-11)
    if family == "hdbscan":
        assert np.all(a["labels"] >= -1)
        assert np.all(np.isfinite(a["strength"]))
        assert np.all((a["strength"] >= 0) & (a["strength"] <= 1))
        assert np.all(a["strength"][a["labels"] < 0] == 0)
        assert all(np.sum(a["labels"] == k) >= 5 for k in set(a["labels"]) - {-1})
        # min_samples includes self: fifth distance is the sorted index four.
        return {
            "fifth_neighbor_min": float(np.sort(distance, axis=1)[:, 4].min()),
            "fifth_neighbor_max": float(np.sort(distance, axis=1)[:, 4].max()),
        }
    weights, means, covariance = a["weights"], a["mixture_means"], a["covariances"]
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1)
    density = []
    for p, mu, cov in zip(weights, means, covariance, strict=True):
        assert np.linalg.eigvalsh(cov).min() > 0
        residual = scores - mu
        density.append(
            np.log(p)
            - 0.5
            * (
                2 * np.log(2 * np.pi)
                + np.linalg.slogdet(cov)[1]
                + np.einsum("ni,ij,nj->n", residual, np.linalg.inv(cov), residual)
            )
        )
    log_joint = np.array(density).T
    log_density = logsumexp(log_joint, axis=1)
    membership = np.exp(log_joint - log_density[:, None])
    np.testing.assert_array_equal(membership.argmax(axis=1), a["labels"])
    np.testing.assert_allclose(membership.max(axis=1), a["strength"], atol=1e-10)
    k = len(weights)
    parameters = k - 1 + 2 * k + 3 * k
    bic = parameters * np.log(len(x)) - 2 * log_density.sum()
    eligible = [r for r in record["candidates"] if r["status"] == "eligible"]
    winner = min(eligible, key=lambda r: (r["bic"], r["k"]))
    assert winner["k"] == k
    np.testing.assert_allclose(winner["bic"], bic, atol=1e-7)
    return {"selected_log_likelihood": float(log_density.sum()), "selected_bic": float(bic)}


def _audit_support(n, records, arrays):
    import numpy as np

    attempted = np.zeros((n, n), dtype=int)
    successful, coassigned = attempted.copy(), attempted.copy()
    included = np.zeros(n, dtype=int)
    done, noise = included.copy(), included.copy()
    for r in records:
        ids = r["training_indices"]
        assert len(set(ids)) == len(ids)
        for i in ids:
            included[i] += 1
            for j in ids:
                attempted[i, j] += 1
        if r["status"] != "complete":
            continue
        labels = dict(zip(ids, r["labels"], strict=True))
        for i in ids:
            done[i] += 1
            noise[i] += labels[i] == -1
            for j in ids:
                successful[i, j] += 1
                coassigned[i, j] += labels[i] >= 0 and labels[i] == labels[j]
    for key, expected in [
        ("unit_included", included),
        ("unit_successful", done),
        ("unit_noise", noise),
        ("pair_included", attempted),
        ("pair_successful", successful),
        ("pair_coassigned", coassigned),
    ]:
        np.testing.assert_array_equal(arrays[key], expected)
    for key, a, b in [
        ("noise_frequency", noise, done),
        ("coassignment", coassigned, successful),
        ("noise_lower", noise, included),
        ("noise_upper", noise + included - done, included),
        ("coassignment_lower", coassigned, attempted),
        ("coassignment_upper", coassigned + attempted - successful, attempted),
    ]:
        expected = np.full(a.shape, np.nan)
        np.divide(a, b, out=expected, where=b > 0)
        np.testing.assert_allclose(arrays[key], expected, equal_nan=True)
    for key, a, b in [("noise_mc_se", noise, done), ("coassignment_mc_se", coassigned, successful)]:
        p = np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
        se = np.sqrt(np.divide(p * (1 - p), b, out=np.full(a.shape, np.nan), where=b > 0))
        np.testing.assert_allclose(arrays[key], se, equal_nan=True)


def audit_density_study(study_dir, *, output_dir):
    import numpy as np
    from scipy.stats import beta

    from super_deli.benchmarking.density_study import (
        DensityStudyConfig,
        generate_raw,
        recovery_metrics,
    )

    if not __debug__:
        raise RuntimeError("Numerical evidence audit requires Python assertions enabled")
    root, out = Path(study_dir).resolve(), Path(output_dir).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Audit must be outside immutable study evidence")
    frozen = read_json(root / "frozen-plan.json")
    config = DensityStudyConfig.from_dict(frozen["configuration"])
    assert frozen["plan"] == config.plan(frozen["phase"])
    assert frozen["density_model"]["model_family"] == "tabular_pca_hdbscan"
    assert frozen["density_model"]["min_samples"] == 5
    assert frozen["gmm_model"]["n_init"] == 10 and frozen["gmm_model"]["max_k"] == 4
    rows, numerical, inventory = [], [], {}
    for plan in frozen["plan"]:
        path = root / plan["key"]
        assert read_json(path / "plan.json") == plan
        for relative, digest in read_json(path / "checksums.json").items():
            assert fingerprint_file(path / relative).sha256 == digest
            inventory[f"{plan['key']}/{relative}"] = digest
        with (path / "raw/traits.tsv").open() as f:
            raw_rows = list(csv.DictReader(f, delimiter="\t"))
        x = np.array([[float(r[f"trait_{j}"]) for j in (1, 2)] for r in raw_rows])
        generated, truth = generate_raw(plan["case"], plan["n"], plan["data_seed"])
        np.testing.assert_array_equal(x, generated)
        if truth is not None:
            with np.load(path / "truth.npz", allow_pickle=False) as a:
                np.testing.assert_array_equal(a["labels"], truth)
        assert len({r["observation_id"] for r in raw_rows}) == len(x)
        hconfig = read_json(path / "density-config.json")
        assert hconfig["n_subsamples"] == plan["draws"]
        seed = int(np.random.SeedSequence(plan["subsampling_seed"]).generate_state(1)[0])
        assert hconfig["random_seed"] == seed
        expected_draws = [
            np.sort(
                np.random.default_rng(s).choice(len(x), int(np.floor(0.8 * len(x))), replace=False)
            )
            for s in np.random.SeedSequence(seed).spawn(plan["draws"])
        ]
        row = read_json(path / "summary.json")
        for family in ("hdbscan", "gmm"):
            records = read_json(path / f"{family}.json")
            with np.load(path / f"{family}.npz", allow_pickle=False) as a:
                arrays = dict(a)
            if records["point"]["status"] == "complete":
                numerical.append(_audit_fit(x, arrays, "point", records["point"], family))
                assert row["families"][family]["metrics"] == recovery_metrics(
                    arrays["point_labels"], truth
                )
            assert len(records["draws"]) in (0, plan["draws"])
            for i, r in enumerate(records["draws"]):
                np.testing.assert_array_equal(r["training_indices"], expected_draws[i])
                if r["status"] == "complete":
                    numerical.append(
                        _audit_fit(x[expected_draws[i]], arrays, f"draw_{i}", r, family)
                    )
            if records["draws"]:
                _audit_support(len(x), records["draws"], arrays)
            frow = row["families"][family]
            assert frow["attempted"] == 1 + len(records["draws"])
            assert frow["successful"] == int(records["point"]["status"] == "complete") + sum(
                r["status"] == "complete" for r in records["draws"]
            )
            assert frow["failed"] == frow["attempted"] - frow["successful"]
        rows.append(row)
    rates, secondary = [], []
    for case in dict.fromkeys(r["case"] for r in rows):
        for n in config.sample_sizes:
            cell = [r for r in rows if r["case"] == case and r["n"] == n]
            for family in ("hdbscan", "gmm"):
                frows = [r["families"][family] for r in cell]
                ok = [r for r in frows if r["status"] == "complete"]
                passed = sum(bool(r["metrics"].get("correct_k")) for r in ok)
                failed = len(cell) - len(ok)
                if case != "cline":
                    # Monotone bounds accommodate unknown outcomes after failed fits.
                    alpha = 0.05 / (5 * len(config.sample_sizes) * 2)
                    lower = (
                        float(beta.ppf(alpha / 2, passed, len(cell) - passed + 1))
                        if passed
                        else 0.0
                    )
                    upper_count = passed + failed
                    upper = (
                        float(beta.ppf(1 - alpha / 2, upper_count + 1, len(cell) - upper_count))
                        if upper_count < len(cell)
                        else 1.0
                    )
                    rates.append(
                        {
                            "case": case,
                            "n": n,
                            "family": family,
                            "datasets": len(cell),
                            "correct_k": passed,
                            "failed": failed,
                            "rate_lower": passed / len(cell),
                            "rate_upper": upper_count / len(cell),
                            "simultaneous_95_lower": lower,
                            "simultaneous_95_upper": upper,
                        }
                    )
                metrics = [r["metrics"] for r in ok]
                secondary.append(
                    {
                        "case": case,
                        "n": n,
                        "family": family,
                        "selected_k_counts": dict(Counter(str(r["selected_k"]) for r in metrics)),
                        **{
                            f"mean_{key}": float(
                                np.mean([r[key] for r in metrics if r.get(key) is not None])
                            )
                            if any(r.get(key) is not None for r in metrics)
                            else None
                            for key in (
                                "ari",
                                "assigned_fraction",
                                "core_assigned_fraction",
                                "background_recall",
                                "noise_precision",
                                "split_generating_groups",
                                "merged_predicted_groups",
                            )
                        },
                    }
                )
    summary = {
        "status": "passed",
        "phase": frozen["phase"],
        "observations": len(rows),
        "numerically_audited_fits": len(numerical),
        "configuration_sha256": configuration_hash(config.to_dict()),
        "primary_rates": rates,
        "descriptive_metrics": secondary,
        "independence": "datasets are independent; subsamples are repeated measurements",
        "algorithm_audit_limit": "HDBSCAN was not independently reimplemented or refitted",
        "assessment": "Share with caveats: fixed synthetic laws; no species/significance guarantee",
    }
    assert len(rates) == 5 * len(config.sample_sizes) * 2
    out.mkdir(exist_ok=False)
    write_json(summary, out / "summary.json")
    write_json(inventory, out / "files.json")
    write_json(numerical, out / "numerical-audit.json")
    return summary
