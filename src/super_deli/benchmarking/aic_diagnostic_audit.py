"""No-refit independent law, projection, density and candidate-selection audits."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from super_deli.benchmarking.aic_diagnostic_simulation import heldout_layout
from super_deli.benchmarking.grouped_boundary_reference import parameters
from super_deli.benchmarking.grouped_likelihood_analysis import audit_candidates
from super_deli.benchmarking.grouped_study import grouped_layout
from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.recovery_inputs import DirectoryArchive
from super_deli.io.config import read_json
from super_deli.models.aic_diagnostic_config import ARMS, AICDiagnosticConfig


def independent_raw(case, n, seed):
    import numpy as np

    mean, covariance, within, _, _ = parameters(case)
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, n)
    if case == "zero_conditional_gaussian":
        ecology = rng.standard_normal((n, 4))
        rng.standard_normal((n, 4))
        regression = np.linalg.solve(covariance[4:, 4:], covariance[4:, :4]).T
        latent = ecology @ regression.T
    else:
        both = mean + rng.standard_normal((n, 8)) @ np.linalg.cholesky(covariance).T
        latent, ecology = both[:, :4], both[:, 4:]
    residual = rng.standard_normal((int(counts.sum()), 4)) @ np.linalg.cholesky(within).T
    target = np.repeat(np.arange(n), counts)
    return (
        {"morphological": latent[target] + residual, "ecological": ecology},
        grouped_layout(counts, "locality"),
        latent,
        residual,
    )


def independent_projection(run, values, layout):
    """Average raw rows first, then apply the saved linear training map."""
    import numpy as np

    blocks = []
    for j, kind in enumerate(("morphological", "ecological")):
        raw = values[kind]
        if kind == "morphological":
            indices = np.repeat(np.arange(len(layout["counts"])), layout["counts"])
            raw = np.array([raw[indices == i].mean(0) for i in range(len(layout["counts"]))])
        path = Path(run) / f"view_{j:03d}" / "parameters"
        with (
            np.load(path / "pca.npz", allow_pickle=False) as p,
            np.load(path / "profiles.npz", allow_pickle=False) as s,
        ):
            # An affine matrix/offset construction differs from the writer's row-level path.
            matrix = p["components"].T / p["feature_scales"][:, None]
            offset = p["feature_means"] @ matrix + p["mean"] @ p["components"].T
            projected = raw[:, p["kept_indices"]] @ matrix - offset
            blocks.append((projected - s["mean"]) / s["scale"] * np.sqrt(s["normalized_weight"]))
    with np.load(Path(run) / "fusion.npz", allow_pickle=False) as p:
        return np.column_stack(blocks) @ p["components"].T - p["mean"] @ p["components"].T


def check_pool(raw_pool, candidates, *, counts, config, heldout=None):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    with np.load(io.BytesIO(raw_pool), allow_pickle=False) as arrays:
        scores = arrays["scores"]
        n, d = scores.shape
        eligible = []
        density = {}
        for c in candidates:
            k = c["k"]
            if c["status"] == "insufficient_samples":
                if n >= k * config.min_cluster_size:
                    raise ValueError("False candidate infeasibility")
                continue
            w, means, cov, prob, labels = (
                arrays[f"k{k}_{key}"]
                for key in ("weights", "means", "covariances", "membership", "labels")
            )
            np.testing.assert_allclose(w.sum(), 1, atol=1e-12)
            training = np.column_stack(
                [
                    np.log(a) + multivariate_normal.logpdf(scores, mean=m, cov=v)
                    for a, m, v in zip(w, means, cov, strict=True)
                ]
            )
            ll = logsumexp(training, axis=1)
            np.testing.assert_allclose(prob, np.exp(training - ll[:, None]), atol=1e-10)
            np.testing.assert_array_equal(labels, prob.argmax(1))
            sizes = np.bincount(labels, minlength=k)
            if (
                sizes.tolist() != c["hard_cluster_sizes"]
                or sizes.tolist() != c["distinct_unit_sizes"]
            ):
                raise ValueError("Candidate hard/distinct sizes differ")
            admissible = c["converged"] and sizes.min() >= config.min_cluster_size
            if (c["status"] == "eligible") != admissible:
                raise ValueError("Candidate eligibility differs")
            if admissible:
                eligible.append(c)
            tied = getattr(config, "covariance_type", "full") == "tied"
            if tied:
                np.testing.assert_array_equal(cov, np.repeat(cov[:1], k, axis=0))
            q = k - 1 + k * d + (1 if tied else k) * d * (d + 1) // 2
            np.testing.assert_allclose(c["log_likelihood"], ll.sum(), atol=1e-7, rtol=1e-10)
            np.testing.assert_allclose(
                [c["aic"], c["bic"]],
                [-2 * ll.sum() + 2 * q, -2 * ll.sum() + q * np.log(n)],
                atol=1e-7,
            )
            if heldout is not None:
                test = heldout["scores"]
                from super_deli.benchmarking.predictive_density_audit import log_density

                z = log_density(test, w, means, cov)
                np.testing.assert_allclose(heldout[f"k{k}"], z, atol=1e-8, rtol=1e-10)
                density[str(k)] = z
        return {
            name + "_k": min(eligible, key=lambda c: (c[name], c["k"]))["k"]
            for name in ("aic", "bic")
        }, density


def _audit(task):
    import numpy as np

    source, declaration, record, raw_config, scratch = task
    config = AICDiagnosticConfig.from_dict(raw_config)
    for key, value in declaration.items():
        if record[key] != value:
            raise ValueError("Ledger declaration differs")
    expected = f"{declaration['cell']}/dataset_{declaration['repetition']:04d}.tar.gz"
    if record["archive"] != expected or record["manifest"] != expected + ".manifest.json":
        raise ValueError("Archive identity differs")
    with materialize(
        Path(source) / expected, record, scratch=scratch, min_free_bytes=config.min_free_bytes
    ) as base:
        if read_json(base / "declaration.json") != declaration:
            raise ValueError("Archived declaration differs")
        original = read_json(base / "record.json")
        if any(record.get(k) != v for k, v in original.items()):
            raise ValueError("Outer and archived records differ")
        direct = declaration["case"] == "gaussian_scores"
        rows = []
        if direct:
            with np.load(base / "input.npz", allow_pickle=False) as p:
                np.testing.assert_array_equal(
                    p["training"],
                    np.random.default_rng(declaration["data_seed"]).standard_normal(
                        (declaration["n"], 4)
                    )
                    / 2,
                )
                tests = (
                    np.random.default_rng(declaration["test_seed"]).standard_normal(
                        (config.n_test, 4)
                    )
                    / 2
                )
                np.testing.assert_array_equal(p["heldout"], tests)
            expected_arms = ("scores",)
        else:
            values, layout, latent, residual = independent_raw(
                declaration["case"], declaration["n"], declaration["data_seed"]
            )
            tests, test_layout, test_latent, test_residual = independent_raw(
                declaration["case"], config.n_test, declaration["test_seed"]
            )
            counts = np.asarray(layout["counts"])
            r = float(np.mean(1 / counts))
            if read_json(base / "layout.json") != json.loads(json.dumps(layout)):
                raise ValueError("Training identity/count layout differs")
            saved_test_layout = read_json(base / "heldout_layout.json")
            if set(saved_test_layout["unit_ids"]) & set(layout["unit_ids"]):
                raise ValueError("Held-out identities overlap training")
            if saved_test_layout != json.loads(json.dumps(heldout_layout(test_layout))):
                raise ValueError("Held-out identities/counts differ")
            if read_json(base / "intervention.json")["reciprocal_count_mean"] != r:
                raise ValueError("Homogeneous intervention variance differs")
            for file, expected_values, mu, err in (
                ("training.npz", values, latent, residual),
                ("heldout_input.npz", tests, test_latent, test_residual),
            ):
                with np.load(base / file, allow_pickle=False) as p:
                    for name in expected_values:
                        np.testing.assert_allclose(
                            p[name], expected_values[name], atol=3e-14, rtol=3e-14
                        )
                        expected_values[name] = p[name].copy()
                    np.testing.assert_allclose(p["latent"], mu, atol=3e-14, rtol=3e-14)
                    np.testing.assert_allclose(p["residual"], err, atol=3e-14, rtol=3e-14)
            expected_arms = ARMS
        if [p["arm"] for p in original["points"]] != list(expected_arms):
            raise ValueError("Missing or unplanned point arm")
        for row in original["points"]:
            arm = row["arm"]
            if row["status"] != "complete":
                if not row.get("error") or row["aic_k"] is not None:
                    raise ValueError("Malformed failure record")
                rows.append({**declaration, **row, "metrics": None})
                continue
            directory = base / arm
            model = config.model if direct else config.arm(arm)
            candidates = read_json(
                directory / ("candidates.json" if direct else "run/candidates.json")
            )
            run = directory if direct else directory / "run"
            if not direct:
                current = {k: v.copy() for k, v in values.items()}
                test_current = {k: v.copy() for k, v in tests.items()}
                if arm == "homogeneous_mean_noise":
                    for v, design, mu, err in (
                        (current, layout, latent, residual),
                        (test_current, test_layout, test_latent, test_residual),
                    ):
                        count = np.asarray(design["counts"])
                        index = np.repeat(np.arange(len(count)), count)
                        v["morphological"] = mu[index] + err * np.sqrt(count[index, None] * r)
                # CSV ingestion retains the writer's exact floating values; independent law
                # checks above allow only rounding differences in analytic reconstruction.
                actual = {}
                for j, kind in enumerate(("morphological", "ecological")):
                    actual[kind] = np.load(
                        run / f"view_{j:03d}/source_values.npy", allow_pickle=False
                    )
                    np.testing.assert_allclose(actual[kind], current[kind], atol=1e-13, rtol=1e-13)
                point = read_json(directory / "record.json")
                audit_candidates(
                    DirectoryArchive(base),
                    point,
                    SimpleNamespace(model=model),
                    actual,
                    json.loads(json.dumps(layout)),
                )
                projection = independent_projection(run, test_current, test_layout)
            else:
                projection = tests
                expected_seed = int(
                    np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0]
                )
                if row["point_seed"] != expected_seed:
                    raise ValueError("Direct score model seed differs")
                with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
                    with np.load(base / "input.npz", allow_pickle=False) as inputs:
                        np.testing.assert_array_equal(pool["scores"], inputs["training"])
            with np.load(directory / "heldout.npz", allow_pickle=False) as held:
                np.testing.assert_allclose(held["scores"], projection, atol=1e-11, rtol=1e-11)
                choice, density = check_pool(
                    (run / "gmm_candidates.npz").read_bytes(),
                    candidates,
                    counts=None if direct else counts,
                    config=model,
                    heldout=held,
                )
            metrics = read_json(directory / "metrics.json")
            if any(row[k] != v or metrics[k] != v for k, v in choice.items()):
                raise ValueError("Selection differs from independent candidate criteria")
            check_metrics(
                (run / "gmm_candidates.npz").read_bytes(),
                metrics,
                None if direct else counts,
                model.reg_covar,
            )
            for c in metrics["candidates"]:
                if str(c["k"]) in density:
                    gain = density[str(c["k"])] - density["1"]
                    np.testing.assert_allclose(
                        c["heldout_mean_logdensity_gain_over_one"], gain.mean(), atol=1e-10
                    )
                    np.testing.assert_allclose(
                        c["heldout_mcse_conditional_on_training"],
                        gain.std(ddof=1) / np.sqrt(len(gain)),
                        atol=1e-10,
                    )
            rows.append({**declaration, **row, "metrics": metrics})
        if (
            sum(r["status"] == "complete" for r in rows) != record["successful_fits"]
            or len(rows) != record["planned_fits"]
        ):
            raise ValueError("Independent fit accounting differs")
        return rows


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit(task)


def check_metrics(raw_pool, metrics, counts, regularizer):
    """Reconstruct descriptive mechanisms independently of the metrics writer."""
    import numpy as np

    with np.load(io.BytesIO(raw_pool), allow_pickle=False) as p:
        x = p["scores"]
        delta = x - p["k1_means"][0]
        distance = np.sum(delta * np.linalg.solve(p["k1_covariances"][0], delta.T).T, axis=1)
        np.testing.assert_allclose(
            metrics["one_component_mahalanobis_squared"], distance, atol=1e-9
        )
        if (
            metrics["extreme_unit_indices"]
            != np.argsort(-distance, kind="stable")[: min(10, len(x))].tolist()
        ):
            raise ValueError("Extreme observation identities differ")
        for c in metrics["candidates"]:
            if c["status"] == "insufficient_samples":
                continue
            k = c["k"]
            labels, prob = p[f"k{k}_labels"], p[f"k{k}_membership"]
            eigen = np.array([np.linalg.eigh(v)[0] for v in p[f"k{k}_covariances"]])
            np.testing.assert_allclose(c["minimum_eigenvalues"], eigen[:, 0])
            np.testing.assert_allclose(c["condition_numbers"], eigen[:, -1] / eigen[:, 0])
            np.testing.assert_allclose(
                c["minimum_eigenvalue_over_regularizer"], eigen.min() / regularizer
            )
            np.testing.assert_allclose(c["weights"], p[f"k{k}_weights"])
            entropy = -sum(v * np.log(v) for row in prob for v in row if v > 0) / len(x)
            np.testing.assert_allclose(c["mean_assignment_entropy"], entropy, atol=1e-12)
            for j, observed in enumerate(c["component_mean_mahalanobis_squared"]):
                selected = distance[labels == j]
                if len(selected):
                    np.testing.assert_allclose(observed, selected.mean(), atol=1e-9)
                elif observed is not None:
                    raise ValueError("Nonempty outlier summary for empty component")
            if counts is not None:
                table = np.array(
                    [
                        [np.count_nonzero((counts == m) & (labels == j)) for j in range(k)]
                        for m in range(1, 5)
                    ]
                )
                np.testing.assert_array_equal(c["count_by_component"], table)
                probability = table / len(x)
                independent = probability.sum(1)[:, None] * probability.sum(0)[None, :]
                nz = probability > 0
                mi = np.sum(probability[nz] * np.log(probability[nz] / independent[nz]))
                np.testing.assert_allclose(c["count_label_mutual_information"], mi, atol=1e-12)
