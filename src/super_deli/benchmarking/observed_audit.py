"""Independent density/moment/accounting audit; imports no production EM/selector."""

from pathlib import Path

from super_deli.io.config import read_json, write_json


def reference(x, weights, means, covariances, moments=False):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    n, d = x.shape
    k = len(weights)
    logjoint = np.empty((n, k))
    mask_groups = np.unique(~np.isnan(x), axis=0)
    for mask in mask_groups:
        indices = np.flatnonzero(np.all((~np.isnan(x)) == mask, axis=1))
        o = np.flatnonzero(mask)
        if not len(o):
            raise AssertionError("Uninformative rows entered a likelihood")
        for c in range(k):
            logjoint[indices, c] = np.log(weights[c]) + multivariate_normal.logpdf(
                x[np.ix_(indices, o)], means[c, o], covariances[c][np.ix_(o, o)]
            )
    density = logsumexp(logjoint, axis=1)
    r = np.exp(logjoint - density[:, None])
    if not moments:
        return density, r
    first = np.zeros((k, d))
    second = np.zeros((k, d, d))
    mass = r.sum(axis=0)
    for mask in mask_groups:
        indices = np.flatnonzero(np.all((~np.isnan(x)) == mask, axis=1))
        o = np.flatnonzero(mask)
        m = np.flatnonzero(~mask)
        for c in range(k):
            filled = np.repeat(means[c][None, :], len(indices), axis=0)
            filled[:, o] = x[np.ix_(indices, o)]
            extra = np.zeros((d, d))
            if len(m):
                # Deliberately use a dense inverse in this independent low-d reference.
                inverse = np.linalg.inv(covariances[c][np.ix_(o, o)])
                cross = covariances[c][np.ix_(m, o)] @ inverse
                filled[:, m] += (filled[:, o] - means[c, o]) @ cross.T
                extra[np.ix_(m, m)] = (
                    covariances[c][np.ix_(m, m)] - cross @ covariances[c][np.ix_(o, m)]
                )
            probability = r[indices, c]
            first[c] += probability @ filled
            second[c] += (filled.T * probability) @ filled + probability.sum() * extra
    fitted_mean = first / mass[:, None]
    scatter = second / mass[:, None, None] - fitted_mean[:, :, None] * fitted_mean[:, None, :]
    return mass / n, fitted_mean, scatter


def reference_update(x, parameters, bound):
    import numpy as np

    weights, means, scatter = reference(x, *parameters, moments=True)
    covariance = []
    for s in scatter:
        eigen, vectors = np.linalg.eigh((s + s.T) / 2)
        covariance.append((vectors * np.maximum(eigen, bound)) @ vectors.T)
    return weights, means, np.asarray(covariance)


def change(a, b):
    import numpy as np

    return max(
        np.max(np.abs(x - y) / np.maximum(1.0, np.maximum(np.abs(x), np.abs(y))))
        for x, y in zip(a, b, strict=True)
    )


def source_arrays(task):
    """Reconstruct the frozen laws without importing the study generator."""
    import numpy as np

    ns = [
        0 if task["phase"] == "preflight" else 1,
        ["spherical", "correlated", "four_groups"].index(task["case"]),
        task["n"],
        task["replicate"],
    ]

    def sample(n, seed):
        rng = np.random.default_rng(seed)
        if task["case"] == "spherical":
            return rng.normal(size=(n, 4)), np.zeros(n, dtype=int)
        rho = 0.6 if task["case"] == "correlated" else 0.3
        covariance = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (4, 4))
        if task["case"] == "correlated":
            return rng.multivariate_normal(np.zeros(4), covariance, n), np.zeros(n, dtype=int)
        labels = rng.integers(0, 4, n)
        left = 2 * (labels // 2) - 1
        right = 2 * (labels % 2) - 1
        mean = np.array([3 * left, 3 * right, 1.5 * left, 1.5 * right]).T
        return mean + rng.multivariate_normal(np.zeros(4), covariance, n), labels

    def masks(x, seed):
        uniform = np.random.default_rng(seed + [0]).random(x.shape)
        mar = np.random.default_rng(seed + [1]).random(x.shape) < np.where(
            x[:, :1] <= 0, 0.15, 0.65
        )
        mar[:, 0] = False
        mnar = np.random.default_rng(seed + [2]).random(x.shape) < np.where(x <= 0, 0.1, 0.5)
        rng = np.random.default_rng(seed + [3])
        flag = rng.integers(2, size=len(x))
        structured = (np.arange(4)[None, :] // 2 == flag[:, None]) & (rng.random(x.shape) < 0.6)
        return dict(
            mcar10=uniform < 0.1, mcar30=uniform < 0.3, mar=mar, mnar=mnar, pattern=structured
        )

    x, truth = sample(task["n"], [80403] + ns)
    ev, ev_truth = sample(2048, [80429] + ns + [0])
    return dict(
        complete=x,
        truth=truth,
        evaluation=ev,
        evaluation_truth=ev_truth,
        **{f"mask_{k}": v for k, v in masks(x, [80409] + ns).items()},
        **{f"eval_mask_{k}": v for k, v in masks(ev, [80429] + ns + [1]).items()},
    )


def audit_fit(folder, values, indices, mean, scale, evaluation, mask, truth):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    summary = read_json(folder / "summary.json")
    keep = np.flatnonzero(~np.isnan(values).all(axis=1))
    raw = values[keep]
    config = summary["config"]
    assert config == dict(
        feature_names=[f"trait_{i}" for i in range(4)],
        max_k=4,
        min_cluster_size=5,
        n_init=10,
        max_iter=2000,
        tol=1e-7,
        fixed_point_tol=1e-5,
        covariance_bound=1e-4,
        min_coobserved=5,
        seed=80423,
        max_working_bytes=268435456,
        schema_version=1,
    )
    coverage = (~np.isnan(raw)).astype(int).T @ (~np.isnan(raw)).astype(int)
    guard = (
        len(raw) < max(5, raw.shape[1] + 2)
        or coverage.min() < 5
        or np.any(np.nanvar(raw, axis=0) <= 0)
    )
    if (folder / "rejected-input.npz").exists():
        assert guard and summary["status"] == "failed"
        with np.load(folder / "rejected-input.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["raw"], values)
        return dict(status="rejected_information_guard", starts=0)
    assert not guard
    with np.load(folder / "fit.npz", allow_pickle=False) as arrays:
        a = {key: arrays[key] for key in arrays.files}
    np.testing.assert_array_equal(a["raw"], raw)
    np.testing.assert_array_equal(a["observed"], ~np.isnan(raw))
    np.testing.assert_array_equal(a["coobserved"], coverage)
    np.testing.assert_allclose(a["normalization_means"], mean, rtol=0, atol=0)
    np.testing.assert_allclose(a["scales"], scale, rtol=0, atol=0)
    assert summary["source_indices"] == indices[keep].tolist()
    x = (raw - mean) / scale
    accepted = []
    numerical_failure = False
    starts_checked = 0
    for k, candidate in enumerate(summary["candidates"], 1):
        assert candidate["k"] == k
        if len(x) < k * 5:
            assert candidate["status"] == "insufficient_samples" and not candidate["starts"]
            continue
        assert len(candidate["starts"]) == 10
        converged = []
        for j, start in enumerate(candidate["starts"]):
            assert start["seed"] == int(np.random.SeedSequence([80423, k, j]).generate_state(1)[0])
            prefix = f"k{k}_s{j}"
            if f"{prefix}_trace" not in a:
                assert start["status"] == "failed" and start["error"]
                continue
            starts_checked += 1
            trace = a[f"{prefix}_trace"]
            assert np.isfinite(trace).all() and len(trace) == start["iterations"] + 1
            assert np.all(
                np.diff(trace)
                >= -np.maximum(
                    1e-8, 128 * np.finfo(float).eps * np.maximum(1.0, np.abs(trace[:-1]))
                )
            )
            params = {
                s: tuple(a[f"{prefix}_{s}_{key}"] for key in ("weights", "means", "covariances"))
                for s in ("initial", "previous", "final")
            }
            for stage, p in params.items():
                assert np.all(np.isfinite(p[1])) and np.all(np.isfinite(p[2]))
                assert np.linalg.eigvalsh(p[2]).min() >= 1e-4 * (1 - 1e-7)
                if stage in ("initial", "final"):
                    expected = reference(x, *p)[0].sum()
                    np.testing.assert_allclose(
                        expected, trace[0 if stage == "initial" else -1], atol=3e-6, rtol=1e-10
                    )
            if start["status"] == "converged":
                independent_update = reference_update(x, params["previous"], 1e-4)
                for observed, expected in zip(params["final"], independent_update, strict=True):
                    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=2e-7)
                residual = change(params["final"], reference_update(x, params["final"], 1e-4))
                assert residual <= 1e-5 + 2e-8
                assert abs(trace[-1] - trace[-2]) / len(x) <= 1e-7 + 1e-10
                converged.append((trace[-1], -j, params["final"]))
        if not converged:
            numerical_failure = True
            assert candidate["status"] == "numerical_failure"
            continue
        likelihood, neg, p = max(converged, key=lambda v: (v[0], v[1]))
        assert candidate["best_start"] == -neg
        for key, value in zip(("weights", "means", "covariances"), p, strict=True):
            np.testing.assert_array_equal(a[f"k{k}_{key}"], value)
        density, membership = reference(x, *p)
        np.testing.assert_allclose(a[f"k{k}_membership"], membership, atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(a[f"k{k}_density_scaled"], density, atol=2e-7)
        sizes = np.bincount(membership.argmax(axis=1), minlength=k)
        status = "eligible" if sizes.min() >= 5 else "small_component"
        parameter_count = k * 4 + k * 10 + k - 1
        bic = -2 * likelihood + parameter_count * np.log(len(x))
        assert candidate["status"] == status and candidate["parameter_count"] == parameter_count
        np.testing.assert_allclose(candidate["bic"], bic, atol=2e-6)
        assert candidate["hard_cluster_sizes"] == sizes.tolist()
        np.testing.assert_allclose(candidate["eigenvalues"], np.linalg.eigvalsh(p[2]))
        if status == "eligible":
            accepted.append((bic, k))
    selected = None if numerical_failure or not accepted else min(accepted)[1]
    assert summary["selected_k"] == selected
    assert summary["status"] == ("complete" if selected is not None else "failed")
    if selected is not None:
        p = tuple(a[key] for key in ("weights", "means", "covariances"))
        raw_labels = a[f"k{selected}_membership"].argmax(axis=1)
        order = sorted(range(selected), key=lambda k: int(np.flatnonzero(raw_labels == k)[0]))
        for key, value in zip(("weights", "means", "covariances"), p, strict=True):
            np.testing.assert_array_equal(value, a[f"k{selected}_{key}"][order])
        density, membership = reference(x, *p)
        np.testing.assert_allclose(a["membership"], membership, atol=2e-9, rtol=2e-8)
        np.testing.assert_array_equal(a["labels"], membership.argmax(axis=1))
        np.testing.assert_allclose(
            summary["ari"], adjusted_rand_score(truth[indices[keep]], a["labels"])
        )
        evaluation_masked = np.where(mask, np.nan, evaluation)
        usable = ~mask.all(axis=1)
        for filename, key, data in (
            ("evaluation.npz", "density", evaluation_masked[usable]),
            ("evaluation.npz", "complete_density", evaluation),
        ):
            scaled = (data - mean) / scale
            expected, prob = reference(scaled, *p)
            expected -= np.where(np.isnan(data), 0.0, np.log(scale)).sum(axis=1)
            with np.load(folder / filename, allow_pickle=False) as ev:
                actual = ev[key][usable] if key == "density" else ev[key]
                np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=1e-9)
                actual_prob = (
                    ev["membership"][usable] if key == "density" else ev["complete_membership"]
                )
                np.testing.assert_allclose(actual_prob, prob, atol=2e-9, rtol=2e-8)
                if key == "density":
                    assert np.isnan(ev[key][~usable]).all()
            np.testing.assert_allclose(
                summary["mean_log_density" if key == "density" else "complete_mean_log_density"],
                expected.mean(),
            )
        assert summary["evaluation_coverage"] == int(usable.sum())
    return dict(status=summary["status"], starts=starts_checked)


def audit_observation(directory):
    import numpy as np

    task = read_json(directory / "task.json")
    expected = source_arrays(task)
    with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(arrays[key], value)
    x, truth, ev = expected["complete"], expected["truth"], expected["evaluation"]
    searches = starts = failures = 0
    for mask_name in ("mcar10", "mcar30", "mar", "mnar", "pattern"):
        mask = expected[f"mask_{mask_name}"]
        masked = np.where(mask, np.nan, x)
        count = (~mask).sum(axis=0)
        mean = np.nansum(masked, axis=0) / count
        scale = np.sqrt(np.nansum((masked - mean) ** 2, axis=0) / count)
        for arm in ("observed", "complete_case", "mean", "reference"):
            indices = np.arange(len(x))
            if arm == "observed":
                values = masked
            elif arm == "reference":
                values = x
            else:
                selected = ~mask.any(axis=1) if arm == "complete_case" else ~mask.all(axis=1)
                indices = indices[selected]
                values = masked[selected]
                if arm == "mean":
                    values = np.where(np.isnan(values), mean, values)
            folder = directory / f"{mask_name}_{arm}"
            result = audit_fit(
                folder, values, indices, mean, scale, ev, expected[f"eval_mask_{mask_name}"], truth
            )
            searches += 1
            starts += result["starts"]
            failures += int(result["status"] != "complete")
            if mask_name == "mcar30" and task["replay"]:
                replay = directory / f"replay_{arm}"
                repeated = audit_fit(
                    replay,
                    values,
                    indices,
                    mean,
                    scale,
                    ev,
                    expected[f"eval_mask_{mask_name}"],
                    truth,
                )
                original = read_json(folder / "summary.json")
                other = read_json(replay / "summary.json")
                original.pop("reference_common_n", None)
                original.pop("reference_common_ari", None)
                assert original == other
                for name in ("fit.npz", "rejected-input.npz", "evaluation.npz"):
                    if not (folder / name).exists():
                        assert not (replay / name).exists()
                        continue
                    with (
                        np.load(folder / name, allow_pickle=False) as a,
                        np.load(replay / name, allow_pickle=False) as b,
                    ):
                        assert a.files == b.files
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                searches += 1
                starts += repeated["starts"]
    return dict(key=task["key"], searches=searches, starts=starts, point_failures=failures)


def audit_phase(root, phase, workers=4):
    from concurrent.futures import ProcessPoolExecutor

    root = Path(root)
    assert phase in ("preflight", "main") and 1 <= workers <= 4
    folders = sorted(p for p in (root / phase).iterdir() if p.is_dir())
    assert len(folders) == (6 if phase == "preflight" else 60)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(audit_observation, folders))
    expected = 144 if phase == "preflight" else 1224
    assert sum(r["searches"] for r in results) == expected
    result = dict(
        status="passed",
        phase=phase,
        searches=expected,
        starts=sum(r["starts"] for r in results),
        point_failures=sum(r["point_failures"] for r in results),
        observations=results,
        production_EM_imported=False,
    )
    write_json(result, root / f"{phase}-audit.json")
    return result


def aggregate(root):
    import numpy as np
    from scipy.stats import binomtest

    root = Path(root)
    assert read_json(root / "main-audit.json")["status"] == "passed"
    rows = [
        read_json(p)
        for p in sorted((root / "main").glob("*/*/summary.json"))
        if not p.parent.name.startswith("replay_")
    ]
    paths = [
        p
        for p in sorted((root / "main").glob("*/*/summary.json"))
        if not p.parent.name.startswith("replay_")
    ]
    for row, path in zip(rows, paths, strict=True):
        if (path.parent / "comparison.json").exists():
            row.update(read_json(path.parent / "comparison.json"))
    assert len(rows) == 1200

    def interval(count, n, confidence=0.95):
        ci = binomtest(count, n).proportion_ci(confidence_level=confidence, method="exact")
        return [float(ci.low), float(ci.high)]

    cells = []
    paired = []
    for case in ("spherical", "correlated", "four_groups"):
        for n in (64, 256):
            for mask in ("mcar10", "mcar30", "mar", "mnar", "pattern"):
                for arm in ("observed", "complete_case", "mean", "reference"):
                    group = [
                        r
                        for r in rows
                        if r["task"]["case"] == case
                        and r["task"]["n"] == n
                        and r["mask"] == mask
                        and r["arm"] == arm
                    ]
                    assert len(group) == 10
                    success = [r for r in group if r["status"] == "complete"]
                    counts = {
                        str(k): sum(r["selected_k"] == k for r in success) for k in range(1, 5)
                    }
                    correct = sum(
                        r["selected_k"] == (4 if case == "four_groups" else 1) for r in success
                    )
                    over = sum(r["selected_k"] > 1 for r in success)
                    rates = {
                        name: dict(
                            count=c,
                            successful=len(success),
                            requested=10,
                            conditional_exact95=interval(c, len(success)) if success else None,
                            failure_bounds=[c / 10, (c + 10 - len(success)) / 10],
                        )
                        for name, c in (("correct_k", correct), ("k_gt_1", over))
                    }
                    cells.append(
                        dict(
                            case=case,
                            n=n,
                            mask=mask,
                            arm=arm,
                            successful=len(success),
                            failed=10 - len(success),
                            selected_k_counts=counts,
                            rates=rates,
                            mean_retention=float(
                                np.mean([len(r["source_indices"]) / n for r in group])
                            ),
                            mean_ari=float(np.mean([r["ari"] for r in success]))
                            if success
                            else None,
                            mean_log_density=float(
                                np.mean([r["mean_log_density"] for r in success])
                            )
                            if success
                            else None,
                            mean_reference_common_ari=float(
                                np.mean(
                                    [
                                        r["reference_common_ari"]
                                        for r in success
                                        if "reference_common_ari" in r
                                    ]
                                )
                            )
                            if any("reference_common_ari" in r for r in success)
                            else None,
                        )
                    )
                obs = {
                    r["task"]["replicate"]: r
                    for r in rows
                    if r["task"]["case"] == case
                    and r["task"]["n"] == n
                    and r["mask"] == mask
                    and r["arm"] == "observed"
                }
                mean = {
                    r["task"]["replicate"]: r
                    for r in rows
                    if r["task"]["case"] == case
                    and r["task"]["n"] == n
                    and r["mask"] == mask
                    and r["arm"] == "mean"
                }
                common = [
                    (obs[i], mean[i])
                    for i in range(10)
                    if obs[i]["status"] == mean[i]["status"] == "complete"
                ]
                target = 4 if case == "four_groups" else 1
                plus = sum(
                    a["selected_k"] == target and b["selected_k"] != target for a, b in common
                )
                minus = sum(
                    a["selected_k"] != target and b["selected_k"] == target for a, b in common
                )
                if common:
                    p, m = interval(plus, len(common), 0.975), interval(minus, len(common), 0.975)
                    ci = [p[0] - m[1], p[1] - m[0]]
                else:
                    ci = None
                paired.append(
                    dict(
                        case=case,
                        n=n,
                        mask=mask,
                        paired_successful=len(common),
                        correct_k_difference=(plus - minus) / len(common) if common else None,
                        conservative95_difference=ci,
                        median_predictive_gain=float(
                            np.median(
                                [a["mean_log_density"] - b["mean_log_density"] for a, b in common]
                            )
                        )
                        if common
                        else None,
                    )
                )
    result = dict(
        schema_version=1,
        status="audited",
        datasets=60,
        point_searches=1200,
        replay_searches=24,
        cells=cells,
        observed_minus_mean=paired,
        inference="Pointwise descriptive intervals across independent datasets; masks are paired",
        p_values=None,
        defaults_changed=False,
        empirical_fits=False,
    )
    write_json(result, root / "aggregate.json")
    return result
