"""Independent raw-mask, training-moment, likelihood and support accounting.

No missing-data selection or orchestration module is imported. Density labels
are checked with the declared sklearn estimator on independently audited scores;
this verifies the adapter, not an independent reimplementation of HDBSCAN.
"""

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.core.provenance import configuration_hash
from super_deli.io.config import read_json, write_json


def _array_table(path):
    import numpy as np

    table = read_json(path)
    return table, np.array(
        [[np.nan if v is None else v for v in row] for row in table["values"]], dtype=float
    )


def _training_fit(raw, a, prefix, record, family):
    import numpy as np
    from scipy.special import logsumexp
    from sklearn.cluster import HDBSCAN

    np.testing.assert_array_equal(a[prefix + "raw"], raw)
    np.testing.assert_array_equal(a[prefix + "missing"], np.isnan(raw))
    assert record["missing_cells"] == int(np.isnan(raw).sum())
    if record["preprocessing_status"] != "complete":
        assert "error" in record
        return False
    assert not np.isnan(raw).all(axis=0).any() and not np.isnan(raw).all(axis=1).any()
    mean = np.nansum(raw, axis=0) / np.sum(~np.isnan(raw), axis=0)
    centered = np.where(np.isnan(raw), 0.0, raw - mean)
    variance = (centered**2).mean(axis=0)
    kept = np.flatnonzero(variance > 0)
    np.testing.assert_array_equal(a[prefix + "kept_indices"], kept)
    np.testing.assert_allclose(a[prefix + "means"], mean[kept], atol=1e-12)
    np.testing.assert_allclose(a[prefix + "scales"], np.sqrt(variance[kept]), atol=1e-12)
    z = centered[:, kept] / np.sqrt(variance[kept])
    c = a[prefix + "components"]
    np.testing.assert_allclose(a[prefix + "pca_mean"], z.mean(0), atol=1e-12)
    np.testing.assert_allclose(c @ c.T, np.eye(2), atol=1e-11)
    scores = (z - z.mean(0)) @ c.T
    np.testing.assert_allclose(a[prefix + "scores"], scores, atol=1e-11)
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1][:2]
    np.testing.assert_allclose(np.var(scores, axis=0, ddof=1), eig, atol=1e-10)
    if prefix + "labels" not in a:
        return False
    labels = a[prefix + "labels"]
    if family == "density":
        est = HDBSCAN(
            min_cluster_size=5,
            min_samples=5,
            metric="euclidean",
            alpha=1.0,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
            allow_single_cluster=True,
            n_jobs=1,
            copy=True,
        ).fit(a[prefix + "scores"])
        np.testing.assert_array_equal(est.labels_, labels)
        np.testing.assert_allclose(est.probabilities_, a[prefix + "strength"], atol=1e-12)
        assert np.all(a[prefix + "strength"][labels < 0] == 0)
    else:
        w, mu, cov = (a[prefix + k] for k in ("weights", "mixture_means", "covariances"))
        assert np.all(w > 0)
        np.testing.assert_allclose(w.sum(), 1)
        terms = []
        for weight, m, v in zip(w, mu, cov, strict=True):
            assert np.linalg.eigvalsh(v).min() > 0
            residual = a[prefix + "scores"] - m
            terms.append(
                np.log(weight)
                - 0.5
                * (
                    2 * np.log(2 * np.pi)
                    + np.linalg.slogdet(v)[1]
                    + np.einsum("ni,ij,nj->n", residual, np.linalg.inv(v), residual)
                )
            )
        joint = np.array(terms).T
        ll = logsumexp(joint, axis=1)
        probabilities = np.exp(joint - ll[:, None])
        np.testing.assert_allclose(a[prefix + "membership"], probabilities, atol=1e-9)
        np.testing.assert_array_equal(labels, probabilities.argmax(1))
        winner = min(
            (r for r in record["candidates"] if r["status"] == "eligible"),
            key=lambda r: (r["bic"], r["k"]),
        )
        k = len(w)
        assert winner["k"] == record["selected_k"] == k
        np.testing.assert_allclose(
            winner["bic"], (6 * k - 1) * np.log(len(raw)) - 2 * ll.sum(), atol=1e-6
        )
    return True


def audit_policy(root, source, mask, policy, truth):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.density_audit import _audit_support
    from super_deli.benchmarking.tabular_comparison_audit import _bootstrap_audit

    records = read_json(root / "eligibility.json")
    expected = np.flatnonzero((~mask).all(1) if policy == "complete_case" else (~mask).any(1))
    assert [r["source_index"] for r in records if r["eligible"]] == expected.tolist()
    for i, r in enumerate(records):
        assert r["unit_id"] == r["observation_id"] == f"unit_{i:04}"
        assert r["missing_features"] == [f"trait_{j}" for j in range(4) if mask[i, j]]
        assert r["values"] == [None if mask[i, j] else source[i, j] for j in range(4)]
        reason = (
            "all_selected_missing"
            if mask[i].all()
            else "incomplete_selected_features"
            if mask[i].any() and policy == "complete_case"
            else None
        )
        assert r["exclusion_reason"] == reason
    x = source[expected].copy()
    x[mask[expected]] = np.nan
    outcomes = []
    checks = draws = 0
    for family in ("gmm", "density"):
        arm = root / "analysis/arms" / family
        metric = dict(
            policy=policy,
            arm=family,
            status="prerequisite_blocked",
            eligible=len(expected),
            retention=len(expected) / len(source),
            selected_k=None,
            correct_k=None,
            ari=None,
            resolved_fraction=None,
            noise_fraction=None,
            imputed_fraction=None,
            selected_missing_fraction=float(mask[expected].mean()) if len(expected) else None,
            training_fits=0,
            draws=0,
        )
        context = arm / "training-context.json"
        native = read_json(arm / "summary.json") if (arm / "summary.json").exists() else None
        point = (
            read_json(arm / "comparison-point.json")
            if (arm / "comparison-point.json").exists()
            else None
        )
        bootfile = arm / ("bootstrap_replicates.json" if family == "gmm" else "subsamples.json")
        boot = read_json(bootfile) if bootfile.exists() else []
        if context.exists():
            metadata = read_json(context)
            assert metadata["declaration"]["policy"] == policy
            assert metadata["declaration"]["missing_value_uncertainty_integrated"] is False
            with np.load(arm / "training-context.npz", allow_pickle=False) as arrays:
                for i, fit in enumerate(metadata["fits"]):
                    ids = np.arange(len(x)) if not i else np.array(boot[i - 1]["training_indices"])
                    checks += _training_fit(x[ids], arrays, f"fit_{i}_", fit, family)
                    if i and family == "gmm" and boot[i - 1]["status"] == "complete":
                        prefix = f"fit_{i}_"
                        keep = arrays[prefix + "kept_indices"]
                        raw = x[:, keep]
                        means = arrays[prefix + "means"]
                        scales = arrays[prefix + "scales"]
                        assert not np.isnan(raw).all(1).any()
                        projected = (
                            (np.where(np.isnan(raw), means, raw) - means) / scales
                            - arrays[prefix + "pca_mean"]
                        ) @ arrays[prefix + "components"].T
                        parts = []
                        for weight, mu, cov in zip(
                            arrays[prefix + "weights"],
                            arrays[prefix + "mixture_means"],
                            arrays[prefix + "covariances"],
                            strict=True,
                        ):
                            residual = projected - mu
                            parts.append(
                                np.log(weight)
                                - 0.5
                                * (
                                    2 * np.log(2 * np.pi)
                                    + np.linalg.slogdet(cov)[1]
                                    + np.einsum(
                                        "ni,ij,nj->n", residual, np.linalg.inv(cov), residual
                                    )
                                )
                            )
                        canonical = np.array(parts).T.argmax(1)
                        original = np.array(fit["component_order"])[canonical]
                        np.testing.assert_array_equal(original, boot[i - 1]["labels"])
                if metadata["fits"] and metadata["fits"][0]["preprocessing_status"] == "complete":
                    metric["imputed_fraction"] = metadata["fits"][0]["imputed_cells"] / (
                        len(x) * len(arrays["fit_0_kept_indices"])
                    )
                metric["training_fits"] = len(metadata["fits"])
        if native is not None:
            provenance = native["provenance"]
            assert provenance["configuration_sha256"] == configuration_hash(
                provenance["configuration"]
            )
        if boot:
            cfg = native["provenance"]["configuration"]["model"]
            count = cfg["n_bootstrap" if family == "gmm" else "n_subsamples"]
            assert len(boot) == count
            seeds = np.random.SeedSequence(cfg["random_seed"]).spawn(count + (family == "gmm"))
            for i, draw in enumerate(boot):
                if family == "gmm":
                    draw_seed, fit_seed = map(int, seeds[i + 1].generate_state(2))
                    assert (draw["draw_seed"], draw["fit_seed"]) == (draw_seed, fit_seed)
                    expected_ids = np.random.default_rng(draw_seed).integers(len(x), size=len(x))
                else:
                    expected_ids = np.sort(
                        np.random.default_rng(seeds[i]).choice(
                            len(x), int(np.floor(len(x) * cfg["subsample_fraction"])), replace=False
                        )
                    )
                np.testing.assert_array_equal(draw["training_indices"], expected_ids)
            draws += len(boot)
            with np.load(arm / "stability.npz") as arrays:
                if family == "gmm":
                    _bootstrap_audit(len(x), boot, arrays)
                else:
                    _audit_support(len(x), boot, arrays)
            metric["draws"] = len(boot)
            metric["draw_failures"] = sum(r["status"] != "complete" for r in boot)
        if point is not None:
            assert native["unit_ids"] == [f"unit_{i:04}" for i in expected]
            assert native["effective_preprocessing"]["policy"] == policy
            y = np.array(point["labels"])
            resolved = np.array(point["resolved"], dtype=bool) & (y >= 0)
            metric.update(
                status="complete",
                selected_k=int(native["selected_k"]),
                correct_k=int(native["selected_k"]) == len(set(truth.tolist())),
                ari=float(adjusted_rand_score(truth[expected][resolved], y[resolved]))
                if resolved.sum() > 1
                else None,
                resolved_fraction=float(resolved.sum() / len(source)),
                noise_fraction=float(np.sum(y < 0) / len(source)),
                native_stability=native["bootstrap" if family == "gmm" else "stability"],
            )
        elif (root / "analysis/summary.json").exists():
            metric["status"] = "failed"
        outcomes.append(metric)
    return outcomes, checks, draws


def audit_observation(directory):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.benchmarking.density_study import generate_raw

    root = Path(directory)
    task = read_json(root / "task.json")
    validation = read_json(root / "validation.json")
    assert validation["status"] == "retained", validation
    phase = 80309 if task["phase"] == "preflight" else 80317
    ci = ("gaussian", "four_spherical", "moons").index(task["case"])
    assert task["data_seed"] == [80303, phase, ci, task["n"], task["replicate"]]
    assert task["mask_seed"] == [80329, phase, ci, task["n"], task["replicate"]]
    base, truth = generate_raw(task["case"], task["n"], task["data_seed"])
    base -= np.array([0.5, 0.25] if task["case"] == "moons" else [0.0, 0.0])
    x = np.stack(
        [
            base[:, 0],
            base[:, 1],
            (base[:, 0] + base[:, 1]) / np.sqrt(2),
            (base[:, 0] - base[:, 1]) / np.sqrt(2),
        ],
        axis=1,
    )
    x += np.random.default_rng(task["data_seed"] + [41]).normal(0, 0.05, x.shape)
    seed = task["mask_seed"]
    u = np.random.default_rng(seed + [0]).random(x.shape)
    mar = np.random.default_rng(seed + [1]).random(x.shape) < np.where(x[:, 0:1] <= 0, 0.15, 0.65)
    mar[:, 0] = False
    mnar = np.random.default_rng(seed + [2]).random(x.shape) < np.where(x <= 0, 0.1, 0.5)
    rng = np.random.default_rng(seed + [3])
    nuisance = rng.integers(2, size=len(x))
    pattern = (np.arange(4)[None, :] // 2 == nuisance[:, None]) & (rng.random(x.shape) < 0.6)
    masks = dict(mcar10=u < 0.1, mcar30=u < 0.3, mar=mar, mnar=mnar, pattern=pattern)
    with np.load(root / "simulation.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["complete"], x)
        np.testing.assert_array_equal(stored["truth"], truth)
        np.testing.assert_array_equal(stored["nuisance"], nuisance)
        for name, mask in masks.items():
            np.testing.assert_array_equal(stored[name], mask)
    assert not (masks["mcar10"] & ~masks["mcar30"]).any()
    # Oracle is an unchanged complete-data workflow. Its dimensionality is now
    # four, so independently check moments/projections rather than the old 2D distance shortcut.
    oracle = read_json(root / "oracle/summary.json")
    metrics = []
    oracle_points = {}
    checks = draws = 0
    for model in oracle["models"]:
        family = model["name"]
        p = root / "oracle/arms" / family / "comparison-point.json"
        m = dict(
            mask="oracle",
            policy="complete",
            arm=family,
            status="failed",
            eligible=len(x),
            retention=1.0,
            selected_k=None,
            correct_k=None,
            ari=None,
            resolved_fraction=None,
            noise_fraction=None,
            imputed_fraction=0.0,
        )
        if p.exists():
            point = read_json(p)
            armroot = root / "oracle/arms" / family
            f = "preprocessing.npz" if family == "gmm" else "point.npz"
            with np.load(armroot / f, allow_pickle=False) as saved:
                a = {"o_" + k: saved[k] for k in saved.files}
            if family == "gmm":
                with np.load(armroot / "mixture.npz", allow_pickle=False) as saved:
                    for key in saved.files:
                        a["o_" + ("mixture_means" if key == "means" else key)] = saved[key]
            a["o_raw"] = x
            a["o_missing"] = np.isnan(x)
            rec = dict(preprocessing_status="complete", missing_cells=0)
            if family == "gmm":
                rec.update(
                    candidates=read_json(armroot / "candidates.json"),
                    selected_k=model["selected_k"],
                )
            checks += _training_fit(x, a, "o_", rec, family)
            labels = np.array(point["labels"])
            resolved = np.array(point["resolved"]) & (labels >= 0)
            oracle_points[family] = (labels, resolved)
            m.update(
                status="complete",
                selected_k=model["selected_k"],
                correct_k=model["selected_k"] == len(set(truth.tolist())),
                ari=float(adjusted_rand_score(truth[resolved], labels[resolved]))
                if resolved.sum() > 1
                else None,
                resolved_fraction=float(resolved.mean()),
                noise_fraction=float((labels < 0).mean()),
            )
        metrics.append(m)
    for name, mask in masks.items():
        analysis = root / "analyses" / name
        manifest = read_json(analysis / "checksums.json")
        for path, digest in manifest.items():
            assert hashlib.sha256((analysis / path).read_bytes()).hexdigest() == digest
        meta = read_json(analysis / "manifest.json")
        assert meta["configuration_sha256"] == configuration_hash(meta["analysis"])
        for model in meta["analysis"]["arms"]:
            cfg = model["model"]
            assert cfg["random_seed"] == 80323 and cfg["n_pcs"] == 2
            if model["family"] == "gmm":
                assert cfg["max_k"] == 4 and cfg["min_cluster_size"] == 5 and cfg["n_init"] == 10
        table, raw = _array_table(analysis / "source-ingestion/modality_000/table.json")
        expected = x.copy()
        expected[mask] = np.nan
        np.testing.assert_array_equal(raw, expected)
        for policy in ("complete_case", "mean"):
            rows, n, d = audit_policy(analysis / "policies" / policy, x, mask, policy, truth)
            checks += n
            draws += d
            selected = np.flatnonzero(
                (~mask).all(1) if policy == "complete_case" else (~mask).any(1)
            )
            for row in rows:
                row.update(
                    oracle_common_coverage=None,
                    oracle_ari_on_common=None,
                    masked_ari_on_common=None,
                    ari_loss_on_common=None,
                )
                if row["status"] == "complete" and row["arm"] in oracle_points:
                    point = read_json(
                        analysis
                        / "policies"
                        / policy
                        / "analysis/arms"
                        / row["arm"]
                        / "comparison-point.json"
                    )
                    y = np.array(point["labels"])
                    origin, origin_resolved = oracle_points[row["arm"]]
                    shared = np.array(point["resolved"]) & (y >= 0) & origin_resolved[selected]
                    row["oracle_common_coverage"] = float(shared.sum() / len(x))
                    if shared.sum() > 1:
                        original_ari = float(
                            adjusted_rand_score(truth[selected][shared], origin[selected][shared])
                        )
                        masked_ari = float(adjusted_rand_score(truth[selected][shared], y[shared]))
                        row.update(
                            oracle_ari_on_common=original_ari,
                            masked_ari_on_common=masked_ari,
                            ari_loss_on_common=original_ari - masked_ari,
                        )
            metrics.extend(dict(row, mask=name) for row in rows)
    replays = validation["replays"]
    assert all(r["status"] in ("exact", "prerequisite_failed") for r in replays), replays
    assert len(replays) == (6 if task["replay"] else 0)
    for replay in replays:
        if replay["status"] != "exact":
            continue
        tag, arm = replay["policy"], replay["arm"]
        original = (
            root / "oracle/arms" / arm
            if tag == "oracle"
            else root / "analyses/mcar30/policies" / tag / "analysis/arms" / arm
        )
        again = root / "replays" / (tag + "_" + arm)
        assert (original / "assignments.csv").read_bytes() == (
            again / "assignments.csv"
        ).read_bytes()
        files = ("mixture.npz", "preprocessing.npz") if arm == "gmm" else ("point.npz",)
        for name in files:
            with (
                np.load(original / name, allow_pickle=False) as a,
                np.load(again / name, allow_pickle=False) as b,
            ):
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
    expected_fits = 22 + (40 if task["stability"] else 0) + (6 if task["replay"] else 0)
    assert validation["planned_pipeline_fits"] == expected_fits
    return dict(
        key=task["key"],
        case=task["case"],
        n=task["n"],
        replicate=task["replicate"],
        metrics=metrics,
        successful_training_model_checks=checks,
        retained_draws=draws,
        exact_replays=sum(r["status"] == "exact" for r in replays),
        blocked_replays=sum(r["status"] == "prerequisite_failed" for r in replays),
        planned_pipeline_fits=expected_fits,
    )


def audit_study(study_dir, *, output_file, workers=1):
    root = Path(study_dir)
    summary = read_json(root / "summary.json")
    directories = [str(root / r["key"]) for r in summary["records"]]
    if workers == 1:
        records = [audit_observation(p) for p in directories]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(audit_observation, directories))
    count = 1 if summary["phase"] == "preflight" else 10
    expected_grid = {
        (case, n, i)
        for case in ("gaussian", "four_spherical", "moons")
        for n in (64, 256)
        for i in range(count)
    }
    assert len(records) == len(expected_grid)
    assert {(r["case"], r["n"], r["replicate"]) for r in records} == expected_grid
    expected = 408 if summary["phase"] == "preflight" else 1836
    assert sum(r["planned_pipeline_fits"] for r in records) == expected
    result = dict(
        status="passed",
        phase=summary["phase"],
        observations=len(records),
        planned_pipeline_fits=expected,
        records=records,
        successful_training_model_checks=sum(
            r["successful_training_model_checks"] for r in records
        ),
        retained_draws=sum(r["retained_draws"] for r in records),
        exact_replays=sum(r["exact_replays"] for r in records),
        blocked_replays=sum(r["blocked_replays"] for r in records),
        new_selection_or_orchestration_imported=False,
    )
    write_json(result, output_file)
    return result


def aggregate(audit):
    import numpy as np

    from super_deli.benchmarking.grouped_false_positive import exact_interval

    groups = defaultdict(list)
    for r in audit["records"]:
        for m in r["metrics"]:
            groups[(r["case"], r["n"], m["mask"], m["policy"], m["arm"])].append(m)
    rows = []
    for key, values in sorted(groups.items()):
        complete = [m for m in values if m["status"] == "complete"]
        n = len(values)
        success = len(complete)
        event = sum(
            (m["selected_k"] > 1 if key[0] == "gaussian" else m["correct_k"]) for m in complete
        )
        interval = exact_interval(event, success)
        rows.append(
            dict(
                case=key[0],
                n=key[1],
                mask=key[2],
                policy=key[3],
                arm=key[4],
                datasets=n,
                successful=success,
                failed_or_blocked=n - success,
                event_count=event,
                event_definition="spurious_k_gt_1"
                if key[0] == "gaussian"
                else "correct_generating_k",
                conditional_rate=event / success if success else None,
                pointwise_exact_95=interval,
                failure_identification_bounds=[event / n, (event + n - success) / n],
                retained_fraction=float(np.mean([m["retention"] for m in values])),
                resolved_fraction_mean=float(np.mean([m["resolved_fraction"] for m in complete]))
                if complete
                else None,
                resolved_ari_median=float(
                    np.median([m["ari"] for m in complete if m["ari"] is not None])
                )
                if any(m["ari"] is not None for m in complete)
                else None,
                imputed_fraction_mean=(
                    float(
                        np.mean(
                            [
                                m["imputed_fraction"]
                                for m in complete
                                if m.get("imputed_fraction") is not None
                            ]
                        )
                    )
                    if any(m.get("imputed_fraction") is not None for m in complete)
                    else None
                ),
                median_ari_loss_on_common=(
                    float(
                        np.median(
                            [
                                m["ari_loss_on_common"]
                                for m in complete
                                if m.get("ari_loss_on_common") is not None
                            ]
                        )
                    )
                    if any(m.get("ari_loss_on_common") is not None for m in complete)
                    else None
                ),
                selected_k_counts=dict(
                    sorted(Counter(str(m["selected_k"]) for m in complete).items())
                ),
            )
        )
    return dict(
        schema_version=1,
        status="passed",
        phase=audit["phase"],
        observations=audit["observations"],
        rates=rows,
        uncertainty_unit="independent raw dataset within each cell; masks/draws repeated",
        intervals=(
            "pointwise descriptive exact intervals conditional on success; "
            "no multiple-testing or species claim"
        ),
        pvalues_generated=0,
        default_promoted=False,
    )
