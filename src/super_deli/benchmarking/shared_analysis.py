"""Independent likelihood/partition/stability audit of the frozen paired study."""

import csv
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.joint_analysis import point
from super_deli.benchmarking.joint_study import simulate_joint
from super_deli.benchmarking.recovery_analysis import _ari, _axis_check, _dense_mixture, _wilson
from super_deli.benchmarking.recovery_inputs import DirectoryArchive, inventory
from super_deli.benchmarking.shared_study import (
    accounting,
    point_metrics,
    simulate_shared,
    study_model,
    study_seed,
)
from super_deli.io.config import read_json, write_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.shared_config import COHORTS, FAMILIES, SharedStudyConfig


def audit_stability(run, model, point_labels, point_k):
    import numpy as np

    summary = read_json(run / "summary.json")["bootstrap"]
    records = read_json(run / "bootstrap_replicates.json")
    if len(records) != model.n_bootstrap or summary["requested"] != model.n_bootstrap:
        raise ValueError("Stability requested/attempted denominators disagree")
    if not model.n_bootstrap:
        if summary["status"] != "not_computed" or (run / "stability.npz").exists():
            raise ValueError("Unexpected stability output when disabled")
        return {"requested": 0, "attempted": 0, "successful": 0, "failed": 0}, []
    n = len(point_labels)
    counts, oob_counts, requested, successful = (
        np.zeros((n, n), dtype=np.uint64) for _ in range(4)
    )
    groups = [frozenset(np.flatnonzero(point_labels == k)) for k in range(point_k)]
    recovered = np.zeros(point_k, dtype=int)
    choices = Counter()
    seeds = np.random.SeedSequence(model.random_seed).spawn(model.n_bootstrap + 1)[1:]
    passed = failed = 0
    for i, (row, child) in enumerate(zip(records, seeds, strict=True)):
        draw_seed, fit_seed = map(int, child.generate_state(2))
        indices = np.random.default_rng(draw_seed).integers(n, size=n)
        absent = np.bincount(indices, minlength=n) == 0
        pairs = np.logical_and.outer(absent, absent)
        if (
            row["replicate"] != i
            or row["draw_seed"] != draw_seed
            or row["fit_seed"] != fit_seed
            or row["training_indices"] != indices.tolist()
            or row["oob_indices"] != np.flatnonzero(absent).tolist()
            or row["unique_training_units"] != len(set(indices))
        ):
            raise ValueError("Bootstrap identities, multiplicities or seeds disagree")
        requested += pairs
        if row["status"] == "failed":
            if not row.get("error") or "labels" in row:
                raise ValueError("Malformed failed bootstrap record")
            failed += 1
            continue
        if row["status"] != "complete":
            raise ValueError("Unknown bootstrap status")
        predicted = np.asarray(row["labels"])
        if (
            predicted.shape != (n,)
            or predicted.dtype.kind not in "iu"
            or predicted.min() < 0
            or predicted.max() >= row["selected_k"]
        ):
            raise ValueError("Invalid bootstrap partition")
        eligible = []
        for c in row["candidates"]:
            if c["status"] == "insufficient_samples":
                if len(set(indices)) >= c["k"] * model.min_cluster_size or c["bic"] is not None:
                    raise ValueError("Invalid distinct-unit infeasibility")
                continue
            sizes, distinct = c["hard_cluster_sizes"], c["distinct_unit_sizes"]
            if (
                sum(sizes) != n
                or sum(distinct) != len(set(indices))
                or len(sizes) != c["k"]
                or len(distinct) != c["k"]
                or not c["converged"]
                or (min(distinct) >= model.min_cluster_size) != (c["status"] == "eligible")
            ):
                raise ValueError("Bootstrap candidate size or eligibility mismatch")
            if c["status"] == "eligible":
                eligible.append(c)
        if [c["k"] for c in row["candidates"]] != list(range(1, model.max_k + 1)) or min(
            eligible, key=lambda c: (c["bic"], c["k"])
        )["k"] != row["selected_k"]:
            raise ValueError("Bootstrap K-selection bookkeeping mismatch")
        same = predicted[:, None] == predicted[None, :]
        counts += same
        oob_counts += same & pairs
        successful += pairs
        observed = {frozenset(np.flatnonzero(predicted == k)) for k in range(row["selected_k"])}
        recovered += [group in observed for group in groups]
        choices[row["selected_k"]] += 1
        passed += 1
    if (
        summary["successful"] != passed
        or summary["failed"] != failed
        or summary["status"] != ("unavailable" if failed else "available")
        or summary["selected_k_counts"] != {str(k): v for k, v in sorted(choices.items())}
    ):
        raise ValueError("Stability summary disagrees with replicate ledger")
    expected = {
        "coassignment_counts": counts,
        "oob_coassignment_counts": oob_counts,
        "oob_requested": requested,
        "oob_successful": successful,
    }

    def bounds(values, total, ok):
        lower = np.full(values.shape, np.nan)
        upper = lower.copy()
        np.divide(values, total, out=lower, where=total > 0)
        np.divide(values + total - ok, total, out=upper, where=total > 0)
        return lower, upper, np.where(total == ok, lower, np.nan)

    lo, hi, freq = bounds(
        counts, np.full(counts.shape, model.n_bootstrap), np.full(counts.shape, passed)
    )
    np.fill_diagonal(lo, 1)
    np.fill_diagonal(hi, 1)
    if not failed:
        np.fill_diagonal(freq, 1)
    expected.update(coassignment_lower=lo, coassignment_upper=hi, coassignment=freq)
    lo, hi, freq = bounds(oob_counts, requested, successful)
    expected.update(oob_coassignment_lower=lo, oob_coassignment_upper=hi, oob_coassignment=freq)
    with np.load(run / "stability.npz", allow_pickle=False) as arrays:
        for key, value in expected.items():
            np.testing.assert_allclose(arrays[key], value, rtol=0, atol=1e-12, equal_nan=True)
    clusters = read_json(run / "clusters.json")
    for k, row in enumerate(clusters):
        if row["exact_recovery_count"] != int(recovered[k]):
            raise ValueError("Exact-set recovery mismatch")
        expected_frequency = recovered[k] / model.n_bootstrap if not failed else None
        if (
            row["frequency"] != expected_frequency
            or row["failure_lower"] != recovered[k] / model.n_bootstrap
            or row["failure_upper"] != (recovered[k] + failed) / model.n_bootstrap
        ):
            raise ValueError("Exact-set failure bounds mismatch")
    return {
        "requested": model.n_bootstrap,
        "attempted": len(records),
        "successful": passed,
        "failed": failed,
    }, recovered.tolist()


def audit_shared_study(root, *, output_dir):
    import numpy as np

    from super_deli.models.joint_null import write_raw_joint

    root, destination = Path(root).resolve(), Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Audit must be separate from immutable study")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    expected = read_json(root / "files.json")
    actual = inventory(root)
    actual.pop("files.json")
    if expected != actual:
        raise ValueError("Study inventory mismatch")
    config = SharedStudyConfig.from_dict(read_json(root / "configuration.json"))
    prior = JointStudyConfig.from_dict(read_json(root / "source_study.json"))
    declarations = read_json(root / "datasets.json")
    if [(r["cohort"], r["repetition"]) for r in declarations] != [
        (c, i) for c in COHORTS for i in range(config.n_datasets)
    ]:
        raise ValueError("Study identities or replacements disagree")
    records, rows, candidates, stability_rows = [], [], [], []
    replayed = 0
    for declaration in declarations:
        cohort, i = declaration["cohort"], declaration["repetition"]
        n = config.large_n if cohort == "fresh_256" else config.small_n
        seed = (
            derived_seed(prior.random_seed, "complementary", i, "observed_data")
            if cohort == "archived"
            else study_seed(config, cohort, i)
        )
        if declaration != {
            "cohort": cohort,
            "repetition": i,
            "n": n,
            "seed": seed,
            "has_groups": cohort not in {"zero_association", "cline"},
        }:
            raise ValueError("Frozen dataset declaration mismatch")
        values, layout, labels = (
            simulate_joint("complementary", n, seed)
            if cohort == "archived"
            else simulate_shared(cohort, n, seed)
        )
        directory = root / cohort / f"dataset_{i:04d}"
        truth = read_json(directory / "truth.json")
        if truth["labels"] != labels.tolist() or truth["unit_ids"] != layout["unit_ids"]:
            raise ValueError("Generating truth or evaluation identities mismatch")
        with np.load(directory / "values.npz", allow_pickle=False) as arrays:
            for key, value in values.items():
                np.testing.assert_array_equal(arrays[key], value)
        with TemporaryDirectory(prefix="super-deli-shared-audit-") as temporary:
            raw = Path(temporary) / "raw"
            write_raw_joint(values, layout, raw)
            if inventory(raw) != inventory(directory / "raw"):
                raise ValueError("Raw generator or input identity mismatch")
        for family in FAMILIES:
            model = study_model(config, cohort, i, family)
            record = read_json(directory / family / "record.json")
            records.append({"cohort": cohort, "repetition": i, "family": family, **record})
            row = {
                "cohort": cohort,
                "repetition": i,
                "family": family,
                "n": n,
                "status": record["status"],
                "selected_k": None,
                "ari": None,
                "k4_status": None,
                "k4_minus_k2_bic": None,
                "bootstrap_requested": model.n_bootstrap,
                "bootstrap_failed": record["bootstrap"]["failed"],
            }
            if record["status"] == "failed":
                if (
                    not record.get("error")
                    or "metrics" in record
                    or record["bootstrap"]
                    != {
                        "requested": model.n_bootstrap,
                        "attempted": 0,
                        "successful": 0,
                        "failed": 0,
                    }
                ):
                    raise ValueError("Malformed failed point ledger")
                rows.append(row)
                continue
            run = directory / family / "run"
            point(
                DirectoryArchive(directory),
                record,
                SimpleNamespace(model=model),
                values,
                layout,
                allow_stability=True,
            )
            metrics = point_metrics(run, labels, declaration["has_groups"])
            if metrics != record["metrics"]:
                raise ValueError("Recovery metrics mismatch")
            with np.load(run / "mixture.npz", allow_pickle=False) as arrays:
                likelihood = _dense_mixture(arrays)
                k = record["selected_k"]
                d = arrays["scores"].shape[1]
                # Independent parameter-count expression for the tied/full families.
                parameters = k * d + k - 1 + d * (d + 1) // 2 * (1 if family == "tied" else k)
                selected = next(c for c in record["candidates"] if c["k"] == k)
                np.testing.assert_allclose(
                    selected["bic"], -2 * likelihood + parameters * np.log(n), atol=1e-7
                )
                if declaration["has_groups"]:
                    np.testing.assert_allclose(
                        metrics["ari"], _ari(labels, arrays["labels"]), atol=1e-12
                    )
                    _axis_check(arrays["scores"], labels, metrics["joint_axis_r2"])
                verified, recoveries = audit_stability(run, model, arrays["labels"], k)
            if verified != record["bootstrap"]:
                raise ValueError("Bootstrap ledger mismatch")
            c2, c4 = (next(c for c in record["candidates"] if c["k"] == j) for j in (2, 4))
            row.update(
                selected_k=record["selected_k"],
                ari=metrics["ari"],
                k4_status=c4["status"],
                k4_minus_k2_bic=c4["bic"] - c2["bic"],
            )
            rows.append(row)
            for c in metrics["bic_terms"]:
                candidates.append(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "family": family,
                        **{
                            k: v
                            for k, v in c.items()
                            if k not in {"hard_cluster_sizes", "distinct_unit_sizes"}
                        },
                    }
                )
            if model.n_bootstrap:
                stability_rows.append(
                    {
                        "cohort": cohort,
                        "repetition": i,
                        "family": family,
                        **verified,
                        "selected_k": record["selected_k"],
                        "mean_exact_set_frequency": float(np.mean(recoveries) / model.n_bootstrap)
                        if not verified["failed"]
                        else None,
                    }
                )
        full = read_json(directory / "full/record.json")
        tied = read_json(directory / "tied/record.json")
        if full["status"] == tied["status"] == "complete":
            with (
                np.load(directory / "full/run/mixture.npz", allow_pickle=False) as a,
                np.load(directory / "tied/run/mixture.npz", allow_pickle=False) as b,
            ):
                np.testing.assert_array_equal(a["scores"], b["scores"])
        if cohort == "archived" and full["status"] == "complete":
            old = read_json(directory / "source_baseline/record.json")
            point(
                DirectoryArchive(directory),
                {**old, "archive_prefix": "source_baseline"},
                SimpleNamespace(model=config.model),
                values,
                layout,
            )
            if old["selected_k"] != full["selected_k"]:
                raise ValueError("Archived baseline selection mismatch")
            for a, b in zip(old["candidates"], full["candidates"], strict=True):
                if {k: v for k, v in a.items() if k != "bic"} != {
                    k: v for k, v in b.items() if k != "bic"
                }:
                    raise ValueError("Archived candidate status mismatch")
                np.testing.assert_allclose(a["bic"], b["bic"], rtol=1e-10, atol=1e-7)
            with (
                np.load(directory / "source_baseline/run/mixture.npz", allow_pickle=False) as a,
                np.load(directory / "full/run/mixture.npz", allow_pickle=False) as b,
            ):
                np.testing.assert_allclose(a["scores"], b["scores"], rtol=1e-9, atol=1e-11)
                np.testing.assert_array_equal(a["labels"], b["labels"])
            replayed += 1
    if records != read_json(root / "points.json"):
        raise ValueError("Global point ledger mismatch")
    accounting_result = {
        "schema_version": 1,
        "datasets": len(declarations),
        "reproduced_archived": replayed,
        **accounting(records),
    }
    if accounting_result != read_json(root / "summary.json"):
        raise ValueError("Study accounting mismatch")
    summaries, paired = [], []
    for cohort in COHORTS:
        for family in FAMILIES:
            selected = [r for r in rows if r["cohort"] == cohort and r["family"] == family]
            available = [r for r in selected if r["status"] == "complete"]
            counts = {str(k): sum(r["selected_k"] == k for r in available) for k in range(1, 5)}
            aris = [r["ari"] for r in available if r["ari"] is not None]
            summaries.append(
                {
                    "cohort": cohort,
                    "family": family,
                    "requested": len(selected),
                    "available": len(available),
                    "failed": len(selected) - len(available),
                    "selected_k_counts": counts,
                    "selected_four_wilson_95": _wilson(counts["4"], len(available)),
                    "selected_one_wilson_95": _wilson(counts["1"], len(available)),
                    "mean_ari": float(np.mean(aris)) if aris else None,
                    "median_ari": float(np.median(aris)) if aris else None,
                }
            )
        full = {
            r["repetition"]: r
            for r in rows
            if r["cohort"] == cohort and r["family"] == "full" and r["status"] == "complete"
        }
        tied = {
            r["repetition"]: r
            for r in rows
            if r["cohort"] == cohort and r["family"] == "tied" and r["status"] == "complete"
        }
        indices = sorted(full.keys() & tied.keys())
        differences = [
            tied[i]["ari"] - full[i]["ari"] for i in indices if full[i]["ari"] is not None
        ]
        paired.append(
            {
                "cohort": cohort,
                "available_pairs": len(indices),
                "mean_ari_change": float(np.mean(differences)) if differences else None,
                "median_ari_change": float(np.median(differences)) if differences else None,
                "ari_improved": sum(d > 1e-9 for d in differences),
                "ari_worsened": sum(d < -1e-9 for d in differences),
                "ari_unchanged": sum(abs(d) <= 1e-9 for d in differences),
                "gained_four": sum(
                    full[i]["selected_k"] != 4 and tied[i]["selected_k"] == 4 for i in indices
                ),
                "lost_four": sum(
                    full[i]["selected_k"] == 4 and tied[i]["selected_k"] != 4 for i in indices
                ),
            }
        )
    result = {
        **accounting_result,
        "status": "complete",
        "verified_files": len(expected),
        "summaries": summaries,
        "paired": paired,
        "audit_scope": (
            "Regenerated inputs/identities/truth; complete inventories, configurations, "
            "paired scores, archived baseline replay, fusion/projection, selected dense "
            "mixture likelihood/BIC/membership and tied covariance equality; independent "
            "ARI/axis checks; bootstrap seeds/multiplicities, distinct-unit constraints, "
            "K-selection, label partitions, exact-set counts, pair-specific OOB denominators "
            "and failure bounds. No numerical model refits. View PCA and "
            "nonselected/bootstrapped component likelihoods preserved as diagnostics, not "
            "independently refitted."
        ),
        "interpretation": (
            "Prespecified covariance-family comparison under constructed laws; opt-in shared "
            "covariance assumes equal component covariance. Pointwise Wilson intervals are "
            "descriptive, not species significance or a calibration guarantee. Bootstrap "
            "support remains separate from p-values."
        ),
    }
    destination.mkdir()
    for name, data in (
        ("points.csv", rows),
        ("candidates.csv", candidates),
        ("stability.csv", stability_rows),
        ("paired.csv", paired),
    ):
        with (destination / name).open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(data[0]) if data else [], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(data)
    write_json(result, destination / "summary.json")
    write_json(expected, destination / "files.json")
    return result
