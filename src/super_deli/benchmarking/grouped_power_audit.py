"""Independent raw-law, density, selection, tail and whole-group stability audits."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from super_deli.benchmarking.config import derived_seed
from super_deli.benchmarking.empirical_analysis import _raw_hashes
from super_deli.benchmarking.grouped_analysis import verify_model
from super_deli.benchmarking.grouped_boundary_analysis import audit_arm, tails
from super_deli.benchmarking.grouped_boundary_reference import load_model
from super_deli.benchmarking.grouped_pca_analysis import _bootstrap_pca
from super_deli.benchmarking.grouped_power_study import (
    COUNT_KEYS,
    _verify_replay,
    policy_for,
    simulate,
)
from super_deli.benchmarking.local_archives import materialize
from super_deli.benchmarking.pipeline_analysis import audit_pipeline_point
from super_deli.benchmarking.recovery_analysis import _ari, _dense_mixture
from super_deli.benchmarking.study_analysis import audit_bootstrap
from super_deli.benchmarking.tabular_pca_audit import audit_tabular_views
from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.grouped_power_config import GroupedPowerStudyConfig
from super_deli.workflows.calibration_artifacts import fit_counts


def independent_values(cell, seed):
    """Construct the declared law directly, independently of the benchmark wrapper."""
    import numpy as np

    n = cell.n_units
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, n)
    targets = np.repeat(np.arange(n), counts)
    z = rng.standard_normal((n, 8))
    errors = rng.standard_normal((int(counts.sum()), 4))
    if cell.law == "independent_gaussian":
        mean, between, within = np.zeros(8), np.eye(8), 0.75 * np.eye(4)
    else:
        scales = np.linspace(0.5, 2, 8)
        between = np.array(
            [[scales[i] * scales[j] * 0.6 ** abs(i - j) for j in range(8)] for i in range(8)]
        )
        within = np.array([[1.5 * 0.35 ** abs(i - j) for j in range(4)] for i in range(4)])
        mean = np.linspace(-2, 2, 8)
    latent = mean + z @ np.linalg.cholesky(between).T
    residual = errors @ np.linalg.cholesky(within).T
    labels = np.zeros(n, dtype=int)
    loading = np.array([1, 0.7, 0.5, 0.3])
    if cell.law == "heavy_tailed":
        latent = mean + (latent - mean) / np.sqrt(rng.chisquare(3, (n, 1)) / 3)
        residual /= np.sqrt(rng.chisquare(3, (len(errors), 1)) / 3)
    elif cell.has_groups:
        labels = rng.permutation(np.repeat(np.arange(4), n // 4))
        effects = np.concatenate(
            [(2 * (labels // 2) - 1)[:, None] * loading, (2 * (labels % 2) - 1)[:, None] * loading],
            axis=1,
        )
        latent += cell.signal * effects
    values = {"morphological": latent[targets, :4] + residual, "ecological": latent[:, 4:]}
    if cell.law == "cline":
        gradient_seed = derived_seed(seed, "cline", 0, "gradient")
        coordinate = np.random.default_rng(gradient_seed).uniform(-1, 1, n)
        values["morphological"] += 3 * coordinate[targets, None] * loading
        values["ecological"] += 3 * coordinate[:, None] * loading
    return values, counts, labels


class _RunArchive:
    """Map the numerical auditor's logical run prefix to a retained standalone run."""

    def __init__(self, run):
        self.run = Path(run)

    def read(self, name):
        prefix = "stability/run/"
        if not name.startswith(prefix):
            raise ValueError("Unexpected logical stability artifact prefix")
        relative = Path(name[len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe logical stability artifact path")
        return (self.run / relative).read_bytes()


def _stability_audit(base, config, layout, values, labels, observed):
    import numpy as np

    record = read_json(base / "stability_record.json")
    if record["status"] in {"point_failed", "not_attempted"}:
        if record.get("point_replay_verified"):
            raise ValueError("Failed replay cannot be marked verified")
        return {
            "status": record["status"],
            "requested": config.stability_draws,
            "successful": 0,
            "failed": record["failed_fits"],
            "point_replay_verified": False,
        }
    from super_deli.models.likelihood_selection import candidate_statistics

    run = base / "stability"
    model = replace(config.calibration.model, n_bootstrap=config.stability_draws)
    if read_json(base / "stability_model.json") != json.loads(json.dumps(model.to_dict())):
        raise ValueError("Stability model differs from the frozen point settings")
    _verify_replay(base / "calibration", run, model)
    summary = read_json(run / "summary.json")
    candidates = read_json(run / "candidates.json")
    point = {
        "archive_prefix": "stability",
        "status": "complete",
        "candidates": candidates,
        **candidate_statistics(candidates),
        "n_retained": len(layout["unit_ids"]),
        "point_seed": summary["point_seed"],
        **{
            k: summary["provenance"][k]
            for k in ("mixture_sha256", "fusion_sha256", "candidate_sha256")
        },
    }
    archive = _RunArchive(run)
    audit_pipeline_point(archive, point, SimpleNamespace(model=model), allow_stability=True)
    projections = audit_tabular_views(archive, "stability/run/", model, values, layout)
    _bootstrap_pca(run, model, layout, projections)
    with np.load(run / "gmm_candidates.npz", allow_pickle=False) as pool:
        n, d = pool["scores"].shape
        for candidate in candidates:
            if candidate["status"] == "insufficient_samples":
                continue
            k = candidate["k"]
            arrays = {
                key: pool[f"k{k}_{key}"]
                for key in ("weights", "means", "covariances", "membership", "labels")
            }
            likelihood = _dense_mixture({"scores": pool["scores"], **arrays})
            parameters = k - 1 + k * d + k * d * (d + 1) // 2
            np.testing.assert_allclose(
                candidate["log_likelihood"], likelihood, atol=1e-7, rtol=1e-10
            )
            np.testing.assert_allclose(
                candidate["aic"], -2 * likelihood + 2 * parameters, atol=1e-7
            )
            np.testing.assert_allclose(
                candidate["bic"], -2 * likelihood + parameters * np.log(n), atol=1e-7
            )
    counts = audit_bootstrap(run, model.random_seed, model.n_bootstrap)
    replicas = read_json(run / "bootstrap_replicates.json")
    for row in replicas:
        if row["status"] != "complete":
            continue
        eligible = []
        for c in row["candidates"]:
            if c["status"] == "insufficient_samples":
                if len(set(row["training_indices"])) >= c["k"] * model.min_cluster_size:
                    raise ValueError("False bootstrap distinct-unit infeasibility")
                continue
            k = c["k"]
            q = k - 1 + k * d + k * d * (d + 1) // 2
            np.testing.assert_allclose(c["aic"], -2 * c["log_likelihood"] + 2 * q, atol=1e-7)
            np.testing.assert_allclose(
                c["bic"], -2 * c["log_likelihood"] + q * np.log(n), atol=1e-7
            )
            admissible = c["converged"] and min(c["distinct_unit_sizes"]) >= model.min_cluster_size
            if (c["status"] == "eligible") != admissible:
                raise ValueError("Bootstrap eligibility differs from recorded distinct sizes")
            if admissible:
                eligible.append(c)
        if min(eligible, key=lambda c: (c["aic"], c["k"]))["k"] != row["selected_k"]:
            raise ValueError("Bootstrap AIC selection differs")
    if (
        record["attempted_fits"] != 1 + len(replicas)
        or record["successful_fits"] != 1 + counts["successful"]
        or record["failed_fits"] != counts["failed"]
    ):
        raise ValueError("Stability fit accounting differs")
    with np.load(run / "stability.npz", allow_pickle=False) as arrays:
        pair = np.triu_indices(len(labels), 1)
        truth = (labels[:, None] == labels[None, :])[pair]
        frequencies = arrays["coassignment"][pair]
        mse = float(np.mean((frequencies - truth) ** 2)) if counts["failed"] == 0 else None
    return {
        "status": record["status"],
        "point_replay_verified": True,
        "requested": counts["requested"],
        "successful": counts["successful"],
        "failed": counts["failed"],
        "coassignment_truth_mse": mse,
        "bootstrap_ari_mean": float(
            np.mean([_ari(labels, r["labels"]) for r in replicas if r["status"] == "complete"])
        )
        if counts["successful"]
        else None,
        "bootstrap_k4_count": sum(
            r.get("selected_k") == 4 for r in replicas if r["status"] == "complete"
        ),
        "density_audit_scope": (
            "All replay-point candidates; resamples audited by seeds, multiplicities, "
            "PCA, criteria, partitions and coassignment. Per-resample Gaussian parameter "
            "arrays are not exported by the unchanged production stability workflow."
        ),
    }


def _audit(task):
    import numpy as np

    from super_deli.models.grouped_null import write_raw_views

    source, declaration, ledger, raw_config, scratch = task
    config = GroupedPowerStudyConfig.from_dict(raw_config)
    cell = next(c for c in config.cells if c.name == declaration["cell"])
    expected_archive = f"{cell.name}/dataset_{declaration['repetition']:04d}.tar.gz"
    if (
        ledger["archive"] != expected_archive
        or ledger["manifest"] != expected_archive + ".manifest.json"
    ):
        raise ValueError("Observation archive identity differs")
    for key, value in declaration.items():
        if ledger[key] != value:
            raise ValueError("Observation declaration/seed differs from the frozen plan")
    policy = policy_for(config, declaration)
    with materialize(
        Path(source) / ledger["archive"],
        ledger,
        scratch=scratch,
        min_free_bytes=config.min_free_bytes,
    ) as base:
        record = read_json(base / "record.json")
        if (
            any(ledger.get(k) != v for k, v in record.items())
            or read_json(base / "declaration.json") != declaration
        ):
            raise ValueError("Archived observation record differs from the execution ledger")
        values, layout, labels = simulate(
            cell, declaration["data_seed"], target_level=policy.model.target_level
        )
        independent, counts, truth_labels = independent_values(cell, declaration["data_seed"])
        np.testing.assert_array_equal(layout["counts"], counts)
        np.testing.assert_array_equal(labels, truth_labels)
        if list(layout["unit_ids"]) != [f"group_{i:04d}" for i in range(cell.n_units)]:
            raise ValueError("Generating group identities differ")
        with np.load(base / "values.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(values):
                raise ValueError("Observed modality fields differ")
            for name, expected in values.items():
                np.testing.assert_array_equal(arrays[name], expected)
                np.testing.assert_allclose(arrays[name], independent[name], atol=2e-14, rtol=2e-14)
        truth = read_json(base / "truth.json")
        if (
            truth["labels"] != labels.tolist()
            or truth["has_groups"] != cell.has_groups
            or truth["unit_ids"] != list(layout["unit_ids"])
        ):
            raise ValueError("Generating truth or evaluation identities differ")
        if fingerprint_file(base / "values.npz").sha256 != ledger["values_sha256"]:
            raise ValueError("Observed array fingerprint differs")
        if read_json(base / "generating_layout.json") != json.loads(json.dumps(layout)):
            raise ValueError("Generating layout differs")
        for name, digest in _raw_hashes(values, layout, write_raw_views).items():
            if fingerprint_file(base / "observed" / name).sha256 != digest:
                raise ValueError("Observed ingested input differs from the declared raw law")
        output = base / "calibration"
        if read_json(output / "configuration.json") != json.loads(json.dumps(policy.to_dict())):
            raise ValueError("Calibration policy differs")
        observed = (
            read_json(output / "observed.json") if (output / "observed.json").exists() else None
        )
        draws = [read_json(p) for p in sorted(output.glob("draw_*.json"))]
        accounting = fit_counts(observed, draws, policy.n_simulations)
        if any(record["calibration"][key] != value for key, value in accounting.items()):
            raise ValueError("Calibration requested/attempted fit accounting differs")
        pipelines, members = [], 0
        if (output / "summary.json").exists():
            result = read_json(output / "summary.json")
            if result["configuration_sha256"] != configuration_hash(policy.to_dict()):
                raise ValueError("Calibration configuration digest differs")
            for name, digest in result["artifacts"].items():
                if (
                    Path(name).is_absolute()
                    or ".." in Path(name).parts
                    or fingerprint_file(output / name).sha256 != digest
                ):
                    raise ValueError("Calibration artifact path/fingerprint differs")
            original = Path(record["original_directory"]) / "observed"
            for name, digest in result["observed_input_sha256"].items():
                path = Path(name)
                if not path.is_relative_to(original) or ".." in path.parts:
                    raise ValueError("Original observed input provenance escapes its declared root")
                if (
                    fingerprint_file(base / "observed" / path.relative_to(original)).sha256
                    != digest
                ):
                    raise ValueError("Relocated observed input fingerprint differs")
            verify_model(base, output, values, layout, policy)
            model = load_model(output)
            pipelines, inference, members = audit_arm(
                output,
                model,
                values,
                policy,
                cohort="power",
                case=cell.name,
                repetition=declaration["repetition"],
                arm="fitted",
            )
            if (
                len(pipelines) != accounting["attempted_fits"]
                or sum(p["status"] == "complete" for p in pipelines)
                != accounting["successful_fits"]
            ):
                raise ValueError("Independent candidate pipeline accounting differs")
            if record["calibration"]["p_value"] != inference[0]["p_value"]:
                raise ValueError("Execution and independently reconstructed tails differ")
            status = "complete_numerical_audit"
        else:
            if (
                not (output / "failure.json").exists()
                or record["calibration"]["p_value"] is not None
            ):
                raise ValueError(
                    "Incomplete calibration must preserve failure and withhold p-value"
                )
            inference = tails(observed or {"status": "failed"}, draws, policy)
            if any(r["p_value"] is not None for r in inference):
                raise ValueError("Failed calibration produced a p-value")
            status = "retained_outer_failure_incomplete_numerical_audit"
        metrics = {"aic_k": None, "bic_k": None, "aic_ari": None, "bic_ari": None}
        if observed is not None and observed["status"] == "complete":
            with ZipFile(output / "pipeline_artifacts.zip") as archive:
                with np.load(
                    io.BytesIO(archive.read("observed/run/gmm_candidates.npz")), allow_pickle=False
                ) as pool:
                    for criterion, key in (("aic", "selected_k"), ("bic", "bic_selected_k")):
                        k = observed[key]
                        metrics[criterion + "_k"] = k
                        metrics[criterion + "_ari"] = (
                            _ari(labels, pool[f"k{k}_labels"]) if cell.has_groups else None
                        )
        stability = (
            _stability_audit(base, config, layout, values, labels, observed)
            if declaration["stability"]
            else None
        )
        if not declaration["stability"] and (base / "stability_record.json").exists():
            raise ValueError("Unexpected stability outside the prespecified subset")
        return {
            "cell": cell.name,
            "repetition": declaration["repetition"],
            "law": cell.law,
            "n_units": cell.n_units,
            "signal": cell.signal,
            "has_groups": cell.has_groups,
            "audit_status": status,
            "likelihood_p": inference[0]["p_value"],
            "likelihood_reject": inference[0]["reject"],
            "bic_p": inference[1]["p_value"],
            "bic_reject": inference[1]["reject"],
            **metrics,
            "tails": inference,
            "stability": stability,
            "pipelines": pipelines,
            "archive_members": members,
            "accounting": {key: ledger[key] for key in COUNT_KEYS},
        }


def audit_observation(task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _audit(task)
