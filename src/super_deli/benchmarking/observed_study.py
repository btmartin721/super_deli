"""Frozen point-only missing-data EM comparison; truth never enters fitting."""

import hashlib
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.io.config import read_json, write_json
from super_deli.models.observed_config import ObservedGMMConfig

CASES = ("spherical", "correlated", "four_groups")
MASKS = ("mcar10", "mcar30", "mar", "mnar", "pattern")
ARMS = ("observed", "complete_case", "mean", "reference")
NAMES = tuple(f"trait_{i}" for i in range(4))
SETTINGS = dict(
    schema_version=1,
    cases=list(CASES),
    masks=list(MASKS),
    arms=list(ARMS),
    sample_sizes=[64, 256],
    replicates=10,
    evaluation_units=2048,
    data_seed=80403,
    mask_seed=80409,
    model_seed=80423,
    evaluation_seed=80429,
    main_searches=1224,
    preflight_searches=144,
    max_workers=4,
    artifact_cap_bytes=2 * 1024**3,
    min_free_bytes=8 * 1024**3,
    allowance=1.25,
    reserve_bytes=128 * 1024**2,
)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def tasks(phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown phase")
    return [
        dict(
            key=f"{case}_n{n}_{i:02}",
            case=case,
            n=n,
            replicate=i,
            phase=phase,
            namespace=[0 if phase == "preflight" else 1, c, n, i],
            replay=i == 0,
        )
        for c, case in enumerate(CASES)
        for n in (64, 256)
        for i in range(1 if phase == "preflight" else 10)
    ]


def sample(case, n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    if case == "spherical":
        return rng.normal(size=(n, 4)), np.zeros(n, dtype=int)
    coefficient = 0.6 if case == "correlated" else 0.3
    covariance = coefficient ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    if case == "correlated":
        return rng.multivariate_normal(np.zeros(4), covariance, n), np.zeros(n, dtype=int)
    labels = rng.integers(0, 4, size=n)
    signs = np.column_stack([2 * (labels // 2) - 1, 2 * (labels % 2) - 1])
    means = np.column_stack([3 * signs, 1.5 * signs])
    return means + rng.multivariate_normal(np.zeros(4), covariance, n), labels


def masks(x, seed):
    import numpy as np

    uniform = np.random.default_rng(seed + [0]).random(x.shape)
    mar = np.random.default_rng(seed + [1]).random(x.shape) < np.where(x[:, 0:1] <= 0, 0.15, 0.65)
    mar[:, 0] = False
    mnar = np.random.default_rng(seed + [2]).random(x.shape) < np.where(x <= 0, 0.1, 0.5)
    rng = np.random.default_rng(seed + [3])
    flag = rng.integers(2, size=len(x))
    pattern = (np.arange(4)[None, :] // 2 == flag[:, None]) & (rng.random(x.shape) < 0.6)
    return dict(mcar10=uniform < 0.1, mcar30=uniform < 0.3, mar=mar, mnar=mnar, pattern=pattern)


def prepared(x, masked, arm, norm):
    import numpy as np

    indices = np.arange(len(x))
    if arm == "observed":
        return masked, indices
    if arm == "reference":
        return x, indices
    keep = (
        ~np.isnan(masked).any(axis=1) if arm == "complete_case" else ~np.isnan(masked).all(axis=1)
    )
    values = masked[keep]
    if arm == "mean":
        values = np.where(np.isnan(values), norm.means, values)
    return values, indices[keep]


def write_once_json(value, path, resume):
    if path.exists() and resume:
        if read_json(path) != value:
            raise ValueError("Resume metadata mismatch")
    else:
        write_json(value, path)


def save_fit(
    values,
    norm,
    config,
    directory,
    metadata,
    source_indices,
    evaluation,
    eval_mask,
    truth,
    resume=False,
):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.models.observed_gmm import fit_observed_matrix

    if directory.exists() and resume:
        stored = read_json(directory / "summary.json")
        if stored["config"] != config.to_dict() or any(stored[k] != v for k, v in metadata.items()):
            raise ValueError("Resume configuration/identity mismatch")
        file = "fit.npz" if (directory / "fit.npz").exists() else "rejected-input.npz"
        with np.load(directory / file, allow_pickle=False) as a:
            keep = a["kept_indices"] if file == "fit.npz" else np.arange(len(values))
            np.testing.assert_array_equal(a["raw"], values[keep])
            if file == "fit.npz":
                np.testing.assert_array_equal(a["normalization_means"], norm.means)
                np.testing.assert_array_equal(a["scales"], norm.scales)
        if stored["source_indices"] != source_indices[keep].tolist():
            raise ValueError("Resume source-row identity mismatch")
        return stored
    directory.mkdir()
    try:
        fit = fit_observed_matrix(values, config=config, fixed_normalization=norm)
        np.savez_compressed(directory / "fit.npz", **fit.arrays)
        summary = {
            **fit.summary,
            **metadata,
            "source_indices": source_indices[fit.kept_indices].tolist(),
        }
        if fit.parameters is not None:
            density, membership = fit.predict(np.where(eval_mask, np.nan, evaluation))
            full_density, full_membership = fit.predict(evaluation)
            np.savez_compressed(
                directory / "evaluation.npz",
                density=density,
                membership=membership,
                complete_density=full_density,
                complete_membership=full_membership,
            )
            summary.update(
                ari=float(
                    adjusted_rand_score(
                        truth[source_indices[fit.kept_indices]], fit.arrays["labels"]
                    )
                ),
                true_groups_retained=int(len(np.unique(truth[source_indices[fit.kept_indices]]))),
                mean_log_density=float(np.nanmean(density)),
                complete_mean_log_density=float(full_density.mean()),
                evaluation_coverage=int(np.isfinite(density).sum()),
                entropy=float(
                    (
                        -fit.arrays["membership"]
                        * np.log(np.maximum(fit.arrays["membership"], np.finfo(float).tiny))
                    )
                    .sum(axis=1)
                    .mean()
                ),
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        np.savez_compressed(directory / "rejected-input.npz", raw=values)
        summary = dict(
            status="failed",
            error=str(exc),
            selected_k=None,
            **metadata,
            config=config.to_dict(),
            source_indices=source_indices.tolist(),
        )
    write_json(summary, directory / "summary.json")
    return summary


def observation(arguments):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from super_deli.models.observed_gmm import normalization

    task, root, resume = arguments
    directory = Path(root) / task["key"]
    directory.mkdir(exist_ok=resume)
    namespace = task["namespace"]
    x, truth = sample(task["case"], task["n"], [80403] + namespace)
    missing = masks(x, [80409] + namespace)
    evaluation, eval_truth = sample(task["case"], 2048, [80429] + namespace + [0])
    eval_missing = masks(evaluation, [80429] + namespace + [1])
    source = dict(
        complete=x,
        truth=truth,
        evaluation=evaluation,
        evaluation_truth=eval_truth,
        **{f"mask_{k}": v for k, v in missing.items()},
        **{f"eval_mask_{k}": v for k, v in eval_missing.items()},
    )
    if (directory / "inputs.npz").exists() and resume:
        with np.load(directory / "inputs.npz", allow_pickle=False) as previous:
            if set(previous.files) != set(source):
                raise ValueError("Resume input manifest mismatch")
            for key, value in source.items():
                np.testing.assert_array_equal(previous[key], value)
    else:
        np.savez_compressed(directory / "inputs.npz", **source)
    write_once_json(task, directory / "task.json", resume)
    config = ObservedGMMConfig(NAMES)
    records = []
    for name, mask in missing.items():
        masked = np.where(mask, np.nan, x)
        norm = normalization(masked)
        for arm in ARMS:
            values, indices = prepared(x, masked, arm, norm)
            metadata = dict(mask=name, arm=arm, task=task, original_n=len(x))
            result = save_fit(
                values,
                norm,
                config,
                directory / f"{name}_{arm}",
                metadata,
                indices,
                evaluation,
                eval_missing[name],
                truth,
                resume=resume,
            )
            records.append(result)
            if task["replay"] and name == "mcar30":
                save_fit(
                    values,
                    norm,
                    config,
                    directory / f"replay_{arm}",
                    metadata,
                    indices,
                    evaluation,
                    eval_missing[name],
                    truth,
                    resume=resume,
                )
        reference = directory / f"{name}_reference"
        if read_json(reference / "summary.json")["status"] == "complete":
            with np.load(reference / "fit.npz", allow_pickle=False) as source:
                ref_labels = source["labels"]
                ref_indices = source["kept_indices"]
            for arm in ARMS:
                output = directory / f"{name}_{arm}"
                summary = read_json(output / "summary.json")
                if summary["status"] != "complete":
                    continue
                with np.load(output / "fit.npz", allow_pickle=False) as source:
                    ids = np.asarray(summary["source_indices"])
                    common, ia, ib = np.intersect1d(ids, ref_indices, return_indices=True)
                    summary["reference_common_n"] = int(len(common))
                    summary["reference_common_ari"] = float(
                        adjusted_rand_score(source["labels"][ia], ref_labels[ib])
                    )
                write_once_json(
                    {key: summary[key] for key in ("reference_common_n", "reference_common_ari")},
                    output / "comparison.json",
                    resume,
                )
    return dict(
        key=task["key"],
        point_searches=20,
        replay_searches=4 if task["replay"] else 0,
        completed=sum(r["status"] == "complete" for r in records),
    )


def run_validation(root, phase, workers=4, resume=False):
    root = Path(root)
    if not 1 <= workers <= 4:
        raise ValueError("Use at most four workers")
    frozen = read_json(
        root
        / (
            "execution-freeze.json"
            if (root / "execution-freeze.json").exists()
            else "frozen-before-generation.json"
        )
    )
    if frozen["source_sha256"] != source_identity():
        raise ValueError("Frozen source mismatch")
    if phase == "main" and read_json(root / "storage-gate.json").get("status") != "passed":
        raise ValueError("Main run requires an audited storage gate")
    destination = root / phase
    destination.mkdir(exist_ok=resume)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(observation, [(t, destination, resume) for t in tasks(phase)]))
    write_json(
        dict(
            status="completed",
            phase=phase,
            outcomes=outcomes,
            searches=sum(x["point_searches"] + x["replay_searches"] for x in outcomes),
        ),
        root / f"{phase}-execution.json",
    )


def storage_gate(root):
    root = Path(root)
    audit = read_json(root / "preflight-audit.json")
    if audit.get("status") != "passed" or audit["searches"] != 144:
        raise ValueError("Storage gate requires the complete independent preflight audit")
    base = replays = 0
    for p in (root / "preflight").rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith("replay_") for part in p.relative_to(root / "preflight").parts):
            replays += p.stat().st_size
        else:
            base += p.stat().st_size
    retained = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    projected = retained + math.ceil((10 * base + replays) * 1.25) + 128 * 1024**2
    free = shutil.disk_usage(root).free
    result = dict(
        status="passed" if projected <= 2 * 1024**3 and free >= 8 * 1024**3 else "failed",
        projected_bytes=projected,
        cap_bytes=2 * 1024**3,
        free_bytes=free,
        base_preflight_bytes=base,
        replay_preflight_bytes=replays,
        retained_before_main=retained,
        allowance=1.25,
        reserve_bytes=128 * 1024**2,
        planned_main_searches=1224,
    )
    write_json(result, root / "storage-gate.json")
    return result
