"""Read retained candidate logs and tails; never fit or recalibrate a model."""

import shutil

from super_deli.benchmarking.config import derived_seed
from super_deli.io.config import read_json
from super_deli.models.grouped_config import study_cases


def _statistic(record):
    if record["status"] != "complete":
        return None
    eligible = [c for c in record["candidates"] if c["status"] == "eligible"]
    best = min(eligible, key=lambda c: (c["bic"], c["k"]))
    one = next(c["bic"] for c in eligible if c["k"] == 1)
    t = one - best["bic"]
    if record["statistic"] != t or record["selected_k"] != best["k"] or t < 0:
        raise ValueError("Retained test statistic or selection mismatch")
    return t


def retain_tests(source, destination, study):
    destination.mkdir()
    for case in study_cases(study):
        (destination / case).mkdir()
        for i in range(study.n_datasets):
            original = source / "study" / case / f"dataset_{i:04d}" / "calibration"
            target = destination / case / f"dataset_{i:04d}"
            target.mkdir()
            for name in ("observed.json", "null_replicates.json", "summary.json"):
                shutil.copyfile(original / name, target / name)
    return audit_tests(destination, study)


def audit_tests(root, study):
    import numpy as np

    rows = []
    for case in study_cases(study):
        for i in range(study.n_datasets):
            base = root / case / f"dataset_{i:04d}"
            observed = read_json(base / "observed.json")
            draws = read_json(base / "null_replicates.json")
            inference = read_json(base / "summary.json")["inference"]
            t = _statistic(observed)
            stream = derived_seed(study.random_seed, case, i, "null_stream")
            seeds = [
                derived_seed(stream, "grouped_raw_gaussian_v2", j, "null_data")
                for j in range(study.calibration.n_simulations)
            ]
            if len(draws) != (len(seeds) if t is not None else 0):
                raise ValueError("Retained draw count mismatch")
            for j, (draw, seed) in enumerate(zip(draws, seeds, strict=t is not None)):
                if (draw["replicate"], draw["simulation_seed"]) != (j, seed):
                    raise ValueError("Retained draw seed/order mismatch")
            stats = [_statistic(r) for r in draws]
            successful = [v for v in stats if v is not None]
            complete = t is not None and len(successful) == len(seeds)
            if complete:
                exceed = sum(v >= t for v in successful)
                p = (1 + exceed) / (1 + len(seeds))
                if (
                    inference["status"] != "available"
                    or inference["p_value"] != p
                    or inference["exceedances"] != exceed
                    or inference["reject_grouped_raw_gaussian"] != (p <= study.calibration.alpha)
                ):
                    raise ValueError("Retained upper-tail inference mismatch")
            elif inference["status"] == "available" or inference.get("p_value") is not None:
                raise ValueError("Incomplete retained test exposes a p-value")
            else:
                exceed = p = None
            rows.append(
                {
                    "case": case,
                    "repetition": i,
                    "status": inference["status"],
                    "selected_k": observed.get("selected_k"),
                    "observed_statistic": t,
                    "exceedances": exceed,
                    "p_value": p,
                    "reject": inference.get("reject_grouped_raw_gaussian"),
                    "successful_draws": len(successful),
                    "failed_draws": len(stats) - len(successful),
                    "null_zero_count": sum(v == 0 for v in successful),
                    "null_max": max(successful) if successful else None,
                    "null_q95": float(np.quantile(successful, 0.95)) if successful else None,
                    "positive_null_count": sum(v > 0 for v in successful),
                }
            )
    return rows
