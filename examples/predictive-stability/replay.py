"""Exact six-search replays of the prespecified archived full/shared point workflows."""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from super_deli.benchmarking.local_archives import compact_directory, materialize
from super_deli.io.config import load_config, read_json, write_json
from super_deli.models.shared_predictive_config import SharedPredictiveConfig

CELLS = (("tied", "four_n64_s3_0"), ("tied", "four_n64_s4_5"), ("full", "independent_gaussian_n64"))


def replay(task):
    import numpy as np

    from super_deli.workflows.predictive import run_predictive

    source, destination, scratch, raw, family, row = task
    c = SharedPredictiveConfig.from_dict(raw).family(family)
    root = Path(destination)
    base = root / ".working" / f"{family}_{row['cell']}_{row['repetition']:04d}"
    base.mkdir(parents=True, exist_ok=False)
    write_json(dict(family=family, source_observation=row), base / "private-source-pin.json")
    with materialize(Path(source) / row["archive"], row, scratch=scratch) as pair:
        arm = row["arms"][family]
        with materialize(pair / family / arm["archive"], arm, scratch=scratch) as original:
            result = run_predictive(
                load_config(original / "observed/dataset.json"),
                model_config=c.model,
                screen_config=replace(c.screening, random_seed=row["split_seed"]),
                base_dir=original / "observed",
                output_dir=base / "workflow",
            )
            assert result["selection"] == arm["selection"]
            assert read_json(base / "workflow/fold_assignments.json") == read_json(
                original / "workflow/fold_assignments.json"
            )
            comparisons = 0
            for prefix in ["full"] + [f"fold_{i:03d}" for i in range(5)]:
                a = original / "workflow" / prefix / "run"
                b = base / "workflow" / prefix / "run"
                assert read_json(a / "candidates.json") == read_json(b / "candidates.json")
                for relative in [
                    "gmm_candidates.npz",
                    "mixture.npz",
                    "fusion.npz",
                    "view_000/parameters/pca.npz",
                    "view_001/parameters/pca.npz",
                    "view_000/parameters/profiles.npz",
                    "view_001/parameters/profiles.npz",
                ]:
                    with (
                        np.load(a / relative, allow_pickle=False) as x,
                        np.load(b / relative, allow_pickle=False) as y,
                    ):
                        assert set(x.files) == set(y.files)
                        for key in x.files:
                            np.testing.assert_array_equal(x[key], y[key])
                comparisons += 1
    record = dict(
        family=family,
        cell=row["cell"],
        repetition=row["repetition"],
        verified=True,
        planned_searches=6,
        comparisons=comparisons,
        **{
            k: result[k]
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
    )
    write_json(record, base / "record.json")
    archive = f"{family}_{row['cell']}_{row['repetition']:04d}.tar.gz"
    packed = compact_directory(base, root / archive)
    shutil.rmtree(base)
    return dict(record, **packed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-study", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four workers")
    source = args.source_study.resolve()
    root = args.output_dir.resolve()
    root.mkdir(exist_ok=False)
    raw = read_json(source / "configuration.json")
    rows = [json.loads(s) for s in (source / "observations.jsonl").read_text().splitlines()]
    plan = [
        (family, row)
        for family, cell in CELLS
        for row in sorted([r for r in rows if r["cell"] == cell], key=lambda r: r["repetition"])[:5]
    ]
    assert len(plan) == 15
    write_json([dict(family=f, row=r) for f, r in plan], root / "private-replay-plan.json")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result = list(
            pool.map(
                replay, [(str(source), str(root), str(args.scratch), raw, f, r) for f, r in plan]
            )
        )
    with (root / "observations.jsonl").open("x") as handle:
        for row in result:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = dict(
        requested_replays=15,
        verified_replays=sum(r["verified"] for r in result),
        planned_searches=90,
        **{
            k: sum(r[k] for r in result)
            for k in ("attempted_fits", "successful_fits", "failed_fits", "unattempted_fits")
        },
    )
    write_json(summary, root / "summary.json")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
