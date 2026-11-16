"""Sealed source access and fit-free retained-draw mechanism traces."""

from collections import Counter
from pathlib import Path

import numpy as np

from super_deli.benchmarking.initialization_source import InitializationArchive
from super_deli.io.config import write_json


class BootstrapSplitArchive(InitializationArchive):
    source_head = "31b7471705e1c2c9709e3d35504a9d695b140345"
    source_version = "0.84.0.dev0"

    def baseline(self, index, arm="legacy100"):
        return Path(f"validation/main/fresh-{index:03d}/legacy100")


def cached_trace(source, output):
    summary = source.json("validation/main/summary.json")
    rows = []
    counts = Counter()
    for case in summary["records"]:
        if case["origin"] != "fresh" or not case["task"]["stability"]:
            continue
        task = case["task"]
        for arm in ("legacy10", "legacy100", "diverse100"):
            prefix = Path(f"validation/main/fresh-{task['index']:03d}/{arm}")
            events = source.json(prefix / "observed-pipeline/events.json")
            fits = source.json(prefix / "observed-candidates/index.json")
            for event in events[1:]:
                draw = np.asarray(event["draw"])
                frequency = np.bincount(draw, minlength=task["n"]) / len(draw)
                row = dict(
                    parent=task["index"],
                    n=task["n"],
                    law=task["law"],
                    arm=arm,
                    replicate=event["index"] - 1,
                    unique_originals=len(set(draw)),
                    effective_weight_support=float(1 / np.sum(frequency**2)),
                    maximum_multiplicity=int(frequency.max() * len(draw)),
                    status=event["status"],
                )
                if event["status"] == "complete":
                    a = source.arrays(prefix / "observed-pipeline" / event["arrays"])
                    covariance = np.linalg.eigvalsh(np.cov(a["scores"], rowvar=False))
                    k1 = next(c for c in event["candidates"] if c["k"] == 1)
                    chosen = next(c for c in event["candidates"] if c["k"] == event["selected_k"])
                    k4 = next(c for c in event["candidates"] if c["k"] == 4)
                    row.update(
                        selected_k=event["selected_k"],
                        covariance_condition=float(covariance[-1] / covariance[0]),
                        view_scales=[float(a[f"v{i}_scale"]) for i in range(4)],
                        bic_improvement=float(k1["bic"] - chosen["bic"]),
                        k4_status=k4["status"],
                        k4_distinct_sizes=k4.get("distinct_unit_sizes"),
                    )
                    # Bind every candidate actually used by this event to its score space.
                    from super_deli.benchmarking.multimodal_validation import array_hash

                    for fit in fits[event["candidate_start"] : event["candidate_stop"]]:
                        assert fit["scores_sha256"] == array_hash(a["scores"])
                rows.append(row)
                counts[arm] += 1
    assert len(rows) == 2250 and counts == Counter(legacy10=750, legacy100=750, diverse100=750)
    source.reverify()
    output = Path(output)
    output.mkdir(parents=True)
    write_json(rows, output / "retained-draw-traces.json")
    result = dict(
        passed=True,
        new_fits=0,
        draws=len(rows),
        ordinary100_primary_draws=750,
        parents=30,
        counts=dict(counts),
        source_files_verified=len(source.consumed),
    )
    write_json(result, output / "summary.json")
    return result
