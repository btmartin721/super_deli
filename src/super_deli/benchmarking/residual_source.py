"""Read-only binding of twelve power-study point replays via audited extracts."""

from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        source != root and not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    entries, files = [], {}
    if fixture:
        marker = read_json(source / "fixture.json")
        require(marker == {"kind": "invented_residual_source_v1"}, "Explicit engineering source")
        files["fixture.json"] = fingerprint_file(source / "fixture.json").sha256
    else:
        complete, frozen = read_json(source / "completion.json"), read_json(source / "frozen.json")
        require(
            complete["completed"]
            and complete["head_sha"] == "8ada12875feb1a0df6952a412cc231b858edfbaf"
            and complete["source_head"] == "cf6082d383f15a5b058c49585f5b4bbbfa4e0aa6"
            and complete["ci_jobs"] == 6,
            "Completed exact diagnostic and power-source heads",
        )
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
            require(
                version(name) == frozen["software"][name], "Unchanged archived numerical runtime"
            )
        for name in (
            "completion.json",
            "frozen.json",
            "replay-binding.json",
            "verification-evidence/ci-verification.json",
        ):
            files[name] = fingerprint_file(source / name).sha256
        old = read_json(source / "replay-binding.json")
        for entry in old["entries"]:
            t = entry["task"]
            if (
                t["case"] != "four_complementary"
                or t["signal_multiplier"] != 0.5
                or t["replicate"] >= 3
            ):
                continue
            relative = "inputs/" + t["key"]
            files.update({relative + "/" + k: v for k, v in inventory(source / relative).items()})
            for arm in ("pca4", "neural"):
                entries.append(
                    dict(
                        key="legacy_" + t["key"] + "_" + arm,
                        path=relative,
                        arm=arm,
                        original_task=t,
                    )
                )
        require(len(entries) == 12, "Exactly twelve observed power-study replays")
    binding = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    root.mkdir(exist_ok=True)
    write_json(binding, root / "replay-binding.json")
    check_binding(source, binding)
    return dict(passed=True, replays=len(entries), bound_files=len(files))
