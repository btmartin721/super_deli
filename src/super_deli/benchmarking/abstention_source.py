"""Read-only binding of eight archived frozen classifiers; never retrain replays."""

from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json


def bind_observation_inventory(source, relative, files):
    """The shared validator requires complete observation trees, not selected files."""
    files.update(
        {relative + "/" + key: value for key, value in file_hashes(Path(source) / relative).items()}
    )


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    files, entries = {}, []

    def keep(path):
        files[path] = fingerprint_file(source / path).sha256

    if fixture:
        require(
            read_json(source / "fixture.json") == dict(kind="invented_abstention_source_v1"),
            "Explicit invented source",
        )
        keep("fixture.json")
    else:
        complete = read_json(source / "completion.json")
        require(
            complete["completed"]
            and complete["head_sha"] == "0a31344387ffc052583bb34d767c7b851d308e36"
            and complete["ci_jobs"] == 6,
            "Completed exact source checkpoint",
        )
        frozen = read_json(source / "frozen.json")
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
            require(
                version(name) == frozen["software"][name], "Unchanged archived numerical runtime"
            )
        for name in (
            "completion.json",
            "frozen.json",
            "verification-evidence/ci-verification.json",
        ):
            keep(name)
        for case in ("linear_weak", "combined"):
            for n in (64, 256):
                path = f"main/{case}_n{n}_0000"
                keep(path + "/task.json")
                for cohort in ("iid", "unfamiliar"):
                    keep(path + f"/{cohort}/generating-reference.npz")
                for arm in ("logistic", "residual_0"):
                    keep(path + f"/{arm}/external-predictions.npz")
                    names = (
                        ("run/model.npz", "run/summary.json")
                        if arm.startswith("residual")
                        else ("model.npz", "model-config.json")
                    )
                    for name in names:
                        keep(path + f"/{arm}/" + name)
                    entries.append(dict(key=f"legacy_{case}_n{n}_{arm}", path=path, arm=arm, n=n))
        for relative in sorted({entry["path"] for entry in entries}):
            bind_observation_inventory(source, relative, files)
    root.mkdir(exist_ok=True)
    result = dict(passed=True, fixture=fixture, replays=entries, source_sha256=files)
    check_binding(source, result)
    write_json(result, root / "replay-binding.json")
    return dict(passed=True, replays=len(entries), bound_files=len(files))
