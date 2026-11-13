"""Complete, read-only source binding of the eighty original archived heads."""

import hashlib
import tarfile
from importlib.metadata import version
from pathlib import Path

from super_deli.benchmarking.abstention_source import bind_observation_inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.neural_false_split_state import verify_seal
from super_deli.benchmarking.view_signal_archive import file_hashes
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json

SOURCE_HEAD = "388a9a1d6f3bc2af028e07d8b151577f5512676c"


def check_completion(complete, ci):
    require(isinstance(complete, dict) and isinstance(ci, dict), "Typed completion/CI receipts")
    require(
        complete.get("completed") is True
        and complete.get("head_sha") == SOURCE_HEAD
        and complete.get("ci_jobs") == 6,
        "Completed exact class-local checkpoint",
    )
    expected = {
        f"{kind} ({version})"
        for kind in ("checks", "genomics", "neural")
        for version in ("3.11", "3.12")
    }
    jobs = ci.get("jobs")
    require(
        ci.get("head_sha") == SOURCE_HEAD
        and ci.get("status") == "completed"
        and ci.get("conclusion") == "success"
        and isinstance(jobs, list)
        and len(jobs) == 6
        and all(isinstance(job, dict) for job in jobs),
        "Source exact-head CI receipt",
    )
    require(
        {job.get("name") for job in jobs} == expected
        and all(
            job.get("head_sha") == SOURCE_HEAD
            and job.get("status") == "completed"
            and job.get("conclusion") == "success"
            for job in jobs
        ),
        "Six completed exact-head source CI jobs",
    )


def check_source(source, binding):
    require(type(binding.get("fixture")) is bool, "Explicit source kind")
    check_binding(source, binding)
    for entry in binding["observations"]:
        path = entry["path"]
        expected = {
            k.removeprefix(path + "/"): v
            for k, v in binding["source_sha256"].items()
            if k.startswith(path + "/")
        }
        require(
            file_hashes(Path(source) / path) == expected, "Complete unchanged source observation"
        )
        if not binding["fixture"]:
            require(path == "main/" + entry["task"]["key"], "Original source task identity")
            require(
                not verify_seal(Path(source), "main", dict(kind="fresh", task=entry["task"]))[
                    "interrupted"
                ],
                "Completed original source seal",
            )


def bind(source, root, *, fixture=False):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    files = {}

    def keep(name):
        files[name] = fingerprint_file(source / name).sha256

    if fixture:
        item = read_json(source / "fixture.json")
        require(
            item["kind"] == "invented_observed_feasibility_source_v1",
            "Explicit invented diagnostic source",
        )
        paths = item["observations"]
        keep("fixture.json")
    else:
        complete = read_json(source / "completion.json")
        ci = read_json(source / "verification-evidence/ci-verification.json")
        check_completion(complete, ci)
        frozen = read_json(source / "frozen.json")
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
            require(version(name) == frozen["software"][name], "Exact source numerical runtime")
        with tarfile.open(source / "frozen-sources.tar.gz") as archive:
            kept = {
                m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive
                if m.isfile()
            }
        require(
            kept == frozen["source_sha256"] and len(kept) == 486, "Complete original source bundle"
        )
        for name, digest in kept.items():
            if name not in ("cli.py", "__init__.py"):
                require(
                    fingerprint_file(Path(__file__).parents[1] / name).sha256 == digest,
                    "Unchanged source module: " + name,
                )
        for name in (
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "main-execution.json",
            "main-audit.json",
            "seed-registry.json",
            "verification-evidence/ci-verification.json",
        ):
            keep(name)
        rows = read_json(source / "main-execution.json")["records"]
        paths = ["main/" + r["key"] for r in rows if r["kind"] == "fresh"]
        require(len(paths) == 80, "Exactly eighty archived original classifier observations")
    entries = []
    for path in sorted(paths):
        task = read_json(source / path / "task.json")
        require(task["fixture"] == fixture, "Source/diagnostic fixture agreement")
        entries.append(dict(path=path, task=task))
        bind_observation_inventory(source, path, files)
        if not fixture:
            keep("ledger/main/" + task["key"] + ".json")
    replays = []
    for entry in entries:
        t = entry["task"]
        if fixture or t["replicate"] == 0 and t["case"] in ("linear_weak", "combined"):
            for arm in ("logistic", "residual_0"):
                replays.append(
                    dict(
                        key=f"frozen_{t['case']}_n{t['n']}_{arm}",
                        path=entry["path"],
                        arm=arm,
                        n=t["n"],
                        case=t["case"],
                    )
                )
    require(len(replays) == (2 if fixture else 8), "Frozen replay budget")
    value = dict(
        passed=True, fixture=fixture, observations=entries, replays=replays, source_sha256=files
    )
    check_source(source, value)
    root.mkdir(exist_ok=True)
    write_json(value, root / "replay-binding.json")
    return dict(
        passed=True, observations=len(entries), replays=len(replays), bound_files=len(files)
    )
