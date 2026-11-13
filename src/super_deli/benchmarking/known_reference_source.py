"""Read-only binding to the completed predecessor and exact published CI head."""

import hashlib
import tarfile
from pathlib import Path

from super_deli.benchmarking.neural_analysis import require
from super_deli.benchmarking.view_weight_archive import check_binding
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json, write_json

SOURCE_HEAD = "2949908b3febbe67c085c022400f6f2c7a981490"


def bind(source, root, *, fixture=False):
    require(type(fixture) is bool, "Explicit source fixture kind")
    source, root = Path(source).resolve(), Path(root).resolve()
    require(
        not source.is_relative_to(root) and not root.is_relative_to(source),
        "Separate source/output roots",
    )
    if fixture:
        require(
            read_json(source / "fixture.json") == dict(kind="invented_known_reference_source_v1"),
            "Explicit invented source",
        )
        names = ["fixture.json"]
    else:
        complete = read_json(source / "completion.json")
        ci = read_json(source / "verification-evidence/ci-verification.json")
        require(
            complete["completed"]
            and complete["head_sha"] == SOURCE_HEAD
            and complete["ci_jobs"] == 6,
            "Completed exact predecessor",
        )
        expected = {
            f"{k} ({v})" for k in ("checks", "genomics", "neural") for v in ("3.11", "3.12")
        }
        require(
            ci["head_sha"] == SOURCE_HEAD
            and ci["status"] == "completed"
            and ci["conclusion"] == "success"
            and len(ci["jobs"]) == 6
            and {j["name"] for j in ci["jobs"]} == expected
            and all(
                j["head_sha"] == SOURCE_HEAD
                and j["status"] == "completed"
                and j["conclusion"] == "success"
                for j in ci["jobs"]
            ),
            "Six exact-head source CI jobs",
        )
        frozen = read_json(source / "frozen.json")
        with tarfile.open(source / "frozen-sources.tar.gz") as z:
            hashes = {
                m.name: hashlib.sha256(z.extractfile(m).read()).hexdigest()
                for m in z.getmembers()
                if m.isfile()
            }
        require(
            hashes == frozen["source_sha256"] and len(hashes) == 497,
            "Complete predecessor source bundle",
        )
        for name, digest in hashes.items():
            if name not in ("cli.py", "__init__.py"):
                require(
                    fingerprint_file(Path(__file__).parents[1] / name).sha256 == digest,
                    "Unchanged predecessor module: " + name,
                )
        names = [
            "completion.json",
            "frozen.json",
            "frozen-sources.tar.gz",
            "verification-evidence/ci-verification.json",
        ]
    value = dict(
        passed=True,
        fixture=fixture,
        replays=[],
        source_sha256={n: fingerprint_file(source / n).sha256 for n in names},
    )
    check_binding(source, value)
    root.mkdir(exist_ok=True)
    write_json(value, root / "replay-binding.json")
    return dict(passed=True, bound_files=len(names), unknown_training_examples=0)
