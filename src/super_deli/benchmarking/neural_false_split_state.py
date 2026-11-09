"""Atomic evidence records and conservative, same-host study ownership."""

import json
import os
import socket
import uuid
from contextlib import AbstractContextManager

from super_deli.benchmarking.matched_study import inventory
from super_deli.benchmarking.neural_analysis import require
from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json


def atomic_json(path, value, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), "Immutable evidence already exists: " + path.name)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def process_alive(pid):
    require(type(pid) is int and pid > 0, "Valid recorded process identity")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StudyLock(AbstractContextManager):
    def __init__(self, root):
        self.root = root
        self.path = root / "study.lock"
        self.token = uuid.uuid4().hex

    def __enter__(self):
        freeze = fingerprint_file(self.root / "frozen.json").sha256
        if self.path.exists():
            previous = read_json(self.path)
            require(previous.get("kind") == "neural_false_split_lock_v1", "Recognized study lock")
            require(previous["host"] == socket.gethostname(), "Cannot recover another host's lock")
            require(previous["freeze_sha256"] == freeze, "Lock belongs to this source freeze")
            require(not process_alive(previous["pid"]), "Another study controller is active")
            for path in (self.root / "state/active").glob("*.json"):
                record = read_json(path)
                require(record["host"] == socket.gethostname(), "Recognized worker host")
                require(
                    record["status"] != "running" or not process_alive(record["pid"]),
                    "An orphaned research worker remains active",
                )
            retired = self.root / "ledger/locks" / (previous["token"] + ".json")
            retired.parent.mkdir(parents=True, exist_ok=True)
            require(not retired.exists(), "Lock recovery cannot overwrite prior evidence")
            os.rename(self.path, retired)
        record = dict(
            kind="neural_false_split_lock_v1",
            token=self.token,
            host=socket.gethostname(),
            pid=os.getpid(),
            freeze_sha256=freeze,
        )
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        require(read_json(self.path)["token"] == self.token, "Controller still owns the lock")
        target = self.root / "ledger/locks" / (self.token + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(self.path, target)
        return False


def seal_path(root, phase, key):
    return root / "ledger" / phase / (key + ".json")


def seal(root, phase, job, result, *, interrupted=False):
    value = dict(
        freeze_sha256=fingerprint_file(root / "frozen.json").sha256,
        kind=job["kind"],
        task=job["task"],
        result=result,
        interrupted=interrupted,
        inventory=inventory(root / phase / job["task"]["key"]),
    )
    atomic_json(seal_path(root, phase, job["task"]["key"]), value)
    return value


def verify_seal(root, phase, job):
    value = read_json(seal_path(root, phase, job["task"]["key"]))
    require(
        value["freeze_sha256"] == fingerprint_file(root / "frozen.json").sha256,
        "Unit belongs to unchanged source freeze",
    )
    require(value["kind"] == job["kind"] and value["task"] == job["task"], "Frozen unit identity")
    require(
        value["inventory"] == inventory(root / phase / job["task"]["key"]),
        "Completed or interrupted evidence changed",
    )
    return value
