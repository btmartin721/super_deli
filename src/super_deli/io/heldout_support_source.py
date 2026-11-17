"""Bind immutable native subsampling artifacts without changing source evidence."""

from dataclasses import dataclass
from pathlib import Path

from super_deli.core.provenance import fingerprint_file
from super_deli.io.config import read_json
from super_deli.io.individual_subsample_source import load_point_source
from super_deli.models.individual_subsample_config import IndividualSubsampleConfig


@dataclass(frozen=True)
class SubsampleSource:
    root: Path
    point: object
    sampling: IndividualSubsampleConfig
    records: tuple
    summary: dict
    files: dict

    def reverify(self):
        self.point.reverify()
        for name, sha in self.files.items():
            path = self.root / name
            if path.is_symlink() or fingerprint_file(path).sha256 != sha:
                raise ValueError("Subsampling source changed: " + name)


def load_subsample_source(directory):
    import numpy as np

    from super_deli.models.individual_subsampling import draw_plan

    root = Path(directory).resolve(strict=True)
    bound = {}

    def file(name):
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or any(p.is_symlink() for p in path.parents if p.is_relative_to(root))
        ):
            raise ValueError("Require regular source artifacts: " + name)
        bound[name] = fingerprint_file(path).sha256
        return path

    summary = read_json(file("summary.json"))
    config = IndividualSubsampleConfig.from_dict(read_json(file("configuration.json")))
    if (
        summary.get("schema_version") != 1
        or summary.get("policy") != config.policy
        or summary.get("status") not in {"complete", "incomplete"}
    ):
        raise ValueError("Require an existing native individual-subsampling result")
    point = load_point_source(root / "reference")
    bound.update({"reference/" + k: v for k, v in point.files.items()})
    records = read_json(file("subsample_replicates.json"))
    n = len(point.units)
    if (
        len(records) != config.n_subsamples
        or summary["individuals"] != n
        or summary["source_point_k"] != point.summary["selected_k"]
        or summary["requested"] != len(records)
    ):
        raise ValueError("Source cohort or replicate declaration differs")
    successes = 0
    for i, (r, (draw, seed)) in enumerate(
        zip(records, draw_plan(n, config, point.model.random_seed), strict=True)
    ):
        absent = np.setdiff1d(np.arange(n), draw, assume_unique=True)
        if (
            r["replicate"] != i
            or r["fit_seed"] != seed
            or r["status"] not in {"complete", "failed"}
            or r["parameters"] != f"draw_{i:04d}"
            or r["training_unit_ids"] != [point.units[j] for j in draw]
            or r["distinct_training_units"] != len(draw)
        ):
            raise ValueError("Invalid source draw identity/seed declaration")
        np.testing.assert_array_equal(r["training_indices"], draw)
        np.testing.assert_array_equal(r["omitted_indices"], absent)
        saved = read_json(file(f"subsamples/{r['parameters']}/replicate.json"))
        if saved != r:
            raise ValueError("Source replicate record differs from its retained log")
        if r["status"] == "complete":
            successes += 1
            if not 1 <= r["selected_k"] <= point.model.max_k:
                raise ValueError("Invalid source candidate count")
            path = file(f"subsamples/{r['parameters']}/fit.npz")
            if path.stat().st_size > 4 * point.model.max_working_bytes:
                raise ValueError("Source fit exceeds its declared working limit")
            k1 = [c for c in r["candidates"] if c["k"] == 1]
            if len(k1) != 1 or k1[0]["status"] != "eligible":
                raise ValueError("Missing source native K=1 candidate")
    if summary["successful"] != successes or summary["failed"] != len(records) - successes:
        raise ValueError("Source success/failure counts differ")
    source = SubsampleSource(root, point, config, tuple(records), summary, bound)
    source.reverify()
    return source
