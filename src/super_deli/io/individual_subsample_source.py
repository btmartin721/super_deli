"""Read and bind native point artifacts without rereading or altering source inputs."""

from dataclasses import dataclass
from pathlib import Path

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json
from super_deli.models.individual_subsample_config import require_supported_point
from super_deli.models.integrative_config import IntegrativeConfig


@dataclass(frozen=True)
class NativePointSource:
    root: Path
    summary: dict
    model: object
    inputs: tuple
    units: tuple
    labels: object
    files: dict

    def reverify(self):
        for name, digest in self.files.items():
            p = self.root / name
            if p.is_symlink() or fingerprint_file(p).sha256 != digest:
                raise ValueError("Point source changed: " + name)


def load_point_source(reference_run):
    import numpy as np

    from super_deli.models.integrative import IntegrationInput, _prepare

    root = Path(reference_run).resolve(strict=True)
    bound = {}

    def file(name):
        p = root / name
        if (
            p.is_symlink()
            or not p.is_file()
            or any(x.is_symlink() for x in p.parents if x.is_relative_to(root))
        ):
            raise ValueError("Point artifacts must be regular source files: " + name)
        bound[name] = fingerprint_file(p).sha256
        return p

    summary = read_json(file("summary.json"))
    if summary.get("status") != "complete":
        raise ValueError("A successful native integrative point run is required")
    declaration = summary["provenance"]["configuration"]
    if configuration_hash(declaration) != summary["provenance"]["configuration_sha256"]:
        raise ValueError("Point configuration checksum mismatch")
    model = IntegrativeConfig.from_dict(declaration["model"])
    require_supported_point(model)
    # Existing ingested dosages and explicit row/landmark identities are the sampling frame.
    inputs = []
    for i, v in enumerate(model.views):
        name = f"view_{i:03d}"
        r = read_json(file(name + "/source_rows.json"))
        p = file(name + "/source_values.npy")
        if p.stat().st_size > model.max_working_bytes + 1024:
            raise ValueError("Source array exceeds the declared working limit")
        if bound[name + "/source_values.npy"] != r["array_sha256"]:
            raise ValueError("Source NPY checksum mismatch")
        a = np.load(p, allow_pickle=False)
        a.setflags(write=False)
        inputs.append(
            IntegrationInput(
                v.modality,
                v.kind,
                r["source_level"],
                a,
                tuple(r["observation_ids"]),
                tuple(r["unit_ids"]),
                tuple(r["target_ids"]),
                tuple(r["feature_names"]),
                tuple(r["feature_types"]),
                tuple(r["landmark_ids"]),
            )
        )
    prepared, units, excluded = _prepare(inputs, model)
    if (
        excluded
        or list(units) != summary["unit_ids"]
        or summary.get("excluded_units")
        or summary.get("excluded_observations")
    ):
        raise ValueError("Require a complete unchanged matched-individual cohort")
    p = file("mixture.npz")
    if bound["mixture.npz"] != summary["provenance"]["mixture_sha256"]:
        raise ValueError("Point mixture checksum mismatch")
    file("fusion.npz")
    if bound["fusion.npz"] != summary["provenance"]["fusion_sha256"]:
        raise ValueError("Point fusion checksum mismatch")
    with np.load(p, allow_pickle=False) as a:
        labels = a["labels"].copy()
        prob = a["membership"].copy()
    k = summary["selected_k"]
    if (
        labels.shape != (len(units),)
        or prob.shape != (len(units), k)
        or not np.isfinite(prob).all()
        or not np.allclose(prob.sum(1), 1)
    ):
        raise ValueError("Invalid native point partition")
    np.testing.assert_array_equal(labels, prob.argmax(1))
    if len(set(labels)) != k or np.bincount(labels, minlength=k).min() < model.min_cluster_size:
        raise ValueError("Native point violates its declared support rule")
    expected = int(np.random.SeedSequence(model.random_seed).spawn(1)[0].generate_state(1)[0])
    if summary["point_seed"] != expected:
        raise ValueError("Native point seed lineage mismatch")
    for name in ("assignments.csv", "result.json", "clusters.json", "candidates.json"):
        if (root / name).exists():
            file(name)
    source = NativePointSource(root, summary, model, prepared, units, labels, bound)
    source.reverify()
    return source
