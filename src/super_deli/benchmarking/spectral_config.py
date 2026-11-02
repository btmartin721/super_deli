"""Dependency-free, versioned contract for a supplied-K archival diagnosis."""

from dataclasses import asdict, dataclass, fields

NEIGHBORS = (5, 15)
TARGETS = (2, 4)
SEEDS = (0, 101, 202)
SPACES = ("source",) + tuple(
    f"{condition}_{arm}"
    for condition in ("clean", "noisy")
    for arm in (
        "traits",
        "density",
        *(f"neural_{s}" for s in SEEDS),
        *(f"denoising_{s}" for s in SEEDS),
    )
)
REPLAY_SPACES = ("source",) + tuple(
    f"{c}_{a}" for c in ("clean", "noisy") for a in ("traits", "density", "neural_0", "denoising_0")
)
SETTINGS = {
    "neighbors_other_units": list(NEIGHBORS),
    "supplied_k": list(TARGETS),
    "affinity": "(directed binary neighbors + transpose)/2; no diagonal",
    "eigen_solver": "arpack",
    "eigen_tol": 0.0,
    "random_state": 71323,
    "drop_first": False,
    "norm_laplacian": True,
    "assign_labels": "cluster_qr",
    "cutoff_gap": 1e-8,
    "assignment_margin": 1e-10,
    "residual_gram_tolerance": 1e-8,
    "projector_tolerance": 1e-6,
    "main_evaluations": 4628,
    "preflight_evaluations": 504,
}


@dataclass(frozen=True)
class SpectralConfig:
    schema_version: int = 1
    fixture: bool = False
    max_workers: int = 2
    artifact_cap_bytes: int = 2 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 3600
    max_working_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported spectral diagnostic schema")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be boolean")
        for name, limit in (
            ("max_workers", 2),
            ("artifact_cap_bytes", 2 * 1024**3),
            ("max_projected_seconds", 3600),
            ("max_working_bytes", 256 * 1024**2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} exceeds the bounded checkpoint")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("At least 8 GiB free required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown spectral configuration fields")
        return cls(**value)


def graph_jobs(binding, phase):
    if phase not in {"preflight", "main"}:
        raise ValueError("Unknown diagnostic phase")
    observations = binding["observations"]
    if phase == "preflight":
        observations = [e for e in observations if e["replicate"] == 0]
    result = []
    for entry in observations:
        spaces = SPACES
        for kind in ("point", "replay") if entry["replicate"] == 0 else ("point",):
            if kind == "replay" and phase == "preflight":
                if (entry["case"], entry["n"]) not in {
                    ("gaussian", binding["sizes"][0]),
                    ("moons", binding["sizes"][1]),
                }:
                    continue
                spaces = REPLAY_SPACES
            for space in spaces:
                for k in NEIGHBORS:
                    result.append(
                        {"key": entry["key"], "space": space, "neighbors": k, "kind": kind}
                    )
    return result


def density_jobs(binding, phase):
    entries = binding["observations"]
    if phase == "preflight":
        entries = [e for e in entries if e["replicate"] == 0]
    elif phase != "main":
        raise ValueError("Unknown diagnostic phase")
    result = []
    special = {("gaussian", binding["sizes"][0]), ("moons", binding["sizes"][1])}
    for entry in entries:
        result.extend(
            {"key": entry["key"], "space": c + "_traits", "kind": "point"}
            for c in ("clean", "noisy")
        )
        if entry["replicate"] == 0 and (phase == "main" or (entry["case"], entry["n"]) in special):
            result.extend(
                {"key": entry["key"], "space": c + "_traits", "kind": "replay"}
                for c in ("clean", "noisy")
            )
        if entry["replicate"] == 0 and (entry["case"], entry["n"]) in special:
            result.extend(
                {"key": entry["key"], "space": s, "kind": "archive-replay"}
                for s in ("source", "noisy_density", "noisy_neural_0", "noisy_denoising_0")
            )
    return result
