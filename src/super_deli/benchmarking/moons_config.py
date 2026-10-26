"""Versioned diagnostic settings, with no changes to production model schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

ARMS = (
    "baseline",
    "no_single",
    "leaf",
    "samples_2",
    "samples_10",
    "cluster_3",
    "cluster_10",
    "center_only",
)
CASES = ("moons_008", "moons_002", "gaussian")


@dataclass(frozen=True)
class MoonsDiagnosticConfig:
    schema_version: int = 1
    datasets_per_cell: int = 25
    sample_sizes: tuple[int, ...] = (64, 256)
    main_seed: int = 129721
    preflight_seed: int = 129737
    data_seed: int = 129749
    artifact_cap_bytes: int = 268435456
    min_free_bytes: int = 8589934592

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported moons diagnostic schema")
        for f in fields(self):
            if f.name in {"sample_sizes", "schema_version"}:
                continue
            if type(getattr(self, f.name)) is not int or getattr(self, f.name) < 1:
                raise ValueError(f"{f.name} must be a positive integer")
        if (
            not isinstance(self.sample_sizes, tuple)
            or not self.sample_sizes
            or any(type(n) is not int or n < 32 or n % 8 for n in self.sample_sizes)
            or len(set(self.sample_sizes)) != len(self.sample_sizes)
        ):
            raise ValueError("Unique sizes divisible by eight and at least 32 required")
        if len({self.main_seed, self.preflight_seed, self.data_seed}) != 3:
            raise ValueError("Distinct seed roots required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or "schema_version" not in value
            or value.keys() - {f.name for f in fields(cls)}
        ):
            raise ValueError("Malformed moons diagnostic configuration")
        result = dict(value)
        if "sample_sizes" in result:
            if not isinstance(result["sample_sizes"], (list, tuple)):
                raise ValueError("sample_sizes must be an array")
            result["sample_sizes"] = tuple(result["sample_sizes"])
        return cls(**result)

    def fresh_plan(self, phase):
        if phase not in {"main", "preflight"}:
            raise ValueError("Unknown diagnostic phase")
        count = self.datasets_per_cell if phase == "main" else 1
        root = self.main_seed if phase == "main" else self.preflight_seed
        return [
            {
                "key": f"fresh_{case}_n{n}_{i:04d}",
                "origin": "fresh",
                "case": case,
                "n": n,
                "dataset": i,
                "seed": [self.data_seed, root, c, n, i],
            }
            for c, case in enumerate(CASES)
            for n in self.sample_sizes
            for i in range(count)
        ]


def arm_settings(arm):
    from dataclasses import replace

    from super_deli.models.hdbscan_config import HDBSCANConfig

    if arm not in ARMS:
        raise ValueError("Unknown frozen diagnostic arm")
    changes = {
        "no_single": {"allow_single_cluster": False},
        "leaf": {"cluster_selection_method": "leaf"},
        "samples_2": {"min_samples": 2},
        "samples_10": {"min_samples": 10},
        "cluster_3": {"min_cluster_size": 3},
        "cluster_10": {"min_cluster_size": 10},
    }
    return replace(HDBSCANConfig(True), **changes.get(arm, {})), (
        "center" if arm == "center_only" else "standard"
    )
