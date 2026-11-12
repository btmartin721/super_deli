"""Dependency-free, fixed archival diagnostic and resource contract."""

from dataclasses import asdict, dataclass, fields

SOURCE_HEAD = "cf6082d383f15a5b058c49585f5b4bbbfa4e0aa6"
ARMS = ("neural", "pca4")
NUMERICAL = ("numpy", "scipy", "scikit-learn", "threadpoolctl")


@dataclass(frozen=True)
class WeakSignalStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 1800
    preflight_max_seconds: int = 600
    max_workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use weak-signal schema 1")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be boolean")
        for key, ceiling in (
            ("artifact_cap_bytes", 1024**3),
            ("max_projected_seconds", 1800),
            ("preflight_max_seconds", 600),
            ("max_workers", 2),
        ):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= ceiling:
                raise ValueError(key + " exceeds approved bounds")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight-GiB free reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown study fields")
        return cls(**value)


def tasks(entries, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Known phase required")
    result = []
    for entry in entries:
        t = entry["task"]
        if phase == "preflight" and t["replicate"] != 0:
            continue
        signal = t["case"] == "four_complementary"
        replay = signal and t["signal_multiplier"] == 0.5 and t["replicate"] < 5
        result.append({**t, "diagnostic_replay": replay, "truth_fits": 2 if signal else 0})
    return result


def counts_from_tasks(grid):
    searches = sum(t["diagnostic_replay"] * 2 for t in grid)
    truth = sum(t["truth_fits"] for t in grid)
    return dict(
        observations=len(grid),
        gmm_searches=searches,
        truth_diagnostics=truth,
        maximum_component_fits=4 * searches + truth,
        maximum_em_starts=40 * searches + truth,
        reference_evaluations=3 * len(grid),
        workflows=4 * searches + truth,
        legacy_replays=0,
        neural_fits=0,
        verified_updates=0,
    )
