"""Prespecified Gaussian validation; resource options never change the design."""

from dataclasses import asdict, dataclass, fields

CASES = ("gaussian_independent", "gaussian_correlated")
ARMS = ("neural", "pca4")
NAMESPACE = (20261108, 690)


@dataclass(frozen=True)
class NeuralFalseSplitStudy:
    schema_version: int = 1
    fixture: bool = False
    artifact_cap_bytes: int = 24 * 1024**3
    min_free_bytes: int = 8 * 1024**3
    max_projected_seconds: int = 40 * 3600
    preflight_max_seconds: int = 2 * 3600
    max_workers: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use false-split study schema 1")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be boolean")
        for name, ceiling in (
            ("artifact_cap_bytes", 24 * 1024**3),
            ("max_projected_seconds", 40 * 3600),
            ("preflight_max_seconds", 2 * 3600),
            ("max_workers", 2),
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= ceiling:
                raise ValueError(name + " exceeds the approved ceiling")
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 8 * 1024**3:
            raise ValueError("Eight GiB free-space reserve required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown false-split study fields")
        return cls(**value)


def make_task(phase, law, replicate, *, n, draws, key):
    names = ("data_seed", "noise_seed", "permutation_seed", "ecology_order_seed", "null_seed_path")
    return dict(
        case=CASES[law],
        n=n,
        replicate=replicate,
        n_simulations=draws,
        key=key,
        **{name: [*NAMESPACE, phase, law, replicate, i] for i, name in enumerate(names)},
    )


def tasks(config, phase):
    if phase not in ("preflight", "main"):
        raise ValueError("Unknown phase")
    n = 32 if config.fixture else 64
    repeats = 1 if config.fixture else (2 if phase == "preflight" else 200)
    out = [
        make_task(
            int(phase == "main"),
            law,
            i,
            n=n,
            draws=2 if config.fixture else 99,
            key=f"{case}_{i:04d}",
        )
        for i in range(repeats)
        for law, case in enumerate(CASES)
    ]
    if phase == "preflight":
        out.append(
            make_task(
                0, 0, repeats, n=n, draws=3 if config.fixture else 999, key="high_draw_preflight"
            )
        )
    return out


def batch_count(config):
    return 1 if config.fixture else 8


def batch_tasks(config, index):
    if type(index) is not int or not 0 <= index < batch_count(config):
        raise ValueError("Unknown fixed batch")
    width = 1 if config.fixture else 25
    return [t for t in tasks(config, "main") if t["replicate"] // width == index]


def counts(config, phase):
    raw = sum(1 + t["n_simulations"] for t in tasks(config, phase))
    replay = (1 if config.fixture else 2) if phase == "main" else 0
    if phase == "main":
        raw += replay * (3 if config.fixture else 100) + (4 if config.fixture else 1000)
    return dict(
        observations=len(tasks(config, phase)),
        raw_inputs=raw,
        legacy_replays=replay,
        sensitivity_tests=int(phase == "main"),
        workflows=2 * raw,
        candidate_searches=2 * raw,
        neural_fits=2 * raw,
        verified_updates=1000 * raw,
    )


def archived_tasks(config, binding):
    out = []
    for entry in binding["replays"]:
        t = dict(entry["task"], key=entry["key"])
        out.append(dict(kind="replay", task=t, source_path=entry["path"]))
    entry = binding["replays"][0]
    t = dict(entry["task"], key="archived_precision", n_simulations=3 if config.fixture else 999)
    t["null_seed_path"] = [*NAMESPACE, 2, 0, entry["task"]["replicate"], 4]
    out.append(dict(kind="sensitivity", task=t, source_path=entry["path"]))
    return out
