"""Explicit reporting policy; independent of legacy model and sampling schemas."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class HeldoutSupportConfig:
    schema_version: int = 1
    policy: str = "native_k1_omitted_v1"
    minimum_coverage: float = 0.95

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Use held-out support schema1")
        if self.policy != "native_k1_omitted_v1":
            raise ValueError("Use the prespecified native K=1 omitted-individual policy")
        if type(self.minimum_coverage) is not float or self.minimum_coverage != 0.95:
            raise ValueError("This reporting policy requires coverage 0.95")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Supply exact held-out support configuration fields")
        return cls(**value)
