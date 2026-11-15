"""Sealed reads of the completed full-retention study; no source mutations."""

import shutil
from pathlib import Path

from super_deli.benchmarking.signal_loss_source import SealedStudy, digest
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig


class InitializationArchive(SealedStudy):
    source_head = "338399a83386058290a52819fca66ecc5708d5c0"
    source_version = "0.83.0.dev0"

    def stage(self, index, directory):
        if self.config.fixture:
            raise ValueError("Invented fixtures cannot supply scientific observations")
        prefix = Path(f"validation/main/fresh-{index:03d}-inputs")
        directory = Path(directory)
        directory.mkdir()
        for name in sorted(self.inventory):
            if name.startswith(prefix.as_posix() + "/"):
                source = self.path(name)
                destination = directory / Path(name).relative_to(prefix)
                destination.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(source, destination)
                assert digest(source) == digest(destination)
        return config_from_dict(
            read_json(directory / "dataset.json")
        ), MultimodalAnalysisConfig.from_dict(read_json(directory / "analysis.json"))

    def baseline(self, index, arm):
        prefix = Path(f"validation/main/fresh-{index:03d}") / (
            "hundred_starts" if arm == "legacy100" else "retained_all"
        )
        return prefix

    def verify_numeric(self, prefix):
        for name in self.inventory:
            # Replays are point-only; snapshots of old bootstrap events are not compared.
            relative = (
                Path(name).relative_to(prefix) if name.startswith(str(prefix) + "/") else None
            )
            if (
                relative is not None
                and (relative.parts[0] not in {"observed-candidates", "observed-pipeline"})
                and name.endswith((".npz", ".npy"))
                and not relative.name.startswith(("stability", "coassignment"))
            ):
                self.path(name)
