"""Reuse the sealed-artifact reader for the completed signal-loss archive."""

import shutil
from pathlib import Path

from super_deli.benchmarking.full_retention_config import SOURCE_HEAD, SOURCE_VERSION
from super_deli.benchmarking.signal_loss_source import SealedStudy, digest
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig


class RetentionArchive(SealedStudy):
    source_head = SOURCE_HEAD
    source_version = SOURCE_VERSION

    def stage(self, index, directory):
        if self.config.fixture:
            raise ValueError("Invented archive fixtures cannot supply scientific observations")
        prefix = Path(f"study/main/fresh-{index:02d}-inputs")
        directory = Path(directory)
        directory.mkdir()
        for name in sorted(self.inventory):
            if name.startswith(prefix.as_posix() + "/"):
                source = self.path(name)
                relative = Path(name).relative_to(prefix)
                destination = directory / relative
                destination.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(source, destination)
                if digest(source) != digest(destination):
                    raise ValueError("Archived input snapshot changed")
        return (
            config_from_dict(read_json(directory / "dataset.json")),
            MultimodalAnalysisConfig.from_dict(read_json(directory / "analysis.json")),
        )

    def baseline(self, index):
        return Path(f"study/main/fresh-{index:02d}/native/combined")

    def verify_numeric(self, prefix):
        for name in self.inventory:
            if name.startswith(str(prefix) + "/") and name.endswith((".npy", ".npz")):
                if not Path(name).name.startswith(("stability", "coassignment")):
                    self.path(name)
