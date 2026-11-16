"""Read only the sealed preceding study and stage exact native replay inputs."""

import shutil
from pathlib import Path

from super_deli.benchmarking.signal_loss_source import SealedStudy, digest
from super_deli.core.schema import config_from_dict
from super_deli.io.config import read_json
from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig


class IndividualSubsampleArchive(SealedStudy):
    source_head = "bf7e801932fbe20c8412061b7f69e66abd3b4fec"
    source_version = "0.85.0.dev0"

    def stage(self, index, directory):
        if self.config.fixture:
            raise ValueError("Invented seals cannot supply scientific replays")
        prefix = Path(f"validation/main/replay-{index:03d}-inputs")
        directory = Path(directory)
        directory.mkdir()
        for name in sorted(self.inventory):
            if not name.startswith(prefix.as_posix() + "/"):
                continue
            source = self.path(name)
            target = directory / Path(name).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert digest(source) == digest(target)
        return config_from_dict(
            read_json(directory / "dataset.json")
        ), MultimodalAnalysisConfig.from_dict(read_json(directory / "analysis.json"))

    def baseline(self, index):
        return Path(f"validation/main/replay-{index:03d}")
