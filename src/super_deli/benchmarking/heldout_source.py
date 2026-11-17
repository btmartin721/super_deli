"""Read sealed native source evidence and stage consumed point snapshots only."""

import shutil
from pathlib import Path

from super_deli.benchmarking.signal_loss_source import SealedStudy, digest
from super_deli.io.individual_subsample_source import load_point_source


class HeldoutArchive(SealedStudy):
    source_head = "f41e0c6cab8179a6d7df82d3c36383ed3535fbee"
    source_version = "0.86.0.dev0"

    def case(self, index):
        if self.config.fixture:
            raise ValueError("Invented seals cannot authorize scientific input")
        prefix = f"validation/main/fresh-{index:03d}"
        if not 0 <= index < 48:
            raise ValueError("Archived source observation index required")
        # Bind only consumed evidence; never mutate or run code from the source archive.
        for name in sorted(self.inventory):
            if name.startswith(prefix + "/") and not name.endswith("stability.npz"):
                if any(
                    name.startswith(prefix + "/" + part)
                    for part in ("subsample/", "evaluation/", "evaluation-ingestion/", "observed/")
                ):
                    self.path(name)
        return self.root / prefix

    def stage_point(self, index, directory):
        case = self.case(index)
        source = load_point_source(case / "ordinary")
        directory = Path(directory)
        directory.mkdir()
        for name, sha in source.files.items():
            old = self.path(f"validation/main/fresh-{index:03d}/ordinary/{name}")
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, target)
            assert digest(target) == sha
        source.reverify()
        return source
