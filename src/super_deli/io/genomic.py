"""Load one declared genomic source through SNPio, with isolated input copies."""

from __future__ import annotations

import csv
import logging
import shutil
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from super_deli.core.data import DatasetConfig, GenomicReaderOptions, Modality
from super_deli.core.provenance import collect_provenance, fingerprint_file
from super_deli.core.schema import config_from_dict
from super_deli.core.validation import resolve_input, validate_dataset
from super_deli.io.config import write_json

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from snpio.read_input.genotype_data import GenotypeData

_READER_LOCK = RLock()


class GenomicIngestionError(ValueError):
    """A genomic source cannot be safely loaded or linked to its declaration."""


@dataclass(frozen=True)
class LoadedGenomicData:
    """An SNPio object plus immutable sample-link snapshots and a missing-call mask.

    Keep output_dir for as long as SNPio's VCF metadata/cache files are needed.
    The SNPio object itself remains mutable; the mask describes ingestion time.
    """

    genotype_data: GenotypeData
    sample_ids: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    missing_mask: NDArray[np.bool_]
    output_dir: Path
    summary: dict


def _snpio_loggers() -> dict[str, logging.Logger]:
    return {
        name: logger
        for name, logger in list(logging.Logger.manager.loggerDict.items())
        if isinstance(logger, logging.Logger) and (name == "snpio" or name.startswith("snpio."))
    }


@contextmanager
def _reader_logging():
    """Scope SNPio's process-global handlers to this serialized ingestion call.

    Existing caller handlers are detached and restored without closing them.
    Only handlers created during this call are closed. No cwd or environment
    variables are changed. Other code using SNPio concurrently needs its own
    process because SNPio's logger configuration is global.
    """
    with _READER_LOCK:
        existing = {
            name: (tuple(logger.handlers), logger.level, logger.propagate)
            for name, logger in _snpio_loggers().items()
        }
        for logger in _snpio_loggers().values():
            for handler in tuple(logger.handlers):
                logger.removeHandler(handler)
        try:
            yield
        finally:
            for name, logger in _snpio_loggers().items():
                handlers, level, propagate = existing.get(name, ((), logging.NOTSET, True))
                for handler in tuple(logger.handlers):
                    logger.removeHandler(handler)
                    if handler not in handlers:
                        handler.close()
                for handler in handlers:
                    logger.addHandler(handler)
                logger.setLevel(level)
                logger.propagate = propagate


def _dependencies() -> tuple[dict, str]:
    try:
        from packaging.specifiers import SpecifierSet

        installed = version("snpio")
        if installed not in SpecifierSet(">=1.7.7,<2"):
            raise GenomicIngestionError(
                f"SNPio {installed} is outside the supported >=1.7.7,<2 range"
            )
        from snpio import GenePopReader, PhylipReader, StructureReader, VCFReader
    except (ImportError, PackageNotFoundError) as exc:
        raise GenomicIngestionError(
            "Genomic ingestion requires the optional dependencies: "
            "pip install 'super-deli[genomics]'"
            f" ({exc})"
        ) from exc

    class PreservingGenePopReader(GenePopReader):
        """Retain the labels SNPio parses from GENEPOP sections before its fallback."""

        def read_popmap(self):
            embedded = tuple(self.populations) if self.popmapfile is None else None
            super().read_popmap()
            if embedded:
                self.populations = list(embedded)
                self.popmap = dict(zip(self.samples, embedded, strict=True))
                inverse = {}
                for sample, population in self.popmap.items():
                    inverse.setdefault(population, []).append(sample)
                self.popmap_inverse = inverse

    return {
        "vcf": VCFReader,
        "phylip": PhylipReader,
        "structure": StructureReader,
        "genepop": PreservingGenePopReader,
    }, installed


def _select_modality(config: DatasetConfig, name: str | None) -> Modality:
    candidates = [
        m for m in config.modalities if m.kind == "genomic" and (name is None or m.name == name)
    ]
    if len(candidates) != 1:
        raise GenomicIngestionError(
            "Select exactly one genomic modality with --modality or modality_name"
        )
    return candidates[0]


def _stage(source: Path, directory: Path) -> tuple[Path, dict]:
    directory.mkdir()
    staged = directory / source.name
    shutil.copyfile(source, staged)
    original = fingerprint_file(source)
    snapshot = fingerprint_file(staged)
    if original.sha256 != snapshot.sha256:
        raise GenomicIngestionError(f"Input changed while staging: {source}")
    return staged, asdict(original)


def _validate_popmap(path: Path, expected_samples: set[str]) -> None:
    """Reject duplicate or incomplete metadata before SNPio's dictionary conversion."""
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = (
        list(csv.reader(lines, skipinitialspace=True))
        if any("," in line for line in lines)
        else [line.split() for line in lines]
    )
    if not rows or any(len(row) != 2 for row in rows):
        raise GenomicIngestionError("Popmaps require two nonempty columns")
    sample_headers = {
        "sample",
        "samples",
        "sampleid",
        "sampleids",
        "samp",
        "samps",
        "sampid",
        "sampids",
    }
    pop_headers = {
        "population",
        "populations",
        "populationid",
        "populationids",
        "pop",
        "pops",
        "popid",
        "popids",
    }
    header = [field.lower() for field in rows[0]]
    sample_column = 0
    if set(header) & sample_headers and set(header) & pop_headers:
        sample_column = next(i for i, field in enumerate(header) if field in sample_headers)
        rows = rows[1:]
    samples = [row[sample_column] for row in rows]
    if any(not value or value != value.strip() for row in rows for value in row):
        raise GenomicIngestionError("Popmap identifiers must be nonempty and unpadded")
    if len(samples) != len(set(samples)):
        raise GenomicIngestionError("Duplicate sample IDs in popmap")
    if set(samples) != expected_samples:
        raise GenomicIngestionError(
            "Popmap sample IDs must exactly match the selected source declaration"
        )


def _options(modality: Modality) -> dict:
    declared = modality.reader_options or GenomicReaderOptions()
    result = {}
    if modality.format == "structure":
        result.update(
            has_popids=declared.has_popids,
            has_marker_names=declared.has_marker_names,
            allele_start_col=declared.allele_start_col,
        )
    if declared.allele_encoding:
        result["allele_encoding"] = dict(declared.allele_encoding)
    if modality.format == "vcf":
        result["disable_progress_bar"] = True
    return result


def ingest_genomic(
    config: DatasetConfig,
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    modality_name: str | None = None,
) -> LoadedGenomicData:
    """Load all samples in one source without filtering, reordering, or imputation.

    Configuration observation IDs must exactly match the source's sample IDs.
    Their order may differ; specimen links are returned in source sample order.
    All writes go to a new output directory. Failures after creation retain a
    failure.json record and staged diagnostics; an existing directory is rejected.
    """
    config = config_from_dict(config.to_dict())
    report = validate_dataset(config, base_dir=base_dir, check_files=False)
    if not report.valid:
        errors = "; ".join(
            f"{i.location}: {i.message}" for i in report.issues if i.severity == "error"
        )
        raise GenomicIngestionError(errors)
    modality = _select_modality(config, modality_name)
    original = resolve_input(modality.path, base_dir)
    original_popmap = (
        resolve_input(modality.popmapfile, base_dir) if modality.popmapfile is not None else None
    )
    for path in (original, original_popmap):
        if path is not None and not path.is_file():
            raise GenomicIngestionError(f"Not a regular input file: {path}")
    destination = Path(output_dir).absolute()
    with _reader_logging():
        readers, snpio_version = _dependencies()
        import numpy as np

        destination.mkdir(exist_ok=False)
        try:
            staged, source_fingerprint = _stage(original, destination / "input")
            staged_popmap = None
            fingerprints = [source_fingerprint]
            if original_popmap is not None:
                staged_popmap, popmap_fingerprint = _stage(original_popmap, destination / "popmap")
                _validate_popmap(staged_popmap, {o.observation_id for o in modality.observations})
                fingerprints.append(popmap_fingerprint)
            data = readers[modality.format](
                filename=str(staged),
                popmapfile=str(staged_popmap) if staged_popmap is not None else None,
                force_popmap=False,
                prefix=str(destination / "snpio"),
                save_plots=False,
                verbose=False,
                **_options(modality),
            )
            calls = data.snp_data
            samples = tuple(data.samples)
            duplicates = [name for name, n in Counter(samples).items() if n > 1]
            if duplicates:
                raise GenomicIngestionError(f"Duplicate source sample IDs: {duplicates}")
            links = {o.observation_id: o.unit_id for o in modality.observations}
            missing, unexpected = (
                sorted(set(links) - set(samples)),
                sorted(set(samples) - set(links)),
            )
            if missing or unexpected:
                raise GenomicIngestionError(
                    f"Source/declaration mismatch: missing samples={missing}; "
                    f"unexpected samples={unexpected}"
                )
            if calls.ndim != 2 or calls.shape[0] != len(samples) or not calls.size:
                raise GenomicIngestionError(
                    "SNPio returned an empty or inconsistent genotype matrix"
                )
            specimens = tuple(links[sample] for sample in samples)
            populations = _population_metadata(config, modality, data, samples, specimens)
            mask = np.isin(np.char.upper(calls), ("N", "-", ".", "?"))
            mask.flags.writeable = False
            np.save(destination / "genotypes.npy", calls, allow_pickle=False)
            np.save(destination / "missing_mask.npy", mask, allow_pickle=False)
            provenance = collect_provenance(config, base_dir=base_dir)
            provenance.update(input_hash_status="computed_selected_modality", inputs=fingerprints)
            provenance["software"]["snpio"] = snpio_version
            summary = {
                "schema_version": 1,
                "status": "complete",
                "dataset_id": config.dataset_id,
                "modality": modality.name,
                "format": modality.format,
                "sample_ids": samples,
                "specimen_ids": specimens,
                "unique_specimen_count": len(set(specimens)),
                "sample_count": len(samples),
                "locus_count": calls.shape[1],
                "missing_call_count": int(mask.sum()),
                "population_metadata": populations,
                "genotype_encoding": "SNPio IUPAC",
                "provenance": provenance,
            }
            write_json(summary, destination / "summary.json")
            return LoadedGenomicData(data, samples, specimens, mask, destination, summary)
        except Exception as exc:
            write_json({"status": "failed", "error": str(exc)}, destination / "failure.json")
            if isinstance(exc, GenomicIngestionError):
                raise
            raise GenomicIngestionError(f"SNPio ingestion failed: {exc}") from exc


def _population_metadata(config, modality, data, samples, specimens) -> dict:
    options = modality.reader_options or GenomicReaderOptions()
    source = (
        "popmap"
        if modality.popmapfile is not None
        else "embedded"
        if modality.format == "genepop" or options.has_popids
        else "none"
    )
    labels = tuple(str(data.popmap[sample]) for sample in samples) if source != "none" else None
    declared = {s.specimen_id: s.population_id for s in config.specimens}
    if labels is not None:
        for sample, specimen, population in zip(samples, specimens, labels, strict=True):
            if declared[specimen] is not None and declared[specimen] != population:
                raise GenomicIngestionError(f"Population metadata conflict for sample {sample!r}")
    return {"source": source, "labels_in_sample_order": labels}
