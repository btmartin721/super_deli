"""Joint ingestion across SNPio formats with optional population maps."""

# ruff: noqa: E402
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from super_deli.benchmarking.joint_study import simulate_joint
from super_deli.io.config import read_json
from super_deli.models.joint_config import JointStudyConfig
from super_deli.models.joint_null import write_raw_joint
from super_deli.workflows.joint_calibration import run_joint_calibration

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.genomic,
    pytest.mark.skipif(importlib.util.find_spec("snpio") is None, reason="Install genomics extra"),
]


@pytest.mark.parametrize("format", ["phylip", "vcf", "structure", "genepop"])
@pytest.mark.parametrize("popmap", [False, True])
def test_joint_readers_and_optional_popmaps(tmp_path, format, popmap):
    c = JointStudyConfig.from_dict(
        read_json(ROOT / "examples/joint-calibration/smoke.json")
    ).calibration
    values, layout, _ = simulate_joint("linear_association", 24, 331)
    dataset = write_raw_joint(values, layout, tmp_path / "raw")
    x, samples = values["genotypes"], layout["genomic"]["sample_ids"]
    path = tmp_path / "raw/genomic/source"
    if format == "phylip":
        path = tmp_path / "raw/genomic/genotypes.phy"
    elif format == "vcf":
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=1>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "\t".join(
                ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]
            ),
        ]
        lines += [
            "\t".join(
                [
                    "1",
                    str(j + 1),
                    ".",
                    "A",
                    "G",
                    ".",
                    "PASS",
                    ".",
                    "GT",
                    *[("0/0", "0/1", "1/1")[int(g)] for g in x[:, j]],
                ]
            )
            for j in range(128)
        ]
        path.write_text("\n".join(lines) + "\n")
    elif format == "structure":
        path.write_text(
            "\n".join(
                s + " " + " ".join(("1 1", "1 3", "3 3")[int(g)] for g in row)
                for s, row in zip(samples, x, strict=True)
            )
            + "\n"
        )
    else:
        lines = ["joint genotypes", *[f"locus_{j}" for j in range(128)], "Pop"]
        lines += [
            s + ", " + " ".join(("0101", "0103", "0303")[int(g)] for g in row)
            for s, row in zip(samples, x, strict=True)
        ]
        path.write_text("\n".join(lines) + "\n")
    popmap_path = tmp_path / "raw/genomic/source.popmap"
    if popmap:
        popmap_path.write_text("\n".join(s + " population" for s in samples) + "\n")
    source = replace(
        dataset.modalities[0],
        format=format,
        path=path.relative_to(tmp_path / "raw").as_posix(),
        popmapfile="genomic/source.popmap" if popmap else None,
        observations=tuple(reversed(dataset.modalities[0].observations)),
    )
    dataset = replace(dataset, modalities=(source, *dataset.modalities[1:]))
    result = run_joint_calibration(
        dataset,
        config=replace(c, n_simulations=1),
        base_dir=tmp_path / "raw",
        output_dir=tmp_path / "fit",
    )
    assert result["successful_fits"] == 2
    null = read_json(tmp_path / "fit/layout.json")
    assert null["genomic"]["sample_ids"] == samples
    assert null["unit_ids"] == layout["unit_ids"]
