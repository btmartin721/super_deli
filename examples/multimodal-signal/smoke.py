"""Invented installed point/CLI smoke; fixture source cannot yield scientific decisions."""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


def fixture_source(result, truth, root):
    from super_deli.io.config import write_json

    root = Path(root)
    root.mkdir()
    destination = root / "validation/main/00-shape_only-analysis/arms"
    shutil.copytree(Path(result) / "native", destination)
    labels = root / "validation/main/00-shape_only-source"
    labels.mkdir()
    shutil.copyfile(truth, labels / "truth-evaluation-only.json")
    write_json(
        dict(passed=True, observations=1, records=[dict(law="shape_only")] * 5),
        root / "validation/main/summary.json",
    )
    inventory = {
        str(p.relative_to(root)): dict(
            bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        )
        for p in (root / "validation").rglob("*")
        if p.is_file()
    }
    write_json(inventory, root / "study-manifest.json")
    write_json(
        dict(completed=True, fixture=True, head_sha="functional_fixture", version="fixture"),
        root / "completion.json",
    )


def smoke(output):
    from super_deli.benchmarking.signal_loss_config import SignalLossConfig
    from super_deli.benchmarking.signal_loss_run import run_case
    from super_deli.cli import main
    from super_deli.io.config import read_json, write_json

    output = Path(output)
    output.mkdir()
    path = Path(__file__).resolve().parents[1] / "multimodal/synthetic.py"
    spec = importlib.util.spec_from_file_location("signal_fixture_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = output / "inputs"
    dataset, analysis = module.invented(source, law="shape_only", seed=63101, draws=0)
    result = run_case(
        dataset,
        analysis,
        base_dir=source,
        directory=output / "point",
        truth_path=source / "truth-evaluation-only.json",
        law="shape_only",
        origin="invented_fixture",
    )
    assert (
        all(r["status"] == "complete" for r in result["native"]) and len(result["variants"]) == 13
    )
    fixture = output / "fixture"
    fixture_source(output / "point", source / "truth-evaluation-only.json", fixture)
    config = SignalLossConfig(
        hashlib.file_digest((fixture / "completion.json").open("rb"), "sha256").hexdigest(),
        hashlib.file_digest((fixture / "study-manifest.json").open("rb"), "sha256").hexdigest(),
        fixture=True,
    )
    write_json(config.to_dict(), output / "config.json")
    status = main(
        [
            "diagnose-multimodal-signal",
            str(output / "config.json"),
            "--source",
            str(fixture),
            "--output-dir",
            str(output / "audit"),
            "--phase",
            "audit",
        ]
    )
    audited = read_json(output / "audit/summary.json")
    assert (
        status == 0
        and audited["fixture"]
        and audited["fit_free"]
        and audited["decision"] == "not_evaluated"
    )
    print(
        json.dumps(
            dict(passed=True, fixture=True, planned_point_searches=15, scientific_decision=False)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
