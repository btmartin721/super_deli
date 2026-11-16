"""Installed invented input/refit contracts, not scientific performance estimates."""

import argparse
import json
from pathlib import Path


def smoke(output):
    from super_deli.benchmarking.bootstrap_split_study import parent
    from super_deli.io.config import read_json
    from super_deli.models.integrative_config import IntegrativeConfig

    output = Path(output)
    output.mkdir()
    generator = Path(__file__).resolve().parents[1] / "multimodal/synthetic.py"
    raw = parent(
        dict(n=64, law="shape_only", seed=8904, index=0), output / "raw", generator, draws=1
    )
    model = IntegrativeConfig.from_dict(read_json(output / "raw/observed/model.json"))
    head = parent(
        dict(n=64, law="Gaussian_score", seed=8905, index=0),
        output / "head",
        generator,
        head=True,
        draws=1,
        head_model=model,
    )
    assert raw["searches"] == head["searches"] == 4
    assert all(
        r["status"] == "complete" for p in (raw, head) for rows in p["arms"].values() for r in rows
    )
    print(
        json.dumps(
            dict(
                passed=True,
                fixture=True,
                scientific_decision=False,
                raw_searches=4,
                head_searches=4,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    smoke(parser.parse_args().output_dir)
