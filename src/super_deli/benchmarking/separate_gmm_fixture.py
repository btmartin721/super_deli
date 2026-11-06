"""Explicit invented engineering source; never a scientific validation replicate."""

import numpy as np

from super_deli.benchmarking.matched_data import write_dataset
from super_deli.benchmarking.separate_gmm_config import SharedEncoderStudy, model_for, tasks
from super_deli.benchmarking.separate_gmm_data import generate
from super_deli.io.config import write_json
from super_deli.workflows.separate_neural import run_separate_neural


def create_fixture(source):
    source.mkdir()
    folder = source / "legacy"
    folder.mkdir()
    task = {
        **tasks(SharedEncoderStudy(fixture=True), "preflight")[0],
        "draws": 0,
        "data_seed": [112011, 32],
        "noise_seed": [112019, 32],
        "permutation_seed": [112027, 32],
        "ecology_order_seed": [112027, 32, 1],
        "subsampling_path": [112031, 32],
    }
    data = generate(task)
    np.savez_compressed(folder / "generating-reference.npz", **data)
    dataset = write_dataset(data, folder / "raw")
    write_dataset(data, folder / "canonical", canonical=True)
    write_json(task, folder / "task.json")
    seed = int(np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0])
    run_separate_neural(
        dataset,
        model_config=model_for("separate_0", seed).encoder_config(),
        base_dir=folder / "raw",
        output_dir=folder / "separate_0",
    )
    write_json(
        dict(
            kind="invented_shared_encoder_source",
            engineering_branch_trainings=2,
            replays=[
                dict(key="legacy_fixture_seed0", path="legacy", task=task, seed=0, arm="separate_0")
            ],
        ),
        source / "fixture.json",
    )
