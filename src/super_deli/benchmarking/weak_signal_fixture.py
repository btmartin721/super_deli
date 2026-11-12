"""Invented numeric source; random frozen encoders are never trained."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.coordinate_fit import mixture
from super_deli.benchmarking.neural_analysis import neural_forward
from super_deli.benchmarking.neural_calibration_config import model_for
from super_deli.benchmarking.neural_false_split_state import atomic_json
from super_deli.benchmarking.neural_power_config import NeuralPowerStudy, tasks
from super_deli.benchmarking.neural_power_data import reference
from super_deli.benchmarking.separate_controls import fit_pca_four


def create(source):
    source = Path(source)
    source.mkdir()
    entries = []
    for task in tasks(NeuralPowerStudy(), "preflight")[:8]:
        t = {**task, "n": 32 if task["n"] == 64 else 64}
        for name in (
            "data_seed",
            "noise_seed",
            "permutation_seed",
            "ecology_order_seed",
            "null_seed_path",
        ):
            t[name] = [20261112, 711, *task[name][2:]]
        out = source / t["key"]
        out.mkdir()
        data = reference(t)
        np.savez_compressed(out / "generating-reference.npz", **data)
        x = np.column_stack((data["morphology"], data["ecology"]))
        pca = fit_pca_four(
            x,
            config=model_for("pca4", 0).model,
            clusterer=lambda scores, config: (
                np.zeros(len(scores), dtype=int),
                np.ones(len(scores)),
            ),
        )
        point = {**pca.model.arrays, "scores": pca.scores}
        neural = {}
        for j, (role, view) in enumerate((("morphology", x[:, :4]), ("ecology", x[:, 4:]))):
            rng = np.random.default_rng(419 + j)
            b = dict(means=view.mean(0), scales=view.std(0), kept_indices=np.arange(view.shape[1]))
            for layer, (left, right) in enumerate(
                zip((view.shape[1], 16, 8), (16, 8, 2), strict=True)
            ):
                b[f"encoder_{layer}_weight"] = rng.normal(0, 0.2, (right, left))
                b[f"encoder_{layer}_bias"] = rng.normal(0, 0.1, right)
            z = (view - b["means"]) / b["scales"]
            raw = neural_forward(z, b, "encoder")
            b.update(
                latent_means=raw.mean(0),
                latent_scales=raw.std(0),
                scores=(raw - raw.mean(0)) / raw.std(0),
            )
            neural.update({role + "__" + k: v for k, v in b.items()})
        neural["scores"] = np.column_stack(
            [neural[r + "__scores"] for r in ("morphology", "ecology")]
        )
        for arm, values in (("neural", neural), ("pca4", point)):
            np.savez_compressed(out / (arm + "-point.npz"), **values)
            folder = out / arm
            folder.mkdir()
            meta, saved = mixture(values["scores"], "tied")
            atomic_json(folder / "execution.json", meta)
            np.savez_compressed(folder / "fit.npz", **saved)
        entries.append(dict(path=t["key"], task=t))
    atomic_json(
        source / "fixture.json",
        dict(
            kind="invented_weak_signal_source_v1",
            entries=entries,
            neural_training=False,
            interpretation="Engineering only, untrained random encoders",
        ),
    )
