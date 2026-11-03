"""Invented, untrained engineering archive; not a scientific validation dataset."""

from pathlib import Path

import numpy as np

from super_deli.benchmarking.view_signal_config import REPRESENTATIONS
from super_deli.benchmarking.view_signal_coordinates import native_arm
from super_deli.benchmarking.view_signal_fit import fit_coordinates
from super_deli.io.config import write_json


def create_fixture(root):
    root = Path(root)
    root.mkdir(exist_ok=False)
    entries = []
    for i, phase in enumerate(("preflight", "main")):
        rng = np.random.default_rng(941 + i)
        truth = np.repeat(np.arange(4), 8)
        latent = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]])[truth] + rng.normal(0, 0.3, (32, 2))
        morph = np.column_stack((latent, latent[:, 0] + latent[:, 1], latent[:, 0] - latent[:, 1]))
        eco = np.column_stack((morph, morph)) + rng.normal(0, 0.2, (32, 8))
        m = (morph - morph.mean(0)) / morph.std(0)
        e = (eco - eco.mean(0)) / eco.std(0)
        context = {
            "morphology": morph,
            "ecology": eco,
            "latent_morphology": latent,
            "latent_ecology": latent + 0.2,
            "truth": truth,
            "unit_ids": np.array([f"invented_{j:04d}" for j in range(32)]),
        }
        for name in REPRESENTATIONS:
            if name == "morphology_standard":
                x = m
            elif name == "ecology_standard":
                x = e
            elif name == "balanced_traits":
                x = np.column_stack((m * np.sqrt(1.5), e * np.sqrt(0.75)))
            elif name == "fused_profiles":
                x = np.column_stack((m[:, :2], e[:, :2]))
            else:
                x = latent + rng.normal(0, 0.05, latent.shape)
            context[name] = x
        for name in REPRESENTATIONS:
            if native_arm(name):
                _, data = fit_coordinates(context[name], context)
                context[name + "_labels"] = data["labels"]
                context[name + "_strength"] = data["strength"]
        relative = phase + "/invented_n32_0000"
        folder = root / relative
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "context.npz", **context)
        entries.append(
            {
                "path": relative,
                "phase": phase,
                "task": {"case": "invented", "n": 32, "replicate": 0},
                "fixture": True,
            }
        )
    write_json(
        {"kind": "invented_untrained_fixture", "observations": entries}, root / "fixture.json"
    )
    return root
