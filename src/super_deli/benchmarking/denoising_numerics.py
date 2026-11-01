"""Read-only NumPy gradient/Adam and independently regenerated corruption audits."""

import hashlib

import numpy as np

from super_deli.benchmarking.neural_analysis import neural_forward, require


def checkpoint(arrays, update):
    prefix = f"checkpoint_{update}_"
    return {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}


def reconstruction(x, weights):
    return neural_forward(neural_forward(x, weights, "encoder"), weights, "decoder")


def gradients(x, target, weights):
    """Independent row-vector backpropagation, with no torch/autograd calls."""
    layers = [(side, i) for side in ("encoder", "decoder") for i in range(3)]
    values = [x]
    for side, i in layers:
        v = values[-1] @ weights[f"{side}_{i}_weight"].T + weights[f"{side}_{i}_bias"]
        values.append(np.tanh(v) if i != 2 else v)
    derivative = 2 * (values[-1] - target) / target.size
    result = {}
    for index in reversed(range(6)):
        side, i = layers[index]
        if i != 2:
            derivative = derivative * (1 - values[index + 1] ** 2)
        result[f"{side}_{i}_weight"] = derivative.T @ values[index]
        result[f"{side}_{i}_bias"] = derivative.sum(axis=0)
        derivative = derivative @ weights[f"{side}_{i}_weight"]
    return result


def audit_training(x, arrays, seed):
    """Regenerate noise only, never retrain or rerun a density fit."""
    import torch

    require(arrays["corrupted_objectives"].shape == (500,), "Corrupted objective history")
    require(arrays["corruption_sha256"].shape == (500,), "Corruption digest history")
    require(arrays["losses"].shape == (501,), "Clean objective history")
    require(
        np.isfinite(arrays["losses"]).all() and np.isfinite(arrays["corrupted_objectives"]).all(),
        "Finite training objectives",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = None
    for i in range(500):
        noise = (
            torch.randn(x.shape, generator=generator, dtype=torch.float64, device="cpu").numpy()
            * 0.15
        )
        require(
            hashlib.sha256(noise.tobytes()).hexdigest() == arrays["corruption_sha256"][i],
            "Corruption stream digest",
        )
        if i == 0:
            first = noise.copy()
        if i in (0, 1, 499):
            weights = checkpoint(arrays, i)
            objective = np.mean((reconstruction(x + noise, weights) - x) ** 2)
            np.testing.assert_allclose(objective, arrays["corrupted_objectives"][i], atol=1e-12)
    for update in (0, 1, 499, 500):
        weights = checkpoint(arrays, update)
        require(len(weights) == 12, "Complete numeric checkpoint")
        np.testing.assert_allclose(
            np.mean((reconstruction(x, weights) - x) ** 2), arrays["losses"][update], atol=1e-12
        )
    initial = checkpoint(arrays, 0)
    gradient = gradients(x + first, x, initial)
    after = checkpoint(arrays, 1)
    for key, parameter in initial.items():
        g = gradient[key] + 0.0001 * parameter
        expected = parameter - 0.001 * g / (np.abs(g) + 1e-8)
        np.testing.assert_allclose(after[key], expected, rtol=1e-11, atol=2e-14)
    for key, value in checkpoint(arrays, 500).items():
        np.testing.assert_array_equal(value, arrays[key])
    return {
        "corruption_matrices_checked": 500,
        "independent_first_adam_update": True,
        "checkpoint_clean_objectives_checked": 4,
        "checkpoint_corrupted_objectives_checked": 3,
    }
