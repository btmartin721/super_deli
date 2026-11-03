"""Read-only, independently differentiated checks on each actual Adam update."""

import hashlib
import math
from contextlib import contextmanager

import numpy as np
from threadpoolctl import threadpool_limits

from super_deli.benchmarking.denoising_numerics import gradients
from super_deli.benchmarking.neural_analysis import require
from super_deli.models.mixture import FIT_LOCK

TOLERANCES = {
    "gradient": (1e-10, 1e-12),
    "moment": (1e-11, 1e-13),
    "parameter": (2e-11, 2e-12),
}


def digest_arrays(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        a = np.asarray(value)
        digest.update(str((key, a.shape, a.dtype.str)).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def compare(actual, expected, kind):
    np.testing.assert_allclose(actual, expected, rtol=TOLERANCES[kind][0], atol=TOLERANCES[kind][1])
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Finite update evidence")
    return float(np.max(np.abs(actual - expected)))


def adam_update(parameter, gradient, first, second, update):
    """Independent coupled-decay Adam equations, without torch/autograd."""
    g = gradient + 0.0001 * parameter
    m = first + (1 - 0.9) * (g - first)
    v = 0.999 * second + (1 - 0.999) * g * g
    denominator = np.sqrt(v) / math.sqrt(1 - 0.999**update) + 1e-8
    result = parameter - (0.001 / (1 - 0.9**update)) * m / denominator
    return result, m, v


class StepChecks:
    def __init__(self, x, initial):
        self.x, self.initial = x, initial
        self.errors, self.hashes = [], []
        self.before = None

    def pre(self, optimizer, args, kwargs):
        require(len(optimizer.param_groups) == 1, "Single fixed Adam parameter group")
        group = optimizer.param_groups[0]
        require(
            group["lr"] == 0.001
            and group["betas"] == (0.9, 0.999)
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0.0001
            and not group["amsgrad"]
            and not group["maximize"]
            and group["foreach"] is False
            and group["fused"] is False
            and not group.get("decoupled_weight_decay", False),
            "Unchanged fixed Adam settings",
        )
        params = group["params"]
        require(len(params) == len(self.initial) == 12, "Ordered network parameters")
        names = tuple(self.initial)
        values = {k: p.detach().numpy().copy() for k, p in zip(names, params, strict=True)}
        reference = gradients(self.x, self.x, values)
        error = 0.0
        expected, states = {}, {}
        current = {"parameters": values}
        for key, p in zip(names, params, strict=True):
            require(p.grad is not None, "Every parameter has a raw gradient")
            gradient = p.grad.detach().numpy().copy()
            error = max(error, compare(gradient, reference[key], "gradient"))
            state = optimizer.state[p]
            require(int(state.get("step", 0)) == len(self.errors), "Contiguous Adam state")
            first = (
                state["exp_avg"].detach().numpy().copy() if state else np.zeros_like(values[key])
            )
            second = (
                state["exp_avg_sq"].detach().numpy().copy() if state else np.zeros_like(values[key])
            )
            expected[key] = adam_update(
                values[key], reference[key], first, second, len(self.errors) + 1
            )
            states[key] = (gradient, first, second)
        current["states"] = states
        self.before = (names, expected, error, current)

    def post(self, optimizer, args, kwargs):
        names, expected, gradient_error, current = self.before
        errors = [gradient_error, 0.0, 0.0, 0.0]
        hashed = {}
        for key, p in zip(names, optimizer.param_groups[0]["params"], strict=True):
            state = optimizer.state[p]
            require(int(state["step"]) == len(self.errors) + 1, "One actual Adam update")
            actual = (
                p.detach().numpy().copy(),
                state["exp_avg"].detach().numpy().copy(),
                state["exp_avg_sq"].detach().numpy().copy(),
            )
            errors[1] = max(errors[1], compare(actual[1], expected[key][1], "moment"))
            errors[2] = max(errors[2], compare(actual[2], expected[key][2], "moment"))
            errors[3] = max(errors[3], compare(actual[0], expected[key][0], "parameter"))
            hashed.update(
                {
                    f"{key}_{i}": a
                    for i, a in enumerate(
                        (current["parameters"][key], *current["states"][key], *actual)
                    )
                }
            )
        self.errors.append(errors)
        self.hashes.append(digest_arrays(hashed))
        self.before = None


@contextmanager
def observe_training(x, initial):
    """Attach instance hooks; restore factories even if an audit rejects a step."""
    from super_deli.models import tabular_neural as numerical

    torch = numerical._torch()
    original_network, original_init = numerical._network, torch.optim.Adam.__init__
    checks = StepChecks(x, initial)
    handles = []

    def network(d):
        result = original_network(d)
        actual = numerical._weights(result)
        require(tuple(actual) == tuple(initial), "Complete ordered initial weight schema")
        for key in actual:
            np.testing.assert_array_equal(actual[key], initial[key])
        return result

    def initialize(optimizer, *args, **kwargs):
        original_init(optimizer, *args, **kwargs)
        handles.extend(
            (
                optimizer.register_step_pre_hook(checks.pre),
                optimizer.register_step_post_hook(checks.post),
            )
        )

    numerical._network, torch.optim.Adam.__init__ = network, initialize
    try:
        yield checks
    finally:
        numerical._network, torch.optim.Adam.__init__ = original_network, original_init
        for handle in handles:
            handle.remove()


def verify_training(x, initial, expected, seed):
    from super_deli.models.tabular_neural import _train

    with FIT_LOCK, threadpool_limits(limits=1), observe_training(x, initial) as checked:
        actual, losses = _train(x, seed)
    require(len(checked.errors) == len(checked.hashes) == 500, "All 500 updates checked")
    for key in initial:
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(losses, expected["losses"])
    return {
        "errors": np.asarray(checked.errors),
        "step_sha256": np.asarray(checked.hashes),
    }
