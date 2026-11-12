"""Read-only checks on every actual residual-network gradient and Adam update."""

import math

import numpy as np

from super_deli.benchmarking.matched_steps import digest_arrays
from super_deli.benchmarking.residual_derivatives import evaluate
from super_deli.benchmarking.view_weight_observer import NumericalAuditFailure
from super_deli.models.residual_network import named_parameters, network_arrays


class ResidualObserver:
    def __init__(self):
        self.errors, self.hashes, self.losses, self.penalties = [], [], [], []
        self.before = None

    @staticmethod
    def _compare(actual, expected):
        try:
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError("Nonfinite independent residual audit")
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
        except (ValueError, AssertionError) as exc:
            raise NumericalAuditFailure(str(exc)) from exc
        return float(np.max(np.abs(actual - expected)))

    def pre(self, network, optimizer, x, y):
        group = optimizer.param_groups[0]
        if (
            len(optimizer.param_groups) != 1
            or any(
                group[k] != v
                for k, v in {
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0001,
                    "foreach": False,
                    "fused": False,
                    "amsgrad": False,
                    "maximize": False,
                }.items()
            )
            or group.get("decoupled_weight_decay", False)
        ):
            raise NumericalAuditFailure("Changed residual Adam settings")
        values = network_arrays(network)
        loss, _, gradients = evaluate(x, y, values, differentiate=True)
        if not self.losses:
            self.losses.append(loss)
            self.penalties.append(0.00005 * sum(np.sum(v * v) for v in values.values()))
        maximum, expected, hashed = 0.0, {}, {}
        for name, p in named_parameters(network):
            raw = p.grad.detach().numpy().copy()
            maximum = max(maximum, self._compare(raw, gradients[name]))
            state = optimizer.state[p]
            step = len(self.errors) + 1
            if int(state.get("step", 0)) != step - 1:
                raise NumericalAuditFailure("Non-contiguous residual Adam updates")
            m = state["exp_avg"].detach().numpy().copy() if state else np.zeros_like(raw)
            v = state["exp_avg_sq"].detach().numpy().copy() if state else np.zeros_like(raw)
            g = gradients[name] + 0.0001 * values[name]
            nm = 0.9 * m + 0.1 * g
            nv = 0.999 * v + 0.001 * g * g
            updated = values[name] - 0.001 * (nm / (1 - 0.9**step)) / (
                np.sqrt(nv / (1 - 0.999**step)) + 1e-8
            )
            expected[name] = (updated, nm, nv)
            hashed.update(
                {
                    name + suffix: a
                    for suffix, a in zip(
                        ("_before", "_gradient", "_m", "_v"), (values[name], raw, m, v), strict=True
                    )
                }
            )
        self.before = (expected, maximum, hashed)

    def post(self, network, optimizer, x, y):
        expected, maximum, hashed = self.before
        error = [maximum, 0.0, 0.0, 0.0]
        for name, p in named_parameters(network):
            state = optimizer.state[p]
            if int(state["step"]) != len(self.errors) + 1:
                raise NumericalAuditFailure("Missing residual optimizer step")
            actual = (
                p.detach().numpy().copy(),
                state["exp_avg"].detach().numpy().copy(),
                state["exp_avg_sq"].detach().numpy().copy(),
            )
            for j in range(3):
                error[j + 1] = max(error[j + 1], self._compare(actual[j], expected[name][j]))
                hashed[f"{name}_after_{j}"] = actual[j]
        self.errors.append(error)
        self.hashes.append(digest_arrays(hashed))
        values = network_arrays(network)
        loss, _ = evaluate(x, y, values)
        self.losses.append(loss)
        self.penalties.append(0.00005 * math.fsum(float(np.sum(v * v)) for v in values.values()))
        self.before = None

    def finish(self, arrays):
        try:
            if len(self.errors) != 500 or len(self.hashes) != 500:
                raise ValueError("Incomplete residual path audit")
            for key, values in (("losses", self.losses), ("regularization", self.penalties)):
                np.testing.assert_allclose(arrays[key], values, rtol=1e-10, atol=1e-10)
        except (ValueError, AssertionError) as exc:
            raise NumericalAuditFailure(str(exc)) from exc
        return {
            "errors": np.asarray(self.errors),
            "step_sha256": np.asarray(self.hashes),
            "reference_losses": np.asarray(self.losses),
            "reference_regularization": np.asarray(self.penalties),
        }
