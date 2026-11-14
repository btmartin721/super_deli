"""Read-only categorical backpropagation and actual-update Adam audits."""

from contextlib import contextmanager

import numpy as np
from scipy.special import logsumexp

from super_deli.benchmarking import matched_steps
from super_deli.models import genomic_autoencoder as numerical


def gradients(x, target, weights):
    layers = [(side, i) for side in ("encoder", "decoder") for i in range(3)]
    values = [x]
    for side, i in layers:
        value = values[-1] @ weights[f"{side}_{i}_weight"].T + weights[f"{side}_{i}_bias"]
        values.append(np.tanh(value) if i != 2 else value)
    logits = values[-1].reshape(*target.shape, 3)
    probability = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
    probability -= np.eye(3)[target]
    derivative = probability.reshape(len(x), -1) / target.size
    result = {}
    for index in reversed(range(6)):
        side, i = layers[index]
        if i != 2:
            derivative *= 1 - values[index + 1] ** 2
        result[f"{side}_{i}_weight"] = derivative.T @ values[index]
        result[f"{side}_{i}_bias"] = derivative.sum(0)
        derivative = derivative @ weights[f"{side}_{i}_weight"]
    return result


class CategoricalChecks(matched_steps.StepChecks):
    def __init__(self, x, target):
        super().__init__(x, {})
        self.target = target

    def pre(self, optimizer, args, kwargs):
        original = matched_steps.gradients
        matched_steps.gradients = lambda x, unused, weights: gradients(x, self.target, weights)
        try:
            super().pre(optimizer, args, kwargs)
        finally:
            matched_steps.gradients = original


@contextmanager
def observe_training(x, target):
    """Attach checks to the actual training call; never run a second optimizer."""
    torch = numerical._torch()
    factory, initialize = numerical._network, torch.optim.Adam.__init__
    checked = CategoricalChecks(x, target)
    handles = []

    def network(d):
        result = factory(d)
        checked.initial = numerical._weights(result)
        return result

    def start(optimizer, *args, **kwargs):
        initialize(optimizer, *args, **kwargs)
        handles.extend(
            (
                optimizer.register_step_pre_hook(checked.pre),
                optimizer.register_step_post_hook(checked.post),
            )
        )

    numerical._network, torch.optim.Adam.__init__ = network, start
    try:
        yield checked
    finally:
        numerical._network, torch.optim.Adam.__init__ = factory, initialize
        for handle in handles:
            handle.remove()
