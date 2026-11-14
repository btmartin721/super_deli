"""Fixed CPU categorical genotype reconstruction; no continuous-value substitution."""

import numpy as np
from scipy.special import logsumexp

from super_deli.models.tabular_neural import NeuralFitError, _torch, _weights, forward_numpy
from super_deli.models.tabular_neural import _network as continuous_network


def _network(d):
    # Preserve the established encoder/hidden initialization and replace only
    # the linear reconstruction head. No legacy module or optimizer is changed.
    torch = _torch()
    network = continuous_network(d)
    network["decoder"][4] = torch.nn.Linear(16, 3 * d, dtype=torch.float64, device="cpu")
    return network


def reconstruction(x, weights):
    z = forward_numpy(x, weights, "encoder")
    logits = forward_numpy(z, weights, "decoder").reshape(len(x), x.shape[1], 3)
    return logits - logsumexp(logits, axis=2, keepdims=True)


def cross_entropy(x, target, weights):
    logp = reconstruction(x, weights)
    return float(-np.take_along_axis(logp, np.asarray(target, dtype=int)[..., None], axis=2).mean())


def _train(x, target, seed):
    torch = _torch()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    network = None
    losses = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            network = _network(x.shape[1])
            tensor = torch.tensor(x, dtype=torch.float64, device="cpu")
            labels = torch.tensor(target.reshape(-1), dtype=torch.long, device="cpu")
            optimizer = torch.optim.Adam(
                network.parameters(), lr=0.001, weight_decay=0.0001, foreach=False, fused=False
            )

            def loss():
                logits = network["decoder"](network["encoder"](tensor)).reshape(-1, 3)
                return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")

            with torch.no_grad():
                losses.append(float(loss()))
            for _ in range(500):
                optimizer.zero_grad(set_to_none=True)
                objective = loss()
                if not bool(torch.isfinite(objective)):
                    raise NeuralFitError("Nonfinite genotype cross entropy")
                objective.backward()
                if any(not bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
                    raise NeuralFitError("Nonfinite categorical gradient")
                optimizer.step()
                with torch.no_grad():
                    value = float(loss())
                if not np.isfinite(value):
                    raise NeuralFitError("Nonfinite categorical update")
                losses.append(value)
            weights = _weights(network)
    except Exception as exc:
        raise NeuralFitError(
            str(exc),
            diagnostics=dict(seed=seed, completed_updates=max(0, len(losses) - 1), losses=losses),
            arrays=_weights(network) if network is not None else {},
        ) from exc
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)
    return weights, np.asarray(losses)
