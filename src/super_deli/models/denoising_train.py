"""Isolated CPU corruption stream; legacy ordinary training is untouched."""

from __future__ import annotations

import hashlib

import numpy as np

from super_deli.models.tabular_neural import NeuralFitError, _network, _torch, _weights


def train_denoising(x, seed, corruption_seed, *, _corruption_sd=0.15):
    """Fixed public objective. Zero corruption is reserved for legacy parity tests."""
    if _corruption_sd not in (0, 0.15):
        raise ValueError("Only the versioned corruption setting is supported")
    torch = _torch()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    losses, objectives, digests, checkpoints = [], [], [], {}
    network = None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            network = _network(x.shape[1])
            generator = torch.Generator(device="cpu").manual_seed(corruption_seed)
            target = torch.tensor(x, dtype=torch.float64, device="cpu")
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=0.001,
                weight_decay=0.0001,
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
                fused=False,
            )

            def loss(value):
                return ((network["decoder"](network["encoder"](value)) - target) ** 2).mean()

            def checkpoint(update):
                checkpoints.update(
                    {f"checkpoint_{update}_{k}": v for k, v in _weights(network).items()}
                )

            checkpoint(0)
            with torch.no_grad():
                losses.append(float(loss(target)))
            for update in range(500):
                optimizer.zero_grad(set_to_none=True)
                noise = (
                    torch.randn(
                        target.shape, generator=generator, dtype=torch.float64, device="cpu"
                    )
                    * _corruption_sd
                    if _corruption_sd
                    else torch.zeros_like(target)
                )
                digests.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
                objective = loss(target + noise)
                if not bool(torch.isfinite(objective)):
                    raise NeuralFitError("Nonfinite corrupted training objective")
                objectives.append(float(objective.detach()))
                objective.backward()
                if any(not bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
                    raise NeuralFitError("Nonfinite denoising gradient")
                optimizer.step()
                with torch.no_grad():
                    clean = float(loss(target))
                if not np.isfinite(clean):
                    raise NeuralFitError("Nonfinite denoising update")
                losses.append(clean)
                if update + 1 in (1, 499, 500):
                    checkpoint(update + 1)
            weights = _weights(network)
    except Exception as exc:
        raise NeuralFitError(
            str(exc),
            diagnostics={
                "seed": seed,
                "corruption_seed": corruption_seed,
                "completed_updates": max(0, len(losses) - 1),
                "losses": losses,
                "corrupted_objectives": objectives,
                "corruption_sha256": digests,
            },
            arrays={**checkpoints, **(_weights(network) if network is not None else {})},
        ) from exc
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)
    return (
        weights,
        np.asarray(losses),
        {
            **checkpoints,
            "corrupted_objectives": np.asarray(objectives),
            "corruption_sha256": np.asarray(digests, dtype="U64"),
        },
    )
