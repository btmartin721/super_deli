"""Worker-local read-only checking during primary training, never a second optimizer."""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from super_deli.benchmarking.denoising_numerics import reconstruction
from super_deli.benchmarking.matched_steps import digest_arrays, observe_training
from super_deli.io.config import write_json


class NumericalAuditFailure(RuntimeError):
    """Abort dependent scientific work; never replace a rejected trajectory."""


@contextmanager
def retain_training(directory, *, paired, guard_state=None, module=None, record_times=False):
    from super_deli.benchmarking import matched_steps
    from super_deli.models import matched_neural, tabular_neural

    module = module if module is not None else matched_neural if paired else tabular_neural
    original_train, original_factory = module._train, tabular_neural._network
    records, saved = [], {}
    guard_state = {} if guard_state is None else guard_state

    def train(x, seed):
        if guard_state.get("failed"):
            raise NumericalAuditFailure("An earlier numerical guard stopped dependent work")
        started = time.monotonic() if record_times else None
        index, initial = len(records), {}
        record = {
            "fit": index,
            "n": len(x),
            "dimensions": x.shape[1],
            "seed": seed,
            "input_sha256": digest_arrays({"input": x}),
            "status": "started",
        }
        records.append(record)
        checked, losses, weights = None, None, None
        network_instance, reference_losses = {}, []
        normalized, comparison_counts = np.zeros((500, 3)), np.zeros((500, 3), dtype=int)
        original_compare = matched_steps.compare

        def compare(actual, expected, kind):
            error = original_compare(actual, expected, kind)
            index = len(checked.errors)
            column = ("gradient", "moment", "parameter").index(kind)
            relative, absolute = matched_steps.TOLERANCES[kind]
            ratio = float(
                np.max(np.abs(actual - expected) / (absolute + relative * np.abs(expected)))
            )
            normalized[index, column] = max(normalized[index, column], ratio)
            comparison_counts[index, column] += 1
            return error

        def factory(d):
            network = original_factory(d)
            network_instance["network"] = network
            initial.update(tabular_neural._weights(network))
            reference_losses.append(float(np.mean((reconstruction(x, initial) - x) ** 2)))
            return network

        tabular_neural._network = factory
        try:
            with observe_training(x, initial) as checked:
                matched_steps.compare = compare

                def guarded(callback, *, post=False):
                    def call(*args, **kwargs):
                        try:
                            result = callback(*args, **kwargs)
                            if post:
                                current = tabular_neural._weights(network_instance["network"])
                                reference_losses.append(
                                    float(np.mean((reconstruction(x, current) - x) ** 2))
                                )
                            return result
                        except Exception as exc:
                            raise NumericalAuditFailure(str(exc)) from exc

                    return call

                checked.pre, checked.post = guarded(checked.pre), guarded(checked.post, post=True)
                weights, losses = original_train(x, seed)
            if len(checked.errors) != 500 or len(checked.hashes) != 500:
                raise NumericalAuditFailure("Incomplete actual optimizer trajectory")
            try:
                np.testing.assert_allclose(losses, reference_losses, rtol=1e-10, atol=1e-12)
            except AssertionError as exc:
                raise NumericalAuditFailure(str(exc)) from exc
            record.update(
                status="complete",
                initial_sha256=digest_arrays(initial),
                final_sha256=digest_arrays(weights),
                loss_sha256=digest_arrays({"losses": losses}),
                max_errors=np.max(checked.errors, axis=0).tolist(),
            )
            return weights, losses
        except Exception as exc:
            cause, numerical = exc, False
            while cause is not None:
                numerical |= isinstance(cause, NumericalAuditFailure)
                cause = cause.__cause__
            record.update(status="failed", error=str(exc), numerical_guard=numerical)
            if numerical:
                guard_state["failed"] = True
                failure = NumericalAuditFailure(str(exc))
                failure.arrays = getattr(exc, "arrays", weights or {})
                failure.diagnostics = getattr(exc, "diagnostics", {})
                raise failure from exc
            raise
        finally:
            matched_steps.compare = original_compare
            tabular_neural._network = original_factory
            saved.update({f"fit_{index}_initial_{k}": v for k, v in initial.items()})
            saved[f"fit_{index}_reference_losses"] = np.asarray(reference_losses)
            if losses is not None:
                saved[f"fit_{index}_reported_losses"] = losses
            if checked is not None:
                saved[f"fit_{index}_errors"] = np.asarray(checked.errors)
                saved[f"fit_{index}_step_sha256"] = np.asarray(checked.hashes)
                saved[f"fit_{index}_normalized_errors"] = normalized
                saved[f"fit_{index}_comparison_counts"] = comparison_counts
                record["updates_checked"] = len(checked.errors)
            if started is not None:
                record["seconds"] = time.monotonic() - started

    module._train = train
    try:
        yield records
    finally:
        module._train, tabular_neural._network = original_train, original_factory
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / "optimizer-checks.npz", **saved)
        write_json(records, directory / "optimizer-checks.json")


@contextmanager
def retain_density(directory, events, *, paired, guard_state, module=None):
    from sklearn.cluster import HDBSCAN

    from super_deli.benchmarking.moons_hierarchy import retained_hierarchy
    from super_deli.models import matched_neural, tabular_neural

    module = module if module is not None else matched_neural if paired else tabular_neural
    original = module.fit_density_scores
    saved, records = {}, []

    def density(scores, config):
        captured = {}
        index = max(len(events) - 1, 0)

        def estimator(**kwargs):
            captured["estimator"] = HDBSCAN(**kwargs)
            return captured["estimator"]

        labels, strength = original(scores, config, estimator_class=estimator)
        try:
            linkage, condensed = retained_hierarchy(captured["estimator"], len(scores), 5)
            saved.update({f"fit_{index}_linkage": linkage, f"fit_{index}_condensed": condensed})
            records.append({"fit": index, "status": "complete"})
        except Exception as exc:
            guard_state["failed"] = True
            records.append({"fit": index, "status": "failed", "error": str(exc)})
            raise NumericalAuditFailure(str(exc)) from exc
        return labels, strength

    module.fit_density_scores = density
    try:
        yield records
    finally:
        module.fit_density_scores = original
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / "density-trace.npz", **saved)
        write_json(records, directory / "density-trace.json")
