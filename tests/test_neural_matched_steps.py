"""Actual-path gradient/update checks, exact parity and intentional corruption."""
# ruff: noqa: E402

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from super_deli.benchmarking.denoising_numerics import gradients, reconstruction
from super_deli.benchmarking.matched_steps import verify_training
from super_deli.models import tabular_neural


@pytest.fixture(scope="module")
def retained():
    x = np.random.default_rng(109071).normal(size=(16, 4))
    x = (x - x.mean(0)) / x.std(0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(202)
        initial = tabular_neural._weights(tabular_neural._network(4))
    weights, losses = tabular_neural._train(x, 202)
    return x, initial, {**weights, "losses": losses}


def test_all_updates_and_exact_read_only_parity(retained):
    x, initial, expected = retained
    original = (tabular_neural._network, torch.optim.Adam.__init__)
    actual = verify_training(x, initial, expected, 202)
    assert actual["errors"].shape == (500, 4) and actual["step_sha256"].shape == (500,)
    assert np.max(actual["errors"]) < 1e-11
    assert (tabular_neural._network, torch.optim.Adam.__init__) == original


def test_independent_backprop_matches_finite_differences(retained):
    x, initial, _ = retained
    reference = gradients(x, x, initial)
    for key, value in initial.items():
        plus, minus = initial.copy(), initial.copy()
        plus[key], minus[key] = value.copy(), value.copy()
        plus[key].flat[0] += 1e-6
        minus[key].flat[0] -= 1e-6
        difference = (
            np.mean((reconstruction(x, plus) - x) ** 2)
            - np.mean((reconstruction(x, minus) - x) ** 2)
        ) / 2e-6
        assert difference == pytest.approx(reference[key].flat[0], abs=1e-8, rel=1e-4)


@pytest.mark.parametrize("corruption", ["gradient", "moment", "parameter"])
def test_bad_equations_are_detected_and_factories_restored(retained, monkeypatch, corruption):
    from super_deli.benchmarking import matched_steps

    x, initial, expected = retained
    original = (tabular_neural._network, torch.optim.Adam.__init__)
    if corruption == "gradient":

        def wrong(*args):
            result = gradients(*args)
            result["encoder_0_bias"] += 1e-4
            return result

        monkeypatch.setattr(matched_steps, "gradients", wrong)
    else:
        original_update = matched_steps.adam_update

        def wrong(*args):
            result = list(original_update(*args))
            result[1 if corruption == "moment" else 0] += 1e-4
            return tuple(result)

        monkeypatch.setattr(matched_steps, "adam_update", wrong)
    with pytest.raises(ValueError):
        verify_training(x, initial, expected, 202)
    assert (tabular_neural._network, torch.optim.Adam.__init__) == original
