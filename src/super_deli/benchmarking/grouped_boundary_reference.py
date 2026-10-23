"""Analytic generating laws for simulation diagnosis, never fitted empirical nulls."""

from dataclasses import replace

from super_deli.models.grouped_config import BOUNDARY_CASES

CASES = ("independent_gaussian", "correlated_gaussian", *BOUNDARY_CASES)
REFERENCE = "known_grouped_gaussian_v1"


def parameters(case):
    import numpy as np

    from super_deli.models.grouped_conditional import psd_root

    if case not in CASES:
        raise ValueError("Only the five declared Gaussian simulation laws are supported")
    if case in BOUNDARY_CASES:
        basis = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
        eigenvalues = {
            "zero_conditional_gaussian": [0, 0, 0, 0],
            "rank_two_gaussian": [0.8, 0.4, 0, 0],
            "weak_conditional_gaussian": [0.8, 0.4, 0.02, 0.005],
        }[case]
        conditional = (basis * eigenvalues) @ basis.T
        regression = np.array(
            [[0.6, 0.2, 0, 0], [0, 0.6, 0.2, 0], [0, 0, 0.6, 0.2], [0.2, 0, 0, 0.6]]
        )
        ecology, within, mean = np.eye(4), 0.75 * np.eye(4), np.zeros(8)
    else:
        scale = np.linspace(0.5, 2, 8)
        between = (
            np.eye(8)
            if case == "independent_gaussian"
            else 0.6 ** np.abs(np.arange(8)[:, None] - np.arange(8)) * np.outer(scale, scale)
        )
        mean = np.zeros(8) if case == "independent_gaussian" else np.linspace(-2, 2, 8)
        within = (
            0.75 * np.eye(4)
            if case == "independent_gaussian"
            else 1.5 * 0.35 ** np.abs(np.arange(4)[:, None] - np.arange(4))
        )
        ecology = between[4:, 4:]
        regression = np.linalg.solve(ecology, between[4:, :4]).T
        conditional = between[:4, :4] - regression @ ecology @ regression.T
    factor = np.zeros((8, 8))
    factor[:4, :4] = psd_root(conditional)
    factor[4:, 4:] = np.linalg.cholesky(ecology)
    factor[:4, 4:] = regression @ factor[4:, 4:]
    return mean, factor @ factor.T, within, factor, np.linalg.cholesky(within)


def known_model(fitted, case):
    mean, covariance, within, factor, residual_factor = parameters(case)
    layout = {
        **fitted.layout,
        "fit": "Known generating parameters; diagnostic simulation reference, not empirical",
        "reference": REFERENCE,
        "reference_case": case,
    }
    layout.pop("diagnostics", None)
    return replace(
        fitted,
        mean=mean,
        covariance=covariance,
        within=within,
        layout=layout,
        sampling_factor=factor,
        within_sampling_factor=residual_factor,
    )


def load_model(directory):
    import numpy as np

    from super_deli.io.config import read_json
    from super_deli.models.grouped_null import GroupedGaussian

    with np.load(directory / "null_model.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    return GroupedGaussian(layout=read_json(directory / "layout.json"), **values)
