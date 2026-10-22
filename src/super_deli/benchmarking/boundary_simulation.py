"""Independent laws with exact zero, reduced-rank and weak full-rank components."""

from super_deli.models.grouped_config import BOUNDARY_CASES


def simulate_boundary(case, n, seed, *, target_level="locality"):
    import numpy as np

    from super_deli.benchmarking.grouped_study import grouped_layout

    if (
        case not in BOUNDARY_CASES
        or type(n) is not int
        or n < 20
        or n % 4
        or target_level not in {"population", "locality"}
    ):
        raise ValueError(
            "Use a declared boundary case, target level and >=20 groups divisible by four"
        )
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, size=n)
    targets = np.repeat(np.arange(n), counts)
    ecology = rng.standard_normal((n, 4))
    # A fixed orthonormal basis avoids aligning every null direction with an axis.
    basis = (
        np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2
    )
    eigenvalues = {
        "zero_conditional_gaussian": [0, 0, 0, 0],
        "rank_two_gaussian": [0.8, 0.4, 0, 0],
        "weak_conditional_gaussian": [0.8, 0.4, 0.02, 0.005],
    }[case]
    conditional = rng.standard_normal((n, 4)) @ (basis * np.sqrt(eigenvalues)).T
    regression = np.array([[0.6, 0.2, 0, 0], [0, 0.6, 0.2, 0], [0, 0, 0.6, 0.2], [0.2, 0, 0, 0.6]])
    latent = ecology @ regression.T + conditional
    residual = rng.standard_normal((len(targets), 4)) * np.sqrt(0.75)
    return (
        {"morphological": latent[targets] + residual, "ecological": ecology},
        grouped_layout(counts, target_level),
        np.zeros(n, dtype=int),
    )
