"""Independent stable mixture density evaluation for no-refit audits."""


def log_density(scores, weights, means, covariances):
    import numpy as np
    from scipy.linalg import cholesky, solve_triangular
    from scipy.special import logsumexp

    terms = []
    for w, mean, cov in zip(weights, means, covariances, strict=True):
        factor = cholesky(cov, lower=True)
        z = solve_triangular(factor, (scores - mean).T, lower=True)
        terms.append(
            np.log(w)
            - 0.5
            * (
                scores.shape[1] * np.log(2 * np.pi)
                + 2 * np.log(np.diag(factor)).sum()
                + (z * z).sum(0)
            )
        )
    return logsumexp(np.column_stack(terms), axis=1)
